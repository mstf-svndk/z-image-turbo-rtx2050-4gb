#!/usr/bin/env python3
"""Generate one image with Z-Image-Turbo (Q8_0) through sd-cli on a 4 GB Ampere GPU (sm_86).

Reference card: NVIDIA RTX 2050 laptop, 4 GB, Windows. The engine in this repository is built for sm_86 only;
the recipe is the one measured upstream on a 4 GB Pascal card and is kept unchanged here.

Runs the fixed recipe (README.md / docs/AGENTS.md): 8 steps, cfg 1.0, euler + simple, no negative prompt,
--offload-to-cpu --mmap --eager-load --flash-attn on. Quality settings are fixed on purpose.
Python 3.10+, standard library only.

VAE decode (the last stage) has three modes, chosen with --vae:
  auto    (default) cuDNN float32 convolution; the whole image is decoded in one piece when it fits in GPU memory,
          otherwise in two large tiles, otherwise in the small default tiles. Needs the cuDNN DLLs in the bin folder.
  tiled   cuDNN float32 convolution, always two large tiles (never tries the single piece).
  legacy  the v0.1.0 path: no cuDNN, automatic small tiles. Used automatically when the cuDNN DLLs are missing.
The three modes give slightly different pixels (see docs/MEASUREMENTS.md); run.json records which path was taken.

Examples (Windows):
  python scripts/generate.py --prompt "A lighthouse on a cliff at dawn, realistic photograph"
  python scripts/generate.py --prompt-file prompts/eiffel.txt --width 1344 --height 768 --seed 7

Output: <out>/<YYYYmmdd_HHMMSS>_<name>/image.png + run.json (seconds, stages, VRAM peak from nvidia-smi, command).
Last line on success: IMAGE_OK <path>.
The upstream numbers (GTX 1050 Ti 4 GB, sm_61) are 193-228 s per 768x1344 picture; this card and this engine have
not been measured yet. Expect the process to hold about 11 GB of RAM. Close browsers and other heavy programs first.
"""
import argparse
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILES = {
    "diffusion": "z_image_turbo-Q8_0.gguf",
    "vae": "ae.safetensors",
    "llm": "Qwen3-4B-Instruct-2507-Q8_0.gguf",
}
STEPS = 8
CFG = "1.0"
DEFAULT_SEED = 42
DEFAULT_W, DEFAULT_H = 768, 1344           # portrait; landscape is 1344x768
TESTED_PIXELS = 1024 * 1024                # tested: 768x1344, 1344x768, 1024x1024
VAE_MODES = ("auto", "tiled", "legacy")
CUDNN_DLL = "cudnn64_9.dll"
MAX_TILE_PIXELS = 1024 * 768               # largest VAE tile area measured to fit on 4 GB (two tiles per image)
TILE_QUANTUM = 16
# Environment for the engine (all read by the patched engine, see patches/README.md).
CUDNN_ENV = {
    "GGML_CUDA_CUDNN_CONV2D": "1",         # VAE convolutions through cuDNN, float32
    "GGML_CUDA_CUDNN_LOG": "1",            # one log line per cuDNN convolution (run.json: cudnn_used)
    "SD_VAE_TILE_FLOOR": "1",              # if the large tile does not fit, fall back once to the default small tile
    "SD_VAE_VRAM_MARGIN_MB": "512",        # GPU memory safety margin during the VAE stage only (default 1024)
    "SD_CAPACITY_LOG": "1",                # log the memory gate (needed / available) so run.json can record it
}
FULL_ENV = {
    "SD_VAE_TRY_FULL": "1",                 # try the untiled decode first, then the requested tile
    "SD_VAE_FULL_MARGIN_MB": "512",         # stricter than the upstream 256: cuDNN picks its own sm_86 engine and
                                            # workspace, which the Pascal tuning did not account for
}
LOW_RAM_MIB = 8000
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def fwd(p):
    return str(p).replace("\\", "/")


def vae_tile(width, height, max_pixels=MAX_TILE_PIXELS):
    """Large VAE tile for an image: (tile_w, tile_h) in pixels, or None when the image needs no tiling.

    Rule: keep the short side whole and cut the long side so that tile area <= max_pixels (multiple of 16).
    A square image gets a full-width tile. The rule looks only at width and height, never at free memory,
    so the same prompt, seed and size always ask for the same tile.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if width * height <= max_pixels:
        return None
    wide = width > height
    long_side, short_side = (width, height) if wide else (height, width)
    t = int(max_pixels / short_side) // TILE_QUANTUM * TILE_QUANTUM
    if t >= long_side:
        return None
    if t < 256:                             # very long and thin: tile both axes
        ts = max(32, int((max_pixels * short_side / long_side) ** 0.5) // TILE_QUANTUM * TILE_QUANTUM)
        tl = max(32, int(max_pixels / ts) // TILE_QUANTUM * TILE_QUANTUM)
        ts, tl = min(ts, short_side), min(tl, long_side)
        return (tl, ts) if wide else (ts, tl)
    return (t, short_side) if wide else (short_side, t)


def find_cudnn(bin_dir):
    """Folder that holds the cuDNN DLLs, or None."""
    for p in sorted(Path(bin_dir).rglob(CUDNN_DLL)):
        return p.parent
    return None


def resolve_vae_mode(requested, cudnn_dir):
    """Effective VAE mode: without the cuDNN DLLs every mode becomes 'legacy'.

    (Passing --conv-direct without cuDNN would select a slow direct convolution kernel: measured 386 s for one decode.)
    """
    return requested if cudnn_dir is not None else "legacy"


def vae_args(mode, width, height):
    """Extra sd-cli arguments for the VAE mode."""
    if mode == "legacy":
        return []
    args = ["--conv-direct", "auto,vae=on"]
    tile = vae_tile(width, height)
    if tile is not None:
        args += ["--vae-tiling", "--vae-tile-size", "%dx%d" % tile]
    return args


def engine_env(mode, base=None):
    """Environment for sd-cli. Engine switches set by the caller are removed first, so a run is reproducible."""
    src = os.environ if base is None else base
    env = {k: v for k, v in src.items() if k not in CUDNN_ENV and k not in FULL_ENV}
    if mode in ("auto", "tiled"):
        env.update(CUDNN_ENV)
    if mode == "auto":
        env.update(FULL_ENV)
    return env


def build_command(sd_cli, models_dir, prompt_file, out_png, width, height, seed, vae_mode="legacy"):
    """The fixed sd-cli command line (list of strings). The tests pin this recipe."""
    m = Path(models_dir)
    return [
        str(sd_cli),
        "--diffusion-model", fwd(m / MODEL_FILES["diffusion"]),
        "--vae", fwd(m / MODEL_FILES["vae"]),
        "--llm", fwd(m / MODEL_FILES["llm"]),
        "--prompt-file", fwd(prompt_file),
        "--cfg-scale", CFG, "--steps", str(STEPS),
        "--sampling-method", "euler", "--scheduler", "simple",
        "--offload-to-cpu", "--mmap", "--eager-load", "--flash-attn", "on",
        "-s", str(seed), "-W", str(width), "-H", str(height),
        "-v", "-o", fwd(out_png),
    ] + vae_args(vae_mode, width, height)


def make_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt", help="prompt text (natural language; the example prompts in this repo are English)")
    g.add_argument("--prompt-file", help="UTF-8 text file with the prompt")
    ap.add_argument("--width", type=int, default=DEFAULT_W, help="pixels, multiple of 16 (default %d)" % DEFAULT_W)
    ap.add_argument("--height", type=int, default=DEFAULT_H, help="pixels, multiple of 16 (default %d)" % DEFAULT_H)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="default %d" % DEFAULT_SEED)
    ap.add_argument("--out", default=str(ROOT / "outputs"), help="output root (default: ./outputs)")
    ap.add_argument("--name", help="short name in the output folder (default: prompt file name, or 'image')")
    ap.add_argument("--models-dir", default=str(ROOT / "models"), help="default: ./models")
    ap.add_argument("--bin-dir", default=str(ROOT / "bin"), help="default: ./bin (sd-cli.exe, CUDA and cuDNN DLLs)")
    ap.add_argument("--vae", choices=VAE_MODES, default="auto",
                    help="VAE decode mode (default auto; see the text above). 'legacy' is the v0.1.0 decode path")
    return ap


def check_size(width, height):
    """Return (errors, warnings) for the requested size."""
    errors, warnings = [], []
    for label, v in (("width", width), ("height", height)):
        if v < 256 or v > 2048 or v % 16:
            errors.append("--%s %d: must be a multiple of 16 between 256 and 2048" % (label, v))
    if not errors and width * height > TESTED_PIXELS:
        warnings.append("%dx%d is larger than the tested sizes (768x1344, 1344x768, 1024x1024); not tested on 4 GB, "
                        "it may run out of memory" % (width, height))
    return errors, warnings


def slug(text):
    s = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return s[:40] or "image"


def find_sd_cli(bin_dir):
    bin_dir = Path(bin_dir)
    if (bin_dir / "sd-cli.exe").is_file():
        return bin_dir / "sd-cli.exe"
    for p in sorted(bin_dir.rglob("sd-cli.exe")):
        return p
    return None


def free_ram_mib():
    if os.name != "nt":
        return None

    class MS(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                    ("avail", ctypes.c_ulonglong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                    ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
    ms = MS()
    ms.l = ctypes.sizeof(MS)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
        return None
    return int(ms.avail // (1024 * 1024))


def gpu_used_mib():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10, creationflags=NO_WINDOW)
        return int(r.stdout.strip().splitlines()[0]) if r.returncode == 0 and r.stdout.strip() else None
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return None


def gpu_info():
    """Name, compute capability, driver and total memory of the first NVIDIA GPU, or None."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "--query-gpu=name,compute_cap,driver_version,memory.total",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10, creationflags=NO_WINDOW)
        if r.returncode != 0 or not r.stdout.strip():
            return None
        row = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        return {"name": row[0], "compute_cap": row[1] if len(row) > 1 else None,
                "driver": row[2] if len(row) > 2 else None,
                "memory_total_mib": int(row[3]) if len(row) > 3 and row[3].isdigit() else None}
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return None


class Monitor(threading.Thread):
    """Samples GPU memory (nvidia-smi) and free RAM every 2 s while sd-cli runs."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop_event = threading.Event()
        self.vram_peak = None
        self.ram_free_min = None
        self.baseline = gpu_used_mib()

    def run(self):
        while not self.stop_event.is_set():
            v = gpu_used_mib()
            if v is not None and (self.vram_peak is None or v > self.vram_peak):
                self.vram_peak = v
            r = free_ram_mib()
            if r is not None and (self.ram_free_min is None or r < self.ram_free_min):
                self.ram_free_min = r
            self.stop_event.wait(2.0)

    def finish(self):
        self.stop_event.set()
        self.join(timeout=10)


def parse_log(text):
    """Pull stage times (seconds) out of the sd-cli verbose log."""
    def sec(pattern):
        m = re.search(pattern + r" completed(?: in|, taking) ([0-9.]+)\s*(ms|s)\b", text)
        if not m:
            return None
        v = float(m.group(1))
        return round(v / 1000 if m.group(2) == "ms" else v, 2)
    out = {
        "text_encode_s": sec(r"get_learned_condition"),
        "sampling_s": sec(r"sampling"),
        "vae_decode_s": sec(r"decode_first_stage"),
        "generate_total_s": sec(r"generate_image"),
        "cuda": "ggml_cuda_init: found" in text,
        "vae_tiling_fallback": "retrying with spatial tiling" in text,
    }
    out.update(parse_vae(text))
    return out


_RE_PATH = re.compile(r"VAE path:\s*(full|tile|floor|retry)\s+(\d+)x(\d+)\s+\(attempt\s+(\d+)\)")
_RE_TILE = re.compile(r"VAE decode tile size:\s*(\d+)x(\d+)\s*pixels")
_RE_GATE = re.compile(r"CAPACITY need ([\d.]+) MB \(pending ([\d.]+) \+ missing ([\d.]+) \+ margin ([\d.]+)\)"
                      r" / available ([\d.]+) MB / residual (-?[\d.]+) MB / (FITS|SHORT)")
_RE_CUDNN = re.compile(r"cudnn conv2d: x\[.*?\] algo (\d+) ws (\d+) KiB")


def parse_vae(text):
    """What the VAE stage did, from the verbose log.

    vae_path: 'full' (one piece), 'tile' (the large tile that was requested), 'floor' / 'retry' (smaller tiles after
    a memory shortfall), or None when the engine did not say (legacy mode).
    vae_quality_drop is True only for 'floor' / 'retry' or an out-of-memory retry: the picture is valid but was
    decoded in small tiles. Falling from 'full' to 'tile' is a planned step, reported as vae_full_fell_back.
    vae_gate_residual_mib: GPU memory left over at the first VAE memory gate (negative when that try did not fit).
    """
    path = _RE_PATH.findall(text)
    tiles = _RE_TILE.findall(text)
    cudnn = _RE_CUDNN.findall(text)
    start = [m.end() for m in re.finditer(r"decoding \d+ latents", text)]
    gate = _RE_GATE.findall(text[start[-1]:]) if start else []
    fell_back = "VAE untiled decode did not fit" in text
    oom = ("VAE decode ran out of memory" in text or "Reducing VAE decode tiles" in text
           or "falling back to the default-equivalent tile" in text)
    vae_path = path[-1][0] if path else None
    return {
        "vae_path": vae_path,
        "vae_path_size": [int(path[-1][1]), int(path[-1][2])] if path else None,
        "vae_tile": [int(tiles[-1][0]), int(tiles[-1][1])] if tiles else None,
        "vae_full_fell_back": fell_back,
        "vae_quality_drop": bool(oom or vae_path in ("floor", "retry")),
        "vae_gate_residual_mib": float(gate[0][5]) if gate else None,
        "cudnn_used": bool(cudnn) and {int(a) for a, _ in cudnn} == {1},
        "cudnn_workspace_max_kib": max((int(k) for _, k in cudnn), default=None),
    }


def main(argv=None):
    a = make_parser().parse_args(argv)
    errors, warnings = check_size(a.width, a.height)
    if errors:
        for e in errors:
            print("ERROR: %s. Fix: use the tested size (no --width/--height) or see docs/TROUBLESHOOTING.md." % e)
        return 2

    sd_cli = find_sd_cli(a.bin_dir)
    if sd_cli is None:
        print("ERROR: sd-cli.exe not found in %s. Run: python scripts/setup.py" % a.bin_dir)
        return 2
    gpu = gpu_info()
    if gpu and gpu.get("compute_cap") and gpu["compute_cap"].replace(".", "") != "86":
        print("WARNING: %s reports compute capability %s; this engine carries sm_86 kernels only and will not run "
              "on it. See docs/TROUBLESHOOTING.md." % (gpu["name"], gpu["compute_cap"]))
    for f in MODEL_FILES.values():
        if not (Path(a.models_dir) / f).is_file():
            print("ERROR: missing model file %s. Run: python scripts/setup.py" % (Path(a.models_dir) / f))
            return 2
    if a.prompt_file:
        try:
            prompt = Path(a.prompt_file).read_text(encoding="utf-8").strip()
        except OSError as e:
            print("ERROR: cannot read prompt file (%s). Fix: pass an existing UTF-8 text file, for example "
                  "--prompt-file prompts/eiffel.txt" % e.__class__.__name__)
            return 2
    else:
        prompt = a.prompt.strip()
    if not prompt:
        print("ERROR: the prompt is empty. Fix: pass --prompt \"...\" or --prompt-file prompts/eiffel.txt")
        return 2
    for w in warnings:
        print("WARNING: " + w)
    ram = free_ram_mib()
    if ram is not None and ram < LOW_RAM_MIB:
        print("WARNING: only %d MiB RAM free; the run uses about 10.5 GB of process memory (measured). Close other programs." % ram)

    name = slug(a.name or (Path(a.prompt_file).stem if a.prompt_file else "image"))
    out = (Path(a.out) / ("%s_%s" % (time.strftime("%Y%m%d_%H%M%S"), name))).resolve()
    out.mkdir(parents=True, exist_ok=True)
    prompt_path = out / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    image = out / "image.png"
    cudnn_dir = find_cudnn(a.bin_dir)
    vae_mode = resolve_vae_mode(a.vae, cudnn_dir)
    if vae_mode != a.vae:
        print("WARNING: %s not found in %s. The engine's CUDA backend imports cudnn64_9.dll, so it may not load "
              "without the cuDNN DLLs (the run can fall back to the CPU and take hours). Fix: python scripts/setup.py "
              "--only cudnn. Continuing with --vae legacy." % (CUDNN_DLL, a.bin_dir))
    cmd = build_command(sd_cli, a.models_dir, prompt_path, image, a.width, a.height, a.seed, vae_mode)
    (out / "command.txt").write_text(subprocess.list2cmdline(cmd) + "\n", encoding="utf-8")
    env = engine_env(vae_mode)
    dll_dirs = [str(sd_cli.parent)]
    if cudnn_dir is not None and cudnn_dir != sd_cli.parent:
        dll_dirs.append(str(cudnn_dir))
    env["PATH"] = os.pathsep.join(dll_dirs + [env.get("PATH", "")])

    print("%dx%d seed=%d steps=%d -> %s" % (a.width, a.height, a.seed, STEPS, out))
    print("Started (VAE mode: %s). Expected 2 to 4 minutes on a 4 GB Ampere laptop card (this engine and card are "
          "not measured yet; upstream measured 193-228 s on a GTX 1050 Ti). Log: %s"
          % (vae_mode, out / "log.txt"), flush=True)
    mon = Monitor()
    mon.start()
    t0 = time.time()
    try:
        with open(out / "log.txt", "w", encoding="utf-8", errors="replace") as log:
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, creationflags=NO_WINDOW)
            try:
                rc = p.wait()
            except KeyboardInterrupt:
                p.terminate()
                print("Interrupted; a generation cannot resume, start it again.")
                return 130
    finally:
        mon.finish()
    secs = round(time.time() - t0, 1)

    try:
        stages = parse_log((out / "log.txt").read_text(encoding="utf-8", errors="replace"))
    except OSError:
        stages = {}
    ok = rc == 0 and image.is_file() and image.stat().st_size > 0
    pixel_sha = None
    if ok:
        try:                                              # pixels only: the PNG text chunk differs between VAE modes
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import compare
            pixel_sha = compare.pixel_sha256(image)[0]
        except Exception as e:                            # never fail a finished image over the hash
            print("WARNING: could not hash the image pixels (%s)" % e.__class__.__name__)
    try:
        engine_zips = json.loads((Path(a.bin_dir) / ".engine_installed.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        engine_zips = None
    info = {
        "result": "OK" if ok else "FAILED", "exit_code": rc, "seconds": secs,
        "width": a.width, "height": a.height, "seed": a.seed, "steps": STEPS, "cfg": float(CFG),
        "gpu": gpu, "engine_arch": "sm_86",
        "vae_mode": vae_mode, "vae_mode_requested": a.vae,
        "engine_env": {k: env[k] for k in sorted(set(CUDNN_ENV) | set(FULL_ENV)) if k in env},
        "image_pixel_sha256": pixel_sha, "engine_zips_sha256": engine_zips,
        "stages": stages,
        "gpu_memory_used_peak_mib": mon.vram_peak, "gpu_memory_used_before_mib": mon.baseline,
        "ram_free_min_mib": mon.ram_free_min,
        "command": cmd,
    }
    (out / "run.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    if not ok:
        print("FAILED: sd-cli exit code %s. Read %s and docs/TROUBLESHOOTING.md." % (rc, out / "log.txt"))
        return 1
    if vae_mode != "legacy" and stages.get("vae_quality_drop"):
        print("NOTE: the VAE stage ran short of GPU memory and used small tiles (see docs/TROUBLESHOOTING.md).")
    elif vae_mode == "auto" and stages.get("vae_full_fell_back"):
        print("NOTE: the one-piece VAE decode did not fit in GPU memory; two large tiles were used instead.")
    elif vae_mode != "legacy" and not stages.get("cudnn_used"):
        print("NOTE: the log has no cuDNN convolution lines; the VAE may have run on the slow path.")
    print("%.0f s" % secs)
    if pixel_sha:
        print("PIXEL_SHA256 %s" % pixel_sha)
    print("IMAGE_OK %s" % image)
    return 0


if __name__ == "__main__":
    sys.exit(main())
