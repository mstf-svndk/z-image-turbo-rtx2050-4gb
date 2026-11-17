# Z-Image-Turbo on a 4 GB RTX 2050 (Ampere, sm_86) — Windows

Z-Image-Turbo (Q8_0 GGUF) text-to-image at 768x1344 on a **4 GB NVIDIA RTX 2050 laptop GPU**, with
stable-diffusion.cpp, on Windows. The engine is **built for compute capability 8.6 (sm_86 / Ampere)** by
GitHub Actions in this repository and shipped as a release asset.

This is a derivative of **[Sinirsiz Foto](https://github.com/alexturkey/z-image-turbo-pascal-4gb)** by Ali Haydar
Öztürk (Apache-2.0), which did the measuring, the patches and the 4 GB recipe on a GTX 1050 Ti (Pascal, sm_61).
What is different here: the engine is compiled for sm_86 instead of sm_61, the Pascal-only kernel patches are left
out, and setup/generate check the card's compute capability.

> **Measured on RTX 2050 (4 GB, driver 566.03, 32 GB RAM):** first verified `768x1344` run (`prompts/eiffel.txt`, seed 42,
> 8 steps, `--vae auto`) completed in **57.1 s** (`text_encode` 1.75 s, `sampling` 44.15 s, `vae_decode` 7.50 s via
> cuDNN 2-tile `768x1024`), GPU peak **2985 MiB**. See `docs/MEASUREMENTS.md`.

## What you need

- Windows 10/11 x64, an **NVIDIA RTX 2050 laptop (or any sm_86 card) with 4 GB of VRAM**
- NVIDIA driver that supports CUDA 12.8 (Windows driver **570.65 or newer**; the RTX 2050 ships on much newer ones)
- 32 GB RAM is comfortable; 16 GB is what the upstream project measured; less is untested
- Python 3.10+ (no pip packages are needed — standard library only)
- git, ~14 GB free disk on an SSD (~12.2 GB of downloads)

## Quick start

```bat
git clone https://github.com/mstf-svndk/z-image-turbo-rtx2050-4gb.git
cd z-image-turbo-rtx2050-4gb
python scripts/setup.py
python scripts/generate.py --prompt-file prompts/eiffel.txt
```

Or launch the local web interface (no extra packages needed):

```bat
python scripts/ui.py
```
(Or double-click `Baslat.cmd` on Windows — opens `http://127.0.0.1:7860` in your browser.)

`setup.py` first prints a **Preflight** block (your GPU, compute capability, driver, RAM, free disk). It downloads
the three model files (~11 GB), the engine zip and the CUDA 12.8 runtime, verifies every file with sha256, and
installs NVIDIA cuDNN (see below). The last line must be `SETUP_OK`.

`generate.py` writes `outputs/<time>_<name>/image.png`, `run.json` and `log.txt`, and prints
`IMAGE_OK <path>`. One picture at a time; close browsers and games first.

## What runs

| Part | File | Source |
|---|---|---|
| Diffusion model | `z_image_turbo-Q8_0.gguf` (6,577,440,704 B) | [leejet/Z-Image-Turbo-GGUF](https://huggingface.co/leejet/Z-Image-Turbo-GGUF) |
| Text encoder | `Qwen3-4B-Instruct-2507-Q8_0.gguf` (4,280,405,600 B) | [unsloth/Qwen3-4B-Instruct-2507-GGUF](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF) |
| VAE | `ae.safetensors` (335,304,388 B) | [Comfy-Org/Z-Image-Turbo](https://huggingface.co/Comfy-Org/Z-Image-Turbo) |
| Engine | `sd-cli.exe` + DLLs, stable-diffusion.cpp `32d516c` + ggml `d803c23` + 8 patches | the release of this repository, built for sm_86 |
| CUDA runtime | cudart / cublas / cublasLt 12.8 | unmodified NVIDIA redistributable zip |
| cuDNN | 9.10.2.21 runtime DLLs, downloaded from NVIDIA | NVIDIA's license, not stored here |

Fixed settings: `--cfg-scale 1.0 --steps 8 --sampling-method euler --scheduler simple --offload-to-cpu --mmap
--eager-load --flash-attn on`. The model is distilled for 8 steps; lowering them is not a speed fix.

One deviation from the upstream recipe is deliberate: the one-piece VAE attempt uses
`SD_VAE_FULL_MARGIN_MB=512` instead of the upstream 256, because cuDNN chooses its own engine and workspace on
sm_86 and that was never tuned for 4 GB Ampere. If the picture falls back to two tiles, that is the margin doing its
job, not an error (`run.json` → `stages.vae_path`).

## cuDNN (required for the GPU path)

The engine is compiled with the cuDNN F32 VAE branch, so `ggml-cuda.dll` **imports `cudnn64_9.dll` at load time**.
Without the cuDNN DLLs the CUDA backend cannot load and the run may fall back to the CPU, which takes hours.
`setup.py` therefore downloads cuDNN from NVIDIA's own server by default and copies only the eight runtime DLLs next
to `sd-cli.exe`; NVIDIA's license applies and the license file is written beside them
(`NVIDIA_cuDNN_LICENSE.txt`).

`python scripts/setup.py --no-cudnn` still exists, but it is only useful together with `--only models` (for example
to pre-download the 11 GB of models on a slow line). It is not a "lighter engine" option.

## Limits (read before you rely on it)

- **Single-run RTX 2050 measurement:** `768x1344` (`prompts/eiffel.txt`, seed 42, 8 steps, `--vae auto`) completed
  in 57.1 s (2985 MiB VRAM peak). Other sizes/prompts and repeat-run pixel determinism have not been measured here yet.
- **sm_86 only.** The engine carries compute-86 kernels; another architecture gives
  `no kernel image is available for execution on the device`. Use the upstream repository or build your own engine
  (`docs/BUILD.md`).
- **cuDNN is required** for the CUDA backend to load (the engine imports `cudnn64_9.dll`); `setup.py` installs it
  from NVIDIA's server. `--no-cudnn` is for `--only models` downloads, not for running pictures.
- **4 GB VRAM is tight.** The engine keeps the weights in system RAM and moves the network to the GPU in 36 slices;
  the VAE tries one piece, then two large tiles, then small tiles. Free RAM drops to a few hundred MiB during a run.
- **Sizes:** 768x1344, 1344x768 and 1024x1024 are the upstream-tested sizes. Bigger sizes print a warning and may
  run out of memory.
- **Prompt fidelity is partial:** a picture does not always follow every detail of the prompt; text inside pictures
  and real landmarks were not evaluated.
- **Unofficial.** Not affiliated with Tongyi-MAI, leejet or NVIDIA. Every picture is synthetic; you are responsible
  for what you publish.

## Troubleshooting

`docs/TROUBLESHOOTING.md` — driver too old, `no kernel image`, cuDNN not loading, out of memory, slow first run.
`docs/MEASUREMENTS.md` — what is measured, what is not, and how to add your run.

## License and credit

- Scripts and documents here: [Apache-2.0](LICENSE), with [NOTICE](NOTICE) keeping the upstream authorship.
- `patches/` are MIT, like the code they modify (stable-diffusion.cpp: leejet; ggml: its authors).
- Model files are not stored here; they are downloaded from Hugging Face under their own licenses.
- cuDNN and the CUDA runtime are NVIDIA's property, under NVIDIA's licenses.

---

## English summary

Derivative of [alexturkey/z-image-turbo-pascal-4gb](https://github.com/alexturkey/z-image-turbo-pascal-4gb)
(Apache-2.0) rebuilt for **sm_86 / Ampere / 4 GB** (reference card: RTX 2050 laptop). The engine is built in CI
(`.github/workflows/build-engine.yml`) and published as a release asset; `scripts/setup.py` verifies every download
with sha256 and checks the GPU's compute capability; `scripts/generate.py` runs the upstream 4 GB recipe unchanged.
**First RTX 2050 measurement:** `57.1 s` for one `768x1344` image (`2985 MiB` VRAM peak, cuDNN tiled VAE). Windows,
NVIDIA, Python 3.10+, ~14 GB disk.
