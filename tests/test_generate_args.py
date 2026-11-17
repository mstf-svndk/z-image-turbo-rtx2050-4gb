"""Tests for scripts/generate.py: fixed recipe, argument checks, log parsing (no GPU, no models)."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("generate", ROOT / "scripts" / "generate.py")
gen = importlib.util.module_from_spec(spec)
sys.modules["generate"] = gen
spec.loader.exec_module(gen)


def cmd(**kw):
    a = dict(sd_cli="bin/sd-cli.exe", models_dir="models", prompt_file="p.txt", out_png="o/image.png",
             width=768, height=1344, seed=42)
    a.update(kw)
    return gen.build_command(**a)


def value(c, flag):
    return c[c.index(flag) + 1]


class Recipe(unittest.TestCase):
    def test_fixed_flags(self):
        c = cmd()
        self.assertEqual(value(c, "--cfg-scale"), "1.0")
        self.assertEqual(value(c, "--steps"), "8")
        self.assertEqual(value(c, "--sampling-method"), "euler")
        self.assertEqual(value(c, "--scheduler"), "simple")
        self.assertEqual(value(c, "--flash-attn"), "on")
        for f in ("--offload-to-cpu", "--mmap", "--eager-load"):
            self.assertIn(f, c)
        for f in ("--conv-direct", "--vae-tiling", "--vae-tile-size"):   # default command is the legacy VAE path
            self.assertNotIn(f, c)
        self.assertNotIn("--negative-prompt", c)
        self.assertNotIn("--negative-prompt-file", c)

    def test_models_and_size(self):
        c = cmd(width=1344, height=768, seed=7)
        self.assertTrue(value(c, "--diffusion-model").endswith("z_image_turbo-Q8_0.gguf"))
        self.assertTrue(value(c, "--vae").endswith("ae.safetensors"))
        self.assertTrue(value(c, "--llm").endswith("Qwen3-4B-Instruct-2507-Q8_0.gguf"))
        self.assertEqual((value(c, "-W"), value(c, "-H"), value(c, "-s")), ("1344", "768", "7"))
        self.assertTrue(all(isinstance(x, str) for x in c))

    def test_no_personal_strings(self):
        import re
        text = (ROOT / "scripts" / "generate.py").read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"(?i)[A-Z]:[\\/]+Users", text))


class Vae(unittest.TestCase):
    def test_tile_rule(self):
        self.assertEqual(gen.vae_tile(768, 1344), (768, 1024))
        self.assertEqual(gen.vae_tile(1344, 768), (1024, 768))
        self.assertEqual(gen.vae_tile(1024, 1024), (1024, 768))
        self.assertIsNone(gen.vae_tile(768, 768))            # fits in one piece
        self.assertIsNone(gen.vae_tile(1024, 768))
        for w, h in ((768, 1344), (1344, 768), (1024, 1024), (1024, 1536), (2048, 512), (512, 2048)):
            tw, th = gen.vae_tile(w, h)
            self.assertLessEqual(tw * th, gen.MAX_TILE_PIXELS)
            self.assertEqual((tw % 16, th % 16), (0, 0))
            self.assertTrue(tw <= w and th <= h)

    def test_mode_arguments(self):
        self.assertEqual(gen.vae_args("legacy", 768, 1344), [])
        for mode in ("auto", "tiled"):
            c = cmd(vae_mode=mode)
            self.assertEqual(value(c, "--conv-direct"), "auto,vae=on")
            self.assertIn("--vae-tiling", c)
            self.assertEqual(value(c, "--vae-tile-size"), "768x1024")
        c = cmd(vae_mode="auto", width=768, height=768)
        self.assertIn("--conv-direct", c)
        self.assertNotIn("--vae-tiling", c)

    def test_mode_needs_cudnn(self):
        self.assertEqual(gen.resolve_vae_mode("auto", None), "legacy")
        self.assertEqual(gen.resolve_vae_mode("tiled", None), "legacy")
        self.assertEqual(gen.resolve_vae_mode("auto", Path("bin")), "auto")
        with tempfile.TemporaryDirectory() as t:
            self.assertIsNone(gen.find_cudnn(t))
            (Path(t) / "sub").mkdir()
            (Path(t) / "sub" / gen.CUDNN_DLL).write_bytes(b"x")
            self.assertEqual(gen.find_cudnn(t), Path(t) / "sub")

    def test_environment(self):
        base = {"PATH": "x", "SD_VAE_TRY_FULL": "1", "GGML_CUDA_CUDNN_CONV2D": "1", "SD_VAE_VRAM_MARGIN_MB": "64"}
        self.assertEqual(gen.engine_env("legacy", base), {"PATH": "x"})   # caller's engine switches never leak in
        tiled = gen.engine_env("tiled", base)
        self.assertEqual(tiled["GGML_CUDA_CUDNN_CONV2D"], "1")
        self.assertEqual(tiled["SD_VAE_VRAM_MARGIN_MB"], "512")
        self.assertNotIn("SD_VAE_TRY_FULL", tiled)
        self.assertEqual(gen.engine_env("auto", base)["SD_VAE_TRY_FULL"], "1")
        self.assertEqual(base["SD_VAE_VRAM_MARGIN_MB"], "64")             # input not modified


class Args(unittest.TestCase):
    def test_defaults(self):
        a = gen.make_parser().parse_args(["--prompt", "x"])
        self.assertEqual((a.width, a.height, a.seed, a.vae), (768, 1344, 42, "auto"))
        with self.assertRaises(SystemExit):
            gen.make_parser().parse_args(["--prompt", "x", "--vae", "fast"])

    def test_prompt_required_and_exclusive(self):
        p = gen.make_parser()
        with self.assertRaises(SystemExit):
            p.parse_args([])
        with self.assertRaises(SystemExit):
            p.parse_args(["--prompt", "a", "--prompt-file", "b"])

    def test_size_checks(self):
        self.assertEqual(gen.check_size(768, 1344), ([], []))
        self.assertEqual(gen.check_size(1344, 768), ([], []))
        self.assertEqual(gen.check_size(1024, 1024), ([], []))
        errors, _ = gen.check_size(770, 1344)
        self.assertEqual(len(errors), 1)
        errors, _ = gen.check_size(128, 1344)
        self.assertEqual(len(errors), 1)
        errors, warnings = gen.check_size(1024, 1536)
        self.assertEqual((errors, len(warnings)), ([], 1))

    def test_slug(self):
        self.assertEqual(gen.slug("Eiffel Tower!"), "eiffel_tower")
        self.assertEqual(gen.slug("***"), "image")

    def test_main_rejects_bad_size_and_missing_engine(self):
        self.assertEqual(gen.main(["--prompt", "x", "--width", "770"]), 2)
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(gen.main(["--prompt", "x", "--bin-dir", t, "--models-dir", t, "--out", t]), 2)

    def test_help_exits_zero(self):
        with self.assertRaises(SystemExit) as cm:
            gen.main(["--help"])
        self.assertEqual(cm.exception.code, 0)


class LogParse(unittest.TestCase):
    LOG = ("ggml_cuda_init: found 1 CUDA devices\n"
           "[I] loading tensors completed, taking 0.21s (read: 0.00s)\n"
           "[I] get_learned_condition completed, taking 13.94s --- image.cpp:662\n"
           "[I] sampling completed, taking 203.95s --- image.cpp:1037\n"
           "[W] VAE decode ran out of memory; retrying with spatial tiling --- x\n"
           "[I] decode_first_stage completed, taking 34.17s --- image.cpp:757\n"
           "[I] generate_image completed in 252.18s --- image.cpp:1227\n")

    def test_stages(self):
        s = gen.parse_log(self.LOG)
        self.assertEqual(s["text_encode_s"], 13.94)
        self.assertEqual(s["sampling_s"], 203.95)
        self.assertEqual(s["vae_decode_s"], 34.17)
        self.assertEqual(s["generate_total_s"], 252.18)
        self.assertTrue(s["cuda"])
        self.assertTrue(s["vae_tiling_fallback"])

    def test_empty(self):
        s = gen.parse_log("")
        self.assertIsNone(s["sampling_s"])
        self.assertFalse(s["cuda"])
        self.assertIsNone(s["vae_path"])
        self.assertFalse(s["cudnn_used"])
        self.assertFalse(s["vae_quality_drop"])

    GATE = ("[W] CAPACITY need %s MB (pending 2816.0 + missing 94.6 + margin %s) / available 3211.4 MB"
            " / residual %s MB / %s --- model_manager.cpp:1900\n")
    CUDNN = "cudnn conv2d: x[1,128,1024,1024] algo 1 ws 32868 KiB\n"

    def test_vae_full(self):
        log = ("[I] decoding 1 latents --- image.cpp:691\n" + self.GATE % ("3166.6", "256.0", "44.8", "FITS")
               + "[W] VAE path: full 1024x1024 (attempt 1) --- diffusion_engine.cpp:3139\n" + self.CUDNN)
        s = gen.parse_vae(log)
        self.assertEqual((s["vae_path"], s["vae_path_size"], s["vae_tile"]), ("full", [1024, 1024], None))
        self.assertEqual(s["vae_gate_residual_mib"], 44.8)
        self.assertTrue(s["cudnn_used"])
        self.assertEqual(s["cudnn_workspace_max_kib"], 32868)
        self.assertFalse(s["vae_full_fell_back"] or s["vae_quality_drop"])

    def test_vae_full_falls_back_to_tile(self):
        log = ("[I] decoding 1 latents --- image.cpp:691\n" + self.GATE % ("3310.6", "400.0", "-99.2", "SHORT")
               + "[W] VAE untiled decode did not fit; falling back to the requested tile --- x\n"
               + "[V] VAE decode tile size: 1024x768 pixels (128x96 latent) --- vae.hpp:348\n"
               + self.GATE % ("2718.6", "512.0", "492.8", "FITS")
               + "[W] VAE path: tile 1024x768 (attempt 2) --- diffusion_engine.cpp:3139\n" + self.CUDNN)
        s = gen.parse_vae(log)
        self.assertEqual((s["vae_path"], s["vae_tile"]), ("tile", [1024, 768]))
        self.assertTrue(s["vae_full_fell_back"])
        self.assertFalse(s["vae_quality_drop"])              # a planned step, not a quality drop
        self.assertEqual(s["vae_gate_residual_mib"], -99.2)

    def test_vae_floor_is_quality_drop(self):
        log = ("[I] decoding 1 latents --- image.cpp:691\n"
               "[W] VAE tile does not fit; falling back to the default-equivalent tile 384x672 --- x\n"
               "[W] VAE path: floor 384x672 (attempt 2) --- x\n")
        self.assertTrue(gen.parse_vae(log)["vae_quality_drop"])


if __name__ == "__main__":
    unittest.main()
