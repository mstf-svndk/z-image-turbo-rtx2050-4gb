"""Tests for the sm_86 specifics: engine arch scan, lock names, preflight warning (no GPU, no network)."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


setup = load("setup")
LOCK = json.loads((ROOT / "models.lock.json").read_text(encoding="utf-8"))


class EngineArch(unittest.TestCase):
    def test_finds_embedded_arch_strings(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "ggml-cuda.dll").write_bytes(b"\x00ggml-cuda compute_86 build\x00")
            (t / "sd-cli.exe").write_bytes(b"MZ sm_86 sm_61")
            (t / "notes.txt").write_bytes(b"sm_120")
            self.assertEqual(setup.engine_arch(t), ["sm_61", "sm_86"])

    def test_empty_folder(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(setup.engine_arch(t), [])

    def test_zip_marker_is_required_per_entry(self):
        # every engine entry must be a github release asset of the named file
        for e in LOCK["engine"]:
            self.assertTrue(e["url"].endswith("/" + e["name"]), e["name"])

    def test_engine_zip_is_named_sm86(self):
        names = [e["name"] for e in LOCK["engine"]]
        self.assertTrue(any("sm86" in n for n in names), names)
        self.assertFalse(any("sm61" in n for n in names), names)


class Preflight(unittest.TestCase):
    def run_preflight(self, report):
        lines = []
        old = setup.machine_report
        setup.machine_report = lambda: report
        try:
            with tempfile.TemporaryDirectory() as t:
                setup.preflight(t, t, out=lines.append)
        finally:
            setup.machine_report = old
        return "\n".join(lines)

    def test_sm86_card_gets_no_arch_warning(self):
        text = self.run_preflight({"gpus": [{"name": "NVIDIA GeForce RTX 2050", "memory_total_mib": 4096,
                                             "driver": "581.29", "compute_cap": "8.6"}],
                                   "ram_total_mib": 32000})
        self.assertIn("Compute  : 8.6", text)
        self.assertNotIn("will not run on compute capability", text)

    def test_other_card_gets_a_warning(self):
        text = self.run_preflight({"gpus": [{"name": "NVIDIA GeForce GTX 1050 Ti", "memory_total_mib": 4096,
                                             "driver": "581.29", "compute_cap": "6.1"}],
                                   "ram_total_mib": 32000})
        self.assertIn("will not run on compute capability 6.1", text)

    def test_unknown_compute_cap_does_not_crash_or_warn(self):
        text = self.run_preflight({"gpus": [{"name": "NVIDIA GeForce RTX 2050", "memory_total_mib": 4096,
                                             "driver": "470.1", "compute_cap": None}],
                                   "ram_total_mib": 32000})
        self.assertIn("Compute  : unknown", text)
        self.assertNotIn("will not run", text)

    def test_no_gpu_does_not_crash(self):
        text = self.run_preflight({"gpus": [], "ram_total_mib": 32000})
        self.assertIn("nvidia-smi not found", text)


class ArchGate(unittest.TestCase):
    def run_main(self, cc):
        lock = {"models": [{"name": "a.gguf", "url": "https://example.invalid/a.gguf", "bytes": 1,
                            "sha256": "0" * 64}],
                "engine": [{"name": "e.zip", "url": "https://example.invalid/e.zip", "bytes": 1,
                            "sha256": "1" * 64}]}
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "lock.json").write_text(json.dumps(lock), encoding="utf-8")
            old = setup.machine_report
            setup.machine_report = lambda: {"gpus": [{"name": "some card", "memory_total_mib": 4096,
                                                      "driver": "x", "compute_cap": cc}], "ram_total_mib": 32000}
            try:
                code = setup.main(["--lock", str(t / "lock.json"), "--models-dir", str(t / "m"),
                                   "--bin-dir", str(t / "b")])
            finally:
                setup.machine_report = old
            downloaded = sorted(p.name for p in (t / "m").glob("*")) if (t / "m").exists() else []
            return code, downloaded

    def test_verdicts(self):
        self.assertEqual(setup.arch_verdict({"gpus": [{"compute_cap": "8.6"}]}), "ok")
        self.assertEqual(setup.arch_verdict({"gpus": [{"compute_cap": "6.1"}]}), "wrong")
        self.assertIsNone(setup.arch_verdict({"gpus": [{"compute_cap": None}]}))
        self.assertIsNone(setup.arch_verdict({"gpus": []}))

    def test_wrong_card_blocks_before_download(self):
        code, downloaded = self.run_main("6.1")
        self.assertEqual(code, 3)
        self.assertEqual(downloaded, [])


class Recipe(unittest.TestCase):
    """The fixed recipe must not drift when adapting the engine."""

    def test_recipe_constants(self):
        gen = load("generate")
        self.assertEqual(gen.STEPS, 8)
        self.assertEqual(gen.CFG, "1.0")
        self.assertEqual(gen.DEFAULT_SEED, 42)
        cmd = gen.build_command("bin/sd-cli.exe", "models", "p.txt", "o.png", 768, 1344, 42, "legacy")
        joined = " ".join(cmd)
        for needle in ("--offload-to-cpu", "--mmap", "--eager-load", "--flash-attn on",
                       "--sampling-method euler", "--scheduler simple", "--cfg-scale 1.0", "--steps 8"):
            self.assertIn(needle, joined)


if __name__ == "__main__":
    unittest.main()
