"""Tests for models.lock.json and scripts/setup.py (no network, no GPU, no real models)."""
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
import zipfile
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
ENTRIES = LOCK["models"] + LOCK["engine"]


class LockFile(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(len(LOCK["models"]), 3)
        self.assertEqual(len(LOCK["engine"]), 2)

    def test_fields(self):
        for e in ENTRIES:
            for k in ("name", "url", "bytes", "sha256", "license"):
                self.assertIn(k, e, e.get("name"))
            self.assertRegex(e["sha256"], r"^[0-9a-f]{64}$", e["name"])
            self.assertGreater(e["bytes"], 0)
            self.assertTrue(e["url"].startswith("https://"), e["name"])
            self.assertTrue(e["url"].endswith("/" + e["name"]), e["name"])

    def test_no_floating_references(self):
        for e in ENTRIES:
            self.assertNotRegex(e["url"], r"/(main|master|latest)(/|$)", e["name"])
        for m in LOCK["models"]:
            self.assertRegex(m["revision"], r"^[0-9a-f]{40}$", m["name"])
            self.assertIn("/resolve/" + m["revision"] + "/", m["url"], m["name"])
        for e in LOCK["engine"]:
            self.assertRegex(e["url"], r"/releases/download/v\d+\.\d+\.\d+/", e["name"])

    def test_unique_names_and_hashes(self):
        self.assertEqual(len({e["name"] for e in ENTRIES}), len(ENTRIES))
        self.assertEqual(len({e["sha256"] for e in ENTRIES}), len(ENTRIES))

    def test_model_names_match_generate(self):
        gen = load("generate")
        self.assertEqual({m["name"] for m in LOCK["models"]}, set(gen.MODEL_FILES.values()))

    def test_no_personal_strings(self):
        text = (ROOT / "models.lock.json").read_text(encoding="utf-8")
        self.assertNotRegex(text, r"(?i)[A-Z]:[\\/]+Users")


class Download(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.payload = bytes(range(256)) * 2000
        self.src = self.dir / "src.bin"
        self.src.write_bytes(self.payload)
        self.url = self.src.as_uri()
        self.sha = hashlib.sha256(self.payload).hexdigest()

    def tearDown(self):
        self.tmp.cleanup()

    def test_download_and_verify(self):
        dest = self.dir / "out" / "file.bin"
        self.assertTrue(setup.download(self.url, dest, len(self.payload), self.sha))
        self.assertEqual(dest.read_bytes(), self.payload)
        self.assertFalse(dest.with_name("file.bin.part").exists())
        entry = {"name": "file.bin", "bytes": len(self.payload), "sha256": self.sha}
        self.assertTrue(setup.is_verified(dest, entry))

    def test_hash_mismatch_removes_part(self):
        dest = self.dir / "bad.bin"
        with self.assertRaises(RuntimeError):
            setup.download(self.url, dest, len(self.payload), "0" * 64)
        self.assertFalse(dest.exists())
        self.assertFalse(dest.with_name("bad.bin.part").exists())

    def test_wrong_size_not_verified(self):
        dest = self.dir / "short.bin"
        dest.write_bytes(self.payload[:100])
        entry = {"name": "x", "bytes": len(self.payload), "sha256": self.sha}
        self.assertFalse(setup.is_verified(dest, entry))

    def test_complete_part_file_is_finished(self):
        dest = self.dir / "done.bin"
        dest.with_name("done.bin.part").write_bytes(self.payload)
        self.assertTrue(setup.download(self.url, dest, len(self.payload), self.sha))
        self.assertEqual(dest.read_bytes(), self.payload)


class Zip(unittest.TestCase):
    def test_extract_and_find(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            z = t / "a.zip"
            with zipfile.ZipFile(z, "w") as f:
                f.writestr("pkg/sd-cli.exe", b"x")
                f.writestr("cudart64_12.dll", b"y")
            out = t / "bin"
            setup.safe_extract(z, out)
            self.assertEqual(setup.find_sd_cli(out), out / "pkg" / "sd-cli.exe")
            self.assertTrue(setup.has_cudart(out))

    def test_zip_slip_refused(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            z = t / "evil.zip"
            with zipfile.ZipFile(z, "w") as f:
                f.writestr("../evil.txt", b"x")
            with self.assertRaises(RuntimeError):
                setup.safe_extract(z, t / "bin")
            self.assertFalse((t / "evil.txt").exists())


class EndToEnd(unittest.TestCase):
    """Whole setup.main() against a tiny fake lock file with file:// URLs (no network)."""

    def entry(self, d, name, data):
        p = d / name
        p.write_bytes(data)
        return {"name": name, "url": p.as_uri(), "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(), "license": "test"}

    def test_full_setup_then_check(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            src = t / "src"
            src.mkdir()
            with zipfile.ZipFile(src / "eng.zip", "w") as z:
                z.writestr("sd-cli.exe", b"x")
            with zipfile.ZipFile(src / "cu.zip", "w") as z:
                z.writestr("cudart64_12.dll", b"y")
            lock = {"models": [self.entry(src, "a.gguf", b"a" * 5000)],
                    "engine": [self.entry(src, "eng.zip", (src / "eng.zip").read_bytes()),
                               self.entry(src, "cu.zip", (src / "cu.zip").read_bytes())]}
            (t / "lock.json").write_text(json.dumps(lock), encoding="utf-8")
            args = ["--lock", str(t / "lock.json"), "--models-dir", str(t / "m"), "--bin-dir", str(t / "b")]
            self.assertEqual(setup.main(args), 0)
            self.assertTrue((t / "m" / "a.gguf").is_file())
            self.assertTrue((t / "b" / "sd-cli.exe").is_file())
            self.assertFalse((t / "b" / "_downloads").exists())
            self.assertEqual(setup.main(args + ["--check"]), 0)
            (t / "m" / "a.gguf").write_bytes(b"broken")
            self.assertEqual(setup.main(args + ["--check"]), 1)
            self.assertEqual(setup.main(args), 0)          # re-downloads the bad file
            self.assertEqual((t / "m" / "a.gguf").read_bytes(), b"a" * 5000)


class Cudnn(unittest.TestCase):
    """Optional cuDNN step against a tiny fake archive (no network). Consent is never assumed."""

    def make(self, t):
        import zlib
        src = t / "src"
        src.mkdir()
        dll = {"cudnn64_9.dll": b"d" * 300, "cudnn_ops64_9.dll": b"o" * 700}
        with zipfile.ZipFile(src / "eng.zip", "w") as z:
            z.writestr("sd-cli.exe", b"x")
        with zipfile.ZipFile(src / "cu.zip", "w") as z:
            z.writestr("cudart64_12.dll", b"y")
        with zipfile.ZipFile(src / "cudnn.zip", "w") as z:
            for n, b in dll.items():
                z.writestr("arch/bin/" + n, b)
            z.writestr("arch/LICENSE", b"license text")
            z.writestr("arch/include/cudnn.h", b"header: must not be extracted")

        def entry(name):
            data = (src / name).read_bytes()
            return {"name": name, "url": (src / name).as_uri(), "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(), "license": "test"}
        cud = entry("cudnn.zip")
        cud.update({"version": "9.test", "license_url": "https://example.invalid/license",
                    "license_member": "arch/LICENSE", "license_save_as": "NVIDIA_cuDNN_LICENSE.txt",
                    "members": [{"path": "arch/bin/" + n, "bytes": len(b), "crc32": "%08x" % zlib.crc32(b)}
                                for n, b in dll.items()]})
        model_src = src / "a.gguf"
        model_src.write_bytes(b"a" * 100)
        lock = {"models": [entry("a.gguf")], "engine": [entry("eng.zip"), entry("cu.zip")], "cudnn": cud}
        (t / "lock.json").write_text(json.dumps(lock), encoding="utf-8")
        return ["--lock", str(t / "lock.json"), "--models-dir", str(t / "m"), "--bin-dir", str(t / "b")], lock

    def test_no_cudnn_flag_skips(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            args, lock = self.make(t)
            self.assertEqual(setup.main(args + ["--no-cudnn"]), 0)
            self.assertFalse(setup.cudnn_installed(t / "b", lock["cudnn"]))
            self.assertFalse(list((t / "b").rglob(setup.CUDNN_RECORD)))   # not installed, no record
            self.assertEqual(setup.main(args + ["--check"]), 0)               # --check never downloads
            self.assertFalse(setup.cudnn_installed(t / "b", lock["cudnn"]))

    class FakeStdin:
        """Stands in for stdin so the tests do not depend on how they were started."""
        def __init__(self, tty, text=""):
            import io
            self.tty, self.buf = tty, io.StringIO(text)
        def isatty(self):
            return self.tty
        def readline(self, *a):
            return self.buf.readline()

    def setUp(self):
        self._stdin = sys.stdin
        sys.stdin = self.FakeStdin(False)          # default: no terminal
    def tearDown(self):
        sys.stdin = self._stdin

    def run_lines(self, args, stdin=None):
        import contextlib, io
        if stdin is not None:
            sys.stdin = stdin
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = setup.main(args)
        sys.stdin = self.FakeStdin(False)
        return code, [x for x in buf.getvalue().splitlines() if x.strip()]

    def test_markers(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            args, lock = self.make(t)
            code, lines = self.run_lines(args + ["--only", "models"])
            self.assertEqual((code, lines[-1]), (0, "SETUP_PART_OK models"))   # a part is not the whole setup
            self.assertNotIn("SETUP_OK", lines)
            code, lines = self.run_lines(args + ["--no-cudnn"])
            self.assertEqual((code, lines[-1]), (0, "SETUP_OK"))
            self.assertTrue(any(x.startswith("CUDNN_SKIPPED (not installed)") for x in lines), lines)
            self.assertFalse(any("NVIDIA's license" in x for x in lines))      # nothing downloaded, nothing announced

    def test_installed_by_default_and_announced(self):
        for stdin in (self.FakeStdin(False), self.FakeStdin(True), self.FakeStdin(True, "no" + chr(10))):
            with tempfile.TemporaryDirectory() as t:
                t = Path(t)
                args, lock = self.make(t)
                code, lines = self.run_lines(args, stdin)                      # no flag: never asks, never waits
                self.assertEqual((code, lines[-1]), (0, "SETUP_OK"))
                self.assertIn("CUDNN_OK", lines)
                told = [x for x in lines if x.startswith("Downloading NVIDIA cuDNN under NVIDIA's license: ")]
                self.assertEqual(len(told), 1, lines)
                self.assertIn(lock["cudnn"]["license_url"], told[0])
                self.assertIn("Use --no-cudnn to skip.", told[0])
                self.assertLess(lines.index(told[0]), lines.index("CUDNN_OK"))  # said before it happens
                b = t / "b"
                self.assertTrue(setup.cudnn_installed(b, lock["cudnn"]))
                rec = json.loads((b / setup.CUDNN_RECORD).read_text(encoding="utf-8"))
                self.assertEqual(rec["mode"], "default_download_disclosed")
                self.assertNotIn("accepted_by", rec)                           # nobody was asked: no such claim
                code, lines = self.run_lines(args)                             # second run: nothing to download
                self.assertEqual((code, lines[-1]), (0, "SETUP_OK"))
                self.assertFalse(any("NVIDIA's license" in x for x in lines))

    def test_installed_with_flag(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            args, lock = self.make(t)
            self.assertEqual(setup.main(args + ["--accept-cudnn-license"]), 0)
            b = t / "b"
            self.assertTrue(setup.cudnn_installed(b, lock["cudnn"]))
            self.assertEqual((b / "cudnn64_9.dll").read_bytes(), b"d" * 300)
            self.assertEqual((b / "NVIDIA_cuDNN_LICENSE.txt").read_bytes(), b"license text")
            self.assertFalse(list(b.rglob("cudnn.h")))        # only the listed members leave the archive
            rec = json.loads((b / setup.CUDNN_RECORD).read_text(encoding="utf-8"))
            self.assertEqual(rec["license_sha256"], hashlib.sha256(b"license text").hexdigest())
            self.assertEqual(rec["archive_sha256"], lock["cudnn"]["sha256"])
            self.assertEqual(rec["mode"], setup.CUDNN_MODE)   # the old flag changes nothing
            self.assertRegex(rec["installed_utc"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
            self.assertFalse((b / "_downloads").exists())
            self.assertEqual(setup.main(args + ["--check"]), 0)

    def test_wrong_member_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            args, lock = self.make(t)
            lock["cudnn"]["members"][0]["crc32"] = "00000000"
            (t / "lock.json").write_text(json.dumps(lock), encoding="utf-8")
            self.assertEqual(setup.main(args + ["--accept-cudnn-license"]), 1)


class Cli(unittest.TestCase):
    def test_help_exits_zero(self):
        with self.assertRaises(SystemExit) as cm:
            setup.main(["--help"])
        self.assertEqual(cm.exception.code, 0)

    def test_missing_lock_is_error(self):
        self.assertEqual(setup.main(["--lock", str(ROOT / "nope.json")]), 2)


if __name__ == "__main__":
    unittest.main()
