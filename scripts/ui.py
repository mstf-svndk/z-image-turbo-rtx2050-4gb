#!/usr/bin/env python3
"""Simple local web UI for Z-Image-Turbo on RTX 2050 4 GB (Windows, Python 3.10+, stdlib only).

Starts a local HTTP server (default http://127.0.0.1:7860) and opens the browser.
Uses scripts/generate.py under the hood so the fixed 4 GB recipe is always preserved.
"""
import argparse
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate  # noqa: E402

PROMPT_LABELS = {
    "eiffel": "Eyfel Kulesi",
    "istanbul_aerial": "İstanbul Havadan",
    "turkish_tea": "Türk Çayı",
    "fisherman": "Balıkçı Portresi",
    "village": "Yeşil Köy",
}

_RE_STEP = re.compile(r"\|\s*(\d+)/(\d+)\s*-")
_RE_OUT = re.compile(r"->\s*(.+)$")


def load_presets():
    pdir = ROOT / "prompts"
    presets = []
    if not pdir.is_dir():
        return presets
    for p in sorted(pdir.glob("*.txt")):
        try:
            text = p.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text:
            presets.append({
                "id": p.stem,
                "label": PROMPT_LABELS.get(p.stem, p.stem.replace("_", " ").title()),
                "text": text,
            })
    return presets


def setup_status(models_dir, bin_dir):
    sd_cli = generate.find_sd_cli(bin_dir)
    cudnn_dir = generate.find_cudnn(bin_dir)
    missing_models = [f for f in generate.MODEL_FILES.values() if not (Path(models_dir) / f).is_file()]
    return {
        "ready": sd_cli is not None and not missing_models,
        "sd_cli": sd_cli is not None,
        "cudnn": cudnn_dir is not None,
        "missing_models": missing_models,
    }


def parse_live_stage(log_path):
    """Inspect sd-cli log.txt while running and return (stage_label, progress_pct)."""
    if not log_path or not Path(log_path).is_file():
        return "Başlatılıyor...", 3
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "Başlatılıyor...", 3
    if "save result image" in text or "generate_image completed" in text:
        return "Görsel kaydediliyor...", 99
    if "decoding 1 latents" in text or "cudnn conv2d:" in text:
        return "VAE çözülüyor (görüntü oluşturuluyor)...", 90
    if "generating image:" in text:
        matches = _RE_STEP.findall(text)
        if matches:
            cur, total = int(matches[-1][0]), max(1, int(matches[-1][1]))
            pct = 18 + int((cur / total) * 68)
            return f"Örnekleme yapılıyor (Adım {cur}/{total})...", pct
        return "Örnekleme hazırlanıyor (Adım 0/8)...", 16
    if "qwen3" in text:
        return "Prompt kodlanıyor (Qwen3-4B)...", 10
    if "loading tensors" in text or "loading diffusion model" in text:
        return "Modeller belleğe yükleniyor...", 6
    return "Motor başlatılıyor...", 4


class RunnerState:
    """Thread-safe controller ensuring only one 4 GB GPU generation runs at a time."""

    def __init__(self, models_dir, bin_dir, out_dir):
        self.models_dir = Path(models_dir)
        self.bin_dir = Path(bin_dir)
        self.out_dir = Path(out_dir)
        self.lock = threading.Lock()
        self.running = False
        self.proc = None
        self.started_at = None
        self.current_out = None
        self.current_prompt = ""
        self.current_meta = {}
        self.last_result = None
        self.last_error = None
        self.cancelled = False

    def start(self, prompt, width, height, seed, vae, name):
        with self.lock:
            if self.running:
                return False, "Şu anda başka bir görsel üretiliyor. RTX 2050 (4 GB VRAM) için tek seferde bir işlem çalıştırılır."
            self.running = True
            self.cancelled = False
            self.proc = None
            self.started_at = time.time()
            self.current_out = None
            self.current_prompt = prompt
            self.current_meta = {"width": width, "height": height, "seed": seed, "vae": vae}
            self.last_error = None

        t = threading.Thread(
            target=self._run_worker,
            args=(prompt, width, height, seed, vae, name),
            daemon=True,
        )
        t.start()
        return True, None

    def cancel(self):
        with self.lock:
            if not self.running or self.proc is None:
                return False
            self.cancelled = True
            try:
                self.proc.terminate()
            except OSError:
                pass
            return True

    def _run_worker(self, prompt, width, height, seed, vae, name):
        cmd = [
            sys.executable, "-u", str(SCRIPT_DIR / "generate.py"),
            "--prompt", prompt,
            "--width", str(width),
            "--height", str(height),
            "--seed", str(seed),
            "--vae", vae,
            "--models-dir", str(self.models_dir),
            "--bin-dir", str(self.bin_dir),
            "--out", str(self.out_dir),
        ]
        if name:
            cmd += ["--name", name]

        stdout_lines = []
        rc = -1
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=generate.NO_WINDOW,
            )
            with self.lock:
                self.proc = proc
            if proc.stdout is not None:
                for line in proc.stdout:
                    line = line.rstrip("\r\n")
                    stdout_lines.append(line)
                    m = _RE_OUT.search(line)
                    if m and not self.current_out:
                        candidate = Path(m.group(1).strip())
                        with self.lock:
                            self.current_out = candidate
            rc = proc.wait()
        except Exception as e:
            stdout_lines.append(f"Runner error: {e}")

        with self.lock:
            out_path = self.current_out
            was_cancelled = self.cancelled
            elapsed = round(time.time() - (self.started_at or time.time()), 1)
            self.proc = None
            self.running = False

        if was_cancelled:
            with self.lock:
                self.last_error = "Üretim kullanıcı tarafından iptal edildi."
            return

        run_json = (out_path / "run.json") if out_path else None
        if rc == 0 and run_json and run_json.is_file():
            try:
                info = json.loads(run_json.read_text(encoding="utf-8"))
                info["folder"] = out_path.name
                info["image_url"] = f"/outputs/{out_path.name}/image.png"
                info["prompt"] = prompt
                with self.lock:
                    self.last_result = info
                    self.last_error = None
                return
            except (OSError, ValueError):
                pass

        err_msg = "\n".join(stdout_lines[-6:]) or f"sd-cli çıkış kodu: {rc} ({elapsed} sn)"
        with self.lock:
            self.last_error = err_msg

    def status(self):
        with self.lock:
            running = self.running
            started = self.started_at
            cur_out = self.current_out
            meta = dict(self.current_meta)
            last_result = self.last_result
            last_error = self.last_error

        stage, pct = ("Hazır", 0)
        if running:
            stage, pct = parse_live_stage((cur_out / "log.txt") if cur_out else None)
        return {
            "running": running,
            "elapsed_s": round(time.time() - started, 1) if (running and started) else 0.0,
            "stage": stage,
            "progress": pct,
            "meta": meta,
            "vram_used_mib": generate.gpu_used_mib(),
            "ram_free_mib": generate.free_ram_mib(),
            "last_result": last_result,
            "last_error": last_error,
        }


def list_history(out_dir, limit=30):
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return []
    items = []
    for folder in sorted(out_dir.iterdir(), key=lambda p: p.name, reverse=True):
        if not folder.is_dir():
            continue
        rj = folder / "run.json"
        img = folder / "image.png"
        if not rj.is_file() or not img.is_file():
            continue
        try:
            data = json.loads(rj.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("result") != "OK":
            continue
        prompt_file = folder / "prompt.txt"
        prompt_text = ""
        if prompt_file.is_file():
            try:
                prompt_text = prompt_file.read_text(encoding="utf-8").strip()
            except OSError:
                pass
        items.append({
            "folder": folder.name,
            "image_url": f"/outputs/{folder.name}/image.png",
            "seconds": data.get("seconds"),
            "width": data.get("width"),
            "height": data.get("height"),
            "seed": data.get("seed"),
            "vae_mode": data.get("vae_mode"),
            "vae_path": (data.get("stages") or {}).get("vae_path"),
            "vram_peak_mib": data.get("gpu_memory_used_peak_mib"),
            "stages": data.get("stages"),
            "prompt": prompt_text,
        })
        if len(items) >= limit:
            break
    return items


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Z-Image-Turbo — RTX 2050 4 GB</title>
<style>
  :root {
    --bg: #0d1117;
    --panel: #161b22;
    --panel-2: #1c222b;
    --border: #30363d;
    --text: #e6edf3;
    --muted: #8b949e;
    --accent: #3b82f6;
    --accent-hover: #2563eb;
    --ok: #22c55e;
    --warn: #f59e0b;
    --err: #ef4444;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
  }
  header {
    border-bottom: 1px solid var(--border);
    background: var(--panel);
    padding: 14px 24px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }
  .brand h1 {
    margin: 0;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.2px;
  }
  .brand span {
    font-size: 12px;
    color: var(--muted);
  }
  .stats {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    font-size: 12px;
  }
  .pill {
    background: var(--panel-2);
    border: 1px solid var(--border);
    padding: 5px 10px;
    border-radius: 999px;
    color: var(--text);
  }
  .pill.ok { border-color: rgba(34, 197, 94, 0.45); color: #86efac; }
  .pill.warn { border-color: rgba(245, 158, 11, 0.45); color: #fcd34d; }
  .banner {
    margin: 16px 24px 0;
    padding: 12px 16px;
    border-radius: 10px;
    border: 1px solid rgba(239, 68, 68, 0.45);
    background: rgba(239, 68, 68, 0.12);
    color: #fca5a5;
    display: none;
    font-size: 14px;
  }
  main {
    max-width: 1320px;
    margin: 0 auto;
    padding: 20px 24px 40px;
    display: grid;
    grid-template-columns: 430px 1fr;
    gap: 20px;
    align-items: start;
  }
  @media (max-width: 960px) {
    main { grid-template-columns: 1fr; }
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px;
  }
  .card h2 {
    margin: 0 0 12px;
    font-size: 15px;
    font-weight: 600;
    color: var(--text);
  }
  label {
    display: block;
    font-size: 12px;
    font-weight: 600;
    color: var(--muted);
    margin: 12px 0 6px;
  }
  textarea, input, select {
    width: 100%;
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
    font: inherit;
    font-size: 14px;
  }
  textarea { min-height: 130px; resize: vertical; }
  textarea:focus, input:focus, select:focus {
    outline: none;
    border-color: var(--accent);
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 8px;
  }
  .chip {
    background: var(--panel-2);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 12px;
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
  }
  .chip:hover { border-color: var(--accent); background: #232b36; }
  .row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }
  .seed-row {
    display: flex;
    gap: 8px;
  }
  .seed-row input { flex: 1; }
  .btn {
    border: 1px solid var(--border);
    background: var(--panel-2);
    color: var(--text);
    border-radius: 8px;
    padding: 10px 14px;
    font: inherit;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    text-decoration: none;
  }
  .btn:hover:not(:disabled) { border-color: #484f58; background: #252c36; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-primary {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
    width: 100%;
    padding: 12px 16px;
    margin-top: 16px;
    font-size: 15px;
  }
  .btn-primary:hover:not(:disabled) { background: var(--accent-hover); }
  .btn-danger {
    background: rgba(239, 68, 68, 0.15);
    border-color: rgba(239, 68, 68, 0.5);
    color: #fca5a5;
    width: 100%;
    margin-top: 8px;
    display: none;
  }
  .progress-box {
    margin-top: 14px;
    padding: 12px;
    border-radius: 8px;
    background: var(--bg);
    border: 1px solid var(--border);
    display: none;
  }
  .progress-top {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    margin-bottom: 8px;
  }
  .bar-bg {
    height: 8px;
    background: var(--panel-2);
    border-radius: 999px;
    overflow: hidden;
  }
  .bar-fill {
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, #3b82f6, #22c55e);
    transition: width 0.35s ease;
  }
  .error-box {
    margin-top: 12px;
    padding: 10px 12px;
    border-radius: 8px;
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.45);
    color: #fca5a5;
    font-size: 13px;
    white-space: pre-wrap;
    display: none;
  }
  .note {
    margin-top: 12px;
    font-size: 12px;
    color: var(--muted);
  }
  .viewer {
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .img-frame {
    min-height: 460px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    position: relative;
  }
  .img-frame img {
    max-width: 100%;
    max-height: 76vh;
    object-fit: contain;
    display: block;
  }
  .empty-state {
    color: var(--muted);
    font-size: 14px;
    text-align: center;
    padding: 40px;
  }
  .meta-bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    font-size: 13px;
    color: var(--muted);
  }
  .gallery-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: 10px;
    margin-top: 12px;
  }
  .thumb {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    cursor: pointer;
    transition: transform 0.12s, border-color 0.12s;
  }
  .thumb:hover { border-color: var(--accent); transform: translateY(-2px); }
  .thumb.active { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
  .thumb img {
    width: 100%;
    height: 140px;
    object-fit: cover;
    display: block;
  }
  .thumb-cap {
    padding: 6px 8px;
    font-size: 11px;
    color: var(--muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
</style>
</head>
<body>
<header>
  <div class="brand">
    <h1>Z-Image-Turbo — Yerel Arayüz</h1>
    <span>4 GB RTX 2050 (sm_86) · Sabit 8 Adım Turbo Reçetesi</span>
  </div>
  <div class="stats">
    <div class="pill" id="gpuPill">GPU yükleniyor...</div>
    <div class="pill" id="vramPill">VRAM: —</div>
    <div class="pill" id="ramPill">Boş RAM: —</div>
  </div>
</header>

<div class="banner" id="setupBanner"></div>

<main>
  <section class="card">
    <h2>Görsel Oluştur</h2>

    <label>Hazır Örnek Promptlar</label>
    <div class="chips" id="presetChips"></div>

    <label for="prompt">Prompt (İngilizce önerilir)</label>
    <textarea id="prompt" placeholder="Örn: A sunlit street in Istanbul at golden hour, realistic photograph..."></textarea>

    <div class="row">
      <div>
        <label for="sizePreset">Boyut</label>
        <select id="sizePreset">
          <option value="768x1344">768 × 1344 (Dikey - Önerilen)</option>
          <option value="1344x768">1344 × 768 (Yatay)</option>
          <option value="1024x1024">1024 × 1024 (Kare)</option>
          <option value="custom">Özel Boyut...</option>
        </select>
      </div>
      <div>
        <label for="vae">VAE Modu</label>
        <select id="vae">
          <option value="auto">auto (cuDNN - En Kaliteli)</option>
          <option value="tiled">tiled (2 Parça cuDNN)</option>
          <option value="legacy">legacy (cuDNN'siz)</option>
        </select>
      </div>
    </div>

    <div class="row" id="customSizeRow" style="display:none;">
      <div>
        <label for="width">Genişlik (16 katı)</label>
        <input id="width" type="number" min="256" max="2048" step="16" value="768">
      </div>
      <div>
        <label for="height">Yükseklik (16 katı)</label>
        <input id="height" type="number" min="256" max="2048" step="16" value="1344">
      </div>
    </div>

    <label for="seed">Seed (Aynı seed + prompt = aynı görsel)</label>
    <div class="seed-row">
      <input id="seed" type="number" value="42">
      <button type="button" class="btn" id="randSeedBtn" title="Rastgele Seed">Rastgele</button>
    </div>

    <button type="button" class="btn btn-primary" id="generateBtn">Görsel Üret (~57 sn)</button>
    <button type="button" class="btn btn-danger" id="cancelBtn">İptal Et</button>

    <div class="progress-box" id="progressBox">
      <div class="progress-top">
        <span id="stageText">Hazırlanıyor...</span>
        <span id="timerText">0.0 sn</span>
      </div>
      <div class="bar-bg"><div class="bar-fill" id="barFill"></div></div>
    </div>

    <div class="error-box" id="errorBox"></div>

    <div class="note">
      İpucu: 4 GB VRAM çok sıkışık olduğu için tek seferde bir görsel üretilir. Üretim sırasında ağır oyun/tarayıcı sekmelerini kapalı tutmak daha kararlı çalışmasını sağlar.
    </div>
  </section>

  <section class="viewer">
    <div class="card">
      <h2>Sonuç</h2>
      <div class="img-frame" id="imgFrame">
        <div class="empty-state">Henüz seçili bir görsel yok. Soldan bir prompt yazıp <b>Görsel Üret</b> butonuna basın veya aşağıdan geçmiş üretimleri seçin.</div>
      </div>
      <div class="meta-bar" id="metaBar" style="margin-top:12px; display:none;">
        <div id="metaInfo"></div>
        <div style="display:flex; gap:8px;">
          <a class="btn" id="openImgBtn" target="_blank" rel="noopener">Tam Boyut Aç</a>
          <a class="btn" id="dlImgBtn" download="z-image.png">İndir</a>
        </div>
      </div>
    </div>

    <div class="card">
      <h2>Geçmiş Üretimler (<span id="histCount">0</span>)</h2>
      <div class="gallery-grid" id="gallery"></div>
    </div>
  </section>
</main>

<script>
const el = id => document.getElementById(id);
let pollTimer = null;
let selectedFolder = null;

function showResult(item) {
  if (!item || !item.image_url) return;
  selectedFolder = item.folder || null;
  const imgUrl = item.image_url + "?t=" + Date.now();
  el("imgFrame").innerHTML = `<img src="${imgUrl}" alt="Generated image">`;
  el("metaBar").style.display = "flex";
  el("openImgBtn").href = item.image_url;
  el("dlImgBtn").href = item.image_url;
  el("dlImgBtn").download = (item.folder || "z-image") + ".png";
  const st = item.stages || {};
  const vaeTag = st.vae_path ? `VAE: ${st.vae_path}` : `VAE: ${item.vae_mode || "?"}`;
  el("metaInfo").textContent =
    `${item.width}×${item.height} · Seed ${item.seed} · ${item.seconds} sn · VRAM Tepe: ${item.vram_peak_mib || item.gpu_memory_used_peak_mib || "?"} MiB · ${vaeTag}`;
  document.querySelectorAll(".thumb").forEach(t => {
    t.classList.toggle("active", t.dataset.folder === selectedFolder);
  });
}

async function loadInfo() {
  const res = await fetch("/api/info");
  const d = await res.json();
  if (d.gpu) {
    el("gpuPill").textContent = `${d.gpu.name} (sm_${(d.gpu.compute_cap || "").replace(".", "")})`;
    el("gpuPill").className = "pill " + ((d.gpu.compute_cap === "8.6") ? "ok" : "warn");
  } else {
    el("gpuPill").textContent = "NVIDIA GPU algılanamadı";
    el("gpuPill").className = "pill warn";
  }
  updateLiveStats(d.vram_used_mib, d.ram_free_mib);

  if (!d.setup.ready) {
    const b = el("setupBanner");
    b.style.display = "block";
    b.textContent = "Kurulum eksik görünüyor! Terminali açıp 'python scripts/setup.py' çalıştırın. Eksikler: " +
      (!d.setup.sd_cli ? "sd-cli.exe " : "") + (d.setup.missing_models || []).join(", ");
  }

  const chips = el("presetChips");
  chips.innerHTML = "";
  (d.presets || []).forEach((p, idx) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.textContent = p.label;
    b.onclick = () => { el("prompt").value = p.text; };
    chips.appendChild(b);
    if (idx === 0 && !el("prompt").value.trim()) {
      el("prompt").value = p.text;
    }
  });
}

function updateLiveStats(vram, ram) {
  el("vramPill").textContent = vram !== null && vram !== undefined ? `VRAM Kullanımı: ${vram} MiB / 4096 MiB` : "VRAM: —";
  el("ramPill").textContent = ram !== null && ram !== undefined ? `Boş RAM: ${(ram / 1024).toFixed(1)} GB` : "Boş RAM: —";
}

async function loadHistory(selectNewest = false) {
  const res = await fetch("/api/history");
  const items = await res.json();
  el("histCount").textContent = items.length;
  const g = el("gallery");
  g.innerHTML = "";
  items.forEach((item, idx) => {
    const card = document.createElement("div");
    card.className = "thumb" + ((selectNewest && idx === 0) || item.folder === selectedFolder ? " active" : "");
    card.dataset.folder = item.folder;
    card.innerHTML = `
      <img src="${item.image_url}" loading="lazy" alt="${item.folder}">
      <div class="thumb-cap" title="${(item.prompt || "").replace(/"/g, "&quot;")}">${item.seconds || "?"}s · ${item.width}×${item.height}</div>
    `;
    card.onclick = () => {
      if (item.prompt) el("prompt").value = item.prompt;
      showResult(item);
    };
    g.appendChild(card);
  });
  if (selectNewest && items.length > 0) {
    showResult(items[0]);
  }
}

function setRunningUI(isRunning) {
  el("generateBtn").disabled = isRunning;
  el("cancelBtn").style.display = isRunning ? "inline-flex" : "none";
  el("progressBox").style.display = isRunning ? "block" : "none";
}

async function pollStatus() {
  try {
    const res = await fetch("/api/status");
    const st = await res.json();
    updateLiveStats(st.vram_used_mib, st.ram_free_mib);
    if (st.running) {
      setRunningUI(true);
      el("stageText").textContent = st.stage;
      el("timerText").textContent = `${st.elapsed_s.toFixed(1)} sn`;
      el("barFill").style.width = `${st.progress}%`;
    } else {
      setRunningUI(false);
      if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
      if (st.last_error) {
        el("errorBox").style.display = "block";
        el("errorBox").textContent = st.last_error;
      } else if (st.last_result) {
        el("errorBox").style.display = "none";
        showResult(st.last_result);
        loadHistory(false);
      }
    }
  } catch (_) {}
}

el("sizePreset").onchange = () => {
  const val = el("sizePreset").value;
  if (val === "custom") {
    el("customSizeRow").style.display = "grid";
  } else {
    el("customSizeRow").style.display = "none";
    const [w, h] = val.split("x").map(Number);
    el("width").value = w;
    el("height").value = h;
  }
};

el("randSeedBtn").onclick = () => {
  el("seed").value = Math.floor(Math.random() * 2147483647);
};

el("generateBtn").onclick = async () => {
  const prompt = el("prompt").value.trim();
  if (!prompt) {
    el("errorBox").style.display = "block";
    el("errorBox").textContent = "Lütfen bir prompt yazın.";
    return;
  }
  el("errorBox").style.display = "none";
  const body = {
    prompt,
    width: Number(el("width").value),
    height: Number(el("height").value),
    seed: Number(el("seed").value),
    vae: el("vae").value,
  };
  const res = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) {
    el("errorBox").style.display = "block";
    el("errorBox").textContent = data.error || "Başlatılamadı.";
    return;
  }
  setRunningUI(true);
  el("stageText").textContent = "Başlatılıyor...";
  el("timerText").textContent = "0.0 sn";
  el("barFill").style.width = "3%";
  if (!pollTimer) pollTimer = setInterval(pollStatus, 1000);
};

el("cancelBtn").onclick = async () => {
  await fetch("/api/cancel", { method: "POST" });
};

(async () => {
  await loadInfo();
  await loadHistory(true);
  const res = await fetch("/api/status");
  const st = await res.json();
  if (st.running && !pollTimer) {
    setRunningUI(true);
    pollTimer = setInterval(pollStatus, 1000);
  }
})();
</script>
</body>
</html>
"""


def make_handler(state, presets):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return  # keep console clean

        def _send_json(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/":
                body = HTML_PAGE.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/api/info":
                self._send_json(HTTPStatus.OK, {
                    "gpu": generate.gpu_info(),
                    "vram_used_mib": generate.gpu_used_mib(),
                    "ram_free_mib": generate.free_ram_mib(),
                    "setup": setup_status(state.models_dir, state.bin_dir),
                    "presets": presets,
                })
                return

            if path == "/api/status":
                self._send_json(HTTPStatus.OK, state.status())
                return

            if path == "/api/history":
                self._send_json(HTTPStatus.OK, list_history(state.out_dir))
                return

            if path.startswith("/outputs/"):
                rel = unquote(path[len("/outputs/"):])
                base = state.out_dir.resolve()
                target = (base / rel).resolve()
                if base not in target.parents or not target.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
                data = target.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path == "/api/cancel":
                ok = state.cancel()
                self._send_json(HTTPStatus.OK, {"cancelled": ok})
                return

            if parsed.path == "/api/generate":
                length = int(self.headers.get("Content-Length", "0") or "0")
                if length <= 0 or length > 256 * 1024:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Geçersiz istek gövdesi."})
                    return
                try:
                    req = json.loads(self.rfile.read(length).decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "JSON okunamadı."})
                    return

                prompt = str(req.get("prompt", "")).strip()
                if not prompt:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Prompt boş olamaz."})
                    return
                try:
                    width = int(req.get("width", generate.DEFAULT_W))
                    height = int(req.get("height", generate.DEFAULT_H))
                    seed = int(req.get("seed", generate.DEFAULT_SEED))
                except (TypeError, ValueError):
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Genişlik, yükseklik ve seed tam sayı olmalıdır."})
                    return

                vae = str(req.get("vae", "auto"))
                if vae not in generate.VAE_MODES:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Geçersiz VAE modu."})
                    return

                errors, _ = generate.check_size(width, height)
                if errors:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "; ".join(errors)})
                    return

                st = setup_status(state.models_dir, state.bin_dir)
                if not st["ready"]:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Kurulum eksik. Önce 'python scripts/setup.py' çalıştırın."})
                    return

                name = str(req.get("name", "")).strip() or None
                ok, err = state.start(prompt, width, height, seed, vae, name)
                if not ok:
                    self._send_json(HTTPStatus.CONFLICT, {"error": err})
                    return
                self._send_json(HTTPStatus.ACCEPTED, {"started": True})
                return

            self.send_error(HTTPStatus.NOT_FOUND)

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1", help="bind host (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=7860, help="bind port (default 7860)")
    ap.add_argument("--no-browser", action="store_true", help="do not automatically open the browser")
    ap.add_argument("--models-dir", default=str(ROOT / "models"), help="default: ./models")
    ap.add_argument("--bin-dir", default=str(ROOT / "bin"), help="default: ./bin")
    ap.add_argument("--out", default=str(ROOT / "outputs"), help="default: ./outputs")
    a = ap.parse_args(argv)

    presets = load_presets()
    state = RunnerState(a.models_dir, a.bin_dir, a.out)
    server = ThreadingHTTPServer((a.host, a.port), make_handler(state, presets))
    url = f"http://{a.host}:{a.port}"
    print(f"Z-Image-Turbo Web UI hazir: {url}", flush=True)
    print("Kapatmak icin Ctrl+C basin.", flush=True)
    if not a.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nKapatiliyor...", flush=True)
        state.cancel()
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
