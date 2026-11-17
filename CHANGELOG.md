# Changelog

## Unreleased (2026-11-17)

- **cuDNN CUDA 12 fix:** switched `models.lock.json`, `.github/workflows/build-engine.yml`, `THIRD_PARTY.md`, and `docs/BUILD.md` from `cudnn-windows-x86_64-9.10.2.21_cuda11-archive.zip` to `cudnn-windows-x86_64-9.10.2.21_cuda12-archive.zip` (the CUDA 11 archive required `cublasLt64_11.dll` at VAE decode time and crashed with `cudart-cuda12.8-win-x64.zip`).
- `scripts/setup.py`: added an optional `certifi` fallback in `_open()` when Windows' built-in certificate store fails SSL verification on GitHub release CDNs (`SSL: CERTIFICATE_VERIFY_FAILED`), keeping zero required external dependencies.
- Added local web UI (`scripts/ui.py`) and Windows launcher (`Baslat.cmd`) using Python standard library only.
- Recorded first measured run on NVIDIA GeForce RTX 2050 4 GB (`57.1 s` total, `2985 MiB` peak VRAM, `7.50 s` cuDNN tiled VAE decode) in `docs/MEASUREMENTS.md`.

## v0.1.0 (2026-11-16)

First build for Ampere / sm_86, derived from Sinirsiz Foto v0.2.0 (Apache-2.0).

- Engine built by CI (`windows-2022`, CUDA 12.8, MSVC 19.44, `CMAKE_CUDA_ARCHITECTURES=86`) and published as
  `z-image-turbo-engine-v0.1.0-win-x64-cuda12.8-sm86.zip` (59,473,409 B, sha256 `f1fd60a1e85b…07bed6`).
  `cuobjdump -lelf` found the sm_86 image in `ggml-cuda.dll` and `stable-diffusion.dll`.
- 8 of the upstream 12 patches kept (Pascal-only and Wan-only patches dropped); the chain applies cleanly to the
  pinned base commits.
- `scripts/setup.py`: reads `compute_cap` in the preflight (falls back to the query without it on old drivers),
  reads every GPU, refuses the 12 GB download when the card is a known non-sm_86 one (`--force-arch` overrides),
  and scans the installed engine for `sm_86` code (`ENGINE_ARCH sm_86`).
- `scripts/generate.py`: records `gpu` (name, compute capability, driver, VRAM) and `engine_arch` in `run.json`;
  one-piece VAE attempt uses `SD_VAE_FULL_MARGIN_MB=512` (upstream 256) because cuDNN picks its own Ampere workspace.
- cuDNN documented as a load-time requirement of the CUDA backend (verified on the released binary: `ggml-cuda.dll`
  imports `cudnn64_9.dll`).
- Model lock unchanged from upstream (same three files, same sha256), so an existing `models/` folder is reused.
- **Not measured on an RTX 2050 yet.**
