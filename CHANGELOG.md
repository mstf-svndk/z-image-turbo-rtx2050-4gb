# Changelog

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
