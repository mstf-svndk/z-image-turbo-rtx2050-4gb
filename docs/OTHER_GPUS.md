# Other GPUs, other systems

This repository exists for **one row** of the upstream table: an NVIDIA **Ampere sm_86 card with 4 GB**
(reference: RTX 2050 laptop) on **Windows x64**. Everything else belongs upstream.

| Your card | Where to go |
|---|---|
| GTX 1050 Ti 4 GB / other Pascal (sm_61) | [alexturkey/z-image-turbo-pascal-4gb](https://github.com/alexturkey/z-image-turbo-pascal-4gb) — measured there |
| Another Ampere card (RTX 3050/3060/3070/3080, A-series) | This repository's engine may run (compute 8.6 is family-wide) but it was not measured; more VRAM does not hurt, the 4 GB recipe still works |
| RTX 40/50 series (sm_89, sm_120) | No. Compute 8.6 kernels do not run there. Build your own engine from `docs/BUILD.md` with your `-DCMAKE_CUDA_ARCHITECTURES` (e.g. `89`) |
| AMD (ROCm), Intel, Apple, CPU | Upstream's `docs/OTHER_GPUS.md` and the upstream build guide |
| Linux | The upstream repository; the scripts here are Windows-only (`nvidia-smi`, `CREATE_NO_WINDOW`, ctypes) |

## What carries over and what does not

- **Carries over:** the three model files in `models.lock.json` (any backend can use them), the fixed recipe
  (8 steps, cfg 1.0, euler + simple, no negative prompt), the prompts, `scripts/compare.py`.
- **Does not carry over:** the engine zip (sm_86 only), `setup.py`'s cuDNN/engine steps, `generate.py`'s memory
  assumptions, the patch set (several upstream patches are Pascal-tuned; see `patches/README.md`).

Do not edit `scripts/setup.py` to make it run elsewhere; install the models with
`python scripts/setup.py --only models` and call your own `sd-cli` directly.
