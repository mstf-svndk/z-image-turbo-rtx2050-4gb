# Notes for AI coding agents

This repository adapts [Sinirsiz Foto](https://github.com/alexturkey/z-image-turbo-pascal-4gb) from Pascal (sm_61)
to Ampere (sm_86, reference card RTX 2050 4 GB). Read this before changing anything.

## Ground rules

1. **Do not change the recipe.** 8 steps, cfg 1.0, euler + simple, no negative prompt, `--offload-to-cpu --mmap
   --eager-load --flash-attn on`, 768x1344. The model is distilled for 8 steps; lowering them is not a speed fix.
2. **Do not "fix" the 4 GB assumptions.** The weights stay in system RAM; the VAE goes one-piece → two large tiles
   → small tiles. Every one of those steps exists because 4 GB is tight.
3. **The engine is built in CI, not on the user's machine.** `.github/workflows/build-engine.yml` is the only
   supported build path (CUDA 12.8 + MSVC 2022 + `CMAKE_CUDA_ARCHITECTURES=86`). Local recipe: `docs/BUILD.md`.
4. **The lock file is the contract.** `models.lock.json` pins every download with size + sha256. If you change a
   release asset, update the entry in the same commit; never point at `main`/`latest`.
5. **Labels are part of the product.** "built for this card" ≠ "measured". Nothing has been measured on an
   RTX 2050 yet; do not write a number that was not produced by a run.
6. `setup.py` and `generate.py` are Windows-only on purpose (`nvidia-smi`, `CREATE_NO_WINDOW`, ctypes). Do not make
   them pretend to work elsewhere; `docs/OTHER_GPUS.md` says where to send people.

## Checks before a commit

```bat
python -m unittest discover -s tests
python scripts/setup.py --help
python scripts/generate.py --help
```

The last two must exit 0 without a GPU. `scripts/setup.py --check` is the offline lock check.

## If a build fails in CI

Read the failing step's log, not the guess. The usual suspects, in order: CUDA installer, cuDNN paths
(`include\cudnn.h`, `lib\x64\cudnn.lib`), patch application (line endings), then nvcc memory (lower `--parallel`).
