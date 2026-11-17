# Measurements: what is measured and what is not

Three labels are used everywhere in this repository and they are never mixed:

| Label | Meaning |
|---|---|
| **measured upstream** | run by the upstream project on a GTX 1050 Ti 4 GB laptop (Pascal, sm_61), 16 GB RAM, Windows 10 |
| **built for this card** | compiled for sm_86 (Ampere), RTX 2050 4 GB reference card; compiled is not the same as measured |
| **not measured** | nobody has reported it; treat every number about it as a guess |

## This repository (RTX 2050, sm_86)

| What | Value | Label |
|---|---|---|
| Engine target | `CMAKE_CUDA_ARCHITECTURES=86` (compute capability 8.6) | built for this card |
| Engine build | CUDA 12.8, MSVC 19.44 (VS 2022), Release, shared libs, `GGML_BACKEND_DL=ON`, 48 min in CI | built for this card |
| sm_86 in the binary | `cuobjdump -lelf`: `ggml-cuda.dll`, `stable-diffusion.dll` | measured (CI log) |
| Release asset | `z-image-turbo-engine-v0.1.0-win-x64-cuda12.8-sm86.zip`, 59,473,409 B, sha256 `f1fd60a1e85b763ec0c1b90b8624aaae5c0b43622b7a9ed89494fab38307bed6` | measured (asset hashed after upload) |
| Patch set | 8 of the upstream 12 (see `patches/README.md`) | built for this card |
| Seconds per 768x1344 picture | — | **not measured** |
| GPU memory peak | — | **not measured** |
| Process RAM peak | — | **not measured** |
| Same seed, same pixels | — | **not measured** here |

## Upstream numbers (GTX 1050 Ti 4 GB, sm_61) — context, not this card

| | v0.2.0 (upstream) |
|---|---|
| One 768x1344 picture (seed 42, 8 steps) | 193–228 s |
| VAE decode (last stage) | 8.3–15.9 s |
| Peak GPU memory (`nvidia-smi`) | 3057–3059 MiB |
| Process memory, peak | 10 667–10 915 MiB |
| Lowest free RAM seen | 418–545 MiB |

Why it fits in 4 GB at all is upstream's work, not ours: `--offload-to-cpu`, `--mmap`, 36 slices, flash attention
(with it off a run needed 4205 MB and did not fit).

## How to measure your card (and fill this table)

```bat
python scripts/generate.py --prompt-file prompts/eiffel.txt
```

Read `outputs\<time>_eiffel\run.json`: `seconds`, `gpu_memory_used_peak_mib`, `ram_free_min_mib`, `stages`,
`gpu` (name, compute capability, driver), `vae_path`. Run the same command twice: the second `PIXEL_SHA256` must
match the first (same engine, same seed, same VAE path).

Please open an issue with `run.json` and the last 40 lines of `log.txt`. A failure is a finding too.
