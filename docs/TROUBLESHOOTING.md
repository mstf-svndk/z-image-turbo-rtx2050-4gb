# Troubleshooting (Windows, sm_86, 4 GB)

Read the last 40 lines of `outputs\<run>\log.txt` first; `run.json` next to it records the command, the VAE path and
the GPU peak.

## `no kernel image is available for execution on the device`

The engine was built for a different compute capability than your card. This repository ships **sm_86 only**.
Check:

```bat
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

`8.6` is the target. Anything else: use the upstream repository or build your own engine (`docs/BUILD.md`).
`setup.py` and `generate.py` print a warning when they see another compute capability.

## `CUDA driver version is insufficient for CUDA runtime version`

The engine needs a driver that supports CUDA 12.8 (Windows 570.65 or newer; newer is better). Update the NVIDIA
driver for the laptop GPU (not the Intel one). Do not install a driver "for notebooks" from a third-party site.

## `Could not locate cudnn64_9.dll`, `cudnn64_9.dll` not found, or the picture runs for hours

The engine is compiled with cuDNN and its CUDA backend imports `cudnn64_9.dll`, so the DLLs are required — this is
not an optional speed-up.

- `python scripts/setup.py --only cudnn` installs the eight DLLs next to `sd-cli.exe`.
- If NVIDIA's server is unreachable, retry later or from another network. Do not delete the DLLs afterwards: without
  them the CUDA backend cannot load and the run falls back to the CPU (hours).
- `generate.py` prints `NOTE: the log has no cuDNN convolution lines` when the VAE did not use cuDNN, and a warning
  when the DLLs are missing before the run starts.

## Out of GPU memory (`cudaMalloc failed`, `out of memory`)

- One picture at a time. Close browsers, Teams, games, anything with a GPU overlay.
- On Optimus laptops the Intel iGPU drives the desktop; the NVIDIA card still has to hold the desktop's share of
  video memory. Make sure the NVIDIA card is not also driving an external 4K display.
- Try `--vae legacy` (small tiles, less VRAM in the last stage).
- Do not raise the size: 768x1344 / 1344x768 / 1024x1024 are the tested ones.

## Out of system RAM

The process holds about 11 GB (measured upstream). Close other programs; the preflight warns when the machine has
less than 16 GB. Windows pagefile on an SSD must not be disabled.

## Very slow first run

The first picture reads ~10 GB of model pages from disk and builds CUDA kernels; it is slower than the second.
Models must be on an SSD (`--models-dir D:\models` is fine). Do not run two generations at once — they fight for
the same 4 GB.

## Antivirus blocks `sd-cli.exe`

The engine is unsigned (it is built in GitHub Actions from public sources). Add an exclusion for the repo's `bin`
folder, or check the release asset's sha256 against `models.lock.json` and the release notes before allowing it.

## `git apply` fails when rebuilding the engine

Line endings. Clone with `git -c core.autocrlf=false clone`, as in `docs/BUILD.md`. The patches are LF files.
