# Engine patches

8 of the 12 upstream patches are used for this build. All of them are MIT and optional switches: with the
environment variable unset the stock code path runs (except where the table says "always on").

Base: stable-diffusion.cpp `32d516c` (master-1088), ggml `d803c23`.

## Included (8)

| Patch | What it does | Environment variable (default) | Kind |
|---|---|---|---|
| `ggml/0006-cuda-im2col-fast` | im2col fast path, same values (int64 modulo hoisted to int32) | `GGML_CUDA_IM2COL_FAST=0` selects the old kernel | speed, arch-neutral |
| `ggml/0009-cuda-cudnn-conv2d-f32` | optional cuDNN F32 conv2d (needs cuDNN at build and run time) | `GGML_CUDA_CUDNN_CONV2D=1` (off), `GGML_CUDA_CUDNN_ALGO`, `GGML_CUDA_CUDNN_WS_MB` (256), `GGML_CUDA_CUDNN_LOG` | quality (VAE), arch-neutral; **makes cuDNN a load-time dependency of the CUDA backend** |
| `sdcpp/0012-vram-margin-env` | replaces the VRAM safety margin | `SD_VRAM_MARGIN_MB` (unset) | measuring tool |
| `sdcpp/0022b-zimage-refiner-single-pass` | refiner blocks run once per step (group name was used twice) | `SD_ZIMAGE_CONCAT_CUT=1` restores old behaviour | speed, logic fix |
| `sdcpp/0025-vae-tile-floor` | memory fallback for VAE tiles: half latent | `SD_VAE_TILE_FLOOR=1` (off) | quality/robustness |
| `sdcpp/0027-vae-vram-margin` | VRAM margin only while the VAE decodes | `SD_VAE_VRAM_MARGIN_MB` (unset) | quality/robustness |
| `sdcpp/0028b-vae-try-untiled-first` | try one untiled VAE decode before tiling | `SD_VAE_TRY_FULL=1` (off), `SD_VAE_FULL_MARGIN_MB` (256) | quality |
| `sdcpp/0029-capacity-log` | one log line per VRAM capacity check | `SD_CAPACITY_LOG=1` (off) | measuring tool |

## Excluded (4) and why

| Patch | Why it is not here |
|---|---|
| `ggml/0001-cuda-conv3d-direct-pascal` | Wan video models only; Z-Image never calls conv3d. Also sits in the `!use_mma` branch, dead code on Ampere |
| `ggml/0002-cuda-mmq-q8_0-dp4a-tile128` | a 128x64 Q8_0 MMQ tile picked for Pascal's L1/shared balance, "always on" with no switch. On sm_86 it can lower occupancy and it cannot be turned off without a rebuild |
| `ggml/0003-cuda-fattn-tile-a3-pascal` | hand-written fp32 flash-attention tile; on sm_86 `fast_fp16_available()` is true, so this kernel is never selected at run time |
| `sdcpp/0004-wan-vae-resample-conv3d-kd1` | Wan VAE only (Z-Image uses the FLUX-type VAE `ae.safetensors`); the patch also carries a subproject pointer change |

The 8 included patches were applied to a clean checkout of the base commits with `git apply` and applied without
conflicts — that is a check of the patch chain, not a check of the resulting speed on sm_86.

## CMake flags of the release engine

```
-DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
-DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON
-DSD_BUILD_SHARED_LIBS=ON -DSD_BUILD_SHARED_GGML_LIB=ON
-DGGML_CUDA_CUDNN_CONV2D_F32=ON -DCUDNN_ROOT=<cudnn>
```

Full build recipe: [`../docs/BUILD.md`](../docs/BUILD.md).

## License

Every patch is MIT (`SPDX-License-Identifier: MIT`, header of each file), like the code it modifies. Copyright of
the original code stays with its authors (stable-diffusion.cpp: leejet; ggml: the ggml authors); the patches are
Copyright 2026 Ali Haydar Öztürk.
