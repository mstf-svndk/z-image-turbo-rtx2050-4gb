# Building the engine (sm_86, Windows)

The release asset in this repository is built by `.github/workflows/build-engine.yml` on `windows-2022`.
This page is the same recipe for a local build, and the record of what exactly went into the release.

## Inputs

| Need | Version |
|---|---|
| Windows 10/11 x64 | — |
| Visual Studio 2022 Build Tools (MSVC C++ x64) | 14.4x |
| CUDA Toolkit | 12.8 |
| CMake | 3.28+ |
| git | 2.x (LF line endings) |
| cuDNN | 9.10.2.21 (headers + `cudnn.lib`; runtime DLLs are downloaded by `setup.py`) |

A full build takes about 20–60 minutes depending on the machine. The CI build uses two parallel compile jobs to
stay inside 16 GB of RAM (CUDA compiles are memory hungry).

## Base sources

| Repository | Commit |
|---|---|
| [leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) | `32d516c` (master-1088) |
| ggml submodule (leejet/ggml) | `d803c23` |

```powershell
git -c core.autocrlf=false clone https://github.com/leejet/stable-diffusion.cpp.git sdcpp
cd sdcpp
git checkout 32d516c
git submodule update --init --recursive
git -C ggml checkout d803c23
```

## Apply the patches

Only 8 of the upstream 12 are used here; `patches/README.md` says which and why. From the `sdcpp` folder:

```powershell
$P = "<repo>\patches"
cd ggml
git apply "$P\ggml\0006-cuda-im2col-fast.patch"
git apply "$P\ggml\0009-cuda-cudnn-conv2d-f32.patch"
cd ..
git apply "$P\sdcpp\0012-vram-margin-env.patch"
git apply "$P\sdcpp\0022b-zimage-refiner-single-pass.patch"
git apply "$P\sdcpp\0025-vae-tile-floor.patch"
git apply "$P\sdcpp\0027-vae-vram-margin.patch"
git apply "$P\sdcpp\0028b-vae-try-untiled-first.patch"
git apply "$P\sdcpp\0029-capacity-log.patch"
```

This order applies cleanly on the base above (checked with `git apply` on a clean checkout).

## cuDNN for the build

Download NVIDIA's archive (no sign-in needed) and unpack it:

```
https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.10.2.21_cuda12-archive.zip
```

`CUDNN_ROOT` is the folder that contains `include\cudnn.h` and `lib\x64\cudnn.lib`. Only `conv2d.cu` gets these
flags, so toggling cuDNN does not rebuild the rest.

## Configure and build

```powershell
cmake -B build -G "Visual Studio 17 2022" -A x64 ^
  -DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 ^
  -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON ^
  -DSD_BUILD_SHARED_LIBS=ON -DSD_BUILD_SHARED_GGML_LIB=ON ^
  -DGGML_CUDA_CUDNN_CONV2D_F32=ON -DCUDNN_ROOT="<cudnn unpacked folder>"
cmake --build build --config Release --parallel 2
```

Output: `build\bin\Release\` — `sd-cli.exe`, `sd-server.exe` and the ggml/sd DLLs. With `GGML_BACKEND_DL=ON` the
CUDA kernels are **not** in `sd-cli.exe`: they are in a separate DLL (`ggml-cuda.dll`). Package the whole folder,
not just the exe.

### Check that the binary really carries sm_86

```powershell
& "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\cuobjdump.exe" -lelf build\bin\Release\ggml-cuda.dll |
  Select-String "sm_86"
```

`setup.py` does a weaker version of this check at install time (`ENGINE_ARCH sm_86`).

## Release asset

Zip the contents of `build\bin\Release` (exe + dll, flat), add `licenses\` and `THIRD_PARTY.md`, name it
`z-image-turbo-engine-v<tag>-win-x64-cuda12.8-sm86.zip`, compute its sha256 and write the size and hash into
`models.lock.json` → `engine[0]`. Without that lock entry `setup.py` will refuse the download (hash mismatch).
