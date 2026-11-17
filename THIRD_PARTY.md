# Third-party material

This repository is a derivative of [Sinirsiz Foto](https://github.com/alexturkey/z-image-turbo-pascal-4gb); its own scripts and documents are Apache-2.0 (see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)); the engine patches in `patches/` are MIT (`patches/LICENSE-MIT`).
Model files, the engine and NVIDIA software are downloaded by `scripts/setup.py` from the pinned sources in `models.lock.json`. Model files and cuDNN are **not stored in this repository or in its releases**; the engine zip and the license texts of the code compiled into it are. Each component keeps its own license.

License labels below were read from the Hugging Face API (`license` field), the repositories' files and NVIDIA's license texts on 2026-11-13 and 2026-11-14. This is a reading, not legal advice; check the originals.

| Component | Use | License | Where it is | Source |
|---|---|---|---|---|
| Z-Image-Turbo (Tongyi-MAI) | Base image model | Apache-2.0 | downloaded by setup | https://huggingface.co/Tongyi-MAI/Z-Image-Turbo (card: `license: apache-2.0`) |
| `z_image_turbo-Q8_0.gguf` (leejet) | GGUF conversion | Apache-2.0 (the page says the quantized files follow the original model's license) | downloaded by setup | https://huggingface.co/leejet/Z-Image-Turbo-GGUF |
| Qwen3-4B-Instruct-2507 (Qwen) | Text encoder, base model | Apache-2.0 | not used directly | https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 |
| `Qwen3-4B-Instruct-2507-Q8_0.gguf` (unsloth) | GGUF conversion of the text encoder | Apache-2.0 | downloaded by setup | https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF |
| `ae.safetensors` (Comfy-Org repackage) | VAE, a FLUX.1-type autoencoder | Apache-2.0 (repo tag and card) | downloaded by setup | https://huggingface.co/Comfy-Org/Z-Image-Turbo |
| stable-diffusion.cpp (leejet), `32d516c` | Engine source (`sd-cli.exe`) | MIT | compiled into the engine zip; license text in `licenses/` | https://github.com/leejet/stable-diffusion.cpp |
| ggml (leejet fork), `d803c23` | Tensor library inside the engine | MIT | compiled into the engine zip; license text in `licenses/` | https://github.com/leejet/ggml |
| Oniguruma, utf8proc, darts-clone, nlohmann/json, stb, miniz/zip | Code compiled into the engine | BSD-2-Clause, MIT/Unicode, BSD-3-Clause, MIT, public domain/MIT | notices in `licenses/` | see `licenses/THIRD_PARTY_NOTICES.md` |
| Engine zip `z-image-turbo-engine-v0.1.0-win-x64-cuda12.8-sm86.zip` | Prebuilt engine (sm_86 build of the 8 patches in `patches/`) | patches MIT; upstream code as above; NVIDIA CUDA/cuDNN are **not** in this zip | this repository's v0.1.0 release | https://github.com/mstf-svndk/z-image-turbo-rtx2050-4gb/releases |
| NVIDIA CUDA 12.8 runtime DLLs (`cudart`, `cublas`, `cublasLt`) | Needed by the engine; unmodified | NVIDIA CUDA Toolkit EULA, Attachment A | separate zip from the wan22-pascal-4gb v0.1.0 release (not re-uploaded) | https://docs.nvidia.com/cuda/eula/index.html |
| NVIDIA cuDNN 9.10.2.21 (CUDA 12 build), 8 runtime DLLs | Optional convolution library for the faster, closer VAE decode | NVIDIA Software License Agreement + cuDNN Supplement (proprietary; version of 22 February 2022) | **not stored and not redistributed here**; downloaded from NVIDIA's server by `setup.py` by default, under NVIDIA's license (`--no-cudnn` skips it) | https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.10.2.21_cuda12-archive.zip (sha256 in `models.lock.json`) |

Microsoft Visual C++ runtime and ffmpeg are not included.

## NVIDIA cuDNN: what we read and what we do

- cuDNN is proprietary NVIDIA software. It is **not covered by this repository's license**. `scripts/setup.py` shows the license address (`https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/eula.html`) and prints it in one line before the download, then downloads the archive by default; `--no-cudnn` skips it. NVIDIA's license text says that whoever does not accept all its terms must not download, install or use the SDK. The license text from the archive is saved next to the DLLs as `NVIDIA_cuDNN_LICENSE.txt` and a record (license sha256, `mode: default_download_disclosed`, `installed_utc`) as `cudnn_license_record.json`; the record states a disclosed default, not a decision by you.
- Without cuDNN everything works with the slower v0.1.0 VAE path (`--vae legacy`).
- The eight DLLs pinned in `models.lock.json` are byte-identical (size and CRC32) to the files in NVIDIA's **CUDA 12** archive for 9.10.2.21 (`cudnn-windows-x86_64-9.10.2.21_cuda12-archive.zip`). Upstream originally used the CUDA 11 archive, which dynamically loads `cublasLt64_11.dll` at VAE decode time and fails when only CUDA 12.8 runtime DLLs (`cublasLt64_12.dll`) are present; switching to the CUDA 12 archive resolves this on RTX 2050 / CUDA 12.8.
- The agreement's cuDNN supplement lists "the runtime files .so and .dll" as distributable, but section 1.1 gives that right only "as incorporated in object code format into a software application", under the conditions of section 1.2 (among them: the distributed application must have material additional functionality, only the application may access the SDK files, and you must notify NVIDIA of known non-compliant use). Whether a release zip that carries our engine together with those DLLs would meet this, we could not decide. **So we do not ship them.** `setup.py` downloads from NVIDIA instead, so this repository does not distribute cuDNN.
- NVIDIA and cuDNN are trademarks of NVIDIA. This project is not sponsored or endorsed by NVIDIA.

## NVIDIA CUDA runtime

The CUDA Toolkit EULA, Attachment A, lists `cudart`, `cublas` and `cublasLt` among the files that may be redistributed under its conditions (read 2026-11-14). We provide them unmodified, in a separate zip from the earlier v0.1.0 release of our sibling project, which `models.lock.json` pins by sha256. That zip carries only DLLs, not an application; we did not obtain a legal opinion on that point (see below).

## Things we could not verify

- **Ownership of generated pictures.** The Z-Image-Turbo model card and README contain no sentence about who owns generated pictures (searched for "rights", "generated content", "commercial"). The card lists no use restrictions beyond the Apache-2.0 license (read 2026-11-14; the page's "TOS" link was not opened). We make no claim in either direction.
- **VAE identity.** The `ae.safetensors` we use comes from the Comfy-Org repo (Apache-2.0 tagged; sha256 `afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38`). FLUX.1-schnell is tagged Apache-2.0 and FLUX.1-dev is under a non-commercial license. Whether the `ae.safetensors` files in those two repos are the same bytes as ours was not verified (the repositories are gated and their file hashes could not be read). If you need certainty about the VAE's terms, check the FLUX.1 repositories yourself.
- **cuDNN in a release zip.** Whether NVIDIA's agreement allows shipping the cuDNN runtime DLLs together with our engine (see above). Not decided; not done.
- **A DLL-only CUDA runtime zip.** Attachment A names the files, but the grant is written for software applications that include them. Whether a separate unmodified DLL zip fits that wording was not decided. NVIDIA's contact for license questions is given in the agreement; we have not written to NVIDIA.
- **Other CUDA versions of cuDNN.** Their files and license terms were not compared.
- Apache-2.0 asks you to keep the license text and notices when you redistribute the model files. This repository does not redistribute them.
- The BSD license of Oniguruma requires us to keep its copyright notice, which contains an upstream author's e-mail address, in `licenses/LICENSE-oniguruma-BSD-2-Clause.txt`.
