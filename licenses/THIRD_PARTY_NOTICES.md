# Third-party notices for the engine binaries

This repository does not build the engine. `scripts/setup.py` downloads the prebuilt engine zip from the
[wan22-pascal-4gb v0.1.0 release](https://github.com/alexturkey/wan22-pascal-4gb/releases/tag/v0.1.0) (sha256 in `models.lock.json`).
The notices below are copied from that release and cover third-party code **compiled into** the engine (`sd-cli.exe`, `stable-diffusion.dll`,
`ggml*.dll`) when built exactly as in that repository's `docs/BUILD.md` (CUDA backend, `SD_WEBP=OFF`, `SD_WEBM=OFF`, no CUTLASS, examples = `sd-cli` only).
It was derived from the compile commands of the clean-root build (`build.ninja` of stable-diffusion.cpp 32d516c with ggml d803c23),
not from memory. Full license texts that exist as files are in this folder.

Not in the binaries, so not listed as notices: libwebp, libwebm (disabled), CUTLASS (`SD_CUDA_CUTLASS_PATH` empty), cpp-httplib (server example, not built),
and the sd.cpp web UI submodule (not fetched).
NVIDIA CUDA runtime DLLs are not in this repository; `scripts/setup.py` downloads them unmodified from the same release (`cudart-cuda12.8-win-x64.zip`, NVIDIA CUDA EULA, Attachment A).

| Component | Where | License | Text |
|---|---|---|---|
| stable-diffusion.cpp (leejet) | whole engine, `src/`, `examples/` | MIT | `LICENSE-stable-diffusion.cpp.txt` |
| ggml (leejet fork, sd.cpp submodule `ggml`) | `ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll`, `ggml-cuda.dll` | MIT | `LICENSE-ggml.txt` |
| Oniguruma 6.9.10 (K.Kosako) | `thirdparty/oniguruma` -> `stable-diffusion.dll` | BSD-2-Clause | `LICENSE-oniguruma-BSD-2-Clause.txt` |
| utf8proc 2.10.0 (+ Unicode data) | `thirdparty/utf8proc` -> `stable-diffusion.dll` | MIT and Unicode data licenses | `LICENSE-utf8proc.md` |
| darts-clone `darts.h` 0.32 (Susumu Yata) | tokenizer header -> `stable-diffusion.dll` | BSD-3-Clause | `LICENSE-darts-clone-BSD-3-Clause.txt` |
| nlohmann/json `json.hpp` | `stable-diffusion.dll`, `sd-cli.exe` | MIT | below |
| stb_image / stb_image_write / stb_image_resize (Sean Barrett) | `sd-cli.exe` | MIT or public domain (your choice) | below |
| kuba--/zip `zip.c` + miniz `miniz.h` | `stable-diffusion.dll` (model/zip reading) | miniz: public domain (Unlicense); zip: Unlicense per upstream | below |

The engine patches (in the wan22-pascal-4gb repository) and this repository's scripts are MIT, see the repository `LICENSE`. The Apache-2.0 text (`LICENSE-Apache-2.0.txt`) is for the model weights (Z-Image-Turbo, Qwen3-4B-Instruct-2507, the VAE file), which are **not** part of this repository or the binaries.

## nlohmann/json (`thirdparty/json.hpp`)

MIT License. Copyright (c) 2013-2022 Niels Lohmann (the header also carries SPDX lines for bundled parts, for example Evan Nemerson's Hedley, CC0).

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## stb_image.h, stb_image_write.h, stb_image_resize.h

Dual-licensed by the author. The MIT alternative, as printed at the end of `stb_image.h`:

Copyright (c) 2017 Sean Barrett
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
---------------------------------------------------------------------------

(The public-domain alternative, "Unlicense", is the other choice offered by the same headers; `stb_image_resize.h` is in the public domain.)

## miniz.h (used by zip.c) and zip.c

Unlicense text as printed at the end of `miniz.h` ("miniz.c 2.2.0 - public domain", Rich Geldreich). `zip.c` / `zip.h` come from https://github.com/kuba--/zip, which upstream licenses under the Unlicense; the file header only carries the standard warranty disclaimer.

This is free and unencumbered software released into the public domain.

  Anyone is free to copy, modify, publish, use, compile, sell, or
  distribute this software, either in source code form or as a compiled
  binary, for any purpose, commercial or non-commercial, and by any
  means.

  In jurisdictions that recognize copyright laws, the author or authors
  of this software dedicate any and all copyright interest in the
  software to the public domain. We make this dedication for the benefit
  of the public at large and to the detriment of our heirs and
  successors. We intend this dedication to be an overt act of
  relinquishment in perpetuity of all present and future rights to this
  software under copyright law.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
  IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
  OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
  ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
  OTHER DEALINGS IN THE SOFTWARE.

  For more information, please refer to <http://unlicense.org/>
