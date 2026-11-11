# Third-party notices

Oto includes the software and model weights below. Each is listed with its license and the
notice that license asks to be reproduced. Where a component is offered under a choice of
licenses, Oto uses it under the MIT license; the Apache License 2.0 text is in `LICENSE-APACHE`
next to this file.

## DeepFilterNet (libDF code and DeepFilterNet3 model weights)

- Source: https://github.com/Rikorose/DeepFilterNet, commit
  `7b14b895fe7fc42cab72b4a4c1e49c982bd6404b`
- Used: the `libDF` Rust crate (`deep_filter`) and the DeepFilterNet3 model weights it embeds
  (`models/DeepFilterNet3_onnx.tar.gz`, through its `default-model` feature). Present only in
  builds with the `dfn3` feature, which is on by default.
- License: MIT OR Apache-2.0. At this commit the repository states that "All code and model
  weights in this repository are dual-licensed" under these two licenses.

```text
The MIT License (MIT)
Copyright (c) 2021 Hendrik Schröter

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## DPDFNet (DPDFNet-2 48 kHz model weights)

- Source: https://huggingface.co/Ceva-IP/DPDFNet (`onnx/dpdfnet2_48khz_hr.onnx`, SHA-256
  `7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b`); code at
  https://github.com/ceva-ip/DPDFNet.
- Used: the model weights, embedded unmodified. Present only in builds with the `onnx` feature,
  which is on by default.
- License: Apache-2.0 (the model card's licence; the code repository is Apache-2.0 too). The
  licence text is in `LICENSE-APACHE`; upstream gives no copyright line beyond the publisher, Ceva.

## FastEnhancer (FastEnhancer-S 48 kHz model weights)

- Source: https://github.com/aask1357/fastenhancer, release `onnx-48khz-v1`
  (`fastenhancer_s.onnx`, SHA-256
  `f04ece2beed330da367264c54cedded62f65a117fbde5c005d3a88fc796d0ba3`).
- Used: the model weights, embedded unmodified. Present only in builds with the `fastenhancer`
  feature: on for local builds, left out of releases (`scripts/macos/release.sh`).
- License: the code is MIT (below). **The weights' licence is not stated upstream**, and the
  48 kHz training data includes corpora that may not be exported from South Korea. Included
  for local evaluation only; not cleared to ship in a release.

```text
MIT License

Copyright (c) 2025 AHN Sung Hwan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## ONNX Runtime (through the `ort` crate)

- Source: https://github.com/microsoft/onnxruntime, statically linked from pyke's prebuilt
  binaries via `ort` 2.0.0-rc.15 (https://github.com/pykeio/ort, MIT OR Apache-2.0).
- Used: inference for DPDFNet-2 and FastEnhancer-S. Present only in builds with the `onnx`
  feature.
- License: MIT.

```text
MIT License

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## tract

- Source: https://github.com/sonos/tract, version 0.21.4 (`tract-core`, `tract-data`,
  `tract-hir`, `tract-linalg`, `tract-nnef`, `tract-onnx`, `tract-onnx-opl`, `tract-pulse`,
  `tract-pulse-opl`), developed by Sonos, Inc. (author Mathieu Poumeyrol). Present only in
  builds with the `dfn3` feature.
- License: MIT OR Apache-2.0. Its license files carry no copyright line.

```text
Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```

## nnnoiseless (RNNoise)

- Source: https://github.com/jneem/nnnoiseless, version 0.5.2: a Rust port of RNNoise
  (https://gitlab.xiph.org/xiph/rnnoise) by Jean-Marc Valin, Xiph.Org Foundation and Mozilla,
  including RNNoise's trained weights. Always present.
- License: BSD-3-Clause. The notice below is nnnoiseless's `COPYING` file, which carries
  RNNoise's original Xiph.Org and Mozilla copyright lines.

```text
Copyright (c) 2020, Joe Neeman
Copyright (c) 2017, Mozilla
Copyright (c) 2007-2017, Jean-Marc Valin
Copyright (c) 2005-2017, Xiph.Org Foundation
Copyright (c) 2003-2004, Mark Borgerding

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

- Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

- Neither the name of the Xiph.Org Foundation nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
