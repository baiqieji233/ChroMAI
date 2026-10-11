# Third-party notices

The initial parser, ChartIR, dataset preprocessing, audio feature cache, frame
labels, and V2.5 baseline were imported from
[Timsuo/maimai-ai-chart](https://github.com/Timsuo/maimai-ai-chart), whose
`pyproject.toml` declares the project license as MIT. The imported files remain
under the original `maichart` package and were retained as a compatibility
foundation.

This repository does not copy source from `lcebot/maidata-parser`; that project
uses a proprietary license. `Choimoe/PySimaiParser` and
`Shikochin/simai-sharp-ts` were consulted as MIT-licensed parser references.
`HuaHuaOfficial/maimai-chart-studio` was consulted as an Apache-2.0-licensed
architecture reference. No binary model, CUDA harness, or GUI code from those
projects is included here.

## MiaCode sensor reference

Sensor constants, Touch adjacency and the Touch-group threshold in
`src/maichart/generation/judgment_system.py` are adapted from the MIT-licensed
project-owned sources MuriConfig.h, MuriAnalyzer.cpp and MuriRuntimeModelBuilder.cpp
at [Team-MiaCode/MiaCode](https://github.com/Team-MiaCode/MiaCode/tree/6573efb611056bccd046dd17bc2f4cceda78bf88).
No skins, sounds, slide reference assets or application binaries are redistributed.
The legacy Python checker implements a static subset. The embedded generation
checker now links the project-owned parser and Muri analysis sources verbatim
under `native/miacode/upstream/src`, with a small local C ABI/JSON bridge. The
upstream LICENSE, LICENSE_SCOPE and THIRD_PARTY_NOTICES are retained alongside
the source; `UPSTREAM.json` pins every copied source hash.

`tools/build_muri.py` obtains the pinned Slide reference dataset for the local
build and links Qt Core/Gui dynamically. The dataset and runtime binaries stay
under ignored `.native/`; they are not relicensed as MIT or committed here.
Upstream asset distribution restrictions and Qt's own license obligations still
apply when distributing a built runtime. No editor, font, sound, skin, audio
backend or video backend is built by this integration.

MIT License

Copyright (c) 2026 fanfaredash

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
