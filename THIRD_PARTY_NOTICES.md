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
No skins, sounds or application binaries are redistributed. The generation
checker is now a Python port under `src/maichart/generation/muri`, covering
parsing, static references, simple-note pad events, Touch groups, hand actions,
and ordered Slide/Wifi judgments. The original source remains available
under `native/miacode/upstream/src` as a development reference. The
upstream LICENSE, LICENSE_SCOPE and THIRD_PARTY_NOTICES are retained alongside
the source; `UPSTREAM.json` pins every copied source hash.

The optional development ABI3 oracle compiles two generated adaptation files with a zero lower
bound for the configurable collision-reference neighbourhood. Zero disables
static proximity references; the judge's physical tolerances remain unchanged.
This local adaptation is recorded in the CMake source and build provenance;
the vendored upstream files are not edited.

`src/maichart/generation/muri/reference/slide_data.json.gz` contains the numerical
path samples and judgment tables extracted from the pinned upstream
`assets/reference/slide_data.json`. Values are preserved; rendering arrows are
omitted. `provenance.json` records the source commit, original hash, packaged
hash and transformation. These reference assets retain MiaCode's **non-commercial
distribution scope** and are not relicensed as MIT. A copy of upstream
`LICENSE_SCOPE.md` is packaged beside the data. This exception to the source
license also applies when distributing a wheel containing the data.

`tools/build_muri.py` and `tools/compare_muri_python.py` are optional developer
tools for comparing the port against the pinned native ABI3 oracle. Only that
explicit developer workflow uses Qt or a native library. Its binaries remain
under ignored `.native/`; production generation uses neither. The Python port
does not build or launch an editor, font, sound, skin, audio or video backend.

## MiaCode upstream MIT license text

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


## Pretrained music encoder

Composer v2 defaults to [m-a-p/MERT-v1-330M](https://huggingface.co/m-a-p/MERT-v1-330M).
Its model card declares **CC BY-NC 4.0**. The encoder weights are downloaded
separately and are not relicensed by this project's MIT source license.
Keep upstream attribution and consult the model card and license when using
or redistributing these weights. A local model override retains its own terms.

## Optional MiaView preview player

The player is downloaded separately from [MiaView_Web](https://github.com/Team-MiaCode/MiaView_Web).
The installer requires its artifact's `LICENSE`, `LICENSE_SCOPE.md` and
`THIRD_PARTY_NOTICES.md` and retains them in the local player installation.
The locally verified preview artifact also states a non-commercial scope for
packaged assets/releases unless an individual license says otherwise.
The player's source, runtime assets and bundled dependencies retain the terms
in those files; this project's MIT license does not override them.

## Composer checkpoints and user content

No Composer checkpoint license is granted by the source license. Checkpoints
are distributed separately and need their own model card and license, with
training-data and dependency restrictions reviewed before publication.
User-supplied songs, charts and generated outputs are not automatically
relicensed as MIT by this repository.
