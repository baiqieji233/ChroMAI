# Embedded Muri judge

`upstream/` contains unchanged project-owned MiaCode sources from commit
`6573efb611056bccd046dd17bc2f4cceda78bf88`. File hashes and the Slide reference
data hash are pinned in `UPSTREAM.json`. The upstream MIT license and its
asset/dependency scope notices are retained in this directory.

`bridge.cpp` exposes an allocator-paired C ABI for Python ctypes, using the
upstream parser, MuriAnalyzer and diagnostic serializers. It creates no Qt
application, window, event loop, GPU context or subprocess. Qt Core and Gui
provide data/geometry types only. Fonts, Qt Widgets, multimedia, BASS and FFmpeg
are not linked or packaged.

From the project root, run `python tools/build_muri.py --qt <Qt SDK root>`.
The script also discovers the existing local SDK under `.scratch/muri_build`.
Requires CMake, C++20 and Qt 6.8 development files. Local binaries and downloaded
reference assets live in `.native/` and are excluded from Git. Deployment on
another machine must copy the built runtime with applicable dependency/asset
notices, or rebuild there. Python's runtime loader never builds or downloads.
