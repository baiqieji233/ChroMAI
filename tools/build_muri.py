"""Build the embedded MiaCode judge, without building or starting its editor.

Requires CMake, a C++20 compiler and Qt 6 Core/Gui development files. Existing
local reference assets/Qt are reused; the pinned Slide data is fetched only if
absent. Runtime loading never downloads, compiles or starts an application.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--qt", type=Path, default=None)
    args = parser.parse_args()
    source = ROOT / "native/miacode"
    provenance = json.loads((source / "UPSTREAM.json").read_text(encoding="utf-8"))
    for name, expected in provenance["files"].items():
        if hashlib.sha256((source / "upstream" / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Upstream source identity changed: {name}")
    cache = ROOT / ".native/miacode"
    cache.mkdir(parents=True, exist_ok=True)
    data = cache / "slide_data.json"
    if not data.exists():
        local = ROOT / ".scratch/muri_build/source/assets/reference/slide_data.json"
        if local.is_file():
            shutil.copyfile(local, data)
        else:
            url = ("https://raw.githubusercontent.com/Team-MiaCode/MiaCode/"
                   + provenance["commit"] + "/assets/reference/slide_data.json")
            with urllib.request.urlopen(url, timeout=60) as response:
                data.write_bytes(response.read())
    if hashlib.sha256(data.read_bytes()).hexdigest() != provenance["slide_data_sha256"]:
        raise RuntimeError("Slide reference dataset SHA-256 mismatch")
    qt = args.qt or (Path(os.environ["QT_ROOT"]) if os.environ.get("QT_ROOT") else None)
    if qt is None and (ROOT / ".scratch/muri_build/qt-sdk").is_dir():
        qt = ROOT / ".scratch/muri_build/qt-sdk"
    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError("CMake is required to build the embedded judge")
    build = cache / "build"
    cmd = [cmake, "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
           "-DMIACODE_SLIDE_DATA=" + data.as_posix()]
    if qt:
        cmd.append("-DCMAKE_PREFIX_PATH=" + str(qt.resolve()))
    subprocess.run(cmd, check=True)
    subprocess.run([cmake, "--build", str(build), "--config", "Release", "--parallel", "4"], check=True)
    name = "maichart_muri.dll" if sys.platform == "win32" else ("libmaichart_muri.dylib" if sys.platform == "darwin" else "libmaichart_muri.so")
    library = next(build.rglob(name))
    runtime = cache / "runtime"
    runtime.mkdir(exist_ok=True)
    shutil.copyfile(library, runtime / name)
    if sys.platform == "win32":
        if qt is None:
            raise RuntimeError("Pass --qt or QT_ROOT to package the Qt Core/Gui runtime")
        for dll in ("Qt6Core.dll", "Qt6Gui.dll"):
            shutil.copyfile(qt / "bin" / dll, runtime / dll)
        # Retain Qt's license inventory with this local runtime, no editor assets.
        if (qt / "licenses").is_dir():
            shutil.copytree(qt / "licenses", runtime / "qt-licenses", dirs_exist_ok=True)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in runtime.iterdir() if p.is_file() and p.suffix in {".dll", ".so", ".dylib"}}
    (runtime / "build.json").write_text(json.dumps({"schema": "maichart-muri-native-v1",
        "commit": provenance["commit"], "abi": 1, "files": files,
        "slide_data_sha256": provenance["slide_data_sha256"],
        "bridge_sha256": hashlib.sha256((source / "bridge.cpp").read_bytes()).hexdigest()}, indent=2), encoding="utf-8")
    print(f"Embedded MiaCode judge ready: {runtime / name}", flush=True)


if __name__ == "__main__":
    main()
