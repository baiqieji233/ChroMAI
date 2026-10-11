"""In-process MiaCode judge. No editor, subprocess, GPU or event loop is used."""
from __future__ import annotations

import ctypes
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

REFERENCE_COMMIT = "6573efb611056bccd046dd17bc2f4cceda78bf88"
_lock = threading.RLock()
_dll_directory = None


def runtime_directory() -> Path:
    return Path(__file__).resolve().parents[3] / ".native/miacode/runtime"


def library_path() -> Path:
    name = "maichart_muri.dll" if sys.platform == "win32" else (
        "libmaichart_muri.dylib" if sys.platform == "darwin" else "libmaichart_muri.so")
    return runtime_directory() / name


@lru_cache(maxsize=1)
def _library():
    global _dll_directory
    path = library_path()
    if not path.is_file():
        raise RuntimeError("内嵌 MiaCode 检测库未部署；请运行 python tools/build_muri.py。无需安装或打开 MiaCode 编辑器。")
    provenance = json.loads((path.parent / "build.json").read_text(encoding="utf-8"))
    if provenance.get("commit") != REFERENCE_COMMIT or provenance.get("abi") != 1:
        raise RuntimeError("内嵌 MiaCode 检测库版本不匹配，请重新部署检测库")
    for name, expected in provenance["files"].items():
        if hashlib.sha256((path.parent / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"内嵌 MiaCode 检测库校验失败：{name}")
    if sys.platform == "win32":
        _dll_directory = os.add_dll_directory(str(path.parent))
    lib = ctypes.CDLL(str(path))
    lib.maichart_muri_abi.restype = ctypes.c_int
    if lib.maichart_muri_abi() != 1:
        raise RuntimeError("内嵌 MiaCode C ABI 不兼容")
    lib.maichart_muri_analyze.argtypes = [ctypes.c_char_p, ctypes.c_double]
    lib.maichart_muri_analyze.restype = ctypes.c_void_p
    lib.maichart_muri_free.argtypes = [ctypes.c_void_p]
    lib.maichart_muri_free.restype = None
    return lib


def ensure_available() -> dict:
    with _lock:
        _library()
    return {"engine": "miacode_native", "commit": REFERENCE_COMMIT, "in_process": True}


def analyze_inote(inote: str, *, first: float = 0.0) -> dict:
    if not math.isfinite(first) or "\0" in inote:
        raise ValueError("Invalid MiaCode input/offset")
    started = time.perf_counter()
    # Upstream caches reference data and parser settings. Serialize native
    # calls so concurrent WebUI inspection cannot race those mutable caches.
    with _lock:
        lib = _library()
        ptr = lib.maichart_muri_analyze(inote.encode("utf-8"), float(first))
        if not ptr:
            raise RuntimeError("MiaCode native output allocation failed")
        try:
            result = json.loads(ctypes.string_at(ptr))
        finally:
            lib.maichart_muri_free(ptr)
    if result.get("bridge_error"):
        raise RuntimeError("MiaCode native error: " + result["bridge_error"])
    result["engine"] = "miacode_native"
    result["reference_commit"] = REFERENCE_COMMIT
    result["elapsed_sec"] = time.perf_counter() - started
    result["playable_verified"] = False
    return result


def analyze_maidata(text: str, *, difficulty: int = 5, first: float | None = None) -> dict:
    from maichart.maidata import parse_maidata_metadata

    chart = parse_maidata_metadata(text)
    inote = chart.fields.get(f"inote_{difficulty}")
    if inote is None:
        raise ValueError(f"No inote_{difficulty} in maidata")
    # Prefix the default tempo without adding a line; chart directives still
    # override it. This endpoint is diagnostic only, source columns refer to
    # the prefixed inote; generation uses explicit-tempo event serialization.
    if chart.wholebpm:
        inote = f"({float(chart.wholebpm):.12g})" + inote
    return analyze_inote(inote, first=float(chart.first or 0) if first is None else first)
