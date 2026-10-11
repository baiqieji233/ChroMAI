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
    name = "maichart_muri_v2.dll" if sys.platform == "win32" else (
        "libmaichart_muri_v2.dylib" if sys.platform == "darwin" else "libmaichart_muri_v2.so")
    return runtime_directory() / name


@lru_cache(maxsize=1)
def _library():
    global _dll_directory
    path = library_path()
    if not path.is_file():
        raise RuntimeError("内嵌 MiaCode 检测库未部署；请运行 python tools/build_muri.py。无需安装或打开 MiaCode 编辑器。")
    provenance = json.loads((path.parent / "build.json").read_text(encoding="utf-8"))
    if provenance.get("commit") != REFERENCE_COMMIT or provenance.get("abi") != 2:
        raise RuntimeError("内嵌 MiaCode 检测库版本不匹配，请运行 python tools/build_muri.py")
    for name, expected in provenance["files"].items():
        if hashlib.sha256((path.parent / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"内嵌 MiaCode 检测库校验失败：{name}")
    if sys.platform == "win32":
        _dll_directory = os.add_dll_directory(str(path.parent))
    lib = ctypes.CDLL(str(path))
    lib.maichart_muri_abi.restype = ctypes.c_int
    if lib.maichart_muri_abi() != 2:
        raise RuntimeError("内嵌 MiaCode C ABI2 不兼容")
    lib.maichart_muri_analyze_v2.argtypes = [ctypes.c_char_p, ctypes.c_double, ctypes.c_double]
    lib.maichart_muri_analyze_v2.restype = ctypes.c_void_p
    lib.maichart_muri_free.argtypes = [ctypes.c_void_p]
    lib.maichart_muri_free.restype = None
    return lib


def ensure_available() -> dict:
    with _lock:
        _library()
    return {"engine": "miacode_native", "commit": REFERENCE_COMMIT, "in_process": True}


def engine_status() -> dict:
    try:
        return {"available": True, **ensure_available()}
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        return {"available": False, "engine": "miacode_native", "error": str(exc)}


def analyze_inote(inote: str, *, first: float = 0.0,
                  tap_on_slide_threshold_ms: float = 200.0) -> dict:
    from maichart.generation.playability_policy import validate_tap_on_slide_threshold_ms

    if not math.isfinite(first) or "\0" in inote:
        raise ValueError("Invalid MiaCode input/offset")
    threshold = validate_tap_on_slide_threshold_ms(tap_on_slide_threshold_ms)
    started = time.perf_counter()
    # Upstream caches reference data and parser settings. Serialize native
    # calls so concurrent WebUI inspection cannot race those mutable caches.
    with _lock:
        lib = _library()
        ptr = lib.maichart_muri_analyze_v2(
            inote.encode("utf-8"), float(first), threshold / 1000.0)
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
    result["tap_on_slide_threshold_ms"] = threshold
    result["elapsed_sec"] = time.perf_counter() - started
    result["playable_verified"] = False
    return result


def analyze_maidata(text: str, *, difficulty: int = 5, first: float | None = None,
                    tap_on_slide_threshold_ms: float = 200.0) -> dict:
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
    return analyze_inote(
        inote,
        first=float(chart.first or 0) if first is None else first,
        tap_on_slide_threshold_ms=tap_on_slide_threshold_ms,
    )
