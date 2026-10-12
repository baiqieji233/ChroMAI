"""Compatibility facade for the in-process MiaCode-compatible judge.

The public module name is kept because older CLI, WebUI and checkpoint
reports import ``miacode_native``. Runtime analysis now lives in
``miacode_python`` and does not load a DLL, spawn a process, or require a
separately installed MiaCode editor. ``library_path`` remains as a read-only
legacy helper so migration code can inspect an older artifact without using it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from maichart.generation.miacode_python import (
    ENGINE,
    IMPLEMENTATION,
    REFERENCE_COMMIT,
    analyze_inote,
    analyze_maidata,
)


def runtime_directory() -> Path:
    """Return the old artifact directory for migration/debugging only."""
    return Path(__file__).resolve().parents[3] / ".native/miacode/runtime"


def library_path() -> Path:
    """Return the historical DLL path without loading or requiring it."""
    import sys

    name = "maichart_muri_v3.dll" if sys.platform == "win32" else (
        "libmaichart_muri_v3.dylib" if sys.platform == "darwin" else "libmaichart_muri_v3.so"
    )
    return runtime_directory() / name


def ensure_available() -> dict[str, Any]:
    """Return the stable engine contract used by old CLI/WebUI callers."""
    from .muri.data import reference_data
    reference_data()
    return {
        "engine": ENGINE,
        "implementation": IMPLEMENTATION,
        "commit": REFERENCE_COMMIT,
        "in_process": True,
        "dynamic_dependency": False,
        "abi": "python",
    }


def engine_status() -> dict[str, Any]:
    try:
        return {"available": True, **ensure_available()}
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        return {"available": False, "engine": ENGINE,
                "implementation": IMPLEMENTATION, "error": str(exc)}


__all__ = [
    "REFERENCE_COMMIT", "analyze_inote", "analyze_maidata", "ensure_available",
    "engine_status", "library_path", "runtime_directory",
]
