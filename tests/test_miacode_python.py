"""Native-oracle snapshots and absence-of-native-runtime integration checks."""
import ctypes
import json
from pathlib import Path
import subprocess

import pytest

from maichart.generation import miacode_native as compatibility
from maichart.generation.miacode_python import analyze_inote
from maichart.generation.muri.parser import parse_markers
from tools.compare_muri_python import equivalent, normalized

CASES = json.loads((Path(__file__).parent / "fixtures/miacode_python_reference.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['inote']}|{c['first']}|{c['threshold_ms']}")
def test_pinned_abi3_decisions(case):
    actual = analyze_inote(case["inote"], first=case["first"], tap_on_slide_threshold_ms=case["threshold_ms"])
    assert equivalent(normalized(actual), case["expected"]), (normalized(actual), case["expected"])


def test_no_library_or_executable_is_required(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Python engine attempted to load/launch a native runtime")
    monkeypatch.setattr(ctypes, "CDLL", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(compatibility, "runtime_directory", lambda: tmp_path / "missing")
    monkeypatch.setattr(compatibility, "library_path", lambda: tmp_path / "missing.dll")
    status = compatibility.ensure_available()
    assert status["engine"] == "miacode_python" and not status["dynamic_dependency"]
    result = compatibility.analyze_maidata("&wholebpm=120\n&inote_5={4}1h[1:1],,,,1,E")
    assert result["ok"] and any(d["kind"] == "Overlap" for d in result["diagnostics"])


def test_all_packaged_paths_are_parsed():
    from maichart.generation.muri.data import reference_data
    data = reference_data()
    for group in ("slides", "wifi"):
        for key in data[group]:
            markers, errors, _ = parse_markers(f"(120){{4}}{key}[4:1],E")
            assert not errors, (key, errors)
            assert markers and markers[0]["end_second"] > markers[0]["slide_trace_second"]


@pytest.mark.parametrize("text", ["(nan){4}1,E", "(0){4}1,E", "(120){0}1,E",
                                  "(120){4}Z,E", "(120){4}A9,E", "(120){4}1h[#inf],E"])
def test_malformed_input_is_never_a_clean_pass(text):
    report = analyze_inote(text)
    assert not report["ok"] and report["parse"]["errors"]


def test_corrupt_data_is_reported_unavailable(monkeypatch):
    from maichart.generation.muri import data
    def bad_data():
        raise RuntimeError("reference data is corrupt")
    monkeypatch.setattr(data, "reference_data", bad_data)
    status = compatibility.engine_status()
    assert not status["available"] and "corrupt" in status["error"]
