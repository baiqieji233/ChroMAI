import json
from pathlib import Path

import pytest

from maichart.generation import runtime_reference


def _driver_payload(alert_level="Muri"):
    return {
        "ok": True,
        "parse": {"ok": True, "error_count": 0, "warning_count": 0, "errors": [], "warnings": []},
        "summary": {"diagnostic_count": 1, "static_reference_count": 1},
        "diagnostics": [{"kind": "TapOnSlide", "alert_level": alert_level, "detail": "reference"}],
        "static_references": [], "marker_states": [], "visible_panel_entries": [],
    }


def test_reference_report_preserves_raw_schema_and_does_not_reject(monkeypatch, tmp_path):
    executable = tmp_path / "miacode_muri_dump.exe"
    executable.write_bytes(b"reference")

    class Completed:
        returncode = 0
        stdout = json.dumps(_driver_payload())
        stderr = ""

    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        return Completed()

    monkeypatch.setattr(runtime_reference.subprocess, "run", fake_run)
    result = runtime_reference.run_reference(executable, "&first=0\n&wholebpm=120\n", difficulty=5,
                                             first=0.0, mode="report")

    assert result["status"] == "completed"
    assert result["reject_reason"] is None
    assert result["raw"]["diagnostics"][0]["alert_level"] == "Muri"
    assert result["reference"]["engine_reference_only"] is True
    assert result["reference"]["playable_verified"] is False
    assert seen["kwargs"]["shell"] is False
    assert "--maidata" in seen["argv"] and "--with-markers" in seen["argv"]


def test_reference_rejects_only_muri_alerts(monkeypatch, tmp_path):
    executable = tmp_path / "miacode_muri_dump.exe"
    executable.write_bytes(b"reference")

    class Completed:
        returncode = 0
        stdout = json.dumps(_driver_payload("Warning"))
        stderr = ""

    monkeypatch.setattr(runtime_reference.subprocess, "run", lambda *args, **kwargs: Completed())
    result = runtime_reference.run_reference(executable, "chart", difficulty=5, first=0.0, mode="reject")
    assert result["status"] == "completed"
    assert result["reject_reason"] is None

    Completed.stdout = json.dumps(_driver_payload("Muri"))
    result = runtime_reference.run_reference(executable, "chart", difficulty=5, first=0.0, mode="reject")
    assert result["status"] == "rejected"
    assert result["reject_reason"] == "muri_alert"


def test_reference_failure_is_conservative_reject_but_keeps_error(monkeypatch, tmp_path):
    executable = tmp_path / "miacode_muri_dump.exe"
    executable.write_bytes(b"reference")
    monkeypatch.setattr(runtime_reference.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("boom")))

    # TimeoutError is not a subprocess.TimeoutExpired and should surface as a
    # normal driver failure only when wrapped by the caller; this test verifies
    # the explicit OSError path without invoking an executable.
    monkeypatch.setattr(runtime_reference.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("missing dll")))
    result = runtime_reference.run_reference(executable, "chart", difficulty=5, first=0.0, mode="reject")
    assert result["status"] == "failed"
    assert result["reject_reason"] == "reference_failed"
    assert "missing dll" in result["errors"][0]


def test_reference_timeout_is_bounded(tmp_path):
    executable = tmp_path / "reference.exe"
    executable.write_bytes(b"reference")
    with pytest.raises(ValueError):
        runtime_reference.run_reference(executable, "chart", difficulty=5, first=0.0, timeout_sec=0)


def test_subprocess_timeout_is_recorded_as_reference_failure(monkeypatch, tmp_path):
    executable = tmp_path / "miacode_muri_dump.exe"
    executable.write_bytes(b"reference")
    monkeypatch.setattr(runtime_reference.subprocess, "run",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            runtime_reference.subprocess.TimeoutExpired(args[0], kwargs["timeout"])))
    result = runtime_reference.run_reference(executable, "chart", difficulty=5,
                                             first=0.0, mode="reject")
    assert result["status"] == "failed"
    assert result["reject_reason"] == "reference_failed"
    assert result["errors"] and "timed out" in result["errors"][0]
