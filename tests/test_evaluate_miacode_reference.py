import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import evaluate_miacode_reference as reference


def _files(tmp_path):
    executable = tmp_path / "开发 构建" / "miacode_muri_dump.exe"
    executable.parent.mkdir()
    executable.write_bytes(b"reference executable identity")
    maidata = tmp_path / "课程 文件" / "谱面.txt"
    maidata.parent.mkdir()
    maidata.write_text("&inote_5=(120){4}1,2,3,4,", encoding="utf-8")
    return executable, maidata


def _completed(stdout, *, returncode=0, stderr=""):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def _driver_report(**overrides):
    value = {
        "ok": True,
        "parse": {"ok": True, "error_count": 0, "errors": [],
                  "warning_count": 0, "warnings": []},
        "diagnostics": [], "static_references": [], "marker_states": [],
        "summary": {"note_count": 4},
    }
    value.update(overrides)
    return value


def test_calls_headless_reference_with_safe_argv_and_preserves_identity(tmp_path, monkeypatch):
    executable, maidata = _files(tmp_path)
    output = tmp_path / "报告 结果.json"
    engine_report = {
        **_driver_report(),
        "parse": {"ok": True, "error_count": 0, "errors": [],
                  "warning_count": 1, "warnings": ["first metadata missing"]},
        "diagnostics": [{"code": "reference-warning", "severity": "info"}],
        "static_references": [{"notes": 4}], "marker_states": [{"break": [False]}],
        "summary": {"note_count": 4, "hard": 999},
    }
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return _completed(json.dumps(engine_report, ensure_ascii=False))

    monkeypatch.setattr(reference.subprocess, "run", fake_run)
    report = reference.evaluate(executable, [maidata], difficulty=5, first=-0.125,
                                output=output, timeout_sec=30)

    assert report["reference"] == {
        "name": "MiaCode Muri headless reference dumper",
        "declared_commit": reference.REFERENCE_COMMIT, "commit_verification": "declared_only",
        "engine_reference_only": True, "playable_verified": False,
    }
    assert report["summary"] == {"input_count": 1, "completed_count": 1, "error_count": 0}
    chart = report["charts"][0]
    assert chart["input_sha256"] == hashlib.sha256(maidata.read_bytes()).hexdigest()
    assert chart["input_sha256_after"] == chart["input_sha256"]
    assert chart["executable_sha256"] == hashlib.sha256(executable.read_bytes()).hexdigest()
    assert chart["raw"] == engine_report
    assert chart["parse"] == engine_report["parse"]
    assert chart["diagnostics"] == engine_report["diagnostics"]
    assert chart["static_references"] == engine_report["static_references"]
    assert chart["marker_states"] == engine_report["marker_states"]
    assert chart["summary"] == engine_report["summary"]
    assert chart["engine_reference_only"] is True
    assert chart["playable_verified"] is False
    argv, kwargs = calls[0]
    assert argv == [str(executable.resolve()), "--maidata", str(maidata.resolve()),
                    "--difficulty", "5", "--with-markers", "--first", "-0.125"]
    assert kwargs["shell"] is False
    assert kwargs["timeout"] == 30
    assert "shell" not in kwargs or kwargs["shell"] is False
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["charts"][0]["summary"]["hard"] == 999


@pytest.mark.parametrize(("stdout", "returncode", "timed_out", "expected_error"), [
    (json.dumps(_driver_report(parse={"ok": False, "error_count": 1, "errors": ["bad chart"],
                                  "warning_count": 0, "warnings": []})), 0, False, True),
    (json.dumps(_driver_report()), 7, False, True),
    ("not-json", 0, False, True),
    (None, None, True, True),
])
def test_bad_parse_timeout_malformed_or_nonzero_are_saved_and_nonzero(
        tmp_path, monkeypatch, stdout, returncode, timed_out, expected_error):
    executable, maidata = _files(tmp_path)
    output = tmp_path / "report.json"

    def fake_run(argv, **kwargs):
        if timed_out:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"], output=b"partial", stderr=b"still working")
        return _completed(stdout, returncode=returncode, stderr="driver stderr")

    monkeypatch.setattr(reference.subprocess, "run", fake_run)
    code = reference.main(["--executable", str(executable), "--maidata", str(maidata),
                           "--output", str(output)])
    assert code == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    chart = report["charts"][0]
    assert chart["status"] == "failed"
    assert bool(chart["error"]) is expected_error
    assert report["summary"]["error_count"] == 1
    if timed_out:
        assert chart["timed_out"] is True
        assert chart["raw_stdout"] == "partial"
        assert chart["stderr"] == "still working"
    if stdout == "not-json":
        assert "not valid JSON" in chart["error"][0]
    if returncode == 7:
        assert chart["exitcode"] == 7
        assert chart["raw"]["parse"]["ok"] is True
    if stdout and 'bad chart' in stdout:
        assert chart["errors"] == ["bad chart"]


def test_multi_input_keeps_per_chart_errors_and_runs_all_inputs(tmp_path, monkeypatch):
    executable, first = _files(tmp_path)
    second = tmp_path / "second.txt"
    second.write_text("&lv_5=13", encoding="utf-8")
    output = tmp_path / "combined.json"
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        body = (_driver_report()
                if len(calls) == 1 else "broken")
        return _completed(body if isinstance(body, str) else json.dumps(body))

    monkeypatch.setattr(reference.subprocess, "run", fake_run)
    report = reference.evaluate(executable, [first, second], output=output)
    assert len(calls) == 2
    assert "--first" not in calls[0]
    assert report["summary"] == {"input_count": 2, "completed_count": 1, "error_count": 1}
    assert [chart["status"] for chart in report["charts"]] == ["completed", "failed"]
    assert report["charts"][1]["input_sha256"] == hashlib.sha256(second.read_bytes()).hexdigest()


@pytest.mark.parametrize("raw,success", [
    ({}, False),
    (_driver_report(ok=False), False),
    (_driver_report(parse={"ok": True, "error_count": 1, "errors": ["malformed maidata"],
                          "warning_count": 0, "warnings": []}), False),
    (_driver_report(), True),
])
def test_schema_or_engine_parse_errors_fail_but_reference_diagnostics_do_not(
        tmp_path, monkeypatch, raw, success):
    executable, maidata = _files(tmp_path)
    output = tmp_path / "report.json"
    raw = dict(raw)
    if raw.get("ok") is True:
        raw["diagnostics"] = [{"kind": "runtime", "severity": "warning"}]
        raw["static_references"] = [{"kind": "marker"}]
        raw["marker_states"] = [{"state": "available"}]
    monkeypatch.setattr(reference.subprocess, "run",
                        lambda *args, **kwargs: _completed(json.dumps(raw)))
    report = reference.evaluate(executable, [maidata], output=output)
    if success:
        assert report["charts"][0]["status"] == "completed"
        assert report["charts"][0]["diagnostics"]
    else:
        assert report["charts"][0]["status"] == "failed"


def test_positive_parse_error_count_fails_even_when_error_array_is_empty(tmp_path, monkeypatch):
    executable, maidata = _files(tmp_path)
    raw = _driver_report(parse={"ok": True, "error_count": 2, "errors": [],
                                "warning_count": 0, "warnings": []})
    monkeypatch.setattr(reference.subprocess, "run",
                        lambda *args, **kwargs: _completed(json.dumps(raw)))
    report = reference.evaluate(executable, [maidata], output=tmp_path / "report.json")
    assert report["charts"][0]["status"] == "failed"


def test_cli_accepts_multiple_maidata_paths_in_one_or_repeated_options(tmp_path):
    executable, first = _files(tmp_path)
    second = tmp_path / "second.txt"
    second.write_text("", encoding="utf-8")
    parsed = reference._parser().parse_args([
        "--executable", str(executable), "--maidata", str(first), str(second),
        "--maidata", str(first), "--output", str(tmp_path / "report.json"),
    ])
    assert parsed.maidata == [first, second, first]


@pytest.mark.parametrize("target", ["source", "executable"])
def test_output_must_not_alias_executable_or_input(tmp_path, target):
    executable, maidata = _files(tmp_path)
    output = executable if target == "executable" else maidata
    with pytest.raises(ValueError, match="must not overwrite"):
        reference.evaluate(executable, [maidata], output=output)


@pytest.mark.parametrize("timeout", [0, 60.01, float("nan"), float("inf")])
def test_timeout_is_bounded_and_finite(tmp_path, timeout):
    executable, maidata = _files(tmp_path)
    with pytest.raises(ValueError, match="timeout-sec"):
        reference.evaluate(executable, [maidata], output=tmp_path / "report.json", timeout_sec=timeout)


def test_difficulty_range_and_first_override_are_validated(tmp_path):
    executable, maidata = _files(tmp_path)
    with pytest.raises(ValueError, match="difficulty"):
        reference.evaluate(executable, [maidata], difficulty=8, output=tmp_path / "report.json")
    with pytest.raises(ValueError, match="first override"):
        reference.evaluate(executable, [maidata], first=float("nan"), output=tmp_path / "report.json")
