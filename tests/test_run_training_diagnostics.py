import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.run_training_diagnostics import load_config, run_config


DIGEST = "a" * 64


def _write_config(tmp_path, *, kind="event_tokens", pipeline_digest=DIGEST):
    checkpoint = tmp_path / "event_epoch_015.pt"
    config = {
        "pipeline_state": "pipeline_state.json",
        "config_sha256": pipeline_digest,
        "state_path": "diagnostics_state.json",
        "poll_seconds": 0,
        "jobs": [{
            "training_job_name": "train-main", "checkpoint": str(checkpoint), "kind": kind,
            "manifest": "manifest.json", "cache_dir": "cache", "limit": 128,
            "batch_size": 2, "output": "diagnostics/report.json",
        }],
    }
    path = tmp_path / "diagnostics_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path, checkpoint


def _pipeline(path, *, config_digest=DIGEST, status="completed", include_checkpoint=True,
              pipeline_status=None):
    result = {"paths": [str(path.parent / "event_epoch_015.pt")] if include_checkpoint else []}
    pipeline = {
        "config_sha256": config_digest,
        "status": pipeline_status or ("running" if status == "running" else
                                      "failed" if status in {"failed", "cancelled"} else "completed"),
        "jobs": [{
            "name": "train-main", "status": status,
            "last_snapshot": {"status": status, "result": result,
                              "error": "training exploded" if status == "failed" else None},
            "task_result": result,
        }],
    }
    (path.parent / "pipeline_state.json").write_text(json.dumps(pipeline), encoding="utf-8")


def test_uncompleted_training_waits_without_reading_checkpoint(tmp_path, monkeypatch):
    config, checkpoint = _write_config(tmp_path)
    _pipeline(config, status="running")
    calls = []
    monkeypatch.setattr("tools.evaluate_event_tokens.evaluate", lambda args: calls.append(args))

    assert run_config(config, sleep_fn=lambda _: None, max_polls=1) == 3
    state = json.loads((tmp_path / "diagnostics_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "waiting"
    assert state["jobs"][0]["status"] == "pending"
    assert not checkpoint.exists()
    assert calls == []


def test_completed_training_runs_cpu_shuffle_evaluator_once_and_records_identity(tmp_path, monkeypatch):
    config, checkpoint = _write_config(tmp_path)
    checkpoint.write_bytes(b"immutable checkpoint")
    # An unrelated generation failure must not block a completed training job.
    _pipeline(config, pipeline_status="completed_with_errors")
    observed = []

    def evaluate(args):
        observed.append(args)
        return {"schema": "fixture", "checkpoint": str(args.checkpoint),
                "evaluation": {"windows_evaluated": 128}, "audio_ablation_metrics": {"f1": 0.2}}

    monkeypatch.setattr("tools.evaluate_event_tokens.evaluate", evaluate)
    assert run_config(config, sleep_fn=lambda _: None) == 0
    state_path = tmp_path / "diagnostics_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    job = state["jobs"][0]
    expected_hash = hashlib.sha256(b"immutable checkpoint").hexdigest()
    assert state["status"] == "completed"
    assert job["status"] == "completed"
    assert job["checkpoint_sha256"] == job["checkpoint_sha256_after"] == expected_hash
    assert job["report_path"] == str((tmp_path / "diagnostics/report.json").resolve())
    assert json.loads(Path(job["report_path"]).read_text(encoding="utf-8"))["evaluation"]["windows_evaluated"] == 128
    assert len(observed) == 1
    assert observed[0].checkpoint == checkpoint.resolve()
    assert observed[0].limit == 128 and observed[0].batch_size == 2
    assert observed[0].audio_ablation == "shuffle_acoustic"

    assert run_config(config, sleep_fn=lambda _: None) == 0
    assert len(observed) == 1


def test_onset_alignment_requests_both_ablation_modes(tmp_path, monkeypatch):
    config, checkpoint = _write_config(tmp_path, kind="onset_alignment")
    checkpoint.write_bytes(b"immutable checkpoint")
    _pipeline(config)
    observed = []
    monkeypatch.setattr("tools.evaluate_onset_alignment.evaluate",
                        lambda args: observed.append(args) or {"schema": "onset-fixture"})

    assert run_config(config, sleep_fn=lambda _: None) == 0
    assert observed[0].shuffle_acoustic is True
    assert observed[0].time_only is True
    assert observed[0].batch_size == 2


def test_pipeline_digest_mismatch_fails_before_checkpoint_access(tmp_path, monkeypatch):
    config, checkpoint = _write_config(tmp_path)
    _pipeline(config, config_digest="b" * 64)
    calls = []
    monkeypatch.setattr("tools.evaluate_event_tokens.evaluate", lambda args: calls.append(args))

    assert run_config(config, sleep_fn=lambda _: None) == 2
    state = json.loads((tmp_path / "diagnostics_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert "digest mismatch" in state["error"]
    assert not checkpoint.exists()
    assert calls == []


def test_training_failure_and_checkpoint_not_listed_are_explicit(tmp_path, monkeypatch):
    config, checkpoint = _write_config(tmp_path)
    checkpoint.write_bytes(b"checkpoint")
    _pipeline(config, status="failed")
    monkeypatch.setattr("tools.evaluate_event_tokens.evaluate", lambda args: pytest.fail("must not evaluate"))
    assert run_config(config, sleep_fn=lambda _: None) == 2
    failed = json.loads((tmp_path / "diagnostics_state.json").read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert "training job failed" in failed["error"]

    _pipeline(config, status="completed", include_checkpoint=False)
    assert run_config(config, sleep_fn=lambda _: None) == 2
    state = json.loads((tmp_path / "diagnostics_state.json").read_text(encoding="utf-8"))
    assert state["jobs"][0]["status"] == "failed"
    assert "does not list" in state["jobs"][0]["error"]
    assert state["jobs"][0]["checkpoint_sha256"] is None


def test_unknown_kind_is_rejected_as_config_error(tmp_path):
    config, _ = _write_config(tmp_path, kind="arbitrary_command")
    with pytest.raises(ValueError, match="event_tokens or onset_alignment"):
        load_config(config)


def test_state_identity_mismatch_prevents_resume_with_different_config(tmp_path):
    config, checkpoint = _write_config(tmp_path)
    _pipeline(config, status="running")
    assert run_config(config, sleep_fn=lambda _: None, max_polls=1) == 3
    data = json.loads(config.read_text(encoding="utf-8"))
    data["jobs"][0]["limit"] = 64
    config.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="different diagnostics config"):
        run_config(config, sleep_fn=lambda _: None, max_polls=1)


@pytest.mark.parametrize("alias", ["pipeline", "state", "checkpoint", "manifest", "config"])
def test_output_alias_of_protected_path_is_rejected(tmp_path, alias):
    config_path, checkpoint = _write_config(tmp_path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    values = {
        "pipeline": data["pipeline_state"], "state": data["state_path"],
        "checkpoint": data["jobs"][0]["checkpoint"], "manifest": data["jobs"][0]["manifest"],
        "config": str(config_path),
    }
    data["jobs"][0]["output"] = values[alias]
    config_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="output aliases a protected"):
        load_config(config_path)


def test_state_path_cannot_alias_pipeline_state(tmp_path):
    config_path, _ = _write_config(tmp_path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    data["state_path"] = data["pipeline_state"]
    config_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="state_path must not alias pipeline_state"):
        load_config(config_path)


@pytest.mark.parametrize("alias", ["config", "checkpoint", "manifest"])
def test_state_path_cannot_alias_configured_input(tmp_path, alias):
    config_path, checkpoint = _write_config(tmp_path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    values = {"config": str(config_path), "checkpoint": str(checkpoint),
              "manifest": data["jobs"][0]["manifest"]}
    data["state_path"] = values[alias]
    config_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="state_path must not alias"):
        load_config(config_path)


def test_output_is_checked_against_later_jobs_inputs(tmp_path):
    config_path, checkpoint = _write_config(tmp_path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    later_manifest = tmp_path / "second_manifest.json"
    data["jobs"][0]["output"] = str(later_manifest)
    data["jobs"].append({
        "training_job_name": "train-second", "checkpoint": str(checkpoint),
        "kind": "event_tokens", "manifest": str(later_manifest), "cache_dir": "cache",
        "output": "diagnostics/second.json",
    })
    config_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="output aliases a protected"):
        load_config(config_path)


@pytest.mark.parametrize("poll_seconds", [float("nan"), float("inf"), -float("inf")])
def test_poll_interval_must_be_finite(tmp_path, poll_seconds):
    config_path, _ = _write_config(tmp_path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    data["poll_seconds"] = poll_seconds
    config_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="finite and nonnegative"):
        load_config(config_path)


@pytest.mark.parametrize("pipeline_status", ["failed", "cancelled", "dependency_failed", "completed_with_errors"])
@pytest.mark.parametrize("job_present", [True, False])
def test_terminal_upstream_status_fails_pending_or_absent_selected_job(
        tmp_path, pipeline_status, job_present, monkeypatch):
    config_path, checkpoint = _write_config(tmp_path)
    jobs = [{"name": "train-main", "status": "pending"}] if job_present else []
    pipeline = {"config_sha256": DIGEST, "status": pipeline_status, "jobs": jobs}
    (tmp_path / "pipeline_state.json").write_text(json.dumps(pipeline), encoding="utf-8")
    monkeypatch.setattr("tools.evaluate_event_tokens.evaluate",
                        lambda args: pytest.fail("pending training must not evaluate"))

    assert run_config(config_path, sleep_fn=lambda _: None, max_polls=1) == 2
    state = json.loads((tmp_path / "diagnostics_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert pipeline_status in state["jobs"][0]["error"]
    assert not checkpoint.exists()


def test_resume_skips_completed_diagnostic_and_retries_only_failed_one(tmp_path, monkeypatch):
    config, checkpoint = _write_config(tmp_path)
    data = json.loads(config.read_text(encoding="utf-8"))
    data["jobs"].append({
        "training_job_name": "train-main", "checkpoint": str(checkpoint),
        "kind": "onset_alignment", "manifest": "manifest.json", "cache_dir": "cache",
        "limit": 128, "batch_size": 4, "output": "diagnostics/onset.json",
    })
    # Each observer entry refers to its own completed pipeline training job.
    data["jobs"][1]["training_job_name"] = "train-onset"
    config.write_text(json.dumps(data), encoding="utf-8")
    checkpoint.write_bytes(b"immutable checkpoint")
    paths = [str(checkpoint)]
    pipeline = {"config_sha256": DIGEST, "jobs": [
        {"name": name, "status": "completed", "last_snapshot": {
            "status": "completed", "result": {"paths": paths}}, "task_result": {"paths": paths}}
        for name in ("train-main", "train-onset")
    ]}
    (tmp_path / "pipeline_state.json").write_text(json.dumps(pipeline), encoding="utf-8")

    token_calls = []
    onset_calls = []
    monkeypatch.setattr("tools.evaluate_event_tokens.evaluate",
                        lambda args: token_calls.append(args) or {"schema": "tokens"})

    def onset_evaluate(args):
        onset_calls.append(args)
        if len(onset_calls) == 1:
            raise RuntimeError("temporary onset evaluator failure")
        return {"schema": "onset"}

    monkeypatch.setattr("tools.evaluate_onset_alignment.evaluate", onset_evaluate)
    assert run_config(config, sleep_fn=lambda _: None) == 2
    state = json.loads((tmp_path / "diagnostics_state.json").read_text(encoding="utf-8"))
    assert [job["status"] for job in state["jobs"]] == ["completed", "failed"]

    assert run_config(config, sleep_fn=lambda _: None) == 0
    state = json.loads((tmp_path / "diagnostics_state.json").read_text(encoding="utf-8"))
    assert [job["status"] for job in state["jobs"]] == ["completed", "completed"]
    assert len(token_calls) == 1
    assert len(onset_calls) == 2
