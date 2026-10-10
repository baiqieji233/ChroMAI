import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.run_staged_rhythm_validation import (
    StateFileLock,
    StateLockError,
    StagedRhythmValidation,
    _load_config,
    _generation_output_paths,
    run_config,
)


class FakeClient:
    def __init__(self, *, gets=None, posts=None, on_post=None):
        self.gets = {key: list(value) if isinstance(value, list) else [value]
                     for key, value in (gets or {}).items()}
        self.posts = list(posts or [])
        self.get_calls = []
        self.post_calls = []
        self.timeouts = []
        self.on_post = on_post

    def request(self, method, url, *, payload, timeout):
        self.timeouts.append(timeout)
        if method == "GET":
            task_id = url.split("id=")[-1]
            self.get_calls.append(task_id)
            sequence = self.gets[task_id]
            item = sequence.pop(0) if len(sequence) > 1 else sequence[0]
            if isinstance(item, BaseException):
                raise item
            return item
        self.post_calls.append((url, payload))
        if self.on_post is not None:
            self.on_post(payload)
        item = self.posts.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _config(tmp_path, *, jobs=None, evaluation_paths=None, poll_seconds=0, http_timeout_sec=20,
            reference_dir=None):
    config = {
        "base_url": "http://127.0.0.1:7862",
        "initial_task_id": "train-001",
        "jobs": jobs or [],
        "state_path": "controller_state.json",
        "report_path": "controller_report.md",
        "evaluation_report_path": "evaluation.json",
        "evaluation_paths": evaluation_paths or [],
        "poll_seconds": poll_seconds,
        "http_timeout_sec": http_timeout_sec,
    }
    if reference_dir is not None:
        config["reference_dir"] = str(reference_dir)
    path = tmp_path / "controller_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _task(status, **extra):
    return {"task_id": "remote", "status": status, **extra}


def test_initial_training_is_observed_then_jobs_run_once_and_evaluate(tmp_path):
    jobs = [
        {"name": "gen-a", "endpoint": "/api/generate", "payload": {"song": "a"},
         "continue_on_failure": True},
        {"name": "gen-b", "endpoint": "/api/generate", "payload": {"song": "b"}},
    ]
    chart = tmp_path / "song.chart.json"
    chart.write_text(json.dumps({
        "schema": "maichart-event-chart-v1",
        "points": [{"beat": "0", "note_text": "1"}, {"beat": "1", "note_text": "2"},
                   {"beat": "2", "note_text": "3"}, {"beat": "3", "note_text": "4"},
                   {"beat": "4", "note_text": ""}],
        "tempo_changes": [{"beat": "0", "bpm": 120}], "first": 0.0,
        "end_beat": "4", "audio_duration_sec": 2.0,
    }), encoding="utf-8")
    config_path = _config(tmp_path, jobs=jobs, evaluation_paths=[str(chart)])
    client = FakeClient(
        gets={"train-001": _task("completed"), "gen-a": _task("failed", error="generation failed"),
              "gen-b": _task("completed")},
        posts=[{"task_id": "gen-a"}, {"task_id": "gen-b"}],
    )

    assert run_config(config_path, client=client, sleep_fn=lambda _: None) == 2

    assert len(client.post_calls) == 2
    assert client.get_calls == ["train-001", "gen-a", "gen-b"]
    state = json.loads((tmp_path / "controller_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed_with_errors"
    assert [job["status"] for job in state["jobs"]] == ["failed", "completed"]
    assert state["evaluation"]["status"] == "partial"
    assert state["evaluation"]["artifact_completeness"]["unverifiable_count"] == 2
    assert state["evaluation"]["human_review"] == "pending"
    assert (tmp_path / "evaluation.json").is_file()
    markdown = (tmp_path / "controller_report.md").read_text(encoding="utf-8")
    assert "Human review: pending" in markdown
    assert "No pass/fail playability conclusion" in markdown


def test_failed_training_job_stops_dependent_jobs_even_if_continue_is_set(tmp_path):
    config_path = _config(tmp_path, jobs=[
        {"name": "finetune", "kind": "training", "endpoint": "/api/train",
         "payload": {}, "continue_on_failure": True},
        {"name": "generate", "endpoint": "/api/generate", "payload": {}},
    ])
    client = FakeClient(gets={"train-001": _task("completed"), "ft-1": _task("failed")},
                        posts=[{"task_id": "ft-1"}])

    assert run_config(config_path, client=client, sleep_fn=lambda _: None) == 1
    assert len(client.post_calls) == 1
    assert client.get_calls == ["train-001", "ft-1"]
    state = json.loads((tmp_path / "controller_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["jobs"][1]["status"] == "pending"


def test_post_timeout_is_uncertain_and_resume_never_posts_again(tmp_path):
    config_path = _config(tmp_path, jobs=[
        {"name": "generation", "endpoint": "/api/generate", "payload": {}}
    ])
    client = FakeClient(gets={"train-001": _task("completed")}, posts=[TimeoutError("lost response")])

    assert run_config(config_path, client=client, sleep_fn=lambda _: None) == 3
    assert run_config(config_path, client=client, sleep_fn=lambda _: None) == 3
    assert len(client.post_calls) == 1
    state = json.loads((tmp_path / "controller_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "submit_uncertain"
    assert state["jobs"][0]["task_id"] is None


def test_reconciled_task_id_can_resume_without_resubmission(tmp_path):
    config_path = _config(tmp_path, jobs=[
        {"name": "generation", "endpoint": "/api/generate", "payload": {}}
    ])
    client = FakeClient(gets={"train-001": _task("completed")}, posts=[TimeoutError("lost response")])
    assert run_config(config_path, client=client, sleep_fn=lambda _: None) == 3

    state_path = tmp_path / "controller_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["jobs"][0].update(task_id="reconciled-generation-id", status="submitted")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    client.gets["reconciled-generation-id"] = [_task("completed")]

    assert run_config(config_path, client=client, sleep_fn=lambda _: None) == 2
    assert len(client.post_calls) == 1
    assert client.get_calls[-1] == "reconciled-generation-id"


def test_three_failed_observations_keep_same_live_task_id_for_resume(tmp_path):
    config_path = _config(tmp_path)
    client = FakeClient(gets={"train-001": [TimeoutError("offline")] * 3 + [_task("completed")]})

    assert run_config(config_path, client=client, sleep_fn=lambda _: None) == 3
    state = json.loads((tmp_path / "controller_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "observation_unavailable"
    assert state["initial"]["task_id"] == "train-001"
    assert run_config(config_path, client=client, sleep_fn=lambda _: None) == 0
    assert client.get_calls == ["train-001"] * 4


def test_ctrl_c_records_interruption_without_cancelling_remote_task(tmp_path):
    config_path = _config(tmp_path)
    client = FakeClient(gets={"train-001": [_task("running"), _task("completed")]})

    def interrupt(_seconds):
        raise KeyboardInterrupt

    assert run_config(config_path, client=client, sleep_fn=interrupt) == 130
    state = json.loads((tmp_path / "controller_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "controller_interrupted"
    assert state["initial"]["task_id"] == "train-001"
    assert run_config(config_path, client=client, sleep_fn=lambda _: None) == 0
    assert client.post_calls == []


def test_poll_and_http_timeouts_are_capped(tmp_path):
    config_path = _config(tmp_path, poll_seconds=90, http_timeout_sec=90)
    client = FakeClient(gets={"train-001": [_task("running"), _task("completed")]})
    sleeps = []

    assert run_config(config_path, client=client, sleep_fn=sleeps.append) == 0
    assert client.timeouts == [20.0, 20.0]
    assert sleeps == [30.0]


def test_all_generation_failures_return_completed_with_errors(tmp_path):
    config_path = _config(tmp_path, jobs=[
        {"name": "gen-a", "endpoint": "/api/generate", "payload": {}, "continue_on_failure": True},
        {"name": "gen-b", "endpoint": "/api/generate", "payload": {}, "continue_on_failure": True},
    ])
    client = FakeClient(
        gets={"train-001": _task("completed"), "gen-a": _task("failed", error="bad A"),
              "gen-b": _task("failed", error="bad B")},
        posts=[{"task_id": "gen-a"}, {"task_id": "gen-b"}],
    )

    assert run_config(config_path, client=client, sleep_fn=lambda _: None) == 2
    state = json.loads((tmp_path / "controller_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed_with_errors"
    assert [job["status"] for job in state["jobs"]] == ["failed", "failed"]
    assert [job["error"] for job in state["jobs"]] == ["bad A", "bad B"]
    assert state["human_review"] == "pending"


def test_evaluation_exception_and_zero_results_with_errors_are_not_green(tmp_path):
    missing_config = _config(tmp_path / "exception", evaluation_paths=[str(tmp_path / "absent")])
    missing_client = FakeClient(gets={"train-001": _task("completed")})

    assert run_config(missing_config, client=missing_client, sleep_fn=lambda _: None) == 2
    missing_state = json.loads((tmp_path / "exception" / "controller_state.json").read_text(encoding="utf-8"))
    assert missing_state["status"] == "completed_with_errors"
    assert missing_state["evaluation"]["status"] == "failed"
    assert missing_state["evaluation"]["error"]
    assert missing_state["human_review"] == "pending"

    bad_chart = tmp_path / "bad.chart.json"
    bad_chart.write_text("{}", encoding="utf-8")
    bad_config = _config(tmp_path / "zero-results", evaluation_paths=[str(bad_chart)])
    bad_client = FakeClient(gets={"train-001": _task("completed")})
    assert run_config(bad_config, client=bad_client, sleep_fn=lambda _: None) == 2
    bad_state = json.loads((tmp_path / "zero-results" / "controller_state.json").read_text(encoding="utf-8"))
    assert bad_state["evaluation"]["status"] == "failed"
    assert bad_state["evaluation"]["evaluated_count"] == 0
    assert bad_state["evaluation"]["error_count"] == 1
    assert bad_state["evaluation"]["error_details"]


def test_partial_evaluation_is_nonzero_and_keeps_error_details(tmp_path):
    chart_dir = tmp_path / "charts"
    chart_dir.mkdir()
    good = chart_dir / "good.chart.json"
    good.write_text(json.dumps({
        "schema": "maichart-event-chart-v1",
        "points": [{"beat": "0", "note_text": "1"}, {"beat": "1", "note_text": "2"},
                   {"beat": "2", "note_text": "3"}, {"beat": "3", "note_text": "4"},
                   {"beat": "4", "note_text": ""}],
        "tempo_changes": [{"beat": "0", "bpm": 120}], "first": 0.0,
        "end_beat": "4", "audio_duration_sec": 2.0,
    }), encoding="utf-8")
    (chart_dir / "bad.chart.json").write_text("{}", encoding="utf-8")
    config_path = _config(tmp_path, evaluation_paths=[str(chart_dir)])
    client = FakeClient(gets={"train-001": _task("completed")})

    assert run_config(config_path, client=client, sleep_fn=lambda _: None) == 2
    state = json.loads((tmp_path / "controller_state.json").read_text(encoding="utf-8"))
    assert state["evaluation"]["status"] == "partial"
    assert state["evaluation"]["evaluated_count"] == 1
    assert state["evaluation"]["error_count"] == 1
    assert state["status"] == "completed_with_errors"


def test_successful_generation_with_only_stale_outputs_is_incomplete(tmp_path):
    output = tmp_path / "generated" / "v3_full" / "song_00.txt"
    chart = output.with_suffix(".chart.json")
    output.parent.mkdir(parents=True)
    output.write_text("stale chart text", encoding="utf-8")
    chart.write_text(json.dumps({
        "schema": "maichart-event-chart-v1",
        "points": [{"beat": "0", "note_text": "1"}, {"beat": "1", "note_text": "2"},
                   {"beat": "2", "note_text": "3"}, {"beat": "3", "note_text": "4"},
                   {"beat": "4", "note_text": ""}],
        "tempo_changes": [{"beat": "0", "bpm": 120}], "first": 0.0,
        "end_beat": "4", "audio_duration_sec": 2.0,
    }), encoding="utf-8")
    (output.with_suffix(".quality.json")).write_text("{}", encoding="utf-8")
    config_path = _config(tmp_path, jobs=[{
        "name": "generate", "endpoint": "/api/generate",
        "payload": {"output": str(output)}, "continue_on_failure": True,
    }], evaluation_paths=[str(output.parent)])
    client = FakeClient(
        gets={"train-001": _task("completed"),
              "gen-1": _task("completed", result={
                  "model_family": "event_ctx_v1",
                  "paths": [str(output), str(chart), str(output.with_suffix(".quality.json"))],
              })},
        posts=[{"task_id": "gen-1"}],
    )

    assert run_config(config_path, client=client, sleep_fn=lambda _: None) == 2

    state = json.loads((tmp_path / "controller_state.json").read_text(encoding="utf-8"))
    completeness = state["evaluation"]["artifact_completeness"]
    assert completeness["expected_count"] == 3
    assert completeness["produced_count"] == 0
    assert completeness["missing_count"] == 3
    assert set(completeness["stale_paths"]) == {
        str(output.resolve()), str(chart.resolve()), str(output.with_suffix(".quality.json").resolve()),
    }
    assert state["evaluation"]["status"] == "partial"
    assert state["evaluation"]["evaluated_count"] == 0
    metrics = json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8"))
    assert metrics["charts"] == []
    assert metrics["artifact_completeness"]["produced_count"] == 0


def test_expected_candidate_artifact_names_match_webui(tmp_path):
    output = tmp_path / "song.txt"
    artifacts = _generation_output_paths({
        "payload": {"output": str(output), "candidates": 2}
    })

    assert [Path(item["path"]).name for item in artifacts if item["kind"] == "text"] == [
        "song_candidate_001.txt", "song_candidate_002.txt",
    ]
    assert [Path(item["path"]).name for item in artifacts if item["kind"] == "chart_json"] == [
        "song.chart_candidate_001.json", "song.chart_candidate_002.json",
    ]


def test_default_generation_records_returned_paths_as_unverifiable(tmp_path):
    output = tmp_path / "generated" / "server-task_maidata.txt"
    chart = output.with_suffix(".chart.json")
    output.parent.mkdir(parents=True)
    output.write_text("new chart", encoding="utf-8")
    chart.write_text(json.dumps({
        "schema": "maichart-event-chart-v1",
        "points": [{"beat": "0", "note_text": "1"}, {"beat": "1", "note_text": "2"},
                   {"beat": "2", "note_text": "3"}, {"beat": "3", "note_text": "4"},
                   {"beat": "4", "note_text": ""}],
        "tempo_changes": [{"beat": "0", "bpm": 120}], "first": 0.0,
        "end_beat": "4", "audio_duration_sec": 2.0,
    }), encoding="utf-8")
    config_path = _config(tmp_path, jobs=[{
        "name": "generate", "endpoint": "/api/generate", "payload": {},
    }])
    client = FakeClient(
        gets={"train-001": _task("completed"),
              "gen-1": _task("completed", result={
                  "model_family": "event_ctx_v1", "paths": [str(output), str(chart)],
              })},
        posts=[{"task_id": "gen-1"}],
    )

    assert run_config(config_path, client=client, sleep_fn=lambda _: None) == 2

    state = json.loads((tmp_path / "controller_state.json").read_text(encoding="utf-8"))
    completeness = state["evaluation"]["artifact_completeness"]
    assert completeness["expected_count"] == 0
    assert completeness["produced_count"] == 2
    assert completeness["unverifiable_count"] == 1
    assert state["evaluation"]["evaluated_count"] == 1
    assert state["evaluation"]["status"] == "partial"


def test_completed_task_restores_result_from_last_snapshot_on_resume(tmp_path):
    output = tmp_path / "generated" / "server-task_maidata.txt"
    chart = output.with_suffix(".chart.json")
    quality = output.with_suffix(".quality.json")
    output.parent.mkdir(parents=True)
    output.write_text("new chart", encoding="utf-8")
    chart.write_text(json.dumps({
        "schema": "maichart-event-chart-v1",
        "points": [{"beat": "0", "note_text": "1"}, {"beat": "1", "note_text": "2"},
                   {"beat": "2", "note_text": "3"}, {"beat": "3", "note_text": "4"},
                   {"beat": "4", "note_text": ""}],
        "tempo_changes": [{"beat": "0", "bpm": 120}], "first": 0.0,
        "end_beat": "4", "audio_duration_sec": 2.0,
    }), encoding="utf-8")
    quality.write_text("{}", encoding="utf-8")
    config_path = _config(tmp_path, jobs=[{
        "name": "generate", "endpoint": "/api/generate", "payload": {},
    }], evaluation_paths=[str(output.parent)])
    config, config_dir = _load_config(config_path)
    controller = StagedRhythmValidation(config, config_dir)
    controller.state["initial"]["status"] = "completed"
    controller.state["jobs"][0].update({
        "status": "completed", "task_id": "gen-1",
        "last_snapshot": {"result": {
            "model_family": "event_ctx_v1", "paths": [str(output), str(chart), str(quality)],
        }},
    })
    controller.state["status"] = "controller_interrupted"
    controller._persist()

    assert run_config(config_path, client=FakeClient(), sleep_fn=lambda _: None) == 2

    state = json.loads((tmp_path / "controller_state.json").read_text(encoding="utf-8"))
    assert state["jobs"][0]["task_result"]["model_family"] == "event_ctx_v1"
    assert state["evaluation"]["artifact_completeness"]["produced_count"] == 3
    assert state["evaluation"]["artifact_completeness"]["unverifiable_count"] == 1


@pytest.mark.parametrize("reference_case", ["missing", "incompatible"])
def test_missing_or_incompatible_reference_pair_makes_generation_evaluation_partial(tmp_path, reference_case):
    output = tmp_path / "generated" / "arm-a" / "song.txt"
    reference_root = tmp_path / "references"
    reference_root.mkdir()
    chart_data = {
        "schema": "maichart-event-chart-v1",
        "points": [{"beat": "0", "note_text": "1"}, {"beat": "1", "note_text": "2"},
                   {"beat": "2", "note_text": "3"}, {"beat": "3", "note_text": "4"},
                   {"beat": "4", "note_text": ""}],
        "tempo_changes": [{"beat": "0", "bpm": 120}], "first": 0.0,
        "end_beat": "4", "audio_duration_sec": 2.0,
        "audio_sha256": "a" * 64,
    }
    if reference_case == "incompatible":
        reference_chart = reference_root / "arm-a" / "song.chart.json"
        reference_chart.parent.mkdir(parents=True)
        reference_chart.write_text(json.dumps({**chart_data, "audio_sha256": "b" * 64}), encoding="utf-8")

    def write_generated(_payload):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("new chart", encoding="utf-8")
        output.with_suffix(".chart.json").write_text(json.dumps(chart_data), encoding="utf-8")
        output.with_suffix(".quality.json").write_text("{}", encoding="utf-8")

    config_path = _config(tmp_path, jobs=[{
        "name": "generate", "endpoint": "/api/generate",
        "payload": {"output": str(output)},
    }], evaluation_paths=[str(output.parent.parent)], reference_dir=reference_root)
    client = FakeClient(
        gets={"train-001": _task("completed"),
              "gen-1": _task("completed", result={"model_family": "event_ctx_v1"})},
        posts=[{"task_id": "gen-1"}], on_post=write_generated,
    )

    assert run_config(config_path, client=client, sleep_fn=lambda _: None) == 2

    state = json.loads((tmp_path / "controller_state.json").read_text(encoding="utf-8"))
    assert state["evaluation"]["evaluated_count"] == 1
    assert state["evaluation"]["artifact_completeness"]["produced_count"] == 3
    assert state["evaluation"]["artifact_completeness"]["missing_count"] == 0
    assert state["evaluation"]["reference_pairing"][reference_case] == 1
    assert state["evaluation"]["status"] == "partial"


def test_state_lock_rejects_a_second_controller(tmp_path):
    state_path = tmp_path / "controller_state.json"
    with StateFileLock(state_path):
        assert (tmp_path / "controller_state.json.lock").is_file()
        try:
            with StateFileLock(state_path):
                raise AssertionError("second lock acquisition should fail")
        except StateLockError as exc:
            assert "already locked" in str(exc)


def test_state_lock_is_released_after_exception(tmp_path):
    state_path = tmp_path / "controller_state.json"
    try:
        with StateFileLock(state_path):
            raise RuntimeError("test failure")
    except RuntimeError as exc:
        assert str(exc) == "test failure"

    with StateFileLock(state_path):
        pass
