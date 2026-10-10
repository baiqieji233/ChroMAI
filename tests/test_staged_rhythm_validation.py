import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.run_staged_rhythm_validation import run_config


class FakeClient:
    def __init__(self, *, gets=None, posts=None):
        self.gets = {key: list(value) if isinstance(value, list) else [value]
                     for key, value in (gets or {}).items()}
        self.posts = list(posts or [])
        self.get_calls = []
        self.post_calls = []
        self.timeouts = []

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
        item = self.posts.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _config(tmp_path, *, jobs=None, evaluation_paths=None, poll_seconds=0, http_timeout_sec=20):
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
    path = tmp_path / "controller_config.json"
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

    assert run_config(config_path, client=client, sleep_fn=lambda _: None) == 0

    assert len(client.post_calls) == 2
    assert client.get_calls == ["train-001", "gen-a", "gen-b"]
    state = json.loads((tmp_path / "controller_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert [job["status"] for job in state["jobs"]] == ["failed", "completed"]
    assert state["evaluation"]["status"] == "completed"
    assert state["evaluation"]["human_review"] is None
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

    assert run_config(config_path, client=client, sleep_fn=lambda _: None) == 0
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
