"""Independent curve analysis/playback and event generation API contracts."""

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from maichart.webui import server as webui


@pytest.fixture
def api(monkeypatch, tmp_path):
    monkeypatch.setattr(webui, "TASKS", {})
    monkeypatch.setattr(webui, "INTENSITY_ROOT", tmp_path / "curves")
    monkeypatch.setattr(webui, "CONDITION_ROOT", tmp_path / "conditions")
    monkeypatch.setattr(webui, "LOG_ROOT", tmp_path / "logs")
    webui.CONDITION_ROOT.mkdir()
    server = ThreadingHTTPServer(("127.0.0.1", 0), webui.WebUIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(path, body=None, headers=None, method=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(f"http://127.0.0.1:{server.server_port}{path}",
                                     data=data, method=method,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=5)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            raw = response.read()
            content = json.loads(raw) if response.headers.get("Content-Type", "").startswith("application/json") else raw
            return response.status, content, dict(response.headers)

    yield request
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def profile(audio):
    from maichart.generation.music_intensity import audio_sha256
    return {"schema": "maichart-intensity-curve-v1", "audio_sha256": audio_sha256(audio),
            "duration_sec": 10.0, "method": "multifeature_v1",
            "points": [{"time_sec": 0, "intensity": 0.5},
                       {"time_sec": 10, "intensity": 0.4}],
            "spectrogram": {"times": [0, 10], "frequencies_hz": [100], "values": [[0, 255]]}}


def test_analysis_is_separate_and_returns_editable_profile(api, monkeypatch, tmp_path):
    from maichart.generation import music_intensity
    audio = tmp_path / "external_song.mp3"
    audio.write_bytes(b"not decoded in this API contract test")
    expected = profile(audio)
    monkeypatch.setattr(music_intensity, "generate_intensity_profile", lambda *a, **kw: expected)
    training = webui.TaskState("keep_train", "train", status="running")
    webui.TASKS[training.task_id] = training
    code, started, _ = api("/api/intensity/analyze", {"audio": str(audio)})
    assert code == 200
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        _, result, _ = api("/api/task?id=" + started["task_id"])
        if result["status"] not in {"queued", "running"}:
            break
        time.sleep(0.02)
    assert result["status"] == "completed", result
    assert result["result"]["intensity_profile"] == expected
    assert result["result"]["audio_preview_url"].startswith("/api/intensity/audio?id=")
    assert training.status == "running"
    assert webui.TASKS[started["task_id"]].kind == "intensity"
    assert (webui.INTENSITY_ROOT / f"{started['task_id']}_automatic.json").exists()


@pytest.mark.parametrize("values", [{"resolution_sec": 0}, {"resolution_sec": float("nan")},
                                  {"local_context_sec": 0}, {"local_context_sec": 500}])
def test_analysis_rejects_bad_options(api, tmp_path, values):
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"audio")
    status, _, _ = api("/api/intensity/analyze", {"audio": str(audio), **values})
    assert status == 400
    assert not webui.TASKS


def test_audio_preview_supports_external_file_and_seek_ranges(api, tmp_path):
    audio = tmp_path / "outside_project.mp3"
    audio.write_bytes(b"0123456789abcdef")
    task = webui.TaskState("analysis", "intensity", result={"audio": str(audio)})
    webui.TASKS[task.task_id] = task
    code, data, headers = api("/api/intensity/audio?id=analysis", headers={"Range": "bytes=3-7"})
    assert code == 206
    assert data == b"34567"
    assert headers["Content-Range"] == "bytes 3-7/16"
    code, data, _ = api("/api/intensity/audio?id=analysis", headers={"Range": "bytes=-4"})
    assert code == 206 and data == b"cdef"
    code, data, headers = api("/api/intensity/audio?id=analysis", method="HEAD")
    assert code == 200 and not data and headers["Content-Length"] == "16"
    code, _, _ = api("/api/intensity/audio?id=analysis", headers={"Range": "bytes=30-40"})
    assert code == 416
    assert api("/api/intensity/audio?id=unselected")[0] == 404


def setup_event_run(monkeypatch, tmp_path):
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"audio")
    cp = tmp_path / "event.pt"
    cp.write_bytes(b"checkpoint")
    monkeypatch.setattr(webui, "_load_checkpoint_payload", lambda *a: {
        "schema": webui.EVENT_CHECKPOINT_SCHEMA,
        "model_state_dict": {"weight": object()}, "model_config": {"d_model": 32},
        "data_contract": {"target_schema": "event_ctx_v1"}})
    calls = []
    monkeypatch.setattr(webui, "_run_subprocess_task", lambda task, args, **kw: calls.append(args))
    return audio, {"checkpoint": str(cp), "audio": str(audio),
                   "output": str(tmp_path / "output.txt")}, calls


def test_edited_curve_is_forwarded_and_visualization_not_saved_for_decoder(api, monkeypatch, tmp_path):
    audio, payload, calls = setup_event_run(monkeypatch, tmp_path)
    curve = profile(audio)
    curve["points"][1]["intensity"] = 0.2
    code, started, _ = api("/api/generate", {**payload, "intensity_curve_enabled": True,
                                           "intensity_curve": curve})
    assert code == 200, started
    assert "--intensity-curve" in calls[0]
    assert calls[0][calls[0].index("--intensity-curve-strength") + 1] == "0.65"
    assert calls[0][calls[0].index("--intensity-curve-mode") + 1] == "follow"
    saved = webui.TASKS[started["task_id"]].result["intensity_curve_path"]
    with open(saved, encoding="utf-8") as handle:
        result = json.load(handle)
    assert result["points"][1]["intensity"] == 0.2
    assert "spectrogram" not in result


@pytest.mark.parametrize("mode", ["follow", "hard", "soft", "strict"])
def test_rest_regions_are_persisted_and_forwarded_with_zero_curve_strength(api, monkeypatch, tmp_path, mode):
    audio, payload, calls = setup_event_run(monkeypatch, tmp_path)
    curve = profile(audio)
    curve["rest_regions"] = [{"start_sec": 1, "end_sec": 3}, {"start_sec": 2, "end_sec": 4}]
    code, started, _ = api("/api/generate", {**payload, "intensity_curve_enabled": True,
        "intensity_curve": curve, "intensity_curve_mode": mode, "intensity_curve_strength": 0})
    assert code == 200, started
    assert calls[0][calls[0].index("--intensity-curve-mode") + 1] == mode
    assert calls[0][calls[0].index("--intensity-curve-strength") + 1] == "0.0"
    saved = webui.TASKS[started["task_id"]].result["intensity_curve_path"]
    with open(saved, encoding="utf-8") as handle:
        assert json.load(handle)["rest_regions"] == [{"start_sec": 1.0, "end_sec": 4.0}]


@pytest.mark.parametrize("rest", [{"start_sec": -1, "end_sec": 2},
                                  {"start_sec": 3, "end_sec": 2},
                                  {"start_sec": 0, "end_sec": 11}])
def test_bad_rest_regions_fail_before_generation_task(api, monkeypatch, tmp_path, rest):
    audio, payload, calls = setup_event_run(monkeypatch, tmp_path)
    curve = {**profile(audio), "rest_regions": [rest]}
    code, _, _ = api("/api/generate", {**payload, "intensity_curve_enabled": True, "intensity_curve": curve})
    assert code == 400
    assert not calls and not webui.TASKS


def test_changed_audio_cannot_use_previous_curve(api, monkeypatch, tmp_path):
    audio, payload, calls = setup_event_run(monkeypatch, tmp_path)
    curve = profile(audio)
    audio.write_bytes(b"different song same filename")
    code, result, _ = api("/api/generate", {**payload, "intensity_curve_enabled": True,
                                          "intensity_curve": curve})
    assert code == 400
    assert "不匹配" in result["error"]
    assert not calls and not webui.TASKS


def test_enabled_without_saved_curve_is_rejected(api, monkeypatch, tmp_path):
    _, payload, calls = setup_event_run(monkeypatch, tmp_path)
    code, result, _ = api("/api/generate", {**payload, "intensity_curve_enabled": True})
    assert code == 400, result
    assert not calls


def test_disabled_curve_preserves_original_generation_cli(api, monkeypatch, tmp_path):
    _, payload, calls = setup_event_run(monkeypatch, tmp_path)
    assert api("/api/generate", payload)[0] == 200
    assert "--intensity-curve" not in calls[0]
    assert calls[0][calls[0].index("--intensity-mode") + 1] == "auto"


def test_legacy_loudness_mode_is_explicit_and_invalid_mode_fails_before_task(api, monkeypatch, tmp_path):
    _, payload, calls = setup_event_run(monkeypatch, tmp_path)
    assert api("/api/generate", {**payload, "intensity_mode": "loudness"})[0] == 200
    assert calls[0][calls[0].index("--intensity-mode") + 1] == "loudness"
    before = set(webui.TASKS)
    assert api("/api/generate", {**payload, "intensity_mode": "unknown"})[0] == 400
    assert set(webui.TASKS) == before


def test_learned_curve_analysis_uses_generation_condition_and_timing(api, monkeypatch, tmp_path):
    from maichart.generation import learned_curve
    audio, payload, _ = setup_event_run(monkeypatch, tmp_path)
    expected = profile(audio)
    expected["method"] = "learned_music_structure_v2"
    received = {}

    def generate(audio_path, checkpoint_path, **kwargs):
        received.update(kwargs)
        assert str(audio_path) == str(audio)
        assert str(checkpoint_path) == payload["checkpoint"]
        return expected

    monkeypatch.setattr(learned_curve, "generate_learned_profile", generate)
    code, started, _ = api("/api/intensity/analyze", {
        **payload, "curve_source": "model", "condition": {"level": 14},
        "bpm": 150, "first": 1.2, "tempo_mode": "constant",
    })
    assert code == 200, started
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        _, result, _ = api("/api/task?id=" + started["task_id"])
        if result["status"] not in {"queued", "running"}:
            break
        time.sleep(0.02)
    assert result["status"] == "completed", result
    assert result["result"]["curve_source"] == "model"
    assert received["condition"]["level"] == 14
    assert received["bpm"] == 150 and received["first"] == 1.2
    assert received["device"] == "cpu"


@pytest.mark.parametrize("bad", [{"intensity_curve_strength": -1},
                                 {"intensity_curve_strength": float("nan")},
                                 {"intensity_curve_mode": "unknown"}])
def test_curve_controls_reject_invalid_inputs_before_task(api, monkeypatch, tmp_path, bad):
    _, payload, calls = setup_event_run(monkeypatch, tmp_path)
    assert api("/api/generate", {**payload, **bad})[0] == 400
    assert not calls and not webui.TASKS


@pytest.mark.parametrize("method", ["learned_music_structure_v2", "learned_music_structure_v3"])
def test_learned_curve_cannot_be_reused_after_checkpoint_changes(api, monkeypatch, tmp_path, method):
    audio, payload, calls = setup_event_run(monkeypatch, tmp_path)
    curve = profile(audio)
    curve.update({"method": method, "checkpoint": payload["checkpoint"],
                  "checkpoint_identity": {"size": 1, "mtime_ns": 0}})
    code, result, _ = api("/api/generate", {**payload, "intensity_curve_enabled": True,
                                          "intensity_curve": curve})
    assert code == 400
    assert "模型已变化" in result["error"]
    assert not calls and not webui.TASKS


def test_v3_strength_and_static_conflict_controls_reach_cli(api, monkeypatch, tmp_path):
    audio, payload, calls = setup_event_run(monkeypatch, tmp_path)
    monkeypatch.setattr(webui, "_load_checkpoint_payload", lambda *a: {
        "schema": webui.EVENT_CHECKPOINT_SCHEMA,
        "model_state_dict": {"weight": object()},
        "model_config": {"d_model": 32, "planner_profile": "music_structure_v3", "plan_dim": 13},
        "data_contract": {"target_schema": "event_ctx_v1",
                           "planner_params": {"profile": "music_structure_v3", "plan_dim": 13}},
    })
    code, started, _ = api("/api/generate", {**payload, "fine_note_strength": 0.8,
        "conflict_detection_enabled": True, "detect_hard_conflicts": False,
        "detect_soft_conflicts": True, "conflict_repair_mode": "local", "conflict_retries": 2})
    assert code == 200, started
    args = calls[0]
    assert args[args.index("--fine-note-strength") + 1] == "0.8"
    assert "--enable-conflict-detection" in args
    assert "--no-detect-hard-conflicts" in args
    assert "--detect-soft-conflicts" in args
    assert args[args.index("--conflict-repair-mode") + 1] == "local"
    assert args[args.index("--conflict-retries") + 1] == "2"


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_invalid_fine_note_strength_fails_before_generation_task(api, monkeypatch, tmp_path, value):
    _, payload, calls = setup_event_run(monkeypatch, tmp_path)
    code, result, _ = api("/api/generate", {**payload, "fine_note_strength": value})
    assert code == 400
    assert "细碎音符强度" in result["error"]
    assert not calls and not webui.TASKS


@pytest.mark.parametrize("retries", [-1, 6])
def test_conflict_retries_outside_supported_range_fail_before_task(api, monkeypatch, tmp_path, retries):
    _, payload, calls = setup_event_run(monkeypatch, tmp_path)
    code, result, _ = api("/api/generate", {**payload, "conflict_detection_enabled": True,
                                              "conflict_repair_mode": "local",
                                              "conflict_retries": retries})
    assert code == 400
    assert "冲突重试次数" in result["error"]
    assert not calls and not webui.TASKS
