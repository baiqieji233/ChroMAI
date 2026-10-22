"""Independent curve analysis/playback and event generation API contracts."""

import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from maichart.webui import server as webui


@pytest.fixture
def api(monkeypatch, tmp_path):
    # The Python checker is usable without a separately deployed runtime;
    # use its real availability preflight in these HTTP/CLI contract tests.
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


def test_status_advertises_python_engine_without_dynamic_dependency(api):
    code, body, _ = api("/api/status")
    assert code == 200
    status = body["playability_engine"]
    assert status["available"] and status["engine"] == "miacode_python"
    assert status["implementation"] == "python" and not status["dynamic_dependency"]


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


@pytest.mark.parametrize('source', ['auto', 'audio', 'model'])
def test_composer_curve_sources_do_not_enter_event_checkpoint_loader(api, monkeypatch, tmp_path, source):
    from maichart.generation import music_intensity, learned_curve
    audio=tmp_path/'track.wav';audio.write_bytes(b'audio')
    checkpoint=tmp_path/'composer.pt';checkpoint.write_bytes(b'checkpoint')
    def load(_path):
        assert source!='audio', 'Audio reference must not load model or optimizer weights'
        return {'schema':'maichart-chart-composer-checkpoint-v2','family':'chart_composer_v1',
            'input_contract':'chart_composer_inputs_v2_audio_only_history_state',
            'model_config':{'planner_profile':'music_structure_v3'},'model_state_dict':{}}
    def forbidden(*args,**kwargs):
        raise AssertionError('Composer checkpoint was sent to the event curve loader')
    monkeypatch.setattr(webui,'_load_checkpoint_payload',load)
    monkeypatch.setattr(learned_curve,'generate_learned_profile',forbidden)
    monkeypatch.setattr(music_intensity,'generate_intensity_profile',lambda *a,**kw:profile(audio))
    status,started,_=api('/api/intensity/analyze',{'audio':str(audio),'checkpoint':str(checkpoint),
        'curve_source':source,'model_family':'chart_stream_v1'})
    assert status==200,started
    deadline=time.monotonic()+4
    while time.monotonic()<deadline:
        _,result,_=api('/api/task?id='+started['task_id'])
        if result['status'] not in {'queued','running'}:break
        time.sleep(.02)
    if source=='model':
        assert result['status']=='failed'
        assert '没有独立的全曲 Planner 曲线头' in result['error']
        assert '旧固定帧' not in result['error']
    else:
        assert result['status']=='completed',result
        assert result['result']['curve_source']=='audio'
        assert result['result']['intensity_profile']['method']=='multifeature_v1'


@pytest.mark.parametrize("values", [{"resolution_sec": 0}, {"resolution_sec": float("nan")},
                                  {"local_context_sec": 0}, {"local_context_sec": 500}])
def test_analysis_rejects_bad_options(api, tmp_path, values):
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"audio")
    status, _, _ = api("/api/intensity/analyze", {"audio": str(audio), **values})
    assert status == 400
    assert not webui.TASKS


@pytest.mark.parametrize("meter_map", ["not-an-array", [{"beat": "1", "numerator": 3, "denominator": 4}],
                                      [{"beat": "0", "numerator": 7, "denominator": 8, "grouping": [3, 3]}]])
def test_analysis_rejects_bad_meter_before_creating_task(api, tmp_path, meter_map):
    audio = tmp_path / "song.mp3"; audio.write_bytes(b"audio")
    status, result, _ = api("/api/intensity/analyze", {"audio": str(audio), "meter_map": meter_map})
    assert status == 400, result
    assert "拍号表无效" in result["error"]
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
        "tempo_map": [{"beat": "0", "bpm": 120}, {"beat": "32", "bpm": 150}],
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
    assert result["result"]["intensity_profile"]["bpm_input"] == 150
    assert result["result"]["intensity_profile"]["explicit_tempo_map"] == [
        {"beat": "0", "bpm": 120.0}, {"beat": "32", "bpm": 150.0},
    ]
    assert result["result"]["intensity_profile"]["tempo_map_sha256"]
    assert received["condition"].level == 14
    assert received["bpm"] == 150 and received["first"] == 1.2
    assert received["tempo_map"] == [{"beat": "0", "bpm": 120.0},
                                     {"beat": "32", "bpm": 150.0}]
    assert received["device"] == "cpu"


def test_song_form_curve_analysis_normalizes_meter_before_inference(api, monkeypatch, tmp_path):
    from maichart.form import curve

    audio = tmp_path / "song.wav"; audio.write_bytes(b"audio")
    checkpoint = tmp_path / "model.pt"; checkpoint.write_bytes(b"weights")
    received = {}
    expected = profile(audio)
    expected["method"] = "learned_song_form_v2"
    monkeypatch.setattr(webui, "_load_checkpoint_payload", lambda *a: {"family": "song_form_v2"})
    monkeypatch.setattr(webui, "_checkpoint_model_family", lambda *a: "song_form_v2")

    def generate(_audio, _checkpoint, **kwargs):
        received.update(kwargs)
        return expected

    monkeypatch.setattr(curve, "generate_learned_profile", generate)
    status, result, _ = api("/api/intensity/analyze", {
        "audio": str(audio), "checkpoint": str(checkpoint), "curve_source": "model",
        "model_family": "song_form_v2", "meter_map": [{"beat": "0/1", "numerator": 7,
            "denominator": 8, "grouping": [2, 2, 3]}],
    })
    assert status == 200, result
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        _, completed, _ = api("/api/task?id=" + result["task_id"])
        if completed["status"] not in {"queued", "running"}:
            break
        time.sleep(.02)
    assert completed["status"] == "completed", completed
    assert received["meter_map"] == [{"beat": "0", "numerator": 7, "denominator": 8,
                                      "grouping": [2, 2, 3], "confidence": 1., "source": "manual"}]
    assert received["device"] == "cpu"


def test_learned_curve_analysis_resolves_sparse_condition_like_generation(api, monkeypatch, tmp_path):
    from maichart.generation.conditions import GENERATION_DEFAULTS
    from maichart.generation import learned_curve

    audio, payload, _ = setup_event_run(monkeypatch, tmp_path)
    manifest = tmp_path / "profiled" / "training_manifest.json"
    manifest.parent.mkdir()
    manifest.write_text("{}", encoding="utf-8")
    controls = {key: {"median": value} for key, value in GENERATION_DEFAULTS.items()
                if key != "level"}
    controls["melody_following"] = {"median": 0.0}
    (manifest.parent / "condition_profile.json").write_text(json.dumps({
        "schema": "maichart-condition-profile-v1",
        "level": {"median": 13.3}, "controls": controls,
    }), encoding="utf-8")
    monkeypatch.setattr(webui, "_load_checkpoint_payload", lambda *a: {
        "schema": webui.EVENT_CHECKPOINT_SCHEMA,
        "model_state_dict": {"weight": object()}, "model_config": {"planner_profile": "music_structure_v3"},
        "data_contract": {"target_schema": "event_ctx_v1",
                           "condition_params": {"mask_prob": 0.0, "default": 0.5}},
    })
    received = {}
    expected_profile = profile(audio)
    expected_profile["method"] = "learned_music_structure_v3"

    def generate(_audio, _checkpoint, **kwargs):
        received.update(kwargs)
        return expected_profile

    monkeypatch.setattr(learned_curve, "generate_learned_profile", generate)
    payload.update({"manifest": str(manifest), "condition": {"level": 14.0}, "level": 14.0})
    code, started, _ = api("/api/intensity/analyze", {
        **payload, "curve_source": "model", "condition": {"level": 14.0},
    })
    assert code == 200, started
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        _, result, _ = api("/api/task?id=" + started["task_id"])
        if result["status"] not in {"queued", "running"}:
            break
        time.sleep(0.02)
    assert result["status"] == "completed", result
    assert received["condition"].level == 14.0
    assert received["condition"].style["melody_following"] == pytest.approx(0.9)
    assert received["condition"].radar["note_density"] == pytest.approx(
        GENERATION_DEFAULTS["note_density"])


def test_learned_curve_preflight_matches_cli_for_profile_plus_level_only_data(api, monkeypatch, tmp_path):
    from maichart.generation.conditions import GENERATION_DEFAULTS
    from maichart.generation.event_cli import _file_identity, resolve_generation_condition

    audio, payload, calls = setup_event_run(monkeypatch, tmp_path)
    manifest = tmp_path / "dataset" / "training_manifest.json"
    manifest.parent.mkdir()
    manifest.write_text("{}", encoding="utf-8")
    controls = {key: {"median": value} for key, value in GENERATION_DEFAULTS.items()
                if key != "level"}
    controls["melody_following"] = {"median": 0.0}
    profile_path = manifest.parent / "condition_profile.json"
    profile_path.write_text(json.dumps({"schema": "maichart-condition-profile-v1",
                                        "level": {"median": 13.3}, "controls": controls}),
                            encoding="utf-8")
    monkeypatch.setattr(webui, "_load_checkpoint_payload", lambda *a: {
        "schema": webui.EVENT_CHECKPOINT_SCHEMA,
        "model_state_dict": {"weight": object()}, "model_config": {"d_model": 32},
        "data_contract": {"target_schema": "event_ctx_v1",
                           "condition_params": {"mask_prob": 0.0, "default": 0.5}},
    })
    payload.update({"manifest": str(manifest), "condition": {"level": 14.0}, "level": 14.0})
    effective, _ = resolve_generation_condition(
        level=14.0, condition_profile=profile_path, condition_data=payload["condition"],
        contract_params={"mask_prob": 0.0, "default": 0.5})
    identity = _file_identity(payload["checkpoint"])
    curve = {**profile(audio), "method": "learned_music_structure_v3",
             "checkpoint": payload["checkpoint"],
             "checkpoint_identity": {"size": identity["size"], "mtime_ns": identity["mtime_ns"],
                                     "sha256": identity["sha256"]},
             "condition": effective.to_dict(), "first": 0.0,
             "tempo_map_sha256": None, "tempo_mode": "constant", "bpm_input": None}

    code, started, _ = api("/api/generate", {**payload, "intensity_curve_enabled": True,
                                               "intensity_curve": curve})
    assert code == 200, started
    args = calls[0]
    assert args[args.index("--condition-profile") + 1] == str(profile_path)
    condition_file = Path(args[args.index("--condition-file") + 1])
    assert json.loads(condition_file.read_text(encoding="utf-8")) == {"level": 14.0}
    cli_effective, _ = resolve_generation_condition(
        level=14.0, condition_profile=profile_path, condition_file=condition_file,
        contract_params={"mask_prob": 0.0, "default": 0.5})
    assert cli_effective.to_dict() == effective.to_dict()

    wrong_curve = {**curve, "condition": {**effective.to_dict(),
                    "style": {**effective.style, "melody_following": 0.2}}}
    before_calls = len(calls)
    code, result, _ = api("/api/generate", {**payload, "intensity_curve_enabled": True,
                                              "intensity_curve": wrong_curve})
    assert code == 200, result
    assert "Style.melody_following" in result["curve_condition_warnings"][0]
    assert len(calls) == before_calls + 1
    args = calls[-1]
    saved_curve = json.loads(Path(args[args.index("--intensity-curve") + 1]).read_text(encoding="utf-8"))
    assert saved_curve["condition"] == wrong_curve["condition"]
    assert saved_curve["points"] == wrong_curve["points"]
    assert webui.TASKS[result["task_id"]].result["curve_condition_warnings"]


def test_learned_curve_preflight_reports_exact_condition_and_timing_differences(api, monkeypatch, tmp_path):
    """A saved curve can be valid while its generation context is stale.

    The error must identify the changed fields; otherwise users cannot tell
    whether to regenerate for a level, a style slider, or a tempo setting.
    """

    from maichart.generation.event_cli import _file_identity, resolve_generation_condition

    audio, payload, calls = setup_event_run(monkeypatch, tmp_path)
    identity = _file_identity(payload["checkpoint"])
    condition, _ = resolve_generation_condition(
        level=13.3, condition_data=None, condition_profile=None, contract_params={})
    curve = {**profile(audio), "method": "learned_music_structure_v3",
             "checkpoint": payload["checkpoint"],
             "checkpoint_identity": {"size": identity["size"], "mtime_ns": identity["mtime_ns"],
                                      "sha256": identity["sha256"]},
             "condition": condition.to_dict(), "first": 0.0,
             "tempo_map_sha256": None, "tempo_mode": "auto", "bpm_input": None}
    stale = {**curve, "condition": {**curve["condition"],
                                     "level": 12.5,
                                     "style": {**curve["condition"]["style"], "melody_following": 0.2}}}
    code, result, _ = api("/api/generate", {**payload, "intensity_curve_enabled": True,
                                              "intensity_curve": stale})
    assert code == 400
    assert "节拍模式 auto→constant" in result["error"]
    assert not calls

    # Identical condition changes are advisory once timing matches.
    code, result, _ = api("/api/generate", {**payload, "tempo_mode": "auto",
        "intensity_curve_enabled": True, "intensity_curve": stale})
    assert code == 200, result
    warning = result["curve_condition_warnings"][0]
    assert "等级 12.5→13.3" in warning and "Style.melody_following 0.2→0.9" in warning
    before_calls = len(calls)

    stale = {**curve, "bpm_input": 120.0}
    code, result, _ = api("/api/generate", {**payload, "bpm": 150,
                                              "intensity_curve_enabled": True,
                                              "intensity_curve": stale})
    assert code == 400
    assert "节拍模式 auto→constant" in result["error"]
    assert "BPM 120→150" in result["error"]
    assert len(calls) == before_calls


def test_explicit_tempo_map_is_normalized_and_forwarded_to_event_cli(api, monkeypatch, tmp_path):
    _, payload, calls = setup_event_run(monkeypatch, tmp_path)
    tempo_map = [{"beat": 0, "bpm": 120}, {"beat": 32, "bpm": 150}]
    code, started, _ = api("/api/generate", {**payload, "tempo_map": tempo_map})
    assert code == 200, started
    args = calls[0]
    tempo_path = args[args.index("--tempo-map") + 1]
    with open(tempo_path, encoding="utf-8") as handle:
        assert json.load(handle) == [{"beat": "0", "bpm": 120.0},
                                     {"beat": "32", "bpm": 150.0}]


def test_learned_curve_explicit_tempo_map_mismatch_is_rejected(api, monkeypatch, tmp_path):
    audio, payload, calls = setup_event_run(monkeypatch, tmp_path)
    curve = {**profile(audio), "method": "learned_music_structure_v2",
             "checkpoint": payload["checkpoint"], "checkpoint_identity": {"size": 1, "mtime_ns": 0},
             "tempo_map_sha256": "different-map"}
    code, result, _ = api("/api/generate", {
        **payload, "tempo_map": [{"beat": 0, "bpm": 120}],
        "intensity_curve_enabled": True, "intensity_curve": curve,
    })
    assert code == 400
    assert "tempo map 已变化" in result["error"]
    assert not calls and not webui.TASKS


def test_invalid_explicit_tempo_map_fails_before_task(api, monkeypatch, tmp_path):
    audio, payload, calls = setup_event_run(monkeypatch, tmp_path)
    bad_maps = [[], [{"beat": 1, "bpm": 120}],
                [{"beat": 0, "bpm": 120}, {"beat": 0, "bpm": 150}]]
    for tempo_map in bad_maps:
        code, _, _ = api("/api/generate", {**payload, "tempo_map": tempo_map})
        assert code == 400
        assert not calls and not webui.TASKS


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
    assert "覆盖或身份不匹配" in result["error"]
    assert not calls and not webui.TASKS


def test_learned_curve_accepts_browser_rounded_legacy_mtime(api, monkeypatch, tmp_path):
    from maichart.generation.event_cli import _file_identity
    audio, payload, calls = setup_event_run(monkeypatch, tmp_path)
    identity = _file_identity(payload["checkpoint"])
    old_identity = {"size": identity["size"], "mtime_ns": float(identity["mtime_ns"])}
    old_identity = json.loads(json.dumps(old_identity))
    curve = {**profile(audio), "method": "learned_music_structure_v3",
             "checkpoint": payload["checkpoint"], "checkpoint_identity": old_identity,
             "condition": {"level": 14, "difficulty": 5,
                           "radar": {"note_density": 0.3, "peak": 0.5, "sweep": 0.1,
                                     "slide_tricky": 0.1, "slide_sequence": 0.1, "jack": 0.1,
                                     "chord": 0.3},
                           "style": {"tap_bias": 0.7, "slide_bias": 0.1, "hold_bias": 0.1,
                                     "touch_bias": 0.05, "chord_bias": 0.3, "melody_following": 0.8,
                                     "pattern_reuse": 0.1, "directionality": 0.1}},
             "first": 0, "tempo_map_sha256": None, "tempo_mode": "constant", "bpm_input": None}
    code, _, _ = api("/api/generate", {**payload, "intensity_curve_enabled": True,
                                        "intensity_curve": curve, "condition": curve["condition"]})
    assert code == 200
    assert "--intensity-curve" in calls[0]


def test_learned_curve_same_size_checkpoint_replacement_is_rejected(api, monkeypatch, tmp_path):
    from maichart.generation.event_cli import _file_identity
    audio, payload, calls = setup_event_run(monkeypatch, tmp_path)
    identity = _file_identity(payload["checkpoint"])
    curve = {**profile(audio), "method": "learned_music_structure_v3",
             "checkpoint": payload["checkpoint"],
             "checkpoint_identity": {"size": identity["size"], "mtime_ns": str(identity["mtime_ns"]),
                                     "sha256": identity["sha256"]}}
    with open(payload["checkpoint"], "wb") as handle:
        handle.write(b"X" * identity["size"])
    code, result, _ = api("/api/generate", {**payload, "intensity_curve_enabled": True,
                                            "intensity_curve": curve})
    assert code == 400
    assert "覆盖或身份不匹配" in result["error"]
    assert not calls


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


def test_generation_uses_embedded_judge_without_executable(api, monkeypatch, tmp_path):
    _, payload, calls = setup_event_run(monkeypatch, tmp_path)
    code, response, _ = api("/api/generate", {**payload, "conflict_detection_enabled": True,
        "conflict_repair_mode": "local", "conflict_retries": 2})
    assert code == 200, response
    args = calls[0]
    assert args[args.index("--conflict-engine") + 1] == "miacode"
    assert "--runtime-reference-executable" not in args


@pytest.mark.parametrize("budget", [-1, 1000])
def test_zero_threshold_and_unbounded_budget_reach_cli(api, monkeypatch, tmp_path, budget):
    from maichart.generation.event_cli import build_parser
    _, payload, calls = setup_event_run(monkeypatch, tmp_path)
    code, response, _ = api("/api/generate", {**payload,"conflict_detection_enabled":True,
        "conflict_repair_mode":"local","tap_on_slide_threshold_ms":0,
        "conflict_backtrack_budget":budget})
    assert code == 200, response
    args = build_parser().parse_args(calls[0][2:])
    assert args.tap_on_slide_threshold_ms == 0
    assert args.conflict_backtrack_budget == budget
    task = webui.TASKS[response["task_id"]]
    assert task.result["tap_on_slide_threshold_ms"] == 0
    assert task.result["conflict_backtrack_budget"] == budget


def test_old_reference_settings_never_launch_editor(api, monkeypatch, tmp_path):
    _, payload, calls = setup_event_run(monkeypatch, tmp_path)
    code, response, _ = api("/api/generate", {**payload, "runtime_reference_mode": "reject",
        "runtime_reference_executable": "Z:/missing/MiaCode.exe"})
    assert code == 200, response
    assert "--enable-conflict-detection" in calls[0]
    assert calls[0][calls[0].index("--conflict-repair-mode") + 1] == "reject"
    assert "--runtime-reference-executable" not in calls[0]


def test_missing_native_engine_fails_before_generation_task(api, monkeypatch, tmp_path):
    from maichart.generation import miacode_native
    _, payload, calls = setup_event_run(monkeypatch, tmp_path)
    def unavailable():
        raise RuntimeError("内嵌 MiaCode 检测库未部署")
    monkeypatch.setattr(miacode_native, "ensure_available", unavailable)
    code, result, _ = api("/api/generate", {**payload, "conflict_detection_enabled": True})
    assert code >= 400
    assert "未部署" in result["error"]
    assert not calls and not webui.TASKS
