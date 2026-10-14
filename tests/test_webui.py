from pathlib import Path
import io
import json
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile

import pytest
from http.server import ThreadingHTTPServer

from maichart.webui import server as webui

from maichart.webui.server import (
    PROJECT_ROOT,
    TASKS,
    TaskState,
    _candidate_path,
    _find_preprocess_task,
    _multipart_boundary,
    _parse_multipart_stream,
    _safe_filename,
)


def test_multipart_parser_keeps_binary_trailing_hyphen():
    boundary = "----maichart-test"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="kind"\r\n\r\n'
        "audio\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="song.wav"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + b"abc-\r\n" + f"--{boundary}--\r\n".encode()
    fields = _parse_multipart_stream(
        io.BufferedReader(io.BytesIO(body), buffer_size=8),
        _multipart_boundary(f"multipart/form-data; boundary={boundary}"),
        len(body),
        Path(tempfile.mkdtemp()),
    )
    assert fields["kind"] == "audio"
    filename, part_path = fields["file"]
    assert filename == "song.wav"
    assert part_path.read_bytes() == b"abc-"


def test_upload_filename_is_reduced_to_a_safe_name():
    assert _safe_filename(r"..\folder\a:b?.wav") == "a_b_.wav"


def test_maidata_upload_accepts_txt_and_maidata_extensions(webui_api, tmp_path):
    for suffix in (".txt", ".maidata"):
        body = (f'--x\r\nContent-Disposition: form-data; name="kind"\r\n\r\nmaidata\r\n'
                f'--x\r\nContent-Disposition: form-data; name="file"; filename="corrected{suffix}"\r\n'
                "Content-Type: text/plain\r\n\r\n&title=T\r\n--x--\r\n").encode()
        status, result = webui_api("/api/upload", body,
                                   content_type="multipart/form-data; boundary=x")
        assert status == 200, result


def test_project_root_is_absolute():
    assert PROJECT_ROOT.is_absolute()


def test_candidate_paths_match_generation_cli_naming():
    output = Path("generated/maidata.txt")
    assert _candidate_path(output, 0, 1) == output
    assert _candidate_path(output, 0, 2).name == "maidata_candidate_001.txt"
    assert _candidate_path(output.with_suffix(".chart.json"), 1, 2).name == "maidata.chart_candidate_002.json"


def test_find_preprocess_task_by_manifest_path(monkeypatch, tmp_path):
    manifest = (tmp_path / "training_manifest.json").resolve()
    task = TaskState(task_id="prep-test", kind="preprocess")
    task.result["manifest"] = str(manifest)
    monkeypatch.setitem(TASKS, task.task_id, task)

    assert _find_preprocess_task(manifest) is task


@pytest.fixture
def webui_api(monkeypatch, tmp_path):
    monkeypatch.setattr(webui, "TASKS", {})
    monkeypatch.setattr(webui, "WEBUI_ROOT", tmp_path / "workspace")
    for name in ("UPLOAD_ROOT", "DATASET_ROOT"):
        directory = tmp_path / name.lower()
        directory.mkdir()
        monkeypatch.setattr(webui, name, directory)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), webui.WebUIHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{httpd.server_port}"

    def request(path, payload=None, *, content_type="application/json"):
        body = json.dumps(payload).encode() if isinstance(payload, dict) else payload
        req = urllib.request.Request(base_url + path, data=body, headers={"Content-Type": content_type})
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    yield request
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def test_missing_manifest_is_a_prepare_step_not_a_server_error(webui_api, tmp_path):
    manifest = tmp_path / "training_manifest.json"
    status, data = webui_api("/api/inspect-manifest", {"manifest": str(manifest)})
    assert status == 200
    assert data["state"] == "not_preprocessed"
    assert data["ready"] is False
    assert "预处理训练集" in data["message"]
    status, data = webui_api("/api/train", {"manifest": str(manifest)})
    assert status == 400
    assert "预处理训练集" in data["error"]
    assert not webui.TASKS


@pytest.mark.parametrize("task_status", ["running", "failed", "cancelled"])
def test_rebuild_never_checks_or_trains_the_previous_manifest(webui_api, tmp_path, task_status):
    manifest = tmp_path / "training_manifest.json"
    manifest.write_text('{"schema":"maichart-training-manifest-v1","songs":[]}', encoding="utf-8")
    task = TaskState("rebuild", "preprocess", status=task_status)
    task.result["manifest"] = str(manifest)
    webui.TASKS[task.task_id] = task
    status, data = webui_api("/api/inspect-manifest", {"manifest": str(manifest)})
    assert status == 200
    assert data["ready"] is False
    assert data["state"] == ("processing" if task_status == "running" else "preprocess_failed")
    status, data = webui_api("/api/train", {"manifest": str(manifest)})
    assert status == 400
    assert "error" in data
    assert len(webui.TASKS) == 1


def test_manifest_from_another_dataset_is_rejected(webui_api, tmp_path):
    old_root = tmp_path / "old"
    old_root.mkdir()
    manifest = old_root / "training_manifest.json"
    manifest.write_text(json.dumps({
        "schema": "maichart-training-manifest-v1", "source_root": ".", "songs": [],
    }), encoding="utf-8")
    payload = {"manifest": str(manifest), "raw_root": str(tmp_path / "new")}
    for endpoint in ("/api/inspect-manifest", "/api/train"):
        status, data = webui_api(endpoint, payload)
        assert status == 400
        assert "另一份" in data["error"]


def test_new_zip_upload_prepare_check_and_train_use_the_same_root(webui_api, monkeypatch):
    calls = []

    def finish_task(task, args, *, expected_epoch, result_paths):
        calls.append((task.kind, args))
        if task.kind == "preprocess":
            manifest = result_paths[0]
            manifest.write_text(json.dumps({
                "schema": "maichart-training-manifest-v1", "source_root": ".", "cache_dir": "cache",
                "songs": [{"song_id": "song", "difficulties": [{
                    "difficulty_index": 5, "level_raw": "13+", "usable_for_training": True,
                }]}],
            }), encoding="utf-8")
        task.status = "completed"

    monkeypatch.setattr(webui, "_run_subprocess_task", finish_task)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("song/maidata.txt", "&lv_5=13+\n&inote_5=(120){4}1,2,3,4,")
    boundary = "maichart-test-dataset"
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="kind"\r\n\r\ndataset_zip\r\n'
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="新训练集.zip"\r\n'
        'Content-Type: application/zip\r\n\r\n'
    ).encode() + archive.getvalue() + f"\r\n--{boundary}--\r\n".encode()
    status, uploaded = webui_api("/api/upload", body, content_type=f"multipart/form-data; boundary={boundary}")
    assert status == 200
    assert uploaded["extracting"] is True
    # Extraction runs in the background; poll like the UI does until it lands.
    deadline = time.time() + 10
    while time.time() < deadline:
        status, state = webui_api("/api/status", None)
        info = state.get("extractions", {}).get(uploaded["upload_id"], {})
        if info.get("status") == "done":
            break
        if info.get("status") == "failed":
            raise AssertionError(f"解压失败: {info.get('error')}")
        time.sleep(0.05)
    assert Path(uploaded["path"], "song", "maidata.txt").is_file()
    payload = {"raw_root": uploaded["path"], "cache_dir": uploaded["cache_dir"], "manifest": uploaded["manifest"]}
    status, checked = webui_api("/api/inspect-manifest", payload)
    assert status == 200 and checked["state"] == "not_preprocessed"
    status, started = webui_api("/api/preprocess", payload)
    assert status == 200
    status, checked = webui_api("/api/inspect-manifest", payload)
    assert status == 200 and checked["ready"] is True
    assert checked["source_root"] == uploaded["path"]
    assert checked["cache_dir"] == uploaded["cache_dir"]
    assert checked["usable_difficulty_count"] == 1
    status, trained = webui_api("/api/train", payload)
    assert status == 200
    assert [kind for kind, args in calls] == ["preprocess", "train"]
    assert uploaded["path"] in calls[0][1]
    assert uploaded["manifest"] in calls[0][1] and uploaded["manifest"] in calls[1][1]
    status, listed = webui_api("/api/status")
    assert listed["datasets"][0]["path"] == uploaded["path"]


def test_event_preprocess_uses_prepare_events_and_event_profile(webui_api, monkeypatch, tmp_path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    calls = []

    def fake_run(task, args, *, expected_epoch, result_paths):
        calls.append(args)
        result_paths[0].parent.mkdir(parents=True, exist_ok=True)
        result_paths[0].write_text(json.dumps({
            "schema": "maichart-training-manifest-v1",
            "source_root": str(raw_root),
            "cache_dir": str(tmp_path / "cache"),
            "preprocessing": {"target_schema": "event_v1", "feature_profile": "event_audio_v1"},
            "songs": [{"maidata_path": "song/maidata.txt", "audio_path": "song/track.wav",
                        "difficulties": [{"usable_for_training": False}]}],
        }), encoding="utf-8")
        task.status = "completed"

    monkeypatch.setattr(webui, "_run_subprocess_task", fake_run)
    payload = {
        "model_family": "event_v1",
        "raw_root": str(raw_root),
        "cache_dir": str(tmp_path / "cache"),
        "manifest": str(tmp_path / "event_manifest.json"),
        "sample_rate": 22050,
        "limit": 3,
        "force": True,
    }
    status, _ = webui_api("/api/preprocess", payload)
    assert status == 200
    assert calls[0][:4] == ["-m", "maichart.training.prepare_events", str(raw_root), "--cache-dir"]
    assert "--sample-rate" in calls[0] and "22050" in calls[0]
    assert "--force" in calls[0] and "--limit" in calls[0]


def test_song_form_preprocess_reuses_event_source_manifest_path(webui_api, monkeypatch, tmp_path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    calls = []

    def fake_run(task, args, *, expected_epoch, result_paths):
        calls.append(args)
        task.status = "completed"

    monkeypatch.setattr(webui, "_run_subprocess_task", fake_run)
    status, data = webui_api("/api/preprocess", {
        "model_family": "song_form_v1", "raw_root": str(raw_root),
        "cache_dir": str(tmp_path / "cache"), "manifest": str(tmp_path / "form.json"),
        "sample_rate": 22050, "workers": 4,
    })
    assert status == 200, data
    assert calls[0][:4] == ["-m", "maichart.training.prepare_events", str(raw_root), "--cache-dir"]
    # The raw-root temporary path itself may contain the word "preprocess";
    # assert the command shape instead of searching the rendered path text.
    assert calls[0][1] == "maichart.training.prepare_events"


def test_event_train_does_not_reject_old_fixed_frame_usable_flag(webui_api, monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "maichart-training-manifest-v1",
        "source_root": str(tmp_path),
        "preprocessing": {"target_schema": "event_v1", "feature_profile": "event_audio_v1"},
        "songs": [{"maidata_path": "song/maidata.txt", "audio_path": "song/track.wav",
                    "difficulties": [{"usable_for_training": False}]}],
    }), encoding="utf-8")
    calls = []

    def fake_run(task, args, *, expected_epoch, result_paths):
        calls.append(args)
        task.status = "completed"

    monkeypatch.setattr(webui, "_run_subprocess_task", fake_run)
    status, data = webui_api("/api/train", {
        "model_family": "event_v1", "manifest": str(manifest), "raw_root": str(tmp_path),
        "cache_dir": str(tmp_path / "cache"), "checkpoint_dir": str(tmp_path / "checkpoints"),
        "epochs": 1,
    })
    assert status == 200, data
    assert calls[0][:2] == ["-m", "maichart.training.train_events"]
    assert "--segment-beats" in calls[0]
    assert "--cuda-memory-budget-gb" in calls[0]
    assert calls[0][calls[0].index("--cuda-memory-budget-gb") + 1] == "0.0"
    assert calls[0][calls[0].index("--onset-profile") + 1] == "auto"
    assert calls[0][calls[0].index("--onset-loss-weight") + 1] == "0.2"


def test_resume_rejects_onset_structure_change_and_legacy_rejects_onset_options():
    with pytest.raises(ValueError, match="只能是 auto、none 或 chart_onsets_v1"):
        webui._resolve_onset_training_options(
            {"onset_profile": "chart_onsets_v2"}, model_family=webui.EVENT_MODEL_FAMILY,
            training_mode="fresh", inspection=None)
    with pytest.raises(ValueError, match="只支持 event_v1"):
        webui._resolve_onset_training_options(
            {"onset_profile": "none"}, model_family=webui.LEGACY_MODEL_FAMILY,
            training_mode="fresh", inspection=None)
    with pytest.raises(ValueError, match="不能改变 checkpoint 的 onset_profile"):
        webui._resolve_onset_training_options(
            {"onset_profile": "none", "onset_loss_weight": 0.2},
            model_family=webui.EVENT_MODEL_FAMILY, training_mode="resume",
            inspection={"model_config": {"onset_profile": "chart_onsets_v1"},
                        "data_contract": {"onset_params": {"profile": "chart_onsets_v1", "loss_weight": 0.2}}})


def test_status_advertises_chart_onset_training_capability(webui_api):
    status, data = webui_api("/api/status")
    assert status == 200
    assert data["training_capabilities"]["local_onset_training"] is True


def test_checkpoint_inspection_exposes_local_onset_generation_capability(webui_api, monkeypatch, tmp_path):
    checkpoint = tmp_path / "trained-onset.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(webui, "_load_checkpoint_payload", lambda _path: {
        "schema": "maichart-event-generator-checkpoint-v1",
        "model_state_dict": {"weight": object()},
        "model_config": {"planner_profile": "music_structure_v3", "onset_profile": "chart_onsets_v1"},
        "data_contract": {"target_schema": "event_v1", "onset_params": {
            "profile": "chart_onsets_v1", "loss_weight": 0.2,
        }},
        "onset_training": {"profile": "chart_onsets_v1", "trained": True, "optimizer_steps": 8},
    })
    monkeypatch.setattr(webui, "checkpoint_capabilities", lambda _payload: {
        "model_family": webui.EVENT_MODEL_FAMILY, "errors": [],
        "conflict_detection": True, "conflict_repair": True,
        "fine_note_strength": True, "local_onset_guidance": True,
    })

    status, inspected = webui_api("/api/checkpoint/inspect", {
        "checkpoint_path": str(checkpoint), "mode": "generate",
    })

    assert status == 200
    assert inspected["generation_capabilities"]["local_onset_guidance"] is True
    assert inspected["onset_training"]["trained"] is True


def test_resume_rejects_onset_structure_change_and_legacy_rejects_onset_options():
    with pytest.raises(ValueError, match="只能是 auto、none 或 chart_onsets_v1"):
        webui._resolve_onset_training_options(
            {"onset_profile": "chart_onsets_v2"}, model_family=webui.EVENT_MODEL_FAMILY,
            training_mode="fresh", inspection=None)
    with pytest.raises(ValueError, match="只支持 event_v1"):
        webui._resolve_onset_training_options(
            {"onset_profile": "none"}, model_family=webui.LEGACY_MODEL_FAMILY,
            training_mode="fresh", inspection=None)
    with pytest.raises(ValueError, match="不能改变 checkpoint 的 onset_profile"):
        webui._resolve_onset_training_options(
            {"onset_profile": "none", "onset_loss_weight": 0.2},
            model_family=webui.EVENT_MODEL_FAMILY, training_mode="resume",
            inspection={"model_config": {"onset_profile": "chart_onsets_v1"},
                        "data_contract": {"onset_params": {"profile": "chart_onsets_v1", "loss_weight": 0.2}}})


def test_status_advertises_chart_onset_training_capability(webui_api):
    status, data = webui_api("/api/status")
    assert status == 200
    assert data["training_capabilities"]["local_onset_training"] is True


def test_resume_rejects_onset_structure_change_and_legacy_rejects_onset_options():
    with pytest.raises(ValueError, match="只能是 auto、none 或 chart_onsets_v1"):
        webui._resolve_onset_training_options(
            {"onset_profile": "chart_onsets_v2"}, model_family=webui.EVENT_MODEL_FAMILY,
            training_mode="fresh", inspection=None)
    with pytest.raises(ValueError, match="只支持 event_v1"):
        webui._resolve_onset_training_options(
            {"onset_profile": "none"}, model_family=webui.LEGACY_MODEL_FAMILY,
            training_mode="fresh", inspection=None)
    with pytest.raises(ValueError, match="不能改变 checkpoint 的 onset_profile"):
        webui._resolve_onset_training_options(
            {"onset_profile": "none", "onset_loss_weight": 0.2},
            model_family=webui.EVENT_MODEL_FAMILY, training_mode="resume",
            inspection={"model_config": {"onset_profile": "chart_onsets_v1"},
                        "data_contract": {"onset_params": {"profile": "chart_onsets_v1", "loss_weight": 0.2}}})


def test_status_advertises_chart_onset_training_capability(webui_api):
    status, data = webui_api("/api/status")
    assert status == 200
    assert data["training_capabilities"]["local_onset_training"] is True


def test_event_finetune_can_add_chart_onset_profile_without_ui_reverting(webui_api, monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "maichart-training-manifest-v1", "source_root": str(tmp_path),
        "preprocessing": {"target_schema": "event_v1", "feature_profile": "event_audio_v2_rhythm"},
        "songs": [],
    }), encoding="utf-8")
    monkeypatch.setattr(webui, "_checkpoint_inspection", lambda *a, **kw: {
        "errors": [], "model_config": {"planner_profile": "music_structure_v3", "onset_profile": "none"},
        "data_contract": {"onset_params": {"profile": "none", "loss_weight": 0.2}},
    })
    calls = []
    monkeypatch.setattr(webui, "_run_subprocess_task", lambda task, args, **kwargs: calls.append(args))

    status, data = webui_api("/api/train", {
        "model_family": "event_v1", "manifest": str(manifest), "raw_root": str(tmp_path),
        "cache_dir": str(tmp_path / "cache"), "checkpoint_dir": str(tmp_path / "finetune"),
        "training_mode": "finetune", "training_checkpoint": str(checkpoint),
        "onset_profile": "chart_onsets_v1", "onset_loss_weight": 0.35,
    })

    assert status == 200, data
    args = calls[0]
    assert args[args.index("--onset-profile") + 1] == "chart_onsets_v1"
    assert args[args.index("--onset-loss-weight") + 1] == "0.35"


def test_event_resume_inherits_saved_onset_profile_and_weight(webui_api, monkeypatch, tmp_path):
    checkpoint = tmp_path / "event_last.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "maichart-training-manifest-v1", "source_root": str(tmp_path),
        "preprocessing": {"target_schema": "event_v1", "feature_profile": "event_audio_v2_rhythm"},
        "songs": [],
    }), encoding="utf-8")
    onset_params = {"profile": "chart_onsets_v1", "loss_weight": 0.35,
                    "sigma_sec": 0.04, "target_version": "chart_onsets_v1",
                    "local_pooling_policy": "per_sample_v2"}
    monkeypatch.setattr(webui, "_checkpoint_inspection", lambda *a, **kw: {
        "errors": [], "epoch": 4,
        "model_config": {"planner_profile": "music_structure_v3", "onset_profile": "chart_onsets_v1"},
        "data_contract": {"onset_params": onset_params},
        "onset_training": {"profile": "chart_onsets_v1", "trained": True, "optimizer_steps": 3},
    })
    calls = []
    monkeypatch.setattr(webui, "_run_subprocess_task", lambda task, args, **kwargs: calls.append(args))

    status, data = webui_api("/api/train", {
        "model_family": "event_v1", "manifest": str(manifest), "raw_root": str(tmp_path),
        "cache_dir": str(tmp_path / "cache"), "checkpoint_dir": str(tmp_path / "resume"),
        "training_mode": "resume", "training_checkpoint": str(checkpoint), "epochs": 1,
        "onset_profile": "auto", "onset_loss_weight": 0.2,
    })

    assert status == 200, data
    args = calls[0]
    assert args[args.index("--onset-profile") + 1] == "chart_onsets_v1"
    assert args[args.index("--onset-loss-weight") + 1] == "0.35"
    task = webui.TASKS[data["task_id"]]
    assert task.result["onset_profile"] == "chart_onsets_v1"
    assert task.result["onset_loss_weight"] == 0.35


def test_event_train_forwards_difficulty_and_audio_contract(webui_api, monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "maichart-training-manifest-v1", "source_root": str(tmp_path),
        "preprocessing": {"target_schema": "event_v1", "feature_profile": "event_audio_v1"},
        "songs": [{"maidata_path": "song/maidata.txt", "audio_path": "song/track.wav",
                    "difficulties": [{"usable_for_training": False}]}],
    }), encoding="utf-8")
    calls = []

    def fake_run(task, args, *, expected_epoch, result_paths):
        calls.append(args)
        task.status = "completed"

    monkeypatch.setattr(webui, "_run_subprocess_task", fake_run)
    status, data = webui_api("/api/train", {
        "model_family": "event_v1", "manifest": str(manifest), "raw_root": str(tmp_path),
        "cache_dir": str(tmp_path / "cache"), "checkpoint_dir": str(tmp_path / "checkpoints"),
        "epochs": 1, "min_difficulty_index": 2, "max_difficulty_index": 5,
        "min_level": 1, "max_level": 13.5, "max_audio_tokens": 1024,
        "condition_default": 0.37, "cuda_memory_budget_gb": 8.5, "lr": 0.000017,
    })
    assert status == 200, data
    args = calls[0]
    for flag, value in (("--min-difficulty-index", "2"), ("--max-difficulty-index", "5"),
                        ("--min-level", "1.0"), ("--max-level", "13.5"),
                        ("--max-audio-tokens", "1024"), ("--condition-default", "0.37"),
                        ("--cuda-memory-budget-gb", "8.5"), ("--lr", "1.7e-05")):
        assert flag in args
        assert args[args.index(flag) + 1] == value


def test_event_train_forwards_saved_quality_and_preference_settings(webui_api, monkeypatch, tmp_path):
    from maichart.training.quality_annotations import annotation_path, save_annotations

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(webui, "_checkpoint_inspection", lambda *a, **kw: {"errors": [], "model_config": {}})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "maichart-training-manifest-v1", "source_root": str(tmp_path),
        "preprocessing": {"target_schema": "event_v1", "feature_profile": "event_audio_v2_rhythm"},
        "songs": [],
    }), encoding="utf-8")
    corrected = tmp_path / "corrected.txt"
    corrected.write_text("&title=Corrected\n", encoding="utf-8")
    save_annotations(manifest, [{
        "chart_id": "negative-chart", "song_id": "song", "difficulty_index": 5,
        "source_path": str(tmp_path / "maidata.txt"), "chart_sha256": "test-hash",
        "label": "negative", "weight": 0.0, "corrected_path": str(corrected),
        "corrected_difficulty": 5,
    }])
    calls = []

    def fake_run(task, args, *, expected_epoch, result_paths):
        calls.append(args)
        task.status = "completed"

    monkeypatch.setattr(webui, "_run_subprocess_task", fake_run)
    status, data = webui_api("/api/train", {
        "model_family": "event_v1", "manifest": str(manifest), "raw_root": str(tmp_path),
        "cache_dir": str(tmp_path / "cache"), "checkpoint_dir": str(tmp_path / "checkpoints"),
        "epochs": 1, "quality_enabled": True, "preference_enabled": True,
        "preference_weight": 0.12, "preference_max_pairs": 17,
        "training_mode": "finetune", "training_checkpoint": str(checkpoint),
    })
    assert status == 200, data
    args = calls[0]
    assert "--quality-annotations" in args
    assert args[args.index("--quality-annotations") + 1] == str(annotation_path(manifest))
    assert "--preference-enabled" in args
    assert args[args.index("--preference-weight") + 1] == "0.12"
    assert args[args.index("--preference-max-pairs") + 1] == "17"
    assert args[args.index("--lr") + 1] == "2e-05"
    from maichart.training.train_events import build_parser
    parsed = build_parser().parse_args(args[2:])
    assert parsed.lr == 0.00002


def test_event_ctx_family_routes_training_and_generation(webui_api, monkeypatch, tmp_path):
    """event_ctx_v1 shares the event pipeline but not the checkpoint alphabet."""

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "maichart-training-manifest-v1",
        "source_root": str(tmp_path),
        "preprocessing": {"target_schema": "event_v1", "feature_profile": "event_audio_v1"},
        "songs": [{"maidata_path": "song/maidata.txt", "audio_path": "song/track.wav",
                   "difficulties": [{"usable_for_training": False}]}],
    }), encoding="utf-8")
    calls = []

    def fake_run(task, args, *, expected_epoch, result_paths):
        calls.append(args)
        task.status = "completed"

    monkeypatch.setattr(webui, "_run_subprocess_task", fake_run)
    status, data = webui_api("/api/train", {
        "model_family": "event_ctx_v1", "manifest": str(manifest), "raw_root": str(tmp_path),
        "cache_dir": str(tmp_path / "cache"), "checkpoint_dir": str(tmp_path / "checkpoints"),
        "epochs": 1, "context_beats": 4, "audio_margin_beats": 2,
        "context_tokens": 96, "overlap_beats": 8,
        "slide_oversample": 3, "touch_oversample": 1,
    })
    assert status == 200, data
    assert calls[0][:2] == ["-m", "maichart.training.train_event_ctx"]
    for flag, value in (("--context-beats", "4"), ("--audio-margin-beats", "2"),
                        ("--context-tokens", "96"), ("--overlap-beats", "8"),
                        ("--slide-oversample", "3"), ("--touch-oversample", "1")):
        assert flag in calls[0]
        assert calls[0][calls[0].index(flag) + 1] == value
    for flag, value in (("--local-pooling-policy", "per_sample_v2"),
                        ("--context-budget-policy", "bounded_v2"),
                        ("--chart-filter-policy", "drop_headless_stars_v1")):
        assert calls[0][calls[0].index(flag) + 1] == value
    task = webui.TASKS[data["task_id"]]
    assert task.result["selection_metric"] == "training_loss"
    assert task.result["input_policies"]["local_pooling_policy"] == "per_sample_v2"

    # The same family selection routes generation to the matching generator.
    checkpoint = tmp_path / "ctx.pt"
    checkpoint.write_bytes(b"checkpoint")
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"audio")
    monkeypatch.setattr(webui, "_load_checkpoint_payload", lambda path: {
        "schema": "maichart-event-generator-checkpoint-v1",
        "model_state_dict": {"weight": object()},
        "model_config": {"d_model": 32, "global_audio_tokens": 256},
        "data_contract": {"target_schema": "event_ctx_v1"},
    })
    calls.clear()
    status, started = webui_api("/api/generate", {
        "checkpoint": str(checkpoint), "audio": str(audio),
        "output": str(tmp_path / "out.txt"), "candidates": 1,
        "context_beats": 4, "audio_margin_beats": 2,
    })
    assert status == 200, started
    assert calls[0][:2] == ["-m", "maichart.generation.event_ctx_cli"]
    assert "--context-beats" in calls[0]
    task = webui.TASKS[started["task_id"]]
    assert task.result["model_family"] == "event_ctx_v1"


def test_generation_layout_comes_from_the_checkpoint(webui_api, monkeypatch, tmp_path):
    """A stale or empty UI field must not put generation out of distribution.

    The layout is part of the data contract; generating with a different lead-in
    makes a cross-window model emit a couple of notes per window and stop.
    """

    checkpoint = tmp_path / "ctx.pt"
    checkpoint.write_bytes(b"checkpoint")
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"audio")
    monkeypatch.setattr(webui, "_load_checkpoint_payload", lambda path: {
        "schema": "maichart-event-generator-checkpoint-v1",
        "model_state_dict": {"weight": object()},
        "model_config": {"d_model": 32, "global_audio_tokens": 256},
        "data_contract": {
            "target_schema": "event_ctx_v1",
            "family_params": {"context_beats": 8, "audio_margin_beats": 4,
                              "context_tokens": 256, "overlap_beats": 0},
        },
    })
    calls = []

    def fake_run(task, args, *, expected_epoch, result_paths):
        calls.append(args)
        task.status = "completed"

    monkeypatch.setattr(webui, "_run_subprocess_task", fake_run)
    # The UI sends 0 (an empty field used to become zero) and a smaller budget.
    status, started = webui_api("/api/generate", {
        "checkpoint": str(checkpoint), "audio": str(audio),
        "output": str(tmp_path / "out.txt"), "candidates": 1,
        "context_beats": 0, "audio_margin_beats": 0, "context_tokens": 96,
    })
    assert status == 200
    args = calls[0]
    assert args[args.index("--context-beats") + 1] == "8", "the trained layout must win"
    assert args[args.index("--audio-margin-beats") + 1] == "4"
    assert args[args.index("--context-tokens") + 1] == "256"
    task = webui.TASKS[started["task_id"]]
    assert task.result["context_layout"]["context_beats"] == 8
    assert task.result["context_layout_warnings"], "the mismatch must be reported"


def test_local_onset_generation_is_capability_gated_and_forwarded(webui_api, monkeypatch, tmp_path):
    checkpoint = tmp_path / "onset.pt"
    checkpoint.write_bytes(b"checkpoint")
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"audio")
    monkeypatch.setattr(webui, "_load_checkpoint_payload", lambda path: {
        "schema": "maichart-event-generator-checkpoint-v1",
        "model_state_dict": {"weight": object()},
        "model_config": {"d_model": 32, "onset_profile": "chart_onsets_v1"},
        "data_contract": {"target_schema": "event_v1"},
        "onset_training": {"profile": "chart_onsets_v1", "trained": True, "optimizer_steps": 8},
    })
    capability_state = {"local_onset_guidance": False}
    monkeypatch.setattr(webui, "checkpoint_capabilities", lambda _payload: {
        "model_family": webui.EVENT_MODEL_FAMILY, "errors": [],
        "conflict_detection": True, "conflict_repair": True,
        "fine_note_strength": True, "local_onset_guidance": capability_state["local_onset_guidance"],
    })
    calls = []

    def fake_run(task, args, *, expected_epoch, result_paths):
        calls.append(args)
        task.status = "completed"

    monkeypatch.setattr(webui, "_run_subprocess_task", fake_run)
    base = {"checkpoint": str(checkpoint), "audio": str(audio), "output": str(tmp_path / "out.txt")}
    status, zero = webui_api("/api/generate", {**base, "local_onset_bias_strength": 0})
    assert status == 200, zero
    assert "--local-onset-bias-strength" not in calls[-1]

    status, rejected = webui_api("/api/generate", {**base, "local_onset_bias_strength": 0.4})
    assert status == 400
    assert "未训练可用" in rejected["error"]
    assert len(calls) == 1

    capability_state["local_onset_guidance"] = True
    status, started = webui_api("/api/generate", {
        **base, "fine_note_strength": 0.8, "local_onset_bias_strength": 0.45,
    })
    assert status == 200, started
    assert calls[-1][calls[-1].index("--local-onset-bias-strength") + 1] == "0.45"
    assert calls[-1][calls[-1].index("--fine-note-strength") + 1] == "0.8"
    assert webui.TASKS[started["task_id"]].result["local_onset_bias_strength"] == 0.45


def test_checkpoint_family_detection_uses_the_contract(webui_api, monkeypatch, tmp_path):
    checkpoint = tmp_path / "ctx.pt"
    checkpoint.write_bytes(b"checkpoint")
    # Same checkpoint schema as event_v1: only target_schema separates them.
    monkeypatch.setattr(webui, "_load_checkpoint_payload", lambda path: {
        "schema": "maichart-event-generator-checkpoint-v1",
        "model_state_dict": {"weight": object()},
        "model_config": {"d_model": 32},
        "data_contract": {"target_schema": "event_ctx_v1"},
    })
    status, data = webui_api("/api/checkpoint/inspect", {
        "checkpoint_path": str(checkpoint), "mode": "generate",
    })
    assert status == 200
    assert data["model_family"] == "event_ctx_v1"
    assert data["generation_cli"] == "maichart.generation.event_ctx_cli"

    # Asking for the wrong family must be refused instead of silently mixing
    # them.  Generate mode reports the checkpoint's own family, so the mismatch
    # only exists on the training paths, where the requested family is explicit.
    status, data = webui_api("/api/checkpoint/inspect", {
        "checkpoint_path": str(checkpoint), "mode": "finetune", "model_family": "event_v1",
    })
    assert status == 200
    assert data["valid"] is False
    assert any("event_ctx_v1" in message for message in data["errors"])


def test_one_click_preprocess_and_train_uses_a_single_task(webui_api, monkeypatch, tmp_path):
    """The button must not depend on the browser chaining two tasks."""

    raw = tmp_path / "raw"
    raw.mkdir()
    manifest = tmp_path / "manifests" / "training_manifest.json"
    calls = []

    def fake_run(task, args, *, expected_epoch, result_paths):
        calls.append((args, result_paths, expected_epoch))
        task.status = "completed"

    monkeypatch.setattr(webui, "_run_subprocess_task", fake_run)
    status, started = webui_api("/api/train", {
        "model_family": "event_v1", "auto_preprocess": True,
        "raw_root": str(raw), "manifest": str(manifest),
        "cache_dir": str(tmp_path / "cache"), "checkpoint_dir": str(tmp_path / "checkpoints"),
        "epochs": 3, "limit": 20,
    })
    assert status == 200, started
    args, result_paths, expected_epoch = calls[0]
    assert args[:2] == ["-m", "maichart.training.auto_pipeline"]
    assert expected_epoch == 3
    # The manifest is produced by the first stage, so it is watched as a result.
    assert manifest in result_paths

    plan = json.loads(Path(args[args.index("--plan") + 1]).read_text(encoding="utf-8"))
    assert plan["family"] == "event_v1"
    # The wrapper calls the stage in-process, so the plan must NOT embed the
    # subprocess launcher (`-m maichart.training.prepare_events`): argparse used
    # to reject that prefix with exit code 2 before any preprocessing ran.
    assert plan["prepare"][0] == str(raw) and str(manifest) in plan["prepare"]
    assert "--limit" in plan["prepare"] and "20" in plan["prepare"]
    # The trainer args keep everything except the "-m <module>" prefix.
    assert plan["train"][0] == "--manifest" and str(manifest) in plan["train"]
    assert "--epochs" in plan["train"]
    task = webui.TASKS[started["task_id"]]
    assert task.result["auto_preprocess"] is True
    assert task.result["manifest"] == str(manifest)


def test_one_click_requires_a_raw_folder(webui_api, tmp_path):
    status, data = webui_api("/api/train", {
        "model_family": "event_v1", "auto_preprocess": True,
        "manifest": str(tmp_path / "manifests" / "training_manifest.json"),
        "checkpoint_dir": str(tmp_path / "checkpoints"), "epochs": 1,
    })
    assert status == 400
    assert "raw" in data["error"].lower() or "原始" in data["error"]


def test_gold_output_folder_never_nests(webui_api, monkeypatch, tmp_path):
    """Repeated builds must not create gold/gold/gold.

    Nesting is what produced manifests whose paths never existed, so it has to
    stop at the second run, not be repaired afterwards.
    """

    from maichart.webui.server import _default_gold_out_dir

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    fresh = dataset / "training_manifest.json"
    fresh.write_text(json.dumps({
        "schema": "maichart-training-manifest-v1", "source_root": ".", "cache_dir": "cache",
        "preprocessing": {"target_schema": "event_v1", "feature_profile": "event_audio_v1"},
        "songs": [], "summary": {}, "warnings": [], "errors": [],
    }, ensure_ascii=False), encoding="utf-8")
    assert _default_gold_out_dir(fresh) == dataset / "gold_set"

    # Second run: the UI points the training manifest at the split it produced.
    out = dataset / "gold_set"
    out.mkdir()
    (out / "gold_val_manifest.json").write_text("{}", encoding="utf-8")
    assert _default_gold_out_dir(out / "pretrain_manifest.json") == out

    # And a build must refuse to write into the charts folder at all.
    gold_charts = dataset / "gold"
    (gold_charts / "SongA").mkdir(parents=True)
    status, data = webui_api("/api/gold/build", {
        "manifest": str(fresh), "mode": "folder",
        "gold_root": str(gold_charts), "out_dir": str(gold_charts),
        "val_ratio": 0.3, "min_note_count": 0,
    })
    assert status == 400
    assert "gold 文件夹" in data["error"] or "输出目录" in data["error"]


def test_baseline_preprocess_remains_on_legacy_cli(webui_api, monkeypatch, tmp_path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    calls = []

    def fake_run(task, args, *, expected_epoch, result_paths):
        calls.append(args)
        task.status = "completed"

    monkeypatch.setattr(webui, "_run_subprocess_task", fake_run)
    status, _ = webui_api("/api/preprocess", {
        "model_family": "legacy_frame_v2", "raw_root": str(raw_root),
        "cache_dir": str(tmp_path / "cache"), "manifest": str(tmp_path / "manifest.json"),
        "division": 16, "sample_rate": 22050,
    })
    assert status == 200
    assert calls[0][:4] == ["-m", "maichart.cli", "preprocess", "build"]


def test_checkpoint_inspection_rejects_cross_model_family(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "event.pt"
    torch.save({
        "schema": "maichart-event-generator-checkpoint-v1",
        "model_state_dict": {"weight": torch.zeros(1)},
        "model_config": {"d_model": 32},
        "data_contract": {"target_schema": "event_v1"},
    }, path)
    inspection = webui._checkpoint_inspection(
        path, mode="finetune", manifest=None, model_family="legacy_frame_v2",
    )
    assert inspection["valid"] is False
    assert any("跨 schema" in error for error in inspection["errors"])


def test_event_generation_selects_event_cli_and_quality_sidecar(webui_api, monkeypatch, tmp_path):
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"audio")
    checkpoint = tmp_path / "event.pt"
    checkpoint.write_bytes(b"checkpoint")
    calls = []

    monkeypatch.setattr(webui, "_load_checkpoint_payload", lambda path: {
        "schema": "maichart-event-generator-checkpoint-v1",
        "model_state_dict": {"weight": object()},
        "model_config": {"d_model": 32},
        "data_contract": {"target_schema": "event_v1"},
    })

    def fake_run(task, args, *, expected_epoch, result_paths):
        calls.append(args)
        for path in result_paths:
            if path.name.endswith(".quality.json"):
                path.write_text('{"syntax_ok":true,"warnings":["manual review"]}', encoding="utf-8")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("result", encoding="utf-8")
        task.status = "completed"
        webui._collect_task_results(task, result_paths)

    monkeypatch.setattr(webui, "_run_subprocess_task", fake_run)
    status, started = webui_api("/api/generate", {
        "checkpoint": str(checkpoint), "audio": str(audio),
        "output": str(tmp_path / "out.txt"), "candidates": 1,
        "tempo_mode": "auto", "temperature": 0.8, "max_tokens": 2048,
        "device": "cpu",
    })
    assert status == 200
    assert started["task_id"] in webui.TASKS
    assert calls[0][:2] == ["-m", "maichart.generation.event_cli"]
    assert calls[0][calls[0].index("--device") + 1] == "cpu"
    task = webui.TASKS[started["task_id"]]
    assert task.result["quality_report_exists"] is True
    assert task.result["quality_reports"][0]["diagnostics"]["syntax_ok"] is True


def test_event_generation_result_collection_includes_only_present_repair_artifacts(tmp_path):
    output = tmp_path / "chart.txt"
    paths = webui._generation_result_paths(
        output, output.with_suffix(".chart.json"), 1, webui.EVENT_MODEL_FAMILY,
    )
    originals = [path for path in paths if ".original." in path.name or ".repaired." in path.name]
    assert {path.name for path in originals} == {
        "chart.original.txt", "chart.original.chart.json", "chart.repaired.txt",
    }
    originals[0].write_text("original", encoding="utf-8")
    originals[2].write_text("repaired", encoding="utf-8")
    task = TaskState(task_id="repair-artifacts", kind="generate")
    webui._collect_task_results(task, paths)
    assert "chart.original.txt" in task.result["files"]
    assert "chart.repaired.txt" in task.result["files"]
    assert "chart.original.chart.json" not in task.result["files"]


def test_resume_epoch_progress_uses_checkpoint_epoch_plus_requested_epochs(webui_api, monkeypatch, tmp_path):
    manifest = tmp_path / "training_manifest.json"
    manifest.write_text(json.dumps({
        "schema": "maichart-training-manifest-v1",
        "source_root": str(tmp_path), "cache_dir": str(tmp_path / "cache"),
        "preprocessing": {"target_schema": "event_v1", "feature_profile": "event_audio_v2_rhythm"},
        "songs": [], "summary": {}, "warnings": [], "errors": [],
    }), encoding="utf-8")
    checkpoint = tmp_path / "event_last.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(webui, "_checkpoint_inspection", lambda *args, **kwargs: {
        "errors": [], "warnings": [], "epoch": 4, "model_config": {"planner_profile": "legacy_v1"},
        "data_contract": {"target_schema": "event_v1", "condition_params": {
            "mask_prob": 0.5, "dropout": 0.0, "default": 0.5,
            "null_embedding": True, "null_policy": "masked_condition_controls",
        }},
    })
    calls = []

    def fake_run(task, args, *, expected_epoch, result_paths):
        calls.append((args, expected_epoch))
        task.status = "completed"

    monkeypatch.setattr(webui, "_run_subprocess_task", fake_run)
    status, started = webui_api("/api/train", {
        "model_family": "event_v1", "manifest": str(manifest), "raw_root": str(tmp_path),
        "cache_dir": str(tmp_path / "cache"), "checkpoint_dir": str(tmp_path / "checkpoints"),
        "training_mode": "resume", "training_checkpoint": str(checkpoint), "epochs": 3,
    })
    assert status == 200, started
    args, expected_epoch = calls[0]
    assert expected_epoch == 7
    assert "--resume" in args
    task = webui.TASKS[started["task_id"]]
    assert task.result["epoch_start"] == 4
    assert task.result["epochs_requested"] == 3
    assert task.result["epoch_target"] == 7
    assert task.result["epoch_semantics"] == "additional_from_checkpoint"


def test_gold_scan_reports_the_folder_contents(webui_api, tmp_path):
    """The panel previews a gold folder before anything is written."""

    raw = tmp_path / "raw"
    manifest = _manifest_with_designers(raw / "training_manifest.json")
    manifest.write_text(json.dumps({
        "schema": "maichart-training-manifest-v1", "source_root": ".", "cache_dir": "cache",
        "preprocessing": {"target_schema": "event_v1", "feature_profile": "event_audio_v1"},
        "songs": [], "summary": {}, "warnings": [], "errors": [],
    }, ensure_ascii=False), encoding="utf-8")
    chart = raw / "gold" / "GoodOne"
    chart.mkdir(parents=True)
    (chart / "maidata.txt").write_text(
        "&title=GoodOne\n&wholebpm=134\n&first=0\n&lv_4=13\n&inote_4=(134){4}" + "1,2,3,4," * 16 + "\n",
        encoding="utf-8",
    )
    (chart / "track.mp3").write_bytes(b"\x00" * 32)

    status, data = webui_api("/api/gold/scan", {"manifest": str(manifest), "raw_root": str(raw)})
    assert status == 200
    assert data["exists"] is True
    assert data["songs"] == 1
    assert data["usable_difficulties"] >= 1
    assert data["gold_root"].endswith("gold")

    # A missing folder is a normal answer, not an error.
    status, data = webui_api("/api/gold/scan", {"manifest": str(manifest), "gold_root": str(raw / "nope")})
    assert status == 200
    assert data["exists"] is False
    assert "gold" in data["message"]


def test_gold_build_in_folder_mode_requires_the_folder(webui_api, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    manifest = raw / "training_manifest.json"
    manifest.write_text(json.dumps({
        "schema": "maichart-training-manifest-v1", "source_root": ".", "cache_dir": "cache",
        "preprocessing": {"target_schema": "event_v1", "feature_profile": "event_audio_v1"},
        "songs": [], "summary": {}, "warnings": [], "errors": [],
    }), encoding="utf-8")
    status, data = webui_api("/api/gold/build", {
        "manifest": str(manifest), "mode": "folder", "raw_root": str(raw),
    })
    assert status == 400
    assert "gold 文件夹" in data["error"]


def test_gold_build_passes_the_gold_root_to_the_cli(webui_api, monkeypatch, tmp_path):
    raw = tmp_path / "raw"
    manifest = _manifest_with_designers(raw / "training_manifest.json")
    gold_dir = raw / "gold"
    gold_dir.mkdir(parents=True)
    calls = []

    def fake_run(task, args, *, expected_epoch, result_paths):
        calls.append(args)
        for path in result_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
        task.status = "completed"
        webui._collect_task_results(task, result_paths)

    monkeypatch.setattr(webui, "_run_subprocess_task", fake_run)
    status, _started = webui_api("/api/gold/build", {
        "manifest": str(manifest), "mode": "folder",
        "gold_root": str(gold_dir), "val_ratio": 0.3, "min_note_count": 0,
    })
    assert status == 200
    assert calls[0][:2] == ["-m", "maichart.training.gold_set"]
    assert "--gold-root" in calls[0]
    assert str(gold_dir) in calls[0]


def test_unknown_checkpoint_schema_is_not_treated_as_baseline(webui_api, monkeypatch, tmp_path):
    checkpoint = tmp_path / "unknown.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(webui, "_load_checkpoint_payload", lambda path: {
        "schema": "someone-elses-model-v9",
        "model_state_dict": {"weight": object()},
    })
    status, data = webui_api("/api/checkpoint/inspect", {
        "checkpoint_path": str(checkpoint), "mode": "generate",
    })
    assert status == 200
    assert data["valid"] is False
    assert data["model_family"] == "unknown"


def _manifest_with_designers(path, designers=("GoodArtist", "OtherDude")):
    path.parent.mkdir(parents=True, exist_ok=True)
    songs = []
    for index, designer in enumerate(designers * 4):
        songs.append({
            "song_id": f"song{index:03d}",
            "title": f"Song {index}",
            "artist": "Tester",
            "maidata_path": f"song{index:03d}/maidata.txt",
            "audio_path": f"song{index:03d}/track.mp3",
            "wholebpm": 150,
            "dataset_usable": True,
            "audio": {"readable": True, "status": "source_present"},
            "difficulties": [{
                "difficulty_index": 5, "level_raw": "13", "level": 13.0,
                "designer": designer, "note_count": 500,
                "event_usable_for_training": True, "filter_reasons": [],
            }],
        })
    path.write_text(json.dumps({
        "schema": "maichart-training-manifest-v1",
        "source_root": ".", "cache_dir": "cache",
        "song_count": len(songs), "difficulty_count": len(songs),
        "usable_difficulty_count": len(songs),
        "preprocessing": {"target_schema": "event_v1", "feature_profile": "event_audio_v1",
                          "sample_rate": 22050, "hop_length": 441},
        "songs": songs, "summary": {}, "warnings": [], "errors": [],
    }, ensure_ascii=False), encoding="utf-8")
    return path


def test_gold_designers_endpoint_lists_the_corpus(webui_api, tmp_path):
    manifest = _manifest_with_designers(tmp_path / "training_manifest.json")
    status, data = webui_api("/api/gold/designers", {"manifest": str(manifest)})
    assert status == 200
    assert data["usable_difficulties"] == 8
    counts = {item["name"]: item["count"] for item in data["designers"]}
    assert counts == {"GoodArtist": 4, "OtherDude": 4}


def test_gold_build_requires_a_criterion(webui_api, tmp_path):
    manifest = _manifest_with_designers(tmp_path / "training_manifest.json")
    status, data = webui_api("/api/gold/build", {"manifest": str(manifest), "designers": []})
    assert status == 400
    assert "谱师" in data["error"]


def test_gold_build_dispatches_the_cli_and_reports_manifests(webui_api, monkeypatch, tmp_path):
    manifest = _manifest_with_designers(tmp_path / "training_manifest.json")
    calls = []

    def fake_run(task, args, *, expected_epoch, result_paths):
        calls.append(args)
        for path in result_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.name == "gold_summary.json":
                path.write_text(json.dumps({
                    "selection": {"gold_songs": 4, "criteria": {"designers": ["GoodArtist"]}},
                    "pretrain": {"songs": 7}, "gold_finetune": {"songs": 3}, "gold_val": {"songs": 1},
                }), encoding="utf-8")
            else:
                path.write_text("{}", encoding="utf-8")
        task.status = "completed"
        webui._collect_task_results(task, result_paths)

    monkeypatch.setattr(webui, "_run_subprocess_task", fake_run)
    status, started = webui_api("/api/gold/build", {
        "manifest": str(manifest), "mode": "designers", "designers": ["GoodArtist"],
        "song_ids": [], "val_ratio": 0.3, "min_note_count": 0,
    })
    assert status == 200
    assert calls[0][:2] == ["-m", "maichart.training.gold_set"]
    assert "--designers" in calls[0] and "GoodArtist" in calls[0]
    task = webui.TASKS[started["task_id"]]
    assert task.result["pretrain_manifest"].endswith("pretrain_manifest.json")
    assert task.result["gold_val_manifest"].endswith("gold_val_manifest.json")
    assert task.result["gold_summary"]["selection"]["gold_songs"] == 4
    assert task.result["pretrain_manifest_exists"] is True
    assert task.result["gold_stats_exists"] is True


def test_acceptance_endpoint_returns_the_report(webui_api, monkeypatch, tmp_path):
    maidata = tmp_path / "maidata.txt"
    maidata.write_text("&inote_5=(120){4}1,2,3,4,", encoding="utf-8")
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"audio")
    gold = tmp_path / "gold_stats.json"
    gold.write_text('{"charts": 12, "metrics": {}}', encoding="utf-8")
    calls = []

    def fake_run(task, args, *, expected_epoch, result_paths):
        calls.append(args)
        for path in result_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "schema": "maichart-acceptance-v1", "verdict": "warn", "failures": 0, "warnings": 2,
                "criteria": [{"key": "blank_fraction", "label": "空白小节比例", "value": 0.2,
                              "low": 0.01, "high": 0.08, "verdict": "warn", "message": "高于金标 p90"}],
                "timeline": {"points": [{"t": 0.0, "intensity": 1.0, "density": 0.0}], "blank_spans": []},
            }, ensure_ascii=False), encoding="utf-8")
        task.status = "completed"
        webui._collect_task_results(task, result_paths)

    monkeypatch.setattr(webui, "_run_subprocess_task", fake_run)
    status, started = webui_api("/api/acceptance/evaluate", {
        "maidata": str(maidata), "audio": str(audio), "gold_stats": str(gold), "difficulty": 5,
    })
    assert status == 200
    assert calls[0][:2] == ["-m", "maichart.generation.acceptance"]
    assert "--gold-stats" in calls[0] and "--audio" in calls[0]
    task = webui.TASKS[started["task_id"]]
    report = task.result["acceptance_report"]
    assert report["verdict"] == "warn"
    assert report["criteria"][0]["key"] == "blank_fraction"
    assert task.result["files"]


def test_acceptance_requires_existing_paths(webui_api, tmp_path):
    status, data = webui_api("/api/acceptance/evaluate", {"maidata": str(tmp_path / "missing.txt")})
    assert status == 400
    assert "不存在" in data["error"]


def test_generation_passes_the_gold_reference(webui_api, monkeypatch, tmp_path):
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"audio")
    checkpoint = tmp_path / "event.pt"
    checkpoint.write_bytes(b"checkpoint")
    gold = tmp_path / "gold_stats.json"
    gold.write_text('{"charts": 5, "metrics": {}}', encoding="utf-8")
    calls = []
    monkeypatch.setattr(webui, "_load_checkpoint_payload", lambda path: {
        "schema": "maichart-event-generator-checkpoint-v1",
        "model_state_dict": {"weight": object()},
        "model_config": {"d_model": 32},
        "data_contract": {"target_schema": "event_v1"},
    })

    def fake_run(task, args, *, expected_epoch, result_paths):
        calls.append(args)
        for path in result_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
        task.status = "completed"
        webui._collect_task_results(task, result_paths)

    monkeypatch.setattr(webui, "_run_subprocess_task", fake_run)
    status, _started = webui_api("/api/generate", {
        "checkpoint": str(checkpoint), "audio": str(audio),
        "output": str(tmp_path / "out.txt"), "candidates": 1, "gold_stats_path": str(gold),
    })
    assert status == 200
    assert "--gold-stats" in calls[0]
    assert str(gold) in calls[0]


def test_logs_endpoint_lists_persisted_runs(webui_api, monkeypatch, tmp_path):
    """The history panel must be able to read finished runs from disk."""

    monkeypatch.setattr(webui, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(webui, "LOG_ROOT", tmp_path / "logs")
    task = webui.TaskState(task_id="hist1", kind="train")
    webui._open_task_log(task, ["-m", "maichart.training.train_event_ctx", "--epochs", "70"])
    task.log("epoch=001 loss=1.2 val_loss=0.9")
    webui._close_task_log(task, status="completed", exit_code=0)

    status, data = webui_api("/api/logs")
    assert status == 200
    assert data["root"].endswith("logs")
    entry = data["logs"][0]
    assert entry["kind"] == "train" and entry["status"] == "completed"
    assert entry["task_id"] == "hist1" and entry["bytes"] > 0
    assert entry["download"] and "api/download" in entry["download"]
