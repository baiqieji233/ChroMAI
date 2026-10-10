"""Verify that WebUI selections reach the trainer and its inspections."""
import json
from pathlib import Path

import pytest

from maichart.webui import server as webui
from test_webui import webui_api


def make_source(root: Path, title: str) -> Path:
    song = root / "song"
    song.mkdir(parents=True)
    (song / "maidata.txt").write_text(
        f"&title={title}\n&wholebpm=120\n&lv_5=13\n&des_5=Tester\n&inote_5=(120){{4}}1,2,3,4,E\n", encoding="utf-8")
    (song / "track.wav").write_bytes(b"placeholder audio")
    manifest = root / "training_manifest.json"
    manifest.write_text(json.dumps({
        "schema": "maichart-training-manifest-v1", "source_root": ".", "cache_dir": "cache",
        "preprocessing": {"target_schema": "event_v1", "feature_profile": "event_audio_v2_rhythm",
                          "sample_rate": 22050, "hop_length": 441},
        "songs": [{"song_id": "same-song-id", "title": title, "maidata_path": "song/maidata.txt",
                   "audio_path": "song/track.wav", "difficulties": [{"difficulty_index": 5,
                   "level": 13.0, "level_raw": "13", "event_usable_for_training": True}]}],
    }), encoding="utf-8")
    return manifest


def test_multi_dataset_train_and_inspection_use_selection(webui_api, monkeypatch, tmp_path):
    monkeypatch.setattr(webui, "WEBUI_ROOT", tmp_path / "workspace")
    a, b = make_source(tmp_path / "a", "Title A"), make_source(tmp_path / "b", "Title B")
    payload = {"manifests": [str(a), str(b)], "manifest": str(a), "raw_root": str(a.parent),
               "model_family": "event_ctx_v1", "epochs": 1, "checkpoint_dir": str(tmp_path / "new-model")}
    status, inspected = webui_api("/api/inspect-manifest", payload)
    assert status == 200, inspected
    assert inspected["song_count"] == 2
    assert Path(inspected["manifest"]) not in {a, b}
    calls = []
    monkeypatch.setattr(webui, "_run_subprocess_task", lambda task, args, **kwargs: calls.append(args))
    status, result = webui_api("/api/train", payload)
    assert status == 200, result
    args = calls[0]
    assert args[args.index("--manifest") + 1] == inspected["manifest"]
    assert args[args.index("--cache-dir") + 1] == inspected["cache_dir"]
    task = webui.TASKS[result["task_id"]]
    assert task.result["dataset_selection"]["manifest"] == inspected["manifest"]
    assert len(json.loads(Path(inspected["manifest"]).read_text(encoding="utf-8"))["songs"]) == 2


def test_checkpoint_inspection_receives_selected_sources(webui_api, monkeypatch, tmp_path):
    monkeypatch.setattr(webui, "WEBUI_ROOT", tmp_path / "workspace")
    a, b = make_source(tmp_path / "a", "Title A"), make_source(tmp_path / "b", "Title B")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test")
    seen = []
    def inspection(path, **kwargs):
        seen.append(kwargs["manifest"])
        return {"valid": True, "errors": []}
    monkeypatch.setattr(webui, "_checkpoint_inspection", inspection)
    status, result = webui_api("/api/checkpoint/inspect", {
        "checkpoint": str(checkpoint), "mode": "finetune", "manifests": [str(a), str(b)],
        "manifest": str(a), "raw_root": str(a.parent), "model_family": "event_ctx_v1"})
    assert status == 200, result
    assert len(seen[0]["songs"]) == 2
    assert result["dataset_selection"]["manifest"] == result["manifest"]


def test_directory_registration_survives_new_api_instance(webui_api, monkeypatch, tmp_path):
    monkeypatch.setattr(webui, "WEBUI_ROOT", tmp_path / "workspace")
    source = make_source(tmp_path / "outside-datasets", "External")
    status, registered = webui_api("/api/datasets/register", {"path": str(source.parent), "name": "精选素材"})
    assert status == 200, registered
    assert registered["prepared"]
    status, catalog = webui_api("/api/datasets/catalog", {})
    assert status == 200, catalog
    assert catalog["datasets"][0]["name"] == "精选素材"
    assert catalog["datasets"][0]["manifest"] == str(source)


def test_raw_registration_is_visible_before_preprocessing(webui_api, monkeypatch, tmp_path):
    monkeypatch.setattr(webui, "WEBUI_ROOT", tmp_path / "workspace")
    source = make_source(tmp_path / "raw", "Raw")
    source.unlink()
    status, registered = webui_api("/api/datasets/register", {"path": str(source.parent)})
    assert status == 200, registered
    assert not registered["prepared"]
    status, catalog = webui_api("/api/datasets/catalog", {})
    assert len(catalog["datasets"]) == 1
    assert not catalog["datasets"][0]["prepared"]


def test_preprocess_default_output_belongs_to_raw_source(webui_api, monkeypatch, tmp_path):
    source = make_source(tmp_path / "raw", "Raw")
    source.unlink()
    calls = []
    monkeypatch.setattr(webui, "_run_subprocess_task", lambda task, args, **kwargs: calls.append(args))
    status, result = webui_api("/api/preprocess", {"raw_root": str(source.parent), "model_family": "event_ctx_v1"})
    assert status == 200, result
    task = webui.TASKS[result["task_id"]]
    assert task.result["manifest"] == str(source)
    assert calls[0][calls[0].index("-o") + 1] == str(source)


def test_preprocessing_new_directory_does_not_overwrite_another_source(webui_api, tmp_path):
    a, b = make_source(tmp_path / "a", "A"), make_source(tmp_path / "b", "B")
    before = a.read_bytes()
    status, result = webui_api("/api/preprocess", {"raw_root": str(b.parent), "manifest": str(a)})
    assert status == 400
    assert "另一份" in result["error"]
    assert a.read_bytes() == before
    assert not webui.TASKS


def test_cards_preserve_current_pretrain_manifest(webui_api, monkeypatch, tmp_path):
    source = make_source(webui.DATASET_ROOT / "full", "Full")
    pretrain = source.with_name("pretrain_manifest.json")
    pretrain.write_bytes(source.read_bytes())
    status, result = webui_api("/api/datasets/catalog", {"current_manifest": str(pretrain)})
    assert status == 200, result
    assert result["datasets"][0]["manifest"] == str(pretrain)
    status, result = webui_api("/api/datasets/catalog", {"current_manifest": str(source)})
    assert result["datasets"][0]["manifest"] == str(source)


def test_selection_does_not_skip_pending_source(webui_api, monkeypatch, tmp_path):
    a, b = make_source(tmp_path / "a", "A"), make_source(tmp_path / "b", "B")
    task = webui.TaskState("pending", "preprocess", status="running")
    task.result["manifest"] = str(b)
    webui.TASKS[task.task_id] = task
    for endpoint in ("/api/train", "/api/inspect-manifest"):
        status, result = webui_api(endpoint, {"manifests": [str(a), str(b)], "manifest": str(a)})
        assert status == 400, result
        assert "正在预处理" in result["error"]


def test_joint_auto_preprocess_does_not_prepare_only_one_source(webui_api, tmp_path):
    a, b = make_source(tmp_path / "a", "A"), make_source(tmp_path / "b", "B")
    status, result = webui_api("/api/train", {"manifests": [str(a), str(b)],
                            "manifest": str(a), "auto_preprocess": True})
    assert status == 400
    assert "分别预处理" in result["error"]
    assert not webui.TASKS


def test_single_auto_preprocess_can_rebuild_previous_family_manifest(tmp_path):
    source = make_source(tmp_path / "raw", "Raw")
    data = json.loads(source.read_text(encoding="utf-8"))
    data["preprocessing"]["target_schema"] = "frame_v2"
    source.write_text(json.dumps(data), encoding="utf-8")
    payload = {"manifest": str(source), "manifests": [str(source)], "raw_root": str(source.parent),
               "model_family": "event_ctx_v1", "auto_preprocess": True}
    prepared, selection = webui._prepare_training_sources(payload)
    assert prepared["auto_preprocess"]
    assert prepared["manifest"] == str(source)
    assert selection is None


def test_combination_does_not_hide_validation_overlap(webui_api, monkeypatch, tmp_path):
    monkeypatch.setattr(webui, "WEBUI_ROOT", tmp_path / "workspace")
    a, b = make_source(tmp_path / "a", "A"), make_source(tmp_path / "b", "B")
    status, result = webui_api("/api/train", {"manifests": [str(a), str(b)],
                            "manifest": str(a), "validation_manifest": str(a)})
    assert status == 400, result
    assert "重叠" in result["error"]
    assert not webui.TASKS


@pytest.mark.parametrize("invalid", ["not-a-list", [None], [""]])
def test_invalid_selection_is_rejected_before_task(webui_api, invalid):
    status, result = webui_api("/api/train", {"manifests": invalid})
    assert status == 400
    assert not webui.TASKS
