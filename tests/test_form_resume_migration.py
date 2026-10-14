"""Low-cost CPU coverage for explicit song-form resume data migration."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
import torch

from maichart.generation.event_audio import ACOUSTIC_DIM, EventAudio


def _assert_recursive_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_recursive_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_recursive_equal(actual_item, expected_item)
    else:
        assert actual == expected


def _fixture_audio(seconds=3):
    times = np.arange(int(seconds * 50), dtype=np.float32) * 0.02
    features = np.zeros((len(times), ACOUSTIC_DIM), np.float32)
    features[:, 80] = 0.7 + 0.3 * np.sin(times * 3)
    features[:, 92] = 0.5 + 0.5 * np.cos(times * 12)
    features[:, 93] = 0.5
    features[:, 95] = 1
    return EventAudio(features, times, seconds, 120.)


def _manifest(tmp_path):
    songs = []
    for name, notes in (
        ("one", "1,2,3,4,1,2,3,4,E"),
        ("two", "5,6,7,8,5,6,7,8,E"),
    ):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "maidata.txt").write_text(
            f"&title={name}\n&wholebpm=120\n&lv_5=13\n"
            f"&inote_5=(120){{4}}{notes}\n",
            encoding="utf-8",
        )
        (folder / "track.wav").write_bytes(b"fixture")
        songs.append({
            "song_id": name,
            "maidata_path": f"{name}/maidata.txt",
            "audio_path": f"{name}/track.wav",
            "difficulties": [{"difficulty_index": 5, "level_raw": "13"}],
        })
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"songs": songs}), encoding="utf-8")
    return manifest


def _common(manifest, tmp_path):
    return [
        "--manifest", str(manifest), "--cache-dir", str(tmp_path / "cache"),
        "--epochs", "1", "--batch-size", "1", "--grad-accumulation", "1",
        "--d-model", "16", "--nhead", "2", "--layers", "1",
        "--max-tokens", "128", "--max-memory-tokens", "32",
        "--max-region-seconds", "3", "--checkpoint-interval-steps", "1",
        "--precision", "fp32", "--seed", "7", "--max-steps", "2",
        "--device", "cpu",
    ]


def _make_mid_epoch_checkpoint(tmp_path, monkeypatch):
    from maichart.form import data, train

    manifest = _manifest(tmp_path)
    audio = _fixture_audio()
    real_dataset = data.FormDataset

    def build_dataset(*args, **kwargs):
        dataset = real_dataset(*args, **kwargs, audio_loader=lambda *_a, **_kw: audio)
        audio_hash = hashlib.sha256(b"fixture").hexdigest()
        for chart in dataset.audit["charts"]:
            chart["audio_sha256"] = audio_hash
        return dataset

    monkeypatch.setattr(
        train,
        "FormDataset",
        build_dataset,
    )
    common = _common(manifest, tmp_path) + ["--limit-charts", "1"]
    source_dir = tmp_path / "source"
    original_save = train._atomic_save

    def interrupt_after_first_layout_batch(value, path):
        original_save(value, path)
        position = value["training_position"]
        if position["phase"] == "layout_revision" and position["completed"] == 1:
            raise InterruptedError("test interruption after durable migration source")

    monkeypatch.setattr(train, "_atomic_save", interrupt_after_first_layout_batch)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with pytest.raises(InterruptedError):
            train.main(common + ["--checkpoint-dir", str(source_dir)])
    finally:
        torch.set_num_threads(threads)
        monkeypatch.setattr(train, "_atomic_save", original_save)

    source_checkpoint = source_dir / "form_last.pt"
    old = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    assert old["training_position"]["phase"] == "layout_revision"
    assert old["training_position"]["completed"] == 1
    assert len(old["dataset_contract"]["charts"]) == 1
    old["dataset_contract"]["charts"][0]["audio_sha256"] = hashlib.sha256(
        b"fixture"
    ).hexdigest()
    old["epoch"] = 11
    old["training_position"]["epoch"] = 12

    # Keep the manifest and source files unchanged, while simulating the old
    # run's effective selection having accepted only one chart.
    old["args"]["limit_charts"] = 0
    old["training_history"] = [{"migration_test_marker": "old-history"}]
    old["training_position"]["phase"] = "layout_revision"
    old["training_position"]["completed"] = 1
    torch.save(old, source_checkpoint)
    return manifest, source_checkpoint, old


def test_strict_resume_rejects_added_chart_and_writes_diff(tmp_path, monkeypatch):
    from maichart.form import train

    manifest, source_checkpoint, old = _make_mid_epoch_checkpoint(tmp_path, monkeypatch)
    output = tmp_path / "strict"
    with pytest.raises(ValueError, match="新增|restart_epoch|finetune"):
        train.main(_common(manifest, tmp_path) + [
            "--checkpoint-dir", str(output), "--resume", str(source_checkpoint),
        ])

    diff = json.loads((output / "resume_contract_diff.json").read_text(encoding="utf-8"))
    assert diff["old_chart_count"] == 1
    assert diff["new_chart_count"] == 2
    assert len(diff["added"]) == 1
    assert diff["removed"] == []
    assert diff["can_restart_epoch"] is True
    assert old["training_position"]["phase"] == "layout_revision"


def test_restart_epoch_preserves_state_and_starts_planner_from_boundary(tmp_path, monkeypatch):
    from maichart.form import train

    manifest, source_checkpoint, old = _make_mid_epoch_checkpoint(tmp_path, monkeypatch)
    loaded_before_update = {}
    original_load = train.checkpoint.load

    def capture_load(*args, **kwargs):
        model, planner, codec, payload = original_load(*args, **kwargs)
        if kwargs.get("inference") is not True:
            loaded_before_update["model"] = deepcopy(model.state_dict())
            loaded_before_update["planner"] = deepcopy(planner.state_dict())
        return model, planner, codec, payload

    monkeypatch.setattr(train.checkpoint, "load", capture_load)
    optimizer_restore = {}
    original_optimizer_load = torch.optim.AdamW.load_state_dict

    def capture_optimizer_load(optimizer, state_dict):
        optimizer_restore["incoming"] = deepcopy(state_dict)
        result = original_optimizer_load(optimizer, state_dict)
        optimizer_restore["after"] = deepcopy(optimizer.state_dict())
        return result

    monkeypatch.setattr(torch.optim.AdamW, "load_state_dict", capture_optimizer_load)
    planner_indices = []
    original_blueprint_input = train._blueprint_input

    def observe_blueprint_input(dataset, index):
        planner_indices.append(index)
        return original_blueprint_input(dataset, index)

    monkeypatch.setattr(train, "_blueprint_input", observe_blueprint_input)
    output = tmp_path / "migrated"
    assert train.main(_common(manifest, tmp_path) + [
        "--checkpoint-dir", str(output), "--resume", str(source_checkpoint),
        "--resume-data-policy", "restart_epoch",
    ]) == 0

    migrated = torch.load(output / "form_last.pt", map_location="cpu", weights_only=False)
    assert planner_indices[:2] == [0, 1]
    assert optimizer_restore["incoming"] is not None
    _assert_recursive_equal(optimizer_restore["incoming"], old["optimizer_state_dict"])
    _assert_recursive_equal(optimizer_restore["after"], old["optimizer_state_dict"])
    assert migrated["optimizer_steps"] > old["optimizer_steps"]
    assert migrated["blueprint_steps"] > old["blueprint_steps"]
    assert migrated["training_history"][0]["migration_test_marker"] == "old-history"
    assert migrated["training_position"]["phase"] == "complete"
    assert migrated["epoch"] == 12
    assert migrated["data_migrations"][-1]["policy"] == "restart_epoch"
    assert migrated["data_migrations"][-1]["restart_epoch"] == 12
    assert migrated["data_migrations"][-1]["old_position"]["completed"] == 1
    assert migrated["dataset_contract"]["charts"] != old["dataset_contract"]["charts"]
    for name, value in old["blueprint_state_dict"].items():
        torch.testing.assert_close(loaded_before_update["planner"][name], value, rtol=0, atol=0)
    for name, value in old["model_state_dict"].items():
        torch.testing.assert_close(loaded_before_update["model"][name], value, rtol=0, atol=0)
    assert any(
        not torch.equal(migrated["model_state_dict"][name], old["model_state_dict"][name])
        for name in old["model_state_dict"]
    )


def test_restart_epoch_requires_a_new_checkpoint_output_directory(tmp_path, monkeypatch):
    from maichart.form import train

    manifest, source_checkpoint, _ = _make_mid_epoch_checkpoint(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="新的 checkpoint 输出目录"):
        train.main(_common(manifest, tmp_path) + [
            "--checkpoint-dir", str(source_checkpoint.parent),
            "--resume", str(source_checkpoint), "--resume-data-policy", "restart_epoch",
        ])


def test_changed_source_is_rejected_before_formdataset(tmp_path, monkeypatch):
    from maichart.form import train

    manifest, source_checkpoint, _ = _make_mid_epoch_checkpoint(tmp_path, monkeypatch)
    source = tmp_path / "one" / "maidata.txt"
    original = source.read_bytes()
    source.write_bytes(original + b"\n")

    def dataset_must_not_be_constructed(*_args, **_kwargs):
        raise AssertionError("source identity failure must precede FormDataset construction")

    monkeypatch.setattr(train, "FormDataset", dataset_must_not_be_constructed)
    with pytest.raises(ValueError, match="源文件变化或缺失"):
        train.main(_common(manifest, tmp_path) + [
            "--checkpoint-dir", str(tmp_path / "changed-source"),
            "--resume", str(source_checkpoint),
        ])
