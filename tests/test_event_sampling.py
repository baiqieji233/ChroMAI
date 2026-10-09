"""Sampling must not raise Touch frequency through the Slide multiplier.

The migration tests exercise the exact contract barrier that previously
prevented reusing a trained model with a corrected sampler.
"""

import copy
import json
from collections import Counter

import pytest

from maichart.training.event_checkpoints import compare_event_contract
from maichart.training.event_ctx_dataset import EventCtxDataset, OVERSAMPLE_POLICY


@pytest.fixture
def sampling_manifest(tmp_path):
    songs = []
    notes = {
        "plain": "1,2,3,4,E",
        "slide": "1-5[4:1],2,3,4,E",
        "touch": "A1,2,3,4,E",
        "touch_hold": "Ch[4:1],2,3,4,E",
        "mixed": "A1,1-5[4:1],3,4,E",
    }
    for name, pattern in notes.items():
        root = tmp_path / name
        root.mkdir()
        (root / "maidata.txt").write_text(
            "&wholebpm=120\n&first=0\n&lv_5=13\n&inote_5=(120){4}" + pattern + "\n",
            encoding="utf-8",
        )
        (root / "track.mp3").write_bytes(b"placeholder")
        songs.append({"song_id": name, "maidata_path": f"{name}/maidata.txt",
                      "audio_path": f"{name}/track.mp3",
                      "difficulties": [{"difficulty_index": 5, "level_raw": "13"}]})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"songs": songs}), encoding="utf-8")
    return manifest


def make_dataset(manifest, slide=1, touch=1):
    return EventCtxDataset(manifest, cache_dir=manifest.parent / "cache",
                           eager_audio=False, segment_beats=4, max_tokens=128,
                           context_beats=0, context_tokens=16, context_dropout=0,
                           slide_oversample=slide, touch_oversample=touch)


@pytest.mark.parametrize("slide,touch,expected", [
    (4, 1, {"plain": 1, "slide": 4, "touch": 1, "touch_hold": 1, "mixed": 1}),
    (1, 3, {"plain": 1, "slide": 1, "touch": 3, "touch_hold": 3, "mixed": 3}),
    (4, 2, {"plain": 1, "slide": 4, "touch": 2, "touch_hold": 2, "mixed": 2}),
])
def test_types_have_independent_sampling_and_touch_controls_mixed_windows(
    sampling_manifest, slide, touch, expected
):
    dataset = make_dataset(sampling_manifest, slide, touch)
    counts = Counter(dataset.samples[item[0]].song_id for item in dataset.segments)
    assert counts == expected
    audit = dataset.audit()
    assert sum(audit["sampling_windows"].values()) == 5
    assert sum(audit["sampling_segments"].values()) == sum(expected.values())
    assert audit["sampling_segments"]["mixed"] == touch
    assert audit["touch_oversample"] == touch
    assert audit["slide_oversample"] == slide
    assert audit["oversample_policy"] == OVERSAMPLE_POLICY


def test_terminal_e_does_not_trigger_touch_oversampling(sampling_manifest):
    # Every document ends with E, but the plain and Slide-only windows must
    # still use their own weights, not the Touch weight.
    dataset = make_dataset(sampling_manifest, 2, 5)
    counts = Counter(dataset.samples[item[0]].song_id for item in dataset.segments)
    assert counts["plain"] == 1
    assert counts["slide"] == 2


@pytest.mark.parametrize("keyword", ["slide_oversample", "touch_oversample"])
def test_sampling_multipliers_cannot_remove_windows(sampling_manifest, keyword):
    with pytest.raises(ValueError, match=keyword):
        EventCtxDataset(sampling_manifest, eager_audio=False, **{keyword: 0})


def old_checkpoint():
    return {"optimizer_state_dict": {"state": {}}, "data_contract": {
        "target_schema": "event_ctx_v1", "dataset_signature": "same",
        "family_params": {"context_beats": 16, "audio_margin_beats": 8,
                          "context_tokens": 320, "overlap_beats": 0,
                          "context_dropout": 0.15, "slide_oversample": 3},
        "condition_params": {"mask_prob": 0.5, "dropout": 0.15, "default": 0.5},
    }}


def corrected_contract(stored):
    current = copy.deepcopy(stored["data_contract"])
    current["family_params"].update(slide_oversample=2, touch_oversample=1,
                                    oversample_policy=OVERSAMPLE_POLICY)
    return current


def test_old_weights_can_initialize_the_corrected_sampler():
    stored = old_checkpoint()
    current = corrected_contract(stored)
    original = copy.deepcopy(stored)
    compare_event_contract(stored, current, mode="init")
    assert stored == original, "migration must not rewrite the source checkpoint"
    with pytest.raises(ValueError, match="finetune"):
        compare_event_contract(stored, current, mode="resume")


def test_new_sampler_can_resume_only_with_matching_settings():
    stored = old_checkpoint()
    stored["data_contract"] = corrected_contract(stored)
    compare_event_contract(stored, copy.deepcopy(stored["data_contract"]), mode="resume")
    altered = copy.deepcopy(stored["data_contract"])
    altered["family_params"]["touch_oversample"] = 2
    with pytest.raises(ValueError, match="family_params"):
        compare_event_contract(stored, altered, mode="resume")


@pytest.mark.parametrize("key,value", [
    ("context_beats", 8), ("audio_margin_beats", 4), ("context_tokens", 192),
    ("overlap_beats", 4), ("context_dropout", 0.0),
])
def test_finetune_sampling_exception_does_not_relax_context_contract(key, value):
    stored = old_checkpoint()
    current = corrected_contract(stored)
    current["family_params"][key] = value
    with pytest.raises(ValueError, match="family_params"):
        compare_event_contract(stored, current, mode="init")


def test_finetune_sampling_exception_does_not_relax_condition_contract():
    stored = old_checkpoint()
    current = corrected_contract(stored)
    current["condition_params"]["mask_prob"] = 0.2
    with pytest.raises(ValueError, match="condition_params"):
        compare_event_contract(stored, current, mode="init")


def test_training_entrypoint_keeps_old_weights_and_resets_optimization(
    sampling_manifest, tmp_path, monkeypatch
):
    """Exercise actual init/CPU optimization/save after changing a legacy sampler.

    Check the weights before the first new step, the fresh optimizer/LR, and
    the new saved contract. No real data or GPU is used by this test.
    """
    import numpy as np
    import torch
    from maichart.generation.event_audio import ACOUSTIC_DIM, EventAudio
    from maichart.training import event_dataset, train_event_ctx, train_events
    from maichart.training.event_checkpoints import load_event_checkpoint

    audio = EventAudio(features=np.zeros((12, ACOUSTIC_DIM), dtype=np.float32),
                       times=np.arange(12, dtype=np.float32) * 0.1,
                       duration_sec=2.0, tempo_bpm=120.0, tempo_changes=[], metadata={})
    monkeypatch.setattr(event_dataset, "extract_event_audio", lambda *a, **kw: audio)
    # train_event_ctx.configure swaps shared hooks; isolate it from other tests.
    monkeypatch.setattr(train_events, "DATASET_CLASS", EventCtxDataset)
    monkeypatch.setattr(train_events, "TARGET_SCHEMA", "event_ctx_v1")
    monkeypatch.setattr(train_events, "FAMILY_LABEL", "event_ctx_v1")
    monkeypatch.setattr(train_events, "dataset_kwargs", train_event_ctx.family_dataset_kwargs)
    torch_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    source_dir = tmp_path / "source"
    common = ["--manifest", str(sampling_manifest), "--cache-dir", str(tmp_path / "cache"),
              "--device", "cpu", "--epochs", "1", "--max-steps", "1", "--batch-size", "1",
              "--segment-beats", "4", "--max-tokens", "128", "--max-audio-tokens", "32",
              "--global-frames", "4", "--context-beats", "0", "--audio-margin-beats", "0",
              "--context-tokens", "16", "--context-dropout", "0", "--d-model", "16",
              "--nhead", "2", "--num-layers", "1", "--dim-feedforward", "32",
              "--dropout", "0", "--grad-accumulation", "1", "--prewarm-workers", "off",
              "--prefetch-batches", "0", "--slide-oversample", "3", "--touch-oversample", "3"]
    try:
        assert train_events.main(common + ["--checkpoint-dir", str(source_dir)]) == 0
        source = load_event_checkpoint(source_dir / "event_last.pt")
        # Reproduce pre-split metadata without modifying any workspace model.
        source["data_contract"]["family_params"].pop("touch_oversample")
        source["data_contract"]["family_params"].pop("oversample_policy")
        source["epoch"] = 11
        old_path = tmp_path / "old_epoch11.pt"
        torch.save(source, old_path)
        original_epoch = train_events.run_epoch
        observed = {}

        def inspect_first_step(model, loader, device, **kwargs):
            if kwargs.get("optimizer") is None:
                assert loader.dataset.slide_oversample == 1
                assert loader.dataset.touch_oversample == 1
                observed["natural_validation"] = True
                return original_epoch(model, loader, device, **kwargs)
            for name, value in model.state_dict().items():
                assert torch.equal(value.cpu(), source["model_state_dict"][name])
            optimizer = kwargs["optimizer"]
            assert not optimizer.state
            assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-5)
            observed["weights_loaded"] = True
            observed["touch_oversample"] = loader.dataset.touch_oversample
            return original_epoch(model, loader, device, **kwargs)

        monkeypatch.setattr(train_events, "run_epoch", inspect_first_step)
        target_dir = tmp_path / "rebalance"
        validation_root = tmp_path / "heldout"
        validation_root.mkdir()
        (validation_root / "maidata.txt").write_text(
            "&wholebpm=120\n&first=0\n&lv_5=13\n&inote_5=(120){4}A1,1-5[4:1],3,4,E\n",
            encoding="utf-8",
        )
        (validation_root / "track.mp3").write_bytes(b"heldout")
        validation_manifest = tmp_path / "heldout_manifest.json"
        validation_manifest.write_text(json.dumps({"songs": [{
            "song_id": "heldout", "maidata_path": "heldout/maidata.txt",
            "audio_path": "heldout/track.mp3",
            "difficulties": [{"difficulty_index": 5, "level_raw": "13"}],
        }]}), encoding="utf-8")
        changed = common.copy()
        changed[changed.index("--slide-oversample") + 1] = "2"
        changed[changed.index("--touch-oversample") + 1] = "1"
        assert train_events.main(changed + ["--init-checkpoint", str(old_path),
                                           "--lr", "3e-5", "--checkpoint-dir", str(target_dir),
                                           "--validation-manifest", str(validation_manifest),
                                           "--validation-max-steps", "1"]) == 0
        saved = load_event_checkpoint(target_dir / "event_last.pt")
        assert observed == {"weights_loaded": True, "touch_oversample": 1,
                            "natural_validation": True}
        assert saved["epoch"] == 1
        assert saved["training_origin"]["source_epoch"] == 11
        assert saved["training_origin"]["optimizer_reset"] is True
        family = saved["data_contract"]["family_params"]
        assert family["touch_oversample"] == 1
        assert family["slide_oversample"] == 2
        assert family["oversample_policy"] == OVERSAMPLE_POLICY
        assert saved["optimizer_state_dict"]["state"]
    finally:
        torch.set_num_threads(torch_threads)
