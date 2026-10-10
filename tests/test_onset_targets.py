import json
from fractions import Fraction

import numpy as np
import pytest
import torch

from maichart.event_timing import EventChart, EventPoint, EventTempo
from maichart.generation.event_audio import (
    ACOUSTIC_DIM,
    INPUT_DIM,
    EventAudio,
    _sample_audio,
    segment_frame_times,
)
from maichart.training import event_dataset as event_dataset_module
from maichart.training import prepare_events
from maichart.training.event_ctx_dataset import EventCtxDataset
from maichart.training.event_dataset import EventDataset, collate_events
from maichart.training.onset_targets import build_onset_targets, onset_times, quality_onset_params


def _chart(first=0.1):
    return EventChart(
        points=[
            EventPoint(Fraction(0), "1"),
            EventPoint(Fraction(1), "12"),  # same-time chord is one onset
            EventPoint(Fraction(2), "3"),
            EventPoint(Fraction(3), "4"),
        ],
        tempo_changes=[EventTempo(Fraction(0), 120.0), EventTempo(Fraction(2), 60.0)],
        first=first,
        end_beat=Fraction(4),
        difficulty=4,
    )


def test_onset_times_deduplicate_chords_include_first_and_integrate_tempo_change():
    times = onset_times(_chart(first=0.1))

    assert times == pytest.approx([0.1, 0.6, 1.1, 2.1])


def test_segment_frame_times_match_audio_feature_sampling_for_negative_start():
    audio = EventAudio(
        features=np.zeros((20, ACOUSTIC_DIM), dtype=np.float32),
        times=np.arange(20, dtype=np.float32) / 50,
        duration_sec=0.4,
        tempo_bpm=120.0,
        tempo_changes=[],
        metadata={},
    )
    expected, _ = _sample_audio(audio, -0.1, 0.05)

    assert np.array_equal(segment_frame_times(-0.1, 0.05), expected)
    assert expected[0] == pytest.approx(-0.1)
    assert expected[-1] == pytest.approx(0.04)


def test_core_is_half_open_and_audio_validity_clips_negative_and_file_edges():
    chart = _chart(first=0.1)
    times = segment_frame_times(-0.1, 1.4)
    target, weight, valid = build_onset_targets(
        chart, times, core_start_sec=0.5, core_end_sec=1.1, audio_duration_sec=1.05,
    )

    assert target[np.isclose(times, 0.6, atol=1e-6)].max().item() == pytest.approx(1.0)
    assert target[np.isclose(times, 0.1, atol=1e-6)].eq(0).all()  # before core
    assert target[np.isclose(times, 1.1, atol=1e-6)].eq(0).all()  # exclusive core end
    assert valid[times < 0].sum() == 0
    assert valid[times >= 1.05].sum() == 0
    assert torch.all(weight == 1.0)  # quality factor is independent of validity/core
    assert valid[np.isclose(times, 0.5, atol=1e-6)].item() == 1.0
    assert valid[np.isclose(times, 1.04, atol=1e-6)].item() == 1.0


def test_negative_first_and_events_outside_audio_do_not_create_targets():
    chart = _chart(first=-0.1)
    times = segment_frame_times(-0.2, 0.2)
    target, _, valid = build_onset_targets(
        chart, times, core_start_sec=-0.2, core_end_sec=0.2, audio_duration_sec=0.2,
    )

    assert onset_times(chart)[0] == pytest.approx(-0.1)
    assert target.max().item() == 0.0
    assert valid[times < 0].sum().item() == 0


def test_zero_quality_regions_block_events_and_gaussian_spill_across_regions():
    chart = _chart(first=0.1)
    times = segment_frame_times(0.5, 1.4)
    target, weight, valid = build_onset_targets(
        chart,
        times,
        core_start_sec=0.5,
        core_end_sec=1.0,
        audio_duration_sec=2.0,
        quality_base_weight=0.5,
        quality_regions=[
            {"start_sec": 0.55, "end_sec": 0.65, "weight": 0.0},
            {"start_sec": 0.8, "end_sec": 0.9, "weight": 0.25},
        ],
    )

    zero_region = (times >= 0.55) & (times < 0.65)
    weighted_region = (times >= 0.8) & (times < 0.9)
    assert weight[zero_region].eq(0).all()
    assert target[zero_region].eq(0).all()
    assert weight[weighted_region].eq(0.125).all()
    assert target[weighted_region].eq(0).all()
    assert valid.sum().item() == int(((times >= 0.5) & (times < 1.0)).sum())


def test_negative_quality_without_correction_has_zero_weight_and_no_positive_targets():
    base, regions = quality_onset_params({"label": "negative", "regions": []})
    times = segment_frame_times(0.0, 0.5)
    target, weight, valid = build_onset_targets(
        _chart(), times, core_start_sec=0.0, core_end_sec=0.5, audio_duration_sec=1.0,
        quality_base_weight=base, quality_regions=regions,
    )

    assert base == 0.0
    assert valid.max().item() == 1.0
    assert weight.eq(0).all()
    assert target.eq(0).all()


def _collate_item(frame_count, *, onset=False):
    item = {
        "audio": torch.ones((frame_count, INPUT_DIM)),
        "global_audio": torch.zeros((2, INPUT_DIM)),
        "decoder_input": torch.tensor([1, 2]),
        "target": torch.tensor([2, 3]),
        "condition": torch.zeros(16),
        "condition_null": torch.tensor(False),
        "plan_target": torch.zeros(10),
    }
    if onset:
        item.update({
            "onset_target": torch.ones(frame_count),
            "onset_weight": torch.full((frame_count,), 0.5),
            "onset_valid": torch.ones(frame_count),
        })
    return item


def test_collate_onset_tracks_pad_zero_and_none_keeps_old_batch_keys():
    legacy = collate_events([_collate_item(2), _collate_item(4)])
    assert not {"onset_target", "onset_weight", "onset_valid"} & legacy.keys()

    batch = collate_events([_collate_item(2, onset=True), _collate_item(4, onset=True)])
    assert batch["onset_target"].shape == (2, 4)
    assert batch["onset_weight"].shape == (2, 4)
    assert batch["onset_valid"].shape == (2, 4)
    assert batch["onset_target"][0, 2:].eq(0).all()
    assert batch["onset_weight"][0, 2:].eq(0).all()
    assert batch["onset_valid"][0, 2:].eq(0).all()


def _write_chart(root, name):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "maidata.txt").write_text(
        "&title=Onset Test\n&artist=Tester\n&wholebpm=120\n&first=0.1\n"
        "&lv_4=13\n&inote_4=(120){4}1,2,3,4,5,6,7,8,E\n",
        encoding="utf-8",
    )
    (folder / "track.mp3").write_bytes(b"fake audio")


def _fake_audio(_path, **_kwargs):
    return EventAudio(
        features=np.zeros((500, ACOUSTIC_DIM), dtype=np.float32),
        times=np.arange(500, dtype=np.float32) / 50,
        duration_sec=10.0,
        tempo_bpm=120.0,
        tempo_changes=[],
        metadata={},
    )


@pytest.fixture
def event_manifest(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    _write_chart(raw, "song")
    manifest = tmp_path / "manifest.json"
    prepare_events.main([str(raw), "--cache-dir", str(tmp_path / "cache"), "-o", str(manifest)])
    monkeypatch.setattr(event_dataset_module, "extract_event_audio", _fake_audio)
    return manifest, tmp_path / "cache"


def test_event_dataset_profile_is_opt_in_and_signature_changes_only_when_enabled(event_manifest):
    manifest, cache = event_manifest
    baseline = EventDataset(manifest, cache_dir=cache, segment_beats=4, max_tokens=256,
                            eager_audio=False)
    explicit_none = EventDataset(manifest, cache_dir=cache, segment_beats=4, max_tokens=256,
                                 eager_audio=False, onset_profile="none")
    enabled = EventDataset(manifest, cache_dir=cache, segment_beats=4, max_tokens=256,
                           eager_audio=False, onset_profile="chart_onsets_v1")

    assert baseline.signature == explicit_none.signature
    assert baseline.signature != enabled.signature
    assert baseline.audit()["onset_profile"] == "none"
    assert "onset_target" not in baseline[0]
    item = enabled[0]
    assert item["onset_target"].shape == (item["audio"].size(0),)
    assert item["onset_weight"].shape == item["onset_target"].shape
    assert item["onset_valid"].shape == item["onset_target"].shape
    assert enabled.audit()["onset_profile"] == "chart_onsets_v1"


def test_event_ctx_uses_actual_margin_grid_and_profile_in_family_contract(event_manifest):
    manifest, cache = event_manifest
    dataset = EventCtxDataset(
        manifest,
        cache_dir=cache,
        segment_beats=4,
        max_tokens=256,
        context_beats=2,
        audio_margin_beats=1,
        context_tokens=64,
        context_dropout=0,
        eager_audio=False,
        onset_profile="chart_onsets_v1",
    )

    item = dataset[0]
    assert item["onset_target"].shape == (item["audio"].size(0),)
    assert item["onset_valid"].sum().item() < item["audio"].size(0)  # margins are unknown
    assert dataset.family_params()["onset_profile"] == "chart_onsets_v1"
    batch = collate_events([item])
    assert batch["onset_target"].shape == batch["audio"].shape[:2]
