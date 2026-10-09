from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import torch

from maichart.event_timing import parse_event_chart
from maichart.generation.event_model import EventSequence, EventVocabulary
from maichart.training.event_preferences import (
    build_preference_pairs,
    score_pair_logits,
)
from maichart.training.quality_annotations import chart_sha256


def _chart(path: Path, notes: str):
    path.write_text(
        "&title=Preference\n&wholebpm=120\n&first=0\n&lv_4=13\n"
        f"&inote_4=(120){{4}}{notes}\n",
        encoding="utf-8",
    )
    return parse_event_chart(path, 4, strict_notes=True)


def _fake_dataset(tmp_path: Path):
    original_path = tmp_path / "original.txt"
    corrected_path = tmp_path / "corrected.txt"
    original = _chart(original_path, "1,2,3,4,")
    _chart(corrected_path, "1,3,2,4,")
    vocab = EventVocabulary.from_event_charts([original], segment_beats=16)
    tokens = vocab.encode(EventSequence([p.beat for p in original.points], [p.note_text for p in original.points]))
    ref = SimpleNamespace(song_id="song", source_path=original_path, chart=original)
    item = {
        "audio": torch.zeros(3, 2), "global_audio": torch.zeros(2, 2),
        "condition": torch.zeros(16), "condition_null": torch.tensor(False),
        "plan_target": torch.zeros(9), "decoder_input": torch.tensor(tokens[:-1]),
        "target": torch.tensor(tokens[1:]), "meta": {"song_id": "song"},
    }

    class Dataset:
        samples = [ref]
        segments = [(0, Fraction(0), Fraction(4), tokens)]
        vocabulary = vocab

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return item

    return Dataset(), original_path, corrected_path


def test_build_pairs_requires_corrected_path_and_keeps_shared_inputs(tmp_path):
    dataset, original_path, corrected_path = _fake_dataset(tmp_path)
    annotations = [{
        "label": "negative", "source_path": str(original_path), "difficulty_index": 4,
        "corrected_path": str(corrected_path), "regions": [{"start_sec": 0, "end_sec": 1, "label": "bad"}],
    }]
    pairs, audit = build_preference_pairs(dataset, annotations)
    assert len(pairs) == 1
    assert audit["pairs"] == 1
    assert torch.equal(pairs[0].chosen["audio"], pairs[0].rejected["audio"])
    assert torch.equal(pairs[0].chosen["condition"], pairs[0].rejected["condition"])
    assert not torch.equal(pairs[0].chosen["target"], pairs[0].rejected["target"])

    no_correction, no_correction_audit = build_preference_pairs(
        dataset, [{"label": "negative", "source_path": str(original_path), "difficulty_index": 4}])
    assert no_correction == []
    assert no_correction_audit["negative_without_correction"] == 1


def test_score_pair_logits_masks_context_and_padding_stably():
    logits = torch.tensor([[[1000.0, 0.0], [0.0, 1000.0], [0.0, 1000.0]]])
    batch = {
        "target": torch.tensor([[0, 1, 1]]),
        "loss_mask": torch.tensor([[False, True, False]]),
        "decoder_padding_mask": torch.tensor([[False, False, True]]),
    }
    score, stats = score_pair_logits(logits, batch)
    assert torch.isfinite(score).all()
    assert score.item() == 0.0
    assert stats["tokens"] == 1


def test_content_resolver_miss_never_falls_back_to_stale_path_record(tmp_path):
    dataset, original_path, corrected_path = _fake_dataset(tmp_path)
    dataset.quality_record = lambda ref: None
    annotations = [{"label": "negative", "source_path": str(original_path),
                    "song_id": "song", "difficulty_index": 4,
                    "corrected_path": str(corrected_path)}]
    pairs, audit = build_preference_pairs(dataset, annotations)
    assert pairs == []
    assert audit["candidate_windows"] == 0


def test_corrected_window_over_max_tokens_is_skipped(tmp_path):
    dataset, original_path, corrected_path = _fake_dataset(tmp_path)
    dataset.max_tokens = 2
    annotations = [{"label": "negative", "source_path": str(original_path),
                    "difficulty_index": 4, "corrected_path": str(corrected_path)}]
    pairs, audit = build_preference_pairs(dataset, annotations)
    assert pairs == []
    assert any("max_tokens" in item["reason"] for item in audit["skipped"])


def test_changed_corrected_chart_content_is_rejected_by_saved_hash(tmp_path):
    dataset, original_path, corrected_path = _fake_dataset(tmp_path)
    expected = chart_sha256(corrected_path, 4)
    corrected_path.write_text(corrected_path.read_text(encoding="utf-8").replace("1,3,2,4,", "1,4,2,3,"), encoding="utf-8")
    annotations = [{"label": "negative", "source_path": str(original_path),
                    "difficulty_index": 4, "corrected_path": str(corrected_path),
                    "corrected_sha256": expected}]
    pairs, audit = build_preference_pairs(dataset, annotations)
    assert pairs == []
    assert any("hash mismatch" in item["reason"] for item in audit["skipped"])
