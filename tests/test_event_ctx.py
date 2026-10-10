"""The cross-window family must actually carry context, and must not disturb event_v1.

The value of this family rests on four invariants, each of which is checked here:

1. the vocabulary gains exactly the two context delimiters, and ``event_v1``'s
   alphabet is byte-for-byte what it was (otherwise every existing checkpoint
   would be invalidated);
2. a window's lead-in really contains the notes that precede it, and the first
   window of a song has none;
3. the loss ignores the lead-in -- the model is given those tokens, so training
   on them would teach it to copy its own input;
4. the audio slice is wider than the window, which is what lets a window see the
   music that follows it.
"""

import json

import numpy as np
import pytest
import torch

from maichart.generation.event_audio import ACOUSTIC_DIM, INPUT_DIM, EventAudio
from maichart.generation.event_ctx import (
    CTX_CLOSE,
    CTX_OPEN,
    TARGET_SCHEMA,
    lead_in_events,
    loss_mask_after,
    prefix_token_ids,
    truncate_lead_in,
)
from maichart.generation.event_model import EventVocabulary
from maichart.training import event_dataset as event_dataset_module
from maichart.training import prepare_events
from maichart.training.event_ctx_dataset import EventCtxDataset
from maichart.training.event_dataset import EventDataset, collate_events

FEATURE_FRAMES = 500  # 10 s at 50 frames/s


def _write_chart(root, name: str, notes: str = "(134){4}1,2,3,4," * 8) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "maidata.txt").write_text(
        "&title=Test Song\n&artist=Tester\n&wholebpm=134\n&first=0\n"
        f"&lv_4=13\n&inote_4={notes}\n",
        encoding="utf-8",
    )
    (directory / "track.mp3").write_bytes(b"\x00" * 32)


def _fake_audio(_path, **_kwargs) -> EventAudio:
    return EventAudio(
        features=np.zeros((FEATURE_FRAMES, ACOUSTIC_DIM), dtype=np.float32),
        times=np.arange(FEATURE_FRAMES, dtype=np.float32) * 0.02,
        duration_sec=10.0,
        tempo_bpm=134.0,
        tempo_changes=[],
        metadata={},
    )


@pytest.fixture
def manifest(tmp_path):
    raw = tmp_path / "raw"
    for index in range(2):
        _write_chart(raw, f"song-{index}")
    # One song with slides -- the oversampling fixture needs the rarest type.
    _write_chart(raw, "song-slides", notes="(134){4}1,2h[4:1],3-5[8:1],4,")
    output = tmp_path / "manifest.json"
    assert prepare_events.main([str(raw), "--cache-dir", str(tmp_path / "cache"), "-o", str(output)]) == 0
    return output, tmp_path / "cache"


@pytest.fixture
def fake_audio(monkeypatch):
    monkeypatch.setattr(event_dataset_module, "extract_event_audio",
                        lambda path, **kwargs: _fake_audio(path))


def _ctx_dataset(manifest, **kwargs) -> EventCtxDataset:
    options = {"segment_beats": 16, "max_tokens": 512, "eager_audio": False}
    options.update(kwargs)
    return EventCtxDataset(manifest[0], cache_dir=manifest[1], **options)


def test_dataset_signature_tracks_difficulty_filter(manifest, fake_audio):
    default = EventDataset(manifest[0], cache_dir=manifest[1], segment_beats=16, max_tokens=512)
    configured = EventDataset(
        manifest[0], cache_dir=manifest[1], segment_beats=16, max_tokens=512,
        min_difficulty_index=2, max_difficulty_index=6, min_level=1, max_level=15,
    )
    assert default.signature != configured.signature
    assert configured.audit()["difficulty_filter"] == {
        "min_index": 2, "max_index": 6, "min_level": 1.0, "max_level": 15.0,
    }


def test_window_condition_uses_matching_audio_intensity_factor(manifest, fake_audio, monkeypatch):
    from maichart.generation import intensity

    monkeypatch.setattr(intensity, "window_factors", lambda audio, windows, **kwargs: [1.5] * len(windows))
    dataset = EventDataset(manifest[0], cache_dir=manifest[1], segment_beats=16,
                           max_tokens=512, condition_mask_prob=1.0, condition_default=0.4)
    item = dataset[0]
    assert item["condition"][0].item() == pytest.approx(dataset.conditions[0][0].item())
    assert item["condition"][1].item() == pytest.approx(0.6)
    assert item["condition"][2].item() == pytest.approx(0.6)
    assert item["condition"][3:].eq(0.4).all()


    plain = EventDataset(manifest[0], cache_dir=manifest[1], segment_beats=16, max_tokens=512)
    contextual = _ctx_dataset(manifest)
    plain_tokens = {token for token in plain.vocabulary.tokens}
    assert CTX_OPEN not in plain_tokens and CTX_CLOSE not in plain_tokens
    assert CTX_OPEN in {token for token in contextual.vocabulary.tokens}
    assert CTX_CLOSE in {token for token in contextual.vocabulary.tokens}
    # Padding stays index 0 in both families, so one collate function serves both.
    assert plain.vocabulary.pad_id == 0 and contextual.vocabulary.pad_id == 0
    # The cross-window alphabet is strictly larger: two control tokens plus the
    # position tokens for the extended span (block + window share one origin, so
    # positions run from 0 to context_beats + segment_beats).
    assert len(contextual.vocabulary) > len(plain.vocabulary) + 2
    assert contextual.vocabulary_segment_beats() == 24
    assert plain.vocabulary_segment_beats() == 16
    assert contextual.audit()["target_schema"] == TARGET_SCHEMA


def test_first_window_has_no_lead_in_and_later_windows_do(manifest, fake_audio):
    # context_dropout=0 here: the fixture's drop decisions share one RNG across
    # the whole dataset, so a 15% draw could otherwise make this presence test
    # sensitive to how many songs precede it in the manifest.
    dataset = _ctx_dataset(manifest, context_beats=8, context_dropout=0.0)
    per_song: dict[int, list] = {}
    for entry in dataset.segments:
        per_song.setdefault(entry[0], []).append(entry)
    # Only songs that actually have several windows carry the invariant; the
    # one-window song in the fixture has no later window by construction.
    multi = [segments for segments in per_song.values() if len(segments) > 1]
    assert multi, "the fixture must contain multi-window songs"
    for segments in multi:
        segments.sort(key=lambda item: item[1])
        first_prefix = segments[0][4]
        # Only the two delimiters: nothing precedes the first window.
        assert len(first_prefix) == 2
        assert first_prefix == [dataset.vocabulary.token_to_id[CTX_OPEN],
                                dataset.vocabulary.token_to_id[CTX_CLOSE]]
        later = [segment for segment in segments[1:] if len(segment[4]) > 2]
        assert later, "a window after the first must carry the preceding notes"


def test_loss_mask_excludes_the_lead_in(manifest, fake_audio):
    dataset = _ctx_dataset(manifest, context_beats=8)
    # A window in the middle of the song, so the lead-in is non-empty.
    index = max(range(len(dataset)), key=lambda i: len(dataset.segments[i][4]))
    item = dataset[index]
    prefix_len = dataset.segments[index][5]
    window_tokens = len(dataset.segments[index][3])
    mask = item["loss_mask"]
    assert prefix_len > 2, "pick a window with real context"
    assert mask[:prefix_len].sum().item() == 0, "the lead-in must not be trained on"
    # Everything after the lead-in is trained: the window's events plus <eos>.
    assert mask.sum().item() == window_tokens - 1
    assert item["target"].numel() == mask.numel()


def test_context_tokens_reach_the_decoder_input(manifest, fake_audio):
    dataset = _ctx_dataset(manifest, context_beats=8, context_tokens=64)
    index = max(range(len(dataset)), key=lambda i: len(dataset.segments[i][4]))
    item = dataset[index]
    open_id = dataset.vocabulary.token_to_id[CTX_OPEN]
    close_id = dataset.vocabulary.token_to_id[CTX_CLOSE]
    ids = item["decoder_input"].tolist()
    assert ids[0] == dataset.vocabulary.bos_id
    assert open_id in ids and close_id in ids
    assert ids.index(open_id) < ids.index(close_id)
    assert len(ids) > dataset.segments[index][5]


def test_audio_slice_is_wider_than_the_plain_window(manifest, fake_audio):
    plain = EventDataset(manifest[0], cache_dir=manifest[1], segment_beats=16, max_tokens=512)
    contextual = _ctx_dataset(manifest, audio_margin_beats=4)
    plain_frames = plain[0]["audio"].size(0)
    ctx_frames = contextual[0]["audio"].size(0)
    assert ctx_frames > plain_frames, "margins must add audio on both sides"
    assert all(item["audio"].size(1) == INPUT_DIM for item in (plain[0], contextual[0]))


def test_overlap_shortens_the_stride(manifest, fake_audio):
    without = _ctx_dataset(manifest, overlap_beats=0)
    with_overlap = _ctx_dataset(manifest, overlap_beats=8)
    assert without.stride_beats == 16 and with_overlap.stride_beats == 8
    assert len(with_overlap) > len(without)
    # The first window still starts at beat 0 in both.
    assert min(entry[1] for entry in with_overlap.segments) == 0


def test_window_plus_context_must_fit_the_token_budget(manifest, fake_audio):
    # A window is ~50 tokens.  A budget of 60 leaves only ~8 tokens for the
    # lead-in: that must be spent on the *newest* context tokens instead of
    # dropping the window.  Only a window that is itself too long is reported.
    dataset = _ctx_dataset(manifest, context_beats=8, context_tokens=256, max_tokens=60)
    prefix_lengths = {entry[5] for entry in dataset.segments}
    assert any(0 < length <= 10 for length in prefix_lengths), \
        "the lead-in must be cropped to whatever the window leaves, not dropped"
    assert len(dataset) > 0, "crop lets dense windows survive"


def test_window_larger_than_the_budget_is_reported(manifest, fake_audio):
    # A 16-beat window is ~50 tokens (16 events x 3 + bos/eos), so a budget of
    # 51 drops every full window while the short end-of-song sentinels survive;
    # the dropped windows must be reported, not silently gone.
    dataset = _ctx_dataset(manifest, context_beats=0, context_tokens=256, max_tokens=51)
    skips = [item for item in dataset.skipped if item.get("reason") == "segment_token_limit"]
    assert skips, "oversized windows must be reported, not silently truncated"
    assert "max_tokens" in skips[0]["message"]
    assert len(dataset) > 0, "shorter windows still fit and survive"


def test_dense_window_keeps_its_lead_in_at_a_reduced_budget(manifest, fake_audio):
    """A window is never skipped just because its lead-in runs out of room.

    Only the lead-in should be shrunk to fit: dropping the whole window removed
    exactly the densest passages -- the ones that most need their context.
    """

    dataset = _ctx_dataset(manifest, context_beats=8, context_tokens=34, max_tokens=60)
    prefix_lengths = [entry[5] for entry in dataset.segments]
    assert prefix_lengths, "windows must survive with a reduced lead-in budget"
    assert max(prefix_lengths) <= 30, "lead-in must be cropped before the window is dropped"


def test_slide_windows_are_oversampled(manifest, fake_audio):
    """Oversampling multiplies the windows that carry the rarest note type."""

    without = _ctx_dataset(manifest, slide_oversample=1)
    with_boost = _ctx_dataset(manifest, slide_oversample=4)
    assert len(with_boost) >= len(without)
    duplicates = len(with_boost) - len(without)
    assert duplicates > 0, "at least one window in the fixture carries a slide"
    # Every duplicated segment is a duplicate of an existing window.
    keys_without = [(entry[0], str(entry[1])) for entry in without.segments]
    keys_boost = [(entry[0], str(entry[1])) for entry in with_boost.segments]
    assert set(keys_without) <= set(keys_boost)
    from collections import Counter
    counts = Counter(keys_boost)
    repeated = {key for key, count in counts.items() if count > 1}
    assert repeated, "duplicated segments must be exact repeats, not new windows"


def test_collate_keeps_event_v1_items_fully_trainable(manifest, fake_audio):
    plain = EventDataset(manifest[0], cache_dir=manifest[1], segment_beats=16, max_tokens=512)
    contextual = _ctx_dataset(manifest)
    plain_batch = collate_events([plain[0], plain[1]], pad_id=plain.vocabulary.pad_id)
    assert plain_batch["loss_mask"].all(), "event_v1 targets are all trainable"
    ctx_batch = collate_events([contextual[0], contextual[1]], pad_id=contextual.vocabulary.pad_id)
    for row, index in enumerate((0, 1)):
        item = contextual[index]
        assert ctx_batch["loss_mask"][row, :item["target"].numel()].equal(item["loss_mask"])
        assert not ctx_batch["loss_mask"][row, item["target"].numel():].any()
    # The audio mask still describes real frames: the lead-in is not padding.
    assert (~ctx_batch["audio_padding_mask"]).any()


def test_lead_in_positions_are_relative_to_the_block_and_strictly_increasing(manifest, fake_audio):
    dataset = _ctx_dataset(manifest, context_beats=8)
    ref = dataset.samples[0]
    events = lead_in_events(ref.chart.points, start=16, context_beats=8)
    assert events, "beats 8..16 of this chart contain notes"
    beats = [beat for beat, _ in events]
    assert beats == sorted(beats) and len(set(beats)) == len(beats)
    assert beats[0] == 0  # first lead-in event anchors the block at position 0
    assert all(0 <= beat < 8 for beat in beats)


def test_truncate_keeps_the_newest_events_that_fit():
    events = [(float(index), "1") for index in range(20)]
    from fractions import Fraction
    typed = [(Fraction(int(beat)), text) for beat, text in events]
    kept = truncate_lead_in(typed, context_tokens=10)
    assert kept, "some events must fit"
    assert kept[-1] == typed[-1], "the beats nearest the window are kept"
    assert len(kept) < len(typed)


def test_loss_mask_helper_boundaries():
    # Two delimiters and no lead-in: the first two targets are the given ones.
    assert loss_mask_after(2, 5) == [False, False, True, True, True]
    assert loss_mask_after(0, 3) == [True, True, True]
    assert loss_mask_after(7, 3) == [False, False, False]


def test_prefix_token_ids_handles_an_empty_lead_in(manifest, fake_audio):
    dataset = _ctx_dataset(manifest)
    vocabulary = dataset.vocabulary
    ids = prefix_token_ids(vocabulary, [], context_tokens=64)
    assert ids == [vocabulary.token_to_id[CTX_OPEN], vocabulary.token_to_id[CTX_CLOSE]]


def test_family_params_are_recorded_for_resume_checks(manifest, fake_audio):
    plain = EventDataset(manifest[0], cache_dir=manifest[1], segment_beats=16, max_tokens=512)
    contextual = _ctx_dataset(manifest, context_beats=4, audio_margin_beats=2,
                              context_tokens=96, overlap_beats=4)
    assert plain.audit()["target_schema"] == "event_v1"
    audit = contextual.audit()
    assert audit["context_beats"] == 4 and audit["audio_margin_beats"] == 2
    assert audit["context_tokens"] == 96 and audit["overlap_beats"] == 4
    assert audit["stride_beats"] == 12
    assert contextual.family_params() == {
        "context_beats": 4, "audio_margin_beats": 2, "context_tokens": 96,
        "overlap_beats": 4, "context_dropout": 0.15, "slide_oversample": 1,
        "touch_oversample": 1, "fine_rhythm_oversample": 1,
        "fine_rhythm_ratio_threshold": 0.25,
        "oversample_policy": "independent_note_types_touch_priority_fine_rhythm_v2",
    }
    boosted = _ctx_dataset(manifest, context_beats=4, audio_margin_beats=2,
                           context_tokens=96, overlap_beats=4, slide_oversample=3)
    assert boosted.family_params()["slide_oversample"] == 3
    fine = _ctx_dataset(manifest, context_beats=4, audio_margin_beats=2,
                        context_tokens=96, overlap_beats=4, fine_rhythm_oversample=4,
                        fine_rhythm_ratio_threshold=0.5)
    assert fine.family_params()["fine_rhythm_oversample"] == 4
    assert fine.family_params()["fine_rhythm_ratio_threshold"] == 0.5
    assert not hasattr(plain, "family_params")


def test_window_positions_continue_the_lead_in_instead_of_restarting(manifest, fake_audio):
    """C1: one origin for the whole composite, so the gap stays visible.

    Two origins in one sequence is what Transformer-XL had to fix with relative
    positional encodings; the model cannot otherwise see how far the lead-in's
    last note is from the window's first.
    """

    dataset = _ctx_dataset(manifest, context_beats=8, context_dropout=0.0)
    # Window starting at beat 16 has a block starting at beat 8: offset 8.
    target = next(entry for entry in dataset.segments if entry[1] == 16)
    _sample, start, _end, window_tokens, prefix, _prefix_len, offset = target
    assert start == 16 and offset == 8
    assert prefix, "this window must have a lead-in for the test to mean anything"
    # The window's first position token is measured from the block start, i.e.
    # it continues the lead-in's numbering rather than restarting at 0.
    first_position = next(token for token in window_tokens if token != dataset.vocabulary.bos_id
                          and dataset.vocabulary.id_to_token[token].startswith("AT:"))
    label = dataset.vocabulary.id_to_token[first_position]
    value = float(label[3:])
    assert value >= float(offset), f"window restarted at {label}; add the offset first"
    lead_in_positions = [
        float(dataset.vocabulary.id_to_token[token][3:])
        for token in prefix if dataset.vocabulary.id_to_token[token].startswith("AT:")
    ]
    assert lead_in_positions and max(lead_in_positions) < value


def test_context_dropout_empties_the_lead_in(manifest, fake_audio):
    """C2: training must survive having no context, or it learns to copy it."""

    dropped = _ctx_dataset(manifest, context_beats=8, context_dropout=1.0)
    assert all(len(entry[4]) == 2 for entry in dropped.segments), \
        "with dropout 1.0 every prefix is just the two delimiters"
    # The geometry is untouched: the offset still tells the model where it sits.
    assert any(entry[6] == 8 for entry in dropped.segments)
    assert dropped.audit()["context_dropout"] == 1.0

    kept = _ctx_dataset(manifest, context_beats=8, context_dropout=0.0)
    assert any(len(entry[4]) > 2 for entry in kept.segments)
    with pytest.raises(ValueError, match="context_dropout"):
        _ctx_dataset(manifest, context_dropout=1.5)


def test_dataset_rejects_negative_context_settings(manifest, fake_audio):
    with pytest.raises(ValueError, match="negative"):
        _ctx_dataset(manifest, context_beats=-1)
    with pytest.raises(ValueError, match="context_tokens"):
        _ctx_dataset(manifest, context_tokens=0)


def test_target_derived_controls_are_masked_by_default(manifest, fake_audio):
    """The condition must not hand the model the answer's own statistics.

    Fifteen of the sixteen controls are computed from the chart being predicted.
    With the default mask probability only ``level`` survives, so the chart has
    to come from the audio -- which is also what generation can supply.
    """

    averaged = _ctx_dataset(manifest, condition_mask_prob=0.0)
    masked = _ctx_dataset(manifest, condition_mask_prob=1.0)
    raw = averaged[0]["condition"]
    clean = masked[0]["condition"]
    assert raw.shape == clean.shape
    # Level is the first control and is kept; the rest are the 0.5 default.
    assert clean[0].item() == pytest.approx(raw[0].item())
    assert all(value == pytest.approx(0.5) for value in clean[1:].tolist())
    # The unmasked dataset must still carry the real corpus statistics.
    assert any(value != pytest.approx(0.5) for value in raw[1:].tolist())


def test_partial_masking_varies_between_items(manifest, fake_audio):
    dataset = _ctx_dataset(manifest, condition_mask_prob=0.5)
    seen_masked = seen_raw = False
    for index in range(len(dataset)):
        values = dataset[index]["condition"][1:].tolist()
        if all(value == pytest.approx(0.5) for value in values):
            seen_masked = True
        else:
            seen_raw = True
    assert seen_raw, "some items must keep the real controls"
    assert seen_masked or len(dataset) < 8, "and some must be masked"
