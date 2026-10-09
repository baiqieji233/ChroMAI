"""Contract tests for the generation-side condition/layout fixes.

1. A mask_prob=1.0 checkpoint must not receive chart-derived condition values
   (level-only), with the replacement logged.
2. An empty lead-in must still produce the ``<ctx> <ctx_end>`` delimiter pair at
   generation time -- the layout every training window had.
"""

from pathlib import Path
from types import SimpleNamespace

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import maichart.generation.event_cli as event_cli  # noqa: E402
from maichart.generation.conditions import GenerationCondition  # noqa: E402
from maichart.generation.event_ctx import CTX_CLOSE, CTX_OPEN  # noqa: E402
from maichart.generation.event_model import EventVocabulary  # noqa: E402
from maichart.training.event_ctx_dataset import EventCtxDataset  # noqa: E402


def _vocabulary() -> EventVocabulary:
    return EventVocabulary.from_event_charts(
        [], segment_beats=24, extra_specials=EventCtxDataset.EXTRA_SPECIALS)


def test_profile_condition_is_guarded_after_profile_resolution():
    condition = GenerationCondition(level=13.5, radar={"note_density": 0.8}, style={"tap_bias": 0.9})
    contract = {"condition_params": {"mask_prob": 1.0, "default": 0.5}}
    guarded, note = event_cli.apply_contract_condition_guard(condition, contract, default_level=condition.level)
    assert note
    assert guarded.radar["note_density"] == 0.5
    assert guarded.style["tap_bias"] == 0.5


    """mask_prob=1.0: every non-default radar/style value must be dropped."""

    condition = GenerationCondition(level=13.5, radar={"note_density": 0.7, "chord": 0.2},
                                   style={"tap_bias": 0.9})
    contract = {"condition_params": {"mask_prob": 1.0, "default": 0.5}}
    neutralised, note = event_cli.apply_contract_condition_guard(
        condition, contract, default_level=condition.level)
    assert note, "a drop must be reported, not silent"
    assert neutralised.level == condition.level
    assert all(abs(value - 0.5) < 1e-6 for value in {**neutralised.radar, **neutralised.style}.values())


def test_mask_partial_checkpoint_keeps_the_profile():
    """A checkpoint trained with mask<1 learned those values; keep the profile."""

    condition = GenerationCondition(level=13.5, radar={"note_density": 0.7})
    contract = {"condition_params": {"mask_prob": 0.5, "default": 0.5}}
    kept, note = event_cli.apply_contract_condition_guard(
        condition, contract, default_level=condition.level)
    assert kept.radar["note_density"] == 0.7
    assert not note, "a mask<1 checkpoint gets the profile untouched"


def test_missing_condition_contract_does_not_assume_full_mask():
    condition = GenerationCondition(level=13.5, radar={"note_density": 0.7},
                                    style={"tap_bias": 0.9})
    kept, note = event_cli.apply_contract_condition_guard(condition, {}, default_level=condition.level)
    assert kept.radar["note_density"] == 0.7
    assert kept.style["tap_bias"] == 0.9
    assert not note




def test_neutral_condition_is_left_alone_and_flagged_not_needed():
    """Sending the plain UI default (all 0.5) must not be flagged as a drop."""

    condition = GenerationCondition(level=13.0)
    contract = {"condition_params": {"mask_prob": 1.0, "default": 0.5}}
    same, note = event_cli.apply_contract_condition_guard(
        condition, contract, default_level=condition.level)
    assert not note
    assert same is condition or same.radar == condition.radar


def test_context_prefix_empty_windows_still_get_the_delimiters():
    """Song start: no generated notes yet, yet the layout must be the
    ``<ctx> <ctx_end>`` pair training always used."""

    vocabulary = _vocabulary()
    prefix = event_cli._context_prefix(vocabulary, windows=[], start=0, context_beats=8,
                                       context_tokens=64)
    assert prefix == [vocabulary.token_to_id[CTX_OPEN], vocabulary.token_to_id[CTX_CLOSE]]
    assert event_cli._empty_prefix(vocabulary) == prefix


def test_context_prefix_failure_falls_back_to_delimiters_not_silence(monkeypatch):
    """An out-of-vocabulary event drops the *content*, not the layout."""

    import maichart.generation.event_ctx as event_ctx

    vocabulary = _vocabulary()
    window = SimpleNamespace(start_beat=0, end_beat=8,
                             sequence=SimpleNamespace(beats=[4], note_texts=["1"]))

    def explode(vocabulary_, events_, context_tokens_):
        raise ValueError("Target is outside the checkpoint vocabulary: ['CHAR:z']")

    monkeypatch.setattr(event_ctx, "prefix_token_ids", explode)
    prefix = event_cli._context_prefix(vocabulary, windows=[window], start=8, context_beats=8,
                                       context_tokens=64)
    assert prefix[:2] == [vocabulary.token_to_id[CTX_OPEN], vocabulary.token_to_id[CTX_CLOSE]]
    assert event_cli._context_prefix.dropped == 1


def test_final_audio_span_filter_drops_lead_and_tail_events():
    from fractions import Fraction
    from maichart.event_timing import EventPoint, EventTempo

    points = [EventPoint(Fraction(0), "1"), EventPoint(Fraction(4), "2"), EventPoint(Fraction(8), "3")]
    kept = event_cli._filter_points_to_audio_span(
        points, tempo_changes=[EventTempo(Fraction(0), 60.0)], first=0.0,
        span=(1.0, 7.0), tolerance=0.0,
    )
    assert [point.beat for point in kept] == [Fraction(4)]


    """A non-empty lead-in must encode identically in both paths."""

    from maichart.generation.event_ctx import prefix_token_ids

    vocabulary = _vocabulary()
    note = "1<>4[4:1]"
    window = SimpleNamespace(start_beat=0, end_beat=16,
                             sequence=SimpleNamespace(beats=[4], note_texts=[note]))
    prefix = event_cli._context_prefix(vocabulary, windows=[window], start=8, context_beats=8,
                                       context_tokens=64)
    # The same events the helper derives (block-relative), through the training
    # path: layout must be byte-identical.
    reference = prefix_token_ids(vocabulary, [(4, note)], context_tokens=64)
    assert prefix == reference
