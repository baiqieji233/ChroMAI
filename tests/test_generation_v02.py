from fractions import Fraction

import torch
import pytest

from maichart.generation.conditions import GenerationCondition, condition_from_targets
from maichart.generation.model import ConditionedChartTransformer, ConditionedChartTransformerConfig
from maichart.generation.simai_compile import PredictedNote, TempoChange, compile_predictions


def test_condition_vector_accepts_target_wrappers():
    condition = GenerationCondition.from_dict(
        {
            "level": 13.5,
            "radar": {"peak": {"target": 0.8}},
            "style": {"slide_bias": 0.2},
        }
    )
    assert condition.to_vector().shape == (16,)
    assert condition.radar["peak"] == 0.8
    assert condition.style["slide_bias"] == 0.2


def test_condition_can_be_derived_from_frame_targets():
    targets = {
        "note_presence": torch.tensor([[1.0], [1.0], [0.0]]),
        "buttons": torch.tensor([[1.0] + [0.0] * 7, [1.0, 1.0] + [0.0] * 6, [0.0] * 8]),
        "note_type": torch.tensor([1, 4, 0]),
        "density": torch.tensor([[1.0], [2.0], [0.0]]),
    }
    condition = condition_from_targets(targets, ("none", "tap", "break", "hold", "slide", "touch", "touch_hold"), level=13)
    assert 0.0 <= condition.radar["peak"] <= 1.0
    assert condition.style["slide_bias"] > 0.0


def test_conditioned_model_shapes():
    model = ConditionedChartTransformer(ConditionedChartTransformerConfig(d_model=32, nhead=4, num_layers=1, dim_feedforward=64, max_len=8))
    output = model(torch.zeros(2, 8, 7), torch.zeros(2, 16))
    assert output["button_logits"].shape == (2, 8, 8)
    assert output["type_logits"].shape[-1] == 7


def test_simai_compiler_preserves_grid_and_tempo_change():
    text, chart = compile_predictions(
        {0: [PredictedNote("tap", 1)], 4: [PredictedNote("slide", 2, end_position=6)]},
        frame_count=8,
        bpm=120,
        division=16,
        tempo_changes=[TempoChange(Fraction(0), 120), TempoChange(Fraction(1), 150)],
    )
    assert text.startswith("(120){16}1,")
    assert "(150)" in text
    assert chart.notes[1].tick == 4 * 480


def test_tempo_map_integrates_chart_seconds():
    _, chart = compile_predictions(
        {8: [PredictedNote("tap", 1)]},
        frame_count=9,
        bpm=120,
        division=16,
        tempo_changes=[TempoChange(Fraction(0), 120), TempoChange(Fraction(1), 150)],
    )
    assert chart.notes[0].time_sec == 0.9


def test_off_frame_tempo_change_roundtrips_every_clock():
    from maichart.event_timing import parse_event_chart_text, seconds_at_beat
    text, chart = compile_predictions(
        {1:[PredictedNote('tap',1)], 2:[PredictedNote('tap',2)]},
        frame_count=3, bpm=120, division=16, first=0.25,
        tempo_changes=[TempoChange(Fraction(0),120),TempoChange(Fraction(1,8),240)])
    parsed=parse_event_chart_text(f'&first=0.25\n&inote_5={text}\n',5)
    assert '(240)' in text and '{32}' in text
    assert [(tempo.beat,tempo.bpm) for tempo in parsed.tempo_changes] == [
        (Fraction(str(item.beat)),item.bpm) for item in chart.timing.bpms]
    assert [point.beat for point in parsed.points] == [Fraction(str(note.beat)) for note in chart.notes]
    for point,note in zip(parsed.points,chart.notes):
        assert seconds_at_beat(point.beat,parsed.tempo_changes,parsed.first) == pytest.approx(note.time_sec)
