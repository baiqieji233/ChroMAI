"""Small integration checks for the new phrase stage, without training jobs."""
from fractions import Fraction
from types import SimpleNamespace

import torch

from maichart.composer.stream_phrase import PHRASE_FEATURE_DIM
from maichart.composer.stream_phrase_model import phrase_auxiliary_loss
from maichart.composer.stream_train import upgrade_learning_stage
from maichart.composer.guard import EventGuard
from maichart.composer.stream_quality import _comfort
from maichart.event_timing import EventPoint, EventTempo
from maichart.generation.event_quality import parse_event_notes, MIACODE_DURATION_SEMANTICS
from test_composer_stream_model import fixture


def test_new_stage_preserves_base_outputs_and_joint_backward():
    base, batch = fixture()
    model = upgrade_learning_stage(base, dict(phrase_context_policy='causal_phrase_v1',
        duration_audio_policy='interval_v1')).eval()
    size = (1, 4)
    batch.update(phrase_start_beat=torch.zeros(size), phrase_end_beat=torch.full(size, 6.),
        phrase_progress=torch.zeros(*size, PHRASE_FEATURE_DIM),
        phrase_memory_features=torch.zeros(*size, 4, PHRASE_FEATURE_DIM),
        phrase_memory_start_beat=torch.zeros(*size, 4), phrase_memory_end_beat=torch.zeros(*size, 4),
        phrase_memory_mask=torch.zeros(*size, 4, dtype=torch.bool),
        target_phrase_features=torch.ones(*size, PHRASE_FEATURE_DIM),
        target_phrase_mask=torch.ones(size, dtype=torch.bool), target_phrase_weight=torch.full(size, .25))
    original, actual = base(batch), model(batch)
    for key in ('hidden', 'kind_logits', 'delta_logits', 'shape_logits', 'phase_logits'):
        torch.testing.assert_close(actual[key], original[key], rtol=0., atol=0.)
    loss, metrics = model.loss(actual, batch)
    loss.backward()
    assert torch.isfinite(loss) and metrics['phrase_loss'] > 0
    assert model.phrase_context.intent_output.weight.grad.abs().sum() > 0
    assert model.phrase_context.predict.weight.grad.abs().sum() > 0
    # A real sustained detail branch accepts target-time interval context.
    cache = model.encode_audio(batch['song_audio'], batch['song_mask'], batch['song_seconds'], batch['song_beats'])
    event = model.codec.encode('1h[4:1]/5h[8:1]')
    context = model.duration_context_at({}, batch, batch['target_second'], cache)
    from maichart.composer.stream_duration import select_duration_context
    context = select_duration_context(context, torch.tensor([0]))
    shape = event['shape']
    logits = torch.zeros(1, len(model.codec.detail_catalog[shape][0]), requires_grad=True)
    scored = model.duration_detail_logits(logits, shape, 0, context)
    torch.testing.assert_close(scored, logits, rtol=0., atol=0.)
    scored.sum().backward()
    assert model.duration_audio.score.weight.grad.abs().sum() > 0


def test_phrase_weights_keep_their_budget_and_guard_accepts_saved_policy():
    prediction = torch.zeros(1, 2, PHRASE_FEATURE_DIM, requires_grad=True)
    batch = dict(target_phrase_features=torch.ones_like(prediction),
        target_phrase_mask=torch.ones(1, 2, dtype=torch.bool), step_mask=torch.ones(1, 2, dtype=torch.bool),
        target_phrase_weight=torch.ones(1, 2))
    first = phrase_auxiliary_loss({'phrase_prediction': prediction}, batch)
    batch['target_phrase_weight'] *= .25
    torch.testing.assert_close(phrase_auxiliary_loss({'phrase_prediction': prediction}, batch), first*.25)
    guard = EventGuard([EventTempo(Fraction(0), 200.)], 0., 10., physical_policy='legacy_report_v1')
    assert guard.physical_policy == 'legacy'


def test_locked_hold_with_fast_other_hand_is_soft_cost_only():
    tempo = [EventTempo(Fraction(0), 200.)]
    tail = [EventPoint(Fraction(1), '6'), EventPoint(Fraction(5, 4), '5'), EventPoint(Fraction(3, 2), '6')]
    def score(first):
        parsed = parse_event_notes([EventPoint(Fraction(0), first), *tail], tempo, 0.,
            duration_semantics=MIACODE_DURATION_SEMANTICS)
        assert not parsed.fatal_errors
        return _comfort(parsed.notes, 0., 1.)
    constrained, free = score('3h[4:2]'), score('3')
    assert 0 < constrained < free <= 1


def test_sustain_withdrawal_removes_its_dependent_suffix():
    from maichart.composer.stream_data import NOTE, StreamStep
    from maichart.composer.stream_search import _repair_prefix
    from maichart.tempo_clock import TempoClock
    song = SimpleNamespace(tempo=[EventTempo(Fraction(0), 200.)], first=0.)
    guard = EventGuard(song.tempo, 0., 10., physical_policy='runtime_hard_v2')
    points = [EventPoint(Fraction(0), '3h[4:2]'), EventPoint(Fraction(1, 2), '7')]
    steps = [StreamStep(NOTE, Fraction(0), Fraction(0), 0., 0., '3h[4:2]'),
             StreamStep(NOTE, Fraction(0), Fraction(1, 2), 0., .15, '7')]
    for point in points:
        assert guard.accepts(point)
        guard.commit(point)
    assert not guard.accepts(EventPoint(Fraction(1), '3'))
    result = dict(steps=steps, points=points, guard=guard, anchor_second=.15)
    options = list(_repair_prefix(result, song, TempoClock(song.tempo, 0.),
        dict(native_calls=0, native_sec=0.), {}, withdraw=True))
    assert options and options[0][0] == []
    assert options[0][1]['operation'] == 'withdraw_sustain_and_regenerate'
    from maichart.composer.event_set_codec import EventSetCodec
    codec=EventSetCodec.fit(['3h[4:2]','3h[8:1]','3','7'])
    edits=list(_repair_prefix(result,song,TempoClock(song.tempo,0.),
        dict(native_calls=0,native_sec=0.),{},codec=codec,
        duration_catalog={'3h[]':['3h[4:2]','3h[8:1]']}))
    assert edits and len(edits[0][0])==1
    assert edits[0][0][0].text=='3h[8:1]'
    assert codec.encode(edits[0][0][0].text)
