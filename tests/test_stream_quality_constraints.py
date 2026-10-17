from fractions import Fraction as F
from types import SimpleNamespace

import pytest
import torch

from maichart.event_timing import EventPoint, EventTempo
from maichart.generation.event_cli import _NoteGrammar
from maichart.generation.sustain_state import SustainStateIndex
from maichart.stream.constraints import BasicConfigurationGuard, _struck_heads
from maichart.stream.context import prefix_action_state
from maichart.stream.timing import interval_consistency_cost
from maichart.stream.train import _finish_accumulation, _mean_metrics, _resume_start_epoch
from maichart.stream.checkpoint import save
from maichart.stream.data import TRAINING_SEMANTICS
from maichart.stream.model import ChartStreamTransformer, StreamConfig
from maichart.webui.stream_api import inspect
from maichart.form.rational_position_search import _score_proposals
from maichart.form.codec import RationalCodec
from maichart.stream.reference_guard import ReferenceCandidateGuard


TEMPO = [EventTempo(F(0), 120.)]


def guard(atoms=None, **kwargs):
    return BasicConfigurationGuard(_NoteGrammar.from_atoms(atoms or
        ['1', '2', '3', '4', '5', '7', '1h[4:2]', '2-6[8:1]', 'A4', 'B4', 'Ch[4:2]']), TEMPO, 0., **kwargs)


def test_three_button_heads_are_unreachable_while_touch_groups_remain_available():
    grammar = guard().event(F(0))
    assert grammar.is_complete('4/5')
    assert grammar.can_continue('4/5/')
    # Touch can still follow a two-button chord. The slash remains reachable
    # only if a valid Touch atom exists, but a third button is never offered.
    assert grammar.can_continue('4/5/B')
    assert not grammar.can_continue('4/5/7')
    assert not grammar.is_complete('4/5/7')
    assert grammar.is_complete('4/5/B4')


def test_hold_and_slide_heads_count_toward_the_same_explicit_cap():
    grammar = guard().event(F(0))
    assert grammar.is_complete('1h[4:2]/2-6[8:1]')
    assert not grammar.can_continue('1h[4:2]/2-6[8:1]/3')
    assert not grammar.is_complete('1h[4:2]/2-6[8:1]/3')


def test_headless_slide_does_not_consume_a_button_head():
    # The basic head classifier follows the reference parser. The underlying
    # project note grammar may still exclude the optional '?' syntax.
    assert _struck_heads('1/2/3?-7[8:1]') == frozenset({'1', '2'})


def test_hold_occupancy_uses_actual_tempo_and_expires_at_its_end():
    owner = guard()
    owner.commit(EventPoint(F(0), '1h[4:2]'))  # lasts two beats / one second
    active = owner.event(F(1))
    assert not active.can_continue('1')
    assert active.is_complete('2')
    assert owner.event(F(2)).is_complete('1')


def test_touch_hold_is_protected_without_reserving_slide_path_buttons():
    owner = guard()
    owner.commit(EventPoint(F(0), 'Ch[4:2]/2-6[8:1]'))
    active = owner.event(F(1, 2))
    assert not active.can_continue('C')
    assert active.is_complete('2')


def test_button_and_a_touch_aliases_do_not_bypass_hold_or_duplicate_pad_rules():
    owner = guard(['4','A4','4h[4:2]'])
    assert not owner.event(F(0)).is_complete('4/A4')
    owner.commit(EventPoint(F(0),'4h[4:2]'))
    assert not owner.event(F(1)).can_continue('A4')
    assert owner.event(F(2)).is_complete('A4')


def test_hold_and_slide_cannot_extend_past_the_authorized_audio_end():
    owner=guard(['1','1h[4:2]','2-6[8:1]'],max_action_end=1.)
    assert owner.event(F(0)).is_complete('1h[4:2]')
    late=owner.event(F(3,2))  # .75s, .25s left
    assert late.is_complete('1')
    assert not late.is_complete('1h[4:2]')
    assert not late.can_continue('1h')
    assert not late.can_continue('2-6[')


def test_constraint_opt_out_and_reachability_prevent_dead_end_prefixes():
    assert guard(max_button_heads=0).event(F(0)).is_complete('3/5/7')
    simple = guard(['1','2'])
    assert simple.event(F(0)).is_complete('1/2')
    assert not simple.event(F(0)).can_continue('1/2/')
    simple.commit(EventPoint(F(0), '1h[4:2]'))
    assert not simple.event(F(1)).can_continue('2/')
    disabled = guard(protect_holds=False)
    disabled.commit(EventPoint(F(0), '1h[4:2]'))
    assert disabled.event(F(1)).is_complete('1')


def test_prefix_clock_is_independent_of_the_next_authored_event():
    first = [EventPoint(F(0), '1h[1:2]'), EventPoint(F(1), '2')]
    second = [first[0], EventPoint(F(6), '2')]
    a = SustainStateIndex(first, TEMPO, 0.)
    b = SustainStateIndex(second, TEMPO, 0.)
    old_a, _, _ = a.at(first[1].beat)
    old_b, _, _ = b.at(second[1].beat)
    assert not torch.equal(old_a, old_b)  # Previous code exposed target timing.
    values, mask, metadata = prefix_action_state(a, first, 1)
    other, other_mask, _ = prefix_action_state(b, second, 1)
    torch.testing.assert_close(values, other, atol=0, rtol=0)
    assert torch.equal(mask, other_mask)
    assert metadata['boundary_second'] == 0. and metadata['state_count'] == 1
    _, empty, _ = prefix_action_state(a, first, 0)
    assert empty.all()  # First target never enters the memory.


def test_prefix_clock_does_not_expose_a_future_head_within_the_old_epsilon():
    points=[EventPoint(F(0),'1h[4:2]'),EventPoint(F(1,10**10),'2h[4:2]')]
    state=SustainStateIndex(points,TEMPO,0.)
    values,mask,metadata=prefix_action_state(state,points,1)
    assert metadata['state_count'] == 1 and not mask.any()


def test_short_interval_prediction_distinguishes_nine_from_161_beats():
    near = interval_consistency_cost(F(9), 1.53/8, 1.)
    far = interval_consistency_cost(F(161), 1.53/8, 1.)
    correct = interval_consistency_cost(F('1.53'), 1.53/8, 1.)
    assert correct == pytest.approx(0.) and far > near > 0
    assert far > 5.
    assert interval_consistency_cost(F(161), 1., 1.) == 0.  # censored >=8 head
    assert interval_consistency_cost(F(10000), 1., 1.) == 0.
    assert interval_consistency_cost(F(161), .2, 0.) == 0.


def test_actual_partial_accumulation_matches_a_direct_mean_gradient():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad(): model.weight.fill_(.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    for x in [1., 2.]:
        loss = model(torch.tensor([[x]])).square().mean()
        scaler.scale(loss/8).backward()
    assert _finish_accumulation(model, optimizer, scaler, 2, 8)
    assert float(model.weight) == pytest.approx(.05)
    assert _mean_metrics({'loss': 6., 'token_accuracy': 1.5}, 3) == {'loss': 2., 'token_accuracy': .5}


def test_partial_resume_replays_the_unfinished_epoch_instead_of_skipping_it():
    assert _resume_start_epoch({'epoch':20,'training_position':{'chunk':472}}) == 19
    assert _resume_start_epoch({'epoch':20,'training_position':{'complete':True}}) == 20


def test_proposed_fraction_scores_match_full_teacher_forced_probabilities():
    torch.manual_seed(91)
    codec = RationalCodec()
    model = ChartStreamTransformer(StreamConfig(len(codec), d_model=16,nhead=2,layers=1,
        dim_feedforward=32,dropout=0,max_tokens=64)).eval()
    with torch.inference_mode():
        memory, mask = model.encode(torch.randn(1,4,120),torch.randn(1,16))
        caches = None
        for index, token in enumerate([codec.bos_id, codec.at_id]):
            prediction,caches = model.decode_step(torch.tensor([[token]]),index,memory,mask,caches)
        texts = ['135','37/997','1/2']
        scored, count = _score_proposals(model,codec,texts,prediction['logits'][0,-1],
            caches,memory,mask,consumed=2,timing_features=None)
        assert count > 0
        for text in texts:
            target = [codec.ids['P:'+c] for c in text]+[codec.note_id]
            source = torch.tensor([[codec.bos_id,codec.at_id,*target[:-1]]])
            logp = model.decode(source,memory,mask)['logits'].float().log_softmax(-1)
            expected = sum(float(logp[0,1+i,token]) for i,token in enumerate(target))
            assert scored[text] == pytest.approx(expected,abs=1e-5)


def test_old_checkpoint_stays_generatable_but_ui_rejects_strict_resume(tmp_path):
    model = ChartStreamTransformer(StreamConfig(55, d_model=16, nhead=2, layers=1, dim_feedforward=32))
    manifest = tmp_path/'manifest.json'; manifest.write_text('{}')
    path = tmp_path/'stream.pt'
    save(path, model, None, epoch=1, args={}, manifest=manifest)
    old = inspect(path)
    assert old['valid'] and not old['can_resume'] and old['can_init']
    assert old['generation_capabilities']['continuous_stream']
    assert old['warnings']
    save(path, model, None, epoch=1, args={'training_semantics': TRAINING_SEMANTICS}, manifest=manifest)
    current = inspect(path)
    assert current['valid'] and current['can_resume'] and not current['warnings']


def test_reference_guard_rejects_active_hold_plus_two_new_taps_and_accepts_one():
    reference = ReferenceCandidateGuard(TEMPO,0.)
    reference.commit(EventPoint(F(0),'1h[4:3]'))
    accepted,errors = reference.check(EventPoint(F(1),'3/5'))
    assert not accepted and any(error['kind'] == 'MultiTouch' for error in errors)
    assert reference.check(EventPoint(F(1),'3'))[0]


def test_reference_guard_checks_runtime_overlap_before_commit():
    reference = ReferenceCandidateGuard(TEMPO,0.)
    reference.commit(EventPoint(F(0),'1'))
    accepted,errors = reference.check(EventPoint(F(1,100),'1'))
    assert not accepted and any(error['kind'] == 'Overlap' for error in errors)
    assert len(reference.records) == 1  # Rejected candidate cannot pollute memory.


def test_reference_guard_does_not_reject_an_intentional_long_rest():
    reference = ReferenceCandidateGuard(TEMPO,0.)
    reference.commit(EventPoint(F(0),'1h[4:1]'))
    assert reference.check(EventPoint(F(100),'3'))[0]


def test_reference_rejects_indirect_damage_to_an_earlier_slide():
    tempo=[EventTempo(F(0),185.05)]
    reference=ReferenceCandidateGuard(tempo,0.)
    # Touch C changes the earlier two-Slide judgment. The final diagnostic
    # names another Slide as cause, so checking only the candidate's key misses it.
    reference.commit(EventPoint(F(138),'5h[8:5]'))
    reference.commit(EventPoint(F(140),'3/A7'))
    reference.commit(EventPoint(F(141),'3<6[2:1]/7-3[2:1]'))
    accepted,errors=reference.check(EventPoint(F(142),'C'))
    assert not accepted and any(e['kind']=='SlideTooFast' for e in errors)
