from fractions import Fraction as F
import math
import numpy as np
import pytest
import torch

from maichart.stream.ending import LearnedEnding
from maichart.form.rational_position_search import position_complexity_cost,search_position
from maichart.stream.timing import CanonicalPositionGrammar
from maichart.stream.train import _loss,_mean_metrics
from maichart.form.codec import RationalCodec
from maichart.form.curve import encode_density
from maichart.form.physical_guard import PhysicalFeasibilityGuard
from maichart.event_timing import EventPoint,EventTempo
from test_stream_generation_integrity import compose
from test_stream_training_contract import loss_inputs
from test_stream_training_contract import small_model
from test_stream_quality_constraints import guard
from test_stream_timing_search import ScriptedModel


def test_complexity_prior_is_neutral_through_32nd_without_denominator_whitelist():
    for value in [F(1),F(1,2),F(1,3),F(1,4),F(1,5),F(1,6),F(1,7),F(1,8)]:
        assert position_complexity_cost(str(value),F(0),floor=8)==0
    assert position_complexity_cost('1/13',F(0),floor=8)>0
    assert position_complexity_cost('1/2',F(0),floor=1)>0


def test_end_boundary_does_not_rescale_an_intended_whole_beat_to_a_tiny_fraction():
    codec=RationalCodec();model=ScriptedModel(codec,[('399',10.),('5177/13',2.)])
    grammar=CanonicalPositionGrammar(F(398),F('398.3988')+8,max_denominator=384)
    value,_=search_position(model,codec,grammar,model.logits([''])[0,0],None,
        torch.zeros(1,1,1),None,consumed=0,width=4,canonical_only=True,
        complexity_floor=8)
    assert value=='399'


def test_actual_loop_stops_before_outside_position_without_emitting_it(monkeypatch):
    expected=[EventPoint(F(3,2),'1'),EventPoint(F(2),'2')]
    actual,status,_=compose(monkeypatch,expected)
    assert actual==expected[:1]
    assert status['reason']=='predicted_position_after_audio_end' and not status['incomplete']


def test_no_remaining_probability_is_derived_from_learned_heads_not_amplitude():
    # Quiet interlude followed by dense music must not become a premature end.
    times=np.array([0.,20.,21.,30.,31.,40.])
    density=encode_density(np.array([0.,0.,8.,8.,.2,.2]),'musical_log_v1')
    ending=LearnedEnding(times,density,40.)
    assert ending.eos_logit(-10.,10.)==pytest.approx(-10.,abs=1e-8)
    assert ending.remaining_heads(36.)==pytest.approx(.8)
    assert ending.eos_logit(-10.,39.)>0
    assert ending.report()['amplitude_threshold'] is None
    disabled=LearnedEnding(times,density,40.,strength=0.)
    assert disabled.eos_logit(-10.,39.)==pytest.approx(-10.)


def test_curve_mass_integrates_exact_linear_rate_between_nonuniform_nodes():
    ending=LearnedEnding([0.,1.,4.],encode_density([2.,4.,4.],'musical_log_v1'),4.)
    assert ending.remaining_heads(0.)==pytest.approx(15.)
    assert ending.remaining_heads(.5)==pytest.approx(13.75)
    assert ending.remaining_heads(3.)==pytest.approx(4.)


def test_later_speculative_time_cannot_delete_hold_occupancy_for_earlier_retry():
    basic=guard()
    basic.commit(EventPoint(F(0),'1h[4:2]'))
    assert basic.event(F(4)).is_complete('1')
    assert not basic.event(F(1)).can_continue('1')
    physical=PhysicalFeasibilityGuard(1.)
    physical.register(F(0),0.,[EventTempo(F(0),120.)],0.)
    physical.commit('1h[4:2]',F(0))
    physical.register(F(4),2.)
    physical.event(F(4))
    assert physical.occupied_mask(.5)[0]


def test_low_probability_configuration_reconsiders_uncommitted_event_time(monkeypatch):
    import maichart.stream.decoding as module
    choices=iter(['0','1/4','1/2'])
    def choose(*args,**kwargs):
        value=next(choices)
        assert args[2].complete(value)
        return value,{'expanded_hypotheses':0,'candidates':[]}
    monkeypatch.setattr(module,'search_position',choose)
    proposed=[EventPoint(F(0),'1h[8:1]'),EventPoint(F(1,4),'1')]
    actual,status,_=compose(monkeypatch,proposed,max_tokens=64,
        atom_catalog=['1','1h[8:1]'],options_override={'position_beam_size':4,
        'hold_protection':'on','constraint_min_probability':.05})
    assert actual==[proposed[0],EventPoint(F(1,2),'1')]
    assert status['reason']=='eos'
    retries=status['event_reconsideration']['retries']
    assert len(retries)==1 and retries[0]['beat']=='1/4'
    assert retries[0]['retained_model_mass']==0.


def test_training_metrics_do_not_hide_bad_eos_behind_common_note_tokens():
    output,batch=loss_inputs()
    codec=RationalCodec()
    output['logits']=torch.zeros_like(output['logits'])
    output['logits'][...,codec.at_id]=10.
    _,metrics=_loss(output,batch)
    means=_mean_metrics(metrics,1)
    assert means['eos_targets']==1 and means['eos_accuracy']==0.
    assert means['eos_token_loss']>9.
    assert means['position_targets']==1


def test_reconsideration_cache_rewind_matches_clean_committed_prefix():
    from maichart.stream.decoding import rewind_decoder_cache
    torch.manual_seed(13)
    model=small_model()
    with torch.inference_mode():
        memory,mask=model.encode(torch.randn(1,5,120),torch.randn(1,16))
        committed=torch.tensor([[1,3,8,4,18,5]])
        expected=model.decode(committed,memory,mask)['logits'][:,-1]
        speculative=torch.cat((committed,torch.tensor([[3,10,4,20]])),1)
        cache=None
        for index in range(speculative.size(1)):
            _,cache=model.decode_step(speculative[:,index:index+1],index,memory,mask,cache)
        cross=cache[0][2]
        cache,consumed=rewind_decoder_cache(cache,committed.size(1)-1)
        assert cache[0][2] is cross
        output,_=model.decode_step(committed[:,-1:],consumed,memory,mask,cache)
        torch.testing.assert_close(output['logits'][:,-1],expected,rtol=1e-5,atol=1e-5)
