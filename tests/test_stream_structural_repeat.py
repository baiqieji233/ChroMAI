"""Structural proposal feedback, true mixture laws, and explicit action ends."""
from fractions import Fraction as F
from types import SimpleNamespace
import math

import numpy as np
import pytest
import torch

from maichart.event_timing import EventTempo
from maichart.event_timing import EventPoint,seconds_at_beat
from maichart.form.codec import RationalCodec
from maichart.form.type_guidance import TypeGuidance
from maichart.generation.conditions import GenerationCondition,GENERATION_DEFAULTS
from maichart.stream.configuration_search import propose_configurations,_mixture_correction
from maichart.stream.constraints import BasicConfigurationGuard
from maichart.stream.decoding import validate_options
from maichart.stream.generate import parser
from maichart.stream.repetition import RepeatBias,RepetitionGuard
from maichart.stream.reference_guard import ReferenceCandidateGuard
from maichart.stream.timing import release_proposals,CanonicalPositionGrammar
from test_stream_grounding import cpu_threads


def guidance(atoms):
    return TypeGuidance(atoms,GenerationCondition(level=14,radar=GENERATION_DEFAULTS,
        style=GENERATION_DEFAULTS),[],planner_strength=0.,style_strength=0.)


def test_structural_penalty_is_visible_at_slash_and_hold_before_terminal():
    g=guidance(['1','2','3','1h[4:1]','2h[4:1]'])
    base=g.event(F(0))
    chord=RepeatBias(base,(((2,0,0,0,0),True,.8),))
    assert chord.biases('1',['/',None])[0]<chord.biases('1',['/',None])[1]
    hold=RepeatBias(base,(((0,0,1,0,0),True,.8),))
    assert hold.biases('1',['h',None])[0]<hold.biases('1',['h',None])[1]
    value=chord.biases('',['1'])[0]+chord.biases('1',['/'])[0]+chord.biases('1/',['2'])[0]+chord.biases('1/2',[None])[0]
    assert value==pytest.approx(-.8-chord._prefix(''))
    assert chord.terminal_score('1/2')==pytest.approx(-.8)
    assert chord.terminal_score('1h[4:1]')==0.
    assert chord.terminal_exploration_score('1h[4:1]')>chord.terminal_score('1h[4:1]')


@pytest.mark.parametrize('count',[2,4,8])
def test_exploratory_proposal_has_exact_three_law_importance_correction(count):
    p=[.9,.1];guided=[.7,.3];explore=[.4,.6];terminal=[-.8,0.]
    mix=[(p[i]+(count-2)*guided[i]+explore[i])/count for i in range(2)]
    implied=[mix[i]*math.exp(terminal[i]+_mixture_correction(
        math.log(p[i]),math.log(guided[i]),count,math.log(explore[i]))) for i in range(2)]
    target=[p[i]*math.exp(terminal[i]) for i in range(2)]
    assert [x/sum(implied) for x in implied]==pytest.approx([x/sum(target) for x in target])


def fill(guard,texts,gap=F(1,2)):
    for i,text in enumerate(texts):guard.commit(text,i*gap,float(i*gap)*60/190)
    beat=len(texts)*gap
    return guard.event(guidance(['1','2','C','2<7[4:5]']).event(beat),beat,float(beat)*60/190)


def test_configuration_feedback_handles_sparse_and_constant_hold_cycles_without_overload():
    profile=SimpleNamespace(head_rate=lambda _:100.)
    hold=RepetitionGuard(profile,strength=0.,structure_strength=.6)
    bias=fill(hold,['1h[8:1]/5h[8:1]','2h[8:1]/6h[8:1]','3h[8:1]/7h[8:1]']*20)
    assert bias.repeat_penalty('4h[8:1]/8h[8:1]')>0
    assert hold.report()['monotony_events']==1 and hold.report()['overload_events']==0
    sparse=RepetitionGuard(profile,strength=.6,structure_strength=.6)
    bias=fill(sparse,['C','2<7[4:5]']*20,gap=F(2))
    assert bias.repeat_penalty('C')>0 and sparse.report()['overload_events']==0
    zero=RepetitionGuard(profile,strength=0.,structure_strength=.6)
    assert fill(zero,['6h[1:0]']*60).repeat_penalty('6h[1:0]')>0


def test_short_trills_and_generic_single_tap_type_are_not_blanket_rejected():
    short=RepetitionGuard(None,strength=0.,structure_strength=3.)
    assert not fill(short,['1','8']*6).cases
    ordinary=RepetitionGuard(None,strength=0.,structure_strength=.6)
    bias=fill(ordinary,list('12345678')*10)
    assert all(not structural for _,structural,_ in bias.cases)
    assert ordinary.report()['hard_rejections']==0
    off=RepetitionGuard(None,strength=0.,structure_strength=0.)
    assert not fill(off,['4']*100).cases


class ChordModel:
    def __init__(self,codec,slash_logit=1.2):self.codec=codec;self.slash_logit=slash_logit
    def decode_step(self,token,position,memory,mask,cache,**kwargs):
        current=torch.cat((cache[0][0],token.float()[...,None]),1)
        logits=torch.full((token.size(0),1,len(self.codec)),-100.)
        for row in range(len(token)):
            text=''.join(self.codec.tokens[int(v)][2:] for v in current[row,:,0]
                         if self.codec.tokens[int(v)].startswith('N:'))
            if text=='1':logits[row,0,self.codec.end_id]=0.;logits[row,0,self.codec.ids['N:/']]=self.slash_logit
            elif text=='1/':logits[row,0,self.codec.ids['N:2']]=0.
            else:logits[row,0,self.codec.end_id]=0.
        return {'logits':logits},[(current,current,None,None)]


def proposal(seed,slash_logit):
    codec=RationalCodec();g=guidance(['1','2'])
    bias=RepeatBias(g.event(F(0)),(((2,0,0,0,0),True,.8),))
    guard=BasicConfigurationGuard(g.grammar,[EventTempo(F(0),120)],0.)
    ids=[codec.bos_id,*codec.position(F(0))];prefix=torch.tensor(ids,dtype=torch.float32)[None,:,None]
    logits=torch.full((len(codec),),-100.);logits[codec.ids['N:1']]=0.
    return propose_configurations(ChordModel(codec,slash_logit),codec,logits,[(prefix,prefix,None,None)],
        torch.zeros(1,2,3),torch.zeros(1,2,dtype=torch.bool),consumed=len(ids),max_tokens=64,
        grammar=g.grammar,restricted=guard.event(F(0)),event_bias=bias,temperature=.15,
        candidates=4,generator=torch.Generator().manual_seed(seed),exploration_temperature=1.)


def test_exploration_can_cover_alternative_structure_and_reports_when_it_does_not():
    reports=[proposal(seed,1.2)[1] for seed in range(10)]
    assert all(row['exploration_proposals']==1 for row in reports)
    assert any(row['candidate_structures']>1 for row in reports)
    picked,report=proposal(0,100.)
    assert picked['text']=='1/2'
    assert report['all_candidates_repeat'] and report['candidate_structures']==1


def test_explicit_sustain_end_blocks_positive_action_tails_without_clipping_or_banning_zero_holds():
    atoms=['1','2','1h[4:2]','1h[8:1]','1h[1:0]','1-5[8:1]','Ch[4:2]']
    g=guidance(atoms)
    owner=BasicConfigurationGuard(g.grammar,[EventTempo(F(0),120.)],0.,max_sustain_end=1.)
    event=owner.event(F(1))
    assert not event.is_complete('1h[4:2]') and not event.can_continue('1h[4')
    assert not event.is_complete('Ch[4:2]') and not event.is_complete('1-5[8:1]')
    assert event.is_complete('1h[8:1]') and event.is_complete('1h[1:0]')
    later=owner.event(F(4))
    assert later.is_complete('2') and later.is_complete('1h[1:0]')
    assert owner.report()['max_sustain_end_sec']==1.


@pytest.mark.parametrize('key,value',[('repetition_structure_strength',-1),
    ('configuration_exploration_temperature',float('nan')),('sustain_end_sec',0)])
def test_new_controls_validate_and_can_be_disabled(key,value):
    args=parser().parse_args(['--audio','a','--checkpoint','c','--output','o'])
    assert args.repetition_structure_strength==.6 and args.sustain_end_sec is None
    setattr(args,key,value)
    with pytest.raises(ValueError,match=key):validate_options(args)


def test_release_recovery_uses_committed_actions_and_exact_rhythm_candidates():
    tempo=[EventTempo(F(0),120.)];reference=ReferenceCandidateGuard(tempo,0.)
    points=[EventPoint(F(0),'1h[1:1]'),EventPoint(F(1),'5h[1:1]')]
    for point in points:reference.commit(point)
    assert reference.button_capacity(F(2))==0
    release=reference.next_button_release(1.)
    assert 2.<release<2.1
    values=release_proposals(F(1),release,tempo,0.,points)
    assert values and all(seconds_at_beat(value,tempo,0.)>=release for value in values)
    assert all(isinstance(value,F) and reference.button_capacity(value)>0 for value in values)
    grammar=CanonicalPositionGrammar(F(1),F(10),max_denominator=384,
        max_delta_denominator=96,delta_reference=F(1),allowed_ranges=((F(str(release))*2,F(10)),))
    assert all(grammar.complete(str(value)) for value in values)
    # Recovery proposes positions without mutating committed chart/detector state.
    assert len(reference.records)==2 and points[-1].beat==F(1)
