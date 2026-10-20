"""Whole-cell preference and speculative cache regression tests."""
from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch

from maichart.event_timing import EventTempo
from maichart.form.codec import RationalCodec
from maichart.form.type_guidance import TypeGuidance
from maichart.generation.conditions import GenerationCondition, GENERATION_DEFAULTS
from maichart.stream.constraints import BasicConfigurationGuard
from maichart.stream.configuration_search import propose_configurations,resample_terminal


class CellModel:
    def __init__(self,codec):self.codec=codec
    def decode_step(self,token,position,memory,mask,cache,**kwargs):
        previous=cache[0][0]
        current=torch.cat((previous,token.float()[...,None]),1)
        logits=torch.full((token.size(0),1,len(self.codec)),-20.)
        for row in range(token.size(0)):
            text=''.join(self.codec.tokens[int(i)][2:] for i in current[row,:,0] if self.codec.tokens[int(i)].startswith('N:'))
            remaining='2h[4:1]'[len(text):] if text.startswith('2') else ''
            logits[row,0,self.codec.ids['N:'+remaining[0]] if remaining else self.codec.end_id]=20.
        return {'logits':logits},[(current,current,None,None)]


def setup():
    codec=RationalCodec()
    condition=GenerationCondition(level=13,style=GENERATION_DEFAULTS,radar=GENERATION_DEFAULTS)
    guidance=TypeGuidance(['1','2h[4:1]'],condition,[],planner_strength=0,style_strength=.5)
    guard=BasicConfigurationGuard(guidance.grammar,[EventTempo(Fraction(0),120.)],0,max_button_heads=1)
    ids=[codec.bos_id,*codec.position(Fraction(0))]
    prefix=torch.tensor(ids,dtype=torch.float32)[None,:,None]
    logits=torch.full((len(codec),),-20.)
    logits[codec.ids['N:1']]=logits[codec.ids['N:2']]=0.
    return codec,guidance,guard,ids,prefix,logits


def test_complete_proposals_keep_selected_cache_and_explore_with_guided_rows():
    codec,guidance,guard,ids,prefix,logits=setup()
    bias=guidance.event(Fraction(0))
    seen=set();guided_choices=0
    for seed in range(12):
        picked,report=propose_configurations(CellModel(codec),codec,logits,[(prefix,prefix,None,None)],
            torch.zeros(1,2,3),torch.zeros(1,2,dtype=torch.bool),consumed=len(ids),max_tokens=64,
            grammar=guidance.grammar,restricted=guard.event(Fraction(0)),event_bias=bias,
            temperature=1.,candidates=4,generator=torch.Generator().manual_seed(seed))
        assert picked is not None and report['reason']=='complete'
        assert report['guided_proposals']==3
        guided_choices+=report['chosen_guided']
        assert picked['tokens'][-1]==codec.end_id
        expected=ids+picked['tokens'][:-1]
        assert picked['cache'][0][0][0,:,0].tolist()==expected
        assert picked['consumed']==len(expected)
        seen.add(picked['text'])
    assert seen=={'1','2h[4:1]'}


def test_mixture_importance_weights_preserve_terminal_target_law():
    from maichart.stream.configuration_search import _mixture_correction
    import math
    # Two cells, model law p=(.9,.1); guided proposal q=(.5,.5); terminal
    # preferences t=(0, log 3). The target is p*exp(t) normalised = (.75,.25).
    p=[.9,.1];q=[.5,.5];t=[0.,math.log(3.)];count=4
    mixture=[(p[i]+(count-1)*q[i])/count for i in range(2)]
    expected=[p[i]*math.exp(t[i]) for i in range(2)]
    expected=[v/sum(expected) for v in expected]
    implied=[mixture[i]*math.exp(t[i]+_mixture_correction(math.log(p[i]),math.log(q[i]),count)) for i in range(2)]
    implied=[v/sum(implied) for v in implied]
    assert implied==pytest.approx(expected)
    assert _mixture_correction(-1.,-1.,4)==pytest.approx(0.)
    assert _mixture_correction(-3.,-1.,1)==0.


def test_context_overflow_does_not_silently_prefer_short_cells():
    codec,guidance,guard,ids,prefix,logits=setup()
    logits[codec.ids['N:1']]=-100
    picked,report=propose_configurations(CellModel(codec),codec,logits,[(prefix,prefix,None,None)],
        torch.zeros(1,2,3),torch.zeros(1,2,dtype=torch.bool),consumed=len(ids),max_tokens=len(ids)+2,
        grammar=guidance.grammar,restricted=guard.event(Fraction(0)),event_bias=guidance.event(Fraction(0)),
        temperature=1.,candidates=4,generator=torch.Generator().manual_seed(0))
    assert picked is None and report['reason']=='context_space'


def test_equal_terminal_scores_preserve_duplicates_and_no_second_model_probability():
    rng=torch.Generator().manual_seed(77)
    # Three sampled copies of A and one B must retain 3:1 mass, not 1:1 or 9:1.
    selected=[resample_terminal([2.,2.,2.,2.],rng,device='cpu') for _ in range(2000)]
    assert .70 < sum(i<3 for i in selected)/len(selected) < .80


def test_same_types_receive_same_style_preference_regardless_of_button_or_path_spelling():
    condition=GenerationCondition(level=13,style=GENERATION_DEFAULTS,radar=GENERATION_DEFAULTS)
    guidance=TypeGuidance(['1','8','1-5[4:1]','8-4[8:2]'],condition,[],planner_strength=0,style_strength=3)
    bias=guidance.event(Fraction(0))
    assert bias.terminal_score('1')==bias.terminal_score('8')
    assert bias.terminal_score('1-5[4:1]')==bias.terminal_score('8-4[8:2]')


def test_stream_token_guidance_penalizes_break_and_chord_runs_causally():
    condition=GenerationCondition(level=13,style={**GENERATION_DEFAULTS,'chord_bias':.2,'tap_bias':.7},radar=GENERATION_DEFAULTS)
    guidance=TypeGuidance(['1','1b','1/2'],condition,[],planner_strength=0,style_strength=1)
    for index in range(8): guidance.commit('1b' if index < 4 else '1/2',Fraction(index))
    bias=guidance.event(Fraction(6))
    values=bias.token_bias('1',['b','/',''])
    assert values[0] < 0 and values[1] < 0
    assert values[2] == 0
