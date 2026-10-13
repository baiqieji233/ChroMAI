from fractions import Fraction as F
from types import SimpleNamespace

import pytest
import torch

from maichart.form import generate
from maichart.form.codec import RationalCodec
from test_form_onset_selection import mocked_plan


@pytest.mark.parametrize('axis,first,rest', [
    ('audio_absolute_v2',0.,(.330,.335)),
    ('chart_relative_v1',.1,(.230,.235)),
])
def test_final_quantized_onsets_cannot_enter_a_saved_rest_region(monkeypatch,axis,first,rest):
    scores = [.1]*20
    scores[17] = .9
    profile = {'time_axis':axis,'points':[{'time_sec':0,'intensity':.5},{'time_sec':1,'intensity':.5}],
               'rest_regions':[{'start_sec':rest[0],'end_sec':rest[1]}]}
    monkeypatch.setattr(generate,'exact_candidate',lambda second,tempo,_first:
                        F(str((1/3-first)*2)))
    plan = mocked_plan(monkeypatch,scores,curve=profile,merge_ms=0,first=first)
    assert not plan['onsets']
    assert plan['selection']['rest_rejected_after_quantization'] == 1


def test_real_180_bpm_rounding_can_enter_rest_but_is_rejected(monkeypatch):
    from maichart.event_timing import EventTempo
    from maichart.generation import intensity
    from maichart.generation.conditions import GenerationCondition
    import numpy as np
    times = np.arange(25)*.02
    song = SimpleNamespace(spans=[SimpleNamespace(start=F(0),end=F(3,2))],
        tempo=[EventTempo(F(0),180)],first=0.,audio=SimpleNamespace(duration_sec=.5))
    monkeypatch.setattr(generate,'form_audio',lambda *_:torch.zeros(25,120))
    monkeypatch.setattr(generate,'input_example',lambda *_a,**_kw:{'form':torch.zeros(1,507)})
    monkeypatch.setattr(generate,'segment_frame_times',lambda *_:times)
    monkeypatch.setattr(intensity,'music_activity',lambda _:SimpleNamespace(times=times,active=lambda:np.ones(25)))
    scores=torch.full((25,),.1);scores[17]=.9
    planner=lambda *_:{'onset_logits':torch.logit(scores)[None],
        'type_logits':torch.zeros(1,25,4),'sustain_seconds':torch.zeros(1,25)}
    profile={'time_axis':'audio_absolute_v2','points':[{'time_sec':0,'intensity':.5},{'time_sec':.5,'intensity':.5}],
             'rest_regions':[{'start_sec':.330,'end_sec':.335}]}
    assert generate.exact_candidate(.34,song.tempo,0) == F(1)
    plan=generate.plan_onsets(planner,song,GenerationCondition(),intensity_curve=profile)
    assert not plan['onsets'] and plan['selection']['rest_rejected_after_quantization'] == 1


def test_character_limit_fallback_replaces_tokens_and_closes_the_emitted_note(monkeypatch):
    monkeypatch.setattr(generate,'MAX_NOTE_CHARACTERS',3)
    codec=RationalCodec()
    atom='1-5[8:1]'
    class Model:
        config=SimpleNamespace(max_tokens=128)
        def __init__(self): self.before_positions=[]
        def eval(self): pass
        def encode(self,_):return torch.zeros(1,1,1),torch.zeros(1,1,dtype=torch.bool)
        def rewind_decode_cache(self,cache,size):return cache[:size]
        def decode_step(self,token,position,memory,mask,cache=None):
            history=list(cache or [])
            token=int(token.item())
            if token==codec.at_id:self.before_positions.append(list(history))
            history.append(token)
            start=max((i for i,t in enumerate(history) if t==codec.note_id),default=len(history))
            length=len(history)-start-1
            logits=torch.full((1,1,len(codec)),-100.)
            wanted=codec.ids['N:'+atom[min(length,len(atom)-1)]]
            logits[...,wanted]=100.
            return logits,history
    model=Model()
    points,count=generate.arrange_span(model,codec,{'meta':{},'action_state_padding_mask':torch.ones(1,dtype=torch.bool)},
        [F(0),F(1)],['2',atom],origin=F(0),temperature=0)
    assert [p.note_text for p in points] == ['2','2'] and count==2
    assert model.before_positions[1][-2:] == [codec.ids['N:2'],codec.end_id]


def test_rewound_real_decoder_cache_matches_a_clean_causal_prefix():
    from maichart.form.model import FormConfig,FormTransformer
    torch.manual_seed(4)
    model=FormTransformer(FormConfig(vocab_size=20,d_model=8,nhead=2,layers=1,dim_feedforward=16,dropout=0)).eval()
    memory=torch.randn(1,3,8);mask=torch.zeros(1,3,dtype=torch.bool)
    with torch.inference_mode():
        cache=None
        for i,token in enumerate((1,3,7)):
            _,cache=model.decode_step(torch.tensor([[token]]),i,memory,mask,cache)
        rewound=model.rewind_decode_cache(cache,2)
        actual,_=model.decode_step(torch.tensor([[9]]),2,memory,mask,rewound)
        clean=None
        for i,token in enumerate((1,3,9)):
            expected,clean=model.decode_step(torch.tensor([[token]]),i,memory,mask,clean)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
