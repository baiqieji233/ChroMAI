from dataclasses import replace
from fractions import Fraction

import numpy as np
import torch

from maichart.composer.generate import (_filter_rhythm_and_active_span_candidates,
    _pointer_bin_settings, _microstep_head_limit, _grounded_pointer_probabilities)
from maichart.composer.candidates import MUSIC_CANDIDATE_POLICY_V6
from maichart.composer.model import build_model
from test_composer_audio_grounded_v3 import config, inputs, ATOMS


def test_local_lead_retains_recent_real_events_and_masks_older_history(monkeypatch):
    model=build_model(replace(config(),history_events=12,history_dropout=.9,
        ledger_dropout=0.,symbolic_dropout=0.),ATOMS).train()
    data=inputs()
    data.update(history=torch.ones(2,10,3,dtype=torch.long),history_features=torch.zeros(2,10,8),
        history_padding_mask=torch.tensor([[False]*10,[False]*3+[True]*7]),
        symbolic_keep=torch.ones(2,1))
    monkeypatch.setattr(torch,'rand',lambda *shape,**kw:torch.zeros(
        shape[0] if len(shape)==1 else shape,device=kw.get('device')))
    masks=[]
    hook=model.history_encoder.register_forward_pre_hook(
        lambda module,args,kwargs:masks.append(kwargs['src_key_padding_mask'].clone()),with_kwargs=True)
    try:out=model(**data,decode_actions=False)
    finally:hook.remove()
    assert masks[0][0].tolist()==[True]*2+[False]*8
    assert masks[0][1].tolist()==[False]*3+[True]*7
    assert torch.isfinite(out['candidate_context']).all()


def test_global_symbolic_dropout_still_removes_local_lead():
    model=build_model(config(cross_event_relation_policy='relative_heads_v1'),ATOMS).eval()
    with torch.no_grad():model.local_lead_in.out_proj.weight.normal_()
    data=inputs();data['symbolic_keep']=torch.zeros(2,1)
    a=model(**data,decode_actions=False)
    data['history']=data['history'].flip(1)
    b=model(**data,decode_actions=False)
    torch.testing.assert_close(a['candidate_context'],b['candidate_context'])
    torch.testing.assert_close(a['candidate_logits'],b['candidate_logits'])


def test_long_jump_activity_requires_head_and_nonrest_in_same_block():
    candidates=[Fraction(1,4),Fraction(8)]
    plan=np.zeros((2,6),np.float32)
    plan[:,1]=np.log1p([5.,0.]);plan[:,5]=[1.,0.]
    kwargs=dict(policy=MUSIC_CANDIDATE_POLICY_V6,
                plan_centers=np.array([.7,1.5]),plan_values=plan)
    assert _filter_rhythm_and_active_span_candidates(candidates,[True,True],np.array([.1,2.5]),
        Fraction(0),0.,**kwargs)==[True,True]
    plan[0,5]=.1
    assert _filter_rhythm_and_active_span_candidates(candidates,[True,True],np.array([.1,2.5]),
        Fraction(0),0.,**kwargs)==[True,False]


def test_shared_recovery_bin_controls_prune_noise_but_retain_real_long_rest():
    bins=torch.tensor([0,1,5])
    noise=_grounded_pointer_probabilities(torch.tensor([.9,.09,.01]).log(),bins,.5,**_pointer_bin_settings(.5))
    assert noise[2]==0
    rest=_grounded_pointer_probabilities(torch.tensor([.01,.09,.9]).log(),bins,.5,**_pointer_bin_settings(.5))
    assert rest[2]>.8
    assert _microstep_head_limit(Fraction(1,8),Fraction(0))==1
    assert _microstep_head_limit(Fraction(1,4),Fraction(0)) is None
