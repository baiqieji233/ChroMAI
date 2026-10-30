from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from maichart.composer import generate as module
from maichart.composer.sampling_contract import (
    MICROSTEP_ACTION_SAMPLING_POLICY, recovery_sampling_settings,
)
from maichart.generation.event_cli import _NoteGrammar


def test_microstep_entry_is_tempered_but_exit_remains_available():
    entry=module.pointer_switch_levels('grid_family_coherent_v3',[Fraction(0)],'cpu')
    inside=module.pointer_switch_levels('grid_family_coherent_v3',[Fraction(1,8)],'cpu')
    legacy=module.pointer_switch_levels('grid_family_coherent_v2',[Fraction(0)],'cpu')
    assert entry[3] and not inside[3] and not legacy[3]
    assert not entry[0] and not entry[1] and not entry[2]
    assert recovery_sampling_settings(MICROSTEP_ACTION_SAMPLING_POLICY)['pointer_grid_policy']=='grid_family_coherent_v3'


def test_fast_chord_member_and_extended_run_receive_soft_cost_only():
    bits=np.asarray([0,1,2,4],dtype=np.uint64)
    clock=SimpleNamespace(at=lambda beat:float(beat)*.3)
    def bias(history,beat,clock_value=clock):
        return module._official_tail_repetition_bias(history,beat,bits,np.uint64(7),
            device='cpu',per_lane_runs=True,repetition_clock=clock_value)
    assert bias([((1,),Fraction(0))],Fraction(1,4)) is None
    fast=bias([((1,2),Fraction(0))],Fraction(1,4))
    assert fast[0]==0 and fast[1]<0 and fast[2]<0 and fast[3]==0
    assert bias([((1,2),Fraction(0))],Fraction(1,2)) is None
    assert bias([((1,),Fraction(0)),((1,),Fraction(1,4))],Fraction(1,2))[1]<0
    assert bias([((1,2),Fraction(0))],Fraction(1,4),None)[1]<0


@pytest.mark.parametrize('policy,penalized',[('official_tail_v2',True),('legacy_v1',False)])
def test_second_slot_and_sequence_score_share_repetition_cost(monkeypatch,policy,penalized):
    atoms=('1','2','3');grammar=_NoteGrammar.from_atoms(atoms)
    metadata=module._head_metadata(atoms,grammar,touch_layout_policy='open')
    monkeypatch.setattr(module,'_anti_repetition_logit_bias',lambda *a,**k:torch.tensor([0.,-3.,0.,0.]))
    rows=torch.tensor([[-100.,-100.,100.,-100.],[-2.,1.,-100.,0.]])
    captured=[]
    original=module._category_truncate_actions
    def record(values,*args,**kwargs):
        captured.append(values.clone())
        return original(values,*args,**kwargs)
    monkeypatch.setattr(module,'_category_truncate_actions',record)
    options=dict(recent_history=[((1,),Fraction(0))],beat=Fraction(1,4),
                 repetition_guard=policy,head_metadata=metadata)
    selected=module._sample_actions(rows,atoms,grammar,1.,torch.Generator().manual_seed(5),**options)
    expected=-2. if penalized else 1.
    assert captured[1][1]==expected
    captured.clear()
    score_options={k:v for k,v in options.items() if k!='head_metadata'}
    module._action_sequence_log_probability(['2','1'],rows,atoms,grammar,1.,
        slot_mask_decoder=lambda slot,prefix,mask:rows[slot],head_metadata=metadata,**score_options)
    assert captured[1][1]==expected
