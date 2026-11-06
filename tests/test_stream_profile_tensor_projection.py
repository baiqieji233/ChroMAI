"""Projection parity and deployment controls; run after coordinated changes."""
import itertools
import numpy as np
import pytest
import torch

from maichart.composer.stream_song_profile import (SongProfileCompleter,
    feasible_profile,feasible_profile_tensor)


def _cases():
    rng=np.random.default_rng(31);values=[];masks=[]
    rows=np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,1],[0,0,0,1,1]])
    for pattern in itertools.product((False,True),repeat=4):
        for radar in itertools.product((False,True),repeat=2):
            truth=rng.dirichlet(np.ones(5));marginals=rows@truth
            value=rng.uniform(.01,.99,16);known=np.zeros(16,dtype=bool);known[0]=True
            known[8:12]=pattern;value[8:12]=np.where(pattern,marginals,value[8:12])
            known[4:6]=radar;value[4:6]=np.where(radar,truth[1]*.7,value[4:6])
            known[3]=True;value[3]=.2;value[7]=value[12]=.35
            known[7]=known[12]=True
            values.append(value);masks.append(known)
    return np.asarray(values),np.asarray(masks)


def test_all_lock_patterns_match_existing_active_set_projection():
    values,known=_cases()
    expected=np.stack([feasible_profile(value,mask) for value,mask in zip(values,known)])
    actual=feasible_profile_tensor(torch.tensor(values),torch.tensor(known)).numpy()
    np.testing.assert_allclose(actual,expected,atol=2e-7,rtol=0)
    np.testing.assert_array_equal(actual[known],values[known])


def test_completion_uses_identical_projection_and_chord_prediction_to_sampling():
    torch.manual_seed(23);model=SongProfileCompleter(8,components=4)
    values,known=_cases();condition=torch.tensor(values[:4],dtype=torch.float32)
    mask=torch.tensor(known[:4]);audio=torch.randn(4,8);choice=torch.arange(4)
    evidence=model.unified_evidence(audio,condition,known_mask=mask)
    actual=model.execution_controls(evidence,condition,choice)
    expected=model.sample(audio,condition,mask,component_choice=choice)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)


def test_unlocked_chord_uses_the_canonical_radar_distribution():
    torch.manual_seed(24);model=SongProfileCompleter(8,components=2)
    condition=torch.full((2,16),.4);mask=torch.zeros_like(condition,dtype=torch.bool);mask[:,0]=True
    audio=torch.randn(2,8);choice=torch.tensor([0,1])
    evidence=model.unified_evidence(audio,condition,known_mask=mask)
    torch.testing.assert_close(model.execution_controls(evidence,condition,choice),
        model.sample(audio,condition,mask,component_choice=choice),atol=0,rtol=0)


def test_infeasible_locks_are_rejected_by_both_paths():
    condition=torch.full((1,16),.5,dtype=torch.float64)
    mask=torch.ones_like(condition,dtype=torch.bool)
    condition[:,8:12]=1.
    with pytest.raises(ValueError,match='Contradictory note-type'):
        feasible_profile_tensor(condition,mask)
    with pytest.raises(ValueError,match='Contradictory note-type'):
        feasible_profile(condition[0].numpy(),mask[0].numpy())


def test_dtype_cast_does_not_quantize_the_convex_projection_tables():
    values,known=_cases();condition=torch.tensor(values[:4],dtype=torch.bfloat16)
    mask=torch.tensor(known[:4]);model=SongProfileCompleter(8).bfloat16()
    expected=feasible_profile_tensor(condition,mask)
    torch.testing.assert_close(model.project_controls(condition,mask),expected,atol=0,rtol=0)
