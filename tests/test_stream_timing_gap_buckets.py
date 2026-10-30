from fractions import Fraction as Q
from types import SimpleNamespace

import torch

from maichart.composer.stream_model import StreamComposer, StreamComposerConfig
from maichart.composer.stream_timing import (
    AbsoluteTimingHead, GRID_FAMILIES, absolute_target_indices,
    grouped_logsumexp, phase_group, temper_grid_logits, timing_group,
)


def head():
    return AbsoluteTimingHead(8,('0','0','1/2','1','2'),('1/4','3/4','1/3'),group_policy='phase_gap_v2')


def test_exact_one_beat_moves_keep_the_long_bucket_for_nonbinary_phases():
    module=AbsoluteTimingHead(8,('0','1','2'),('1/3','2/3','1/6','5/6'),group_policy='phase_gap_v2')
    cursors=torch.tensor([[float(phase)+origin for origin in (0,10,500)
                           for phase in (Q(1,3),Q(2,3),Q(1,6),Q(5,6))]],dtype=torch.float64)
    beats,gaps=module.candidates(cursors)
    target=cursors+1
    indices=absolute_target_indices(beats,target,torch.ones_like(cursors,dtype=torch.bool))
    groups=module.candidate_groups(gaps).gather(-1,indices[...,None]).squeeze(-1)
    expected=torch.tensor([[phase_group(phase)+11 for _ in (0,10,500)
                            for phase in (Q(1,3),Q(2,3),Q(1,6),Q(5,6))]])
    torch.testing.assert_close(groups,expected)


def test_short_entry_to_integer_does_not_share_the_integer_long_jump_bucket():
    module=head()
    cursor=torch.tensor([[.75,1.]],dtype=torch.float64)
    beats,gaps=module.candidates(cursor)
    groups=module.candidate_groups(gaps)
    indices=absolute_target_indices(beats,torch.tensor([[1.,2.]],dtype=torch.float64),torch.ones(1,2,dtype=torch.bool))
    chosen=groups.gather(-1,indices[...,None]).squeeze(-1)
    assert chosen.tolist()==[[0,len(GRID_FAMILIES)]]
    assert timing_group(Q(0),Q(1,4))==0
    assert timing_group(Q(0),Q(1))==len(GRID_FAMILIES)
    assert timing_group(Q(0),Q(1),'phase_only_v1')==0
    # Both the reserved PAD and real initial-zero columns belong to the short
    # integer bucket; their legality remains a separate support mask.
    assert not groups[...,:2].count_nonzero()
    assert module.num_groups==22


def test_dynamic_temperature_uses_the_final_inference_row_and_22_masses():
    module=head()
    cursor=torch.tensor([[.75,1.]],dtype=torch.float64)
    beats,gaps=module.candidates(cursor)
    groups=module.candidate_groups(gaps)
    allowed=gaps.gt(0);allowed[...,0]=False
    values=torch.tensor([0.,0.,.5,1.,2.],dtype=torch.float64)
    logits,_,_=module(torch.zeros(1,2,8),cursor,torch.zeros(1,2,5),values,
        torch.zeros(1,2,32),torch.zeros_like(beats),allowed)
    scores=logits[0,-1].masked_fill(~allowed[0,-1],-torch.inf)
    sampled=temper_grid_logits(scores,.65,groups,group_count=22)
    masses=grouped_logsumexp(sampled,groups[0,-1],groups=22)
    original=grouped_logsumexp(scores,groups[0,-1],groups=22)
    expected=(original/.65).log_softmax(-1)
    valid=torch.isfinite(expected)
    torch.testing.assert_close(masses[valid],expected[valid])
    assert not torch.isfinite(masses[0])  # No short move to integer from integer.
    assert torch.isfinite(masses[11])


def test_interval_count_is_learned_in_both_candidate_and_macro_scores():
    module=head()
    cursor=torch.tensor([[0.]],dtype=torch.float64)
    beats,gaps=module.candidates(cursor)
    groups=module.candidate_groups(gaps)
    allowed=gaps.gt(0);allowed[...,0]=False
    values=torch.tensor([0.,0.,.5,1.,2.],dtype=torch.float64)
    expected=(gaps.float()*4).requires_grad_()
    logits,_,_=module(torch.zeros(1,1,8),cursor,torch.zeros(1,1,5),values,
        torch.zeros(1,1,32),torch.zeros_like(beats),allowed,candidate_expected_count=expected)
    index=absolute_target_indices(beats,torch.tensor([[.25]],dtype=torch.float64),torch.ones(1,1,dtype=torch.bool))
    loss=-logits.gather(-1,index[...,None]).mean()
    loss.backward()
    assert module.interval_candidate_projection.weight.grad.abs().sum()>0
    assert module.interval_group_projection.weight.grad.abs().sum()>0
    assert expected.grad is None  # Time loss cannot rewrite its own count evidence.
    assert torch.isfinite(module.interval_candidate_projection.weight.grad).all()
    assert torch.isfinite(module.interval_group_projection.weight.grad).all()
    assert groups.gather(-1,index[...,None]).item()==phase_group(Q(1,4))


def test_plan_expected_counts_reuse_prefix_and_align_batch_and_tail_queries():
    # Two conditions share one music clock but have different learned rates.
    cache=dict(seconds=torch.tensor([[0.,.5,1.,1.5]],dtype=torch.float64),
        beats=torch.tensor([[0.,1.,2.,3.]],dtype=torch.float64),mask=torch.ones(1,4,dtype=torch.bool))
    inverse=torch.tensor([0,0])
    logits=torch.zeros(2,4,3)
    logits[0,:,2]=torch.log(torch.expm1(torch.tensor(4.)))
    logits[1,:,2]=torch.log(torch.expm1(torch.tensor(8.)))
    batch=dict(step_second=torch.tensor([[1.25],[1.25]],dtype=torch.float64),
        audio_end_sec=torch.tensor([2.,2.]),audio_end_beat=torch.tensor([4.,4.]))
    state=StreamComposer._plan_integral_state(logits,cache,inverse,batch['audio_end_sec'])
    subject=SimpleNamespace(_read_audio=StreamComposer._read_audio)
    candidates=torch.tensor([[[2.5,3.,3.5,4.]],[[2.5,3.,3.5,4.]]],dtype=torch.float64)
    counts=StreamComposer._plan_candidate_expected_counts(subject,state,cache,batch,inverse,candidates)
    torch.testing.assert_close(counts,torch.tensor([[[0.,1.,2.,3.]],[[0.,2.,4.,6.]]]))
    assert not counts.requires_grad


def test_legacy_group_policy_does_not_gain_new_parameters_on_load():
    assert StreamComposerConfig.from_dict({}).timing_group_policy=='phase_only_v1'
    legacy=AbsoluteTimingHead(8,('0','1','2'))
    assert legacy.num_groups==11 and not hasattr(legacy,'bucket_head')
    assert not hasattr(legacy,'interval_candidate_projection')
