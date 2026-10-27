"""Regression tests for independently learned macro groups and actual sampling."""
import pytest
import torch
from torch.nn import functional as F

from maichart.composer.action_factors import CompositionalActions
from maichart.composer.checkpoint import migrate_audio_grounded
from maichart.composer.conserved_actions import ConservedGroupedActionHead, build_macro_groups, group_logsumexp
from maichart.composer.generate import _conserved_action_probabilities, _sample_actions
from maichart.composer.model import build_model
from maichart.generation.event_cli import _NoteGrammar
from test_composer_audio_grounded_v3 import inputs
from test_composer_horizon_and_conserved_actions import ATOMS, make_config


def small_head(dtype=torch.float32):
    factors=CompositionalActions(ATOMS,16)
    return ConservedGroupedActionHead(factors,16).to(dtype=dtype)


def test_touch_macro_groups_include_actual_pad_and_all_compound_heads():
    atoms=['1','1b','18','1/8','A1','A1f','A2','E1','E1f','E2',
           'A1h[4:1]','A1h[2:1]','A2h[4:1]','A1/A2']
    factors=CompositionalActions(atoms,8)
    groups,count=build_macro_groups(factors)
    gid={atom:int(groups[i+1]) for i,atom in enumerate(atoms)}
    assert int(groups[0])==0
    assert gid['1']==gid['1b']
    assert gid['18']==gid['1/8']
    assert gid['A1']==gid['A1f']
    assert gid['E1']==gid['E1f']
    assert len({gid[x] for x in ('A1','A2','E1','E2')})==4
    assert gid['A1h[4:1]']==gid['A1h[2:1]']
    assert gid['A1h[4:1]']!=gid['A2h[4:1]']
    assert gid['A1/A2'] not in {gid['A1'],gid['A2']}
    assert count==len(set(groups.tolist()))


def test_tap_target_has_no_gradient_to_other_buckets_atom_classifiers():
    torch.manual_seed(101)
    head=small_head()
    base=torch.randn(2,len(ATOMS)+1,requires_grad=True)
    context=torch.randn(2,16)
    loss=F.cross_entropy(head(base,context),torch.tensor([1,2]))
    loss.backward()
    tap_group=head.atom_group_id[1]
    assert torch.count_nonzero(base.grad[:,head.atom_group_id==tap_group])>0
    torch.testing.assert_close(base.grad[:,head.atom_group_id!=tap_group],
        torch.zeros_like(base.grad[:,head.atom_group_id!=tap_group]),atol=1e-7,rtol=0)
    assert head.macro_head.weight.grad.abs().sum()>0


def test_compositional_macro_factors_are_shared_across_groups_and_receive_slot0_gradients():
    torch.manual_seed(106)
    model=build_model(make_config(action_head_policy='conserved_grouped_v2'),ATOMS)
    ctx=torch.randn(2,16,requires_grad=True)
    dec=model.decode_candidate(ctx,action_targets=torch.tensor([[1],[12]]))
    loss=F.cross_entropy(dec['action_logits'][:,0],torch.tensor([1,12]))
    loss.backward()
    for name in ('kind_mask','head_lane_mask','touch_ring'):
        grad=model.action_factors.factors[name].weight.grad
        assert grad is not None and float(grad.abs().sum())>1e-3, f'{name} was canceled out of macro head'
    with torch.no_grad():
        before=model.action_slot(ctx[0],0,[]).clone()
        model.action_factors.factors['kind_mask'].weight[1].add_(5.0)
        after=model.action_slot(ctx[0],0,[]).clone()
        assert float((after-before).abs().max())>0.5


def test_group_quality_is_independent_of_flat_logit_group_offset():
    torch.manual_seed(102)
    head=small_head()
    base=torch.randn(2,len(ATOMS)+1)
    context=torch.randn(2,16)
    shifted=base+head.atom_group_id.float()*30
    expected=group_logsumexp(head(base,context),head.atom_group_id,head.num_groups)
    actual=group_logsumexp(head(shifted,context),head.atom_group_id,head.num_groups)
    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=0)


def test_bfloat16_head_keeps_float32_normalized_group_probabilities():
    torch.manual_seed(5)
    head=small_head(torch.bfloat16)
    with torch.no_grad():
        for embedding in (head.group_traj_gate,head.shape_detail,head.endpoint_detail,head.break_detail,
                          head.group_dur_gate,head.shape_dur_gate,head.end_dur_gate,
                          head.hold_dur_detail,head.slide_dur_detail):
            embedding.weight.normal_(0,.8)
    base=(torch.randn(4,len(ATOMS)+1)*4).to(torch.bfloat16)
    context=torch.randn(4,16).to(torch.bfloat16)
    actual=head(base,context)
    assert actual.dtype==torch.float32
    expected=head.macro_head(head.macro_norm(context)).float().log_softmax(-1)
    torch.testing.assert_close(group_logsumexp(actual,head.atom_group_id,head.num_groups),
        expected,atol=1e-5,rtol=0)


def test_group_logsumexp_preserves_empty_and_all_masked_groups():
    values=torch.tensor([float('-inf'),-3.,1.],requires_grad=True)
    actual=group_logsumexp(values,torch.tensor([0,1,1]),3)
    assert torch.isneginf(actual[[0,2]]).all()
    torch.testing.assert_close(actual[1],torch.logsumexp(values[1:],0))
    actual[1].backward()
    assert torch.isfinite(values.grad).all()


@pytest.mark.parametrize('temperature',[.2,.5,.85,1.,2.])
def test_sampling_temperature_preserves_different_branch_count_bucket_masses(temperature):
    logits=torch.tensor([float('-inf'),.5,.125,.125,.125,.125])
    logits[1:]=logits[1:].log()
    ids=torch.tensor([0,1,2,2,2,2])
    actual=_conserved_action_probabilities(logits,temperature,ids)
    torch.testing.assert_close(actual[1],torch.tensor(.5))
    torch.testing.assert_close(actual[2:].sum(),torch.tensor(.5))


def test_sampling_bucket_mass_survives_tail_truncation_and_legal_filter():
    group_a=torch.tensor([.3,.2])
    group_b=torch.linspace(1.,.001,98);group_b=group_b/group_b.sum()*.5
    logits=torch.cat((torch.tensor([float('-inf')]),group_a.log(),group_b.log()))
    ids=torch.tensor([0,1,1]+[2]*98)
    actual=_conserved_action_probabilities(logits,.4,ids)
    torch.testing.assert_close(actual[1:3].sum(),torch.tensor(.5))
    torch.testing.assert_close(actual[3:].sum(),torch.tensor(.5))
    assert torch.count_nonzero(actual[3:])<=16
    logits[1]=float('-inf')
    actual=_conserved_action_probabilities(logits,.4,ids)
    assert actual[1]==0
    torch.testing.assert_close(actual[2],torch.tensor(2/7))
    torch.testing.assert_close(actual[3:].sum(),torch.tensor(5/7))


def test_type_control_preserves_stop_and_relative_touch_ring_and_pad_mass():
    probabilities=torch.tensor([.1,.2,.1,.2,.06,.04,.18,.12])
    ids=torch.arange(8)
    groups=tuple(torch.tensor([False]+[i==category for i in (0,1,2,3,3,7,7)])
                 for category in range(8))
    actual=_conserved_action_probabilities(probabilities.log(),.3,ids,groups,type_temperature=.5)
    torch.testing.assert_close(actual[0],probabilities[0])
    torch.testing.assert_close(actual[4:6].sum()/actual[6:].sum(),torch.tensor(1/3))
    torch.testing.assert_close(actual[4]/actual[5],torch.tensor(1.5))
    torch.testing.assert_close(actual[4:].sum(),torch.tensor(.576))


def test_zero_temperature_and_actual_sampler_choose_macro_then_atom():
    atoms=['1','2-6[8:1]','2-6[4:1]','2<6[8:1]','2>6[8:1]']
    grammar=_NoteGrammar.from_atoms(atoms)
    ids=torch.tensor([0,1,2,2,2,2])
    logits=torch.tensor([[float('-inf'),.4,.15,.15,.15,.15]])
    logits[:,1:]=logits[:,1:].log()
    assert _sample_actions(logits,atoms,grammar,0.,None)==['1']
    actual=_sample_actions(logits,atoms,grammar,0.,None,macro_group_ids=ids)
    assert actual[0] in atoms[1:]


def test_grouped_migration_requires_learning_independent_macro_head():
    torch.manual_seed(104)
    original=build_model(make_config(),ATOMS).eval()
    upgraded=migrate_audio_grounded(original,ATOMS,action_head_policy='conserved_grouped_v2').eval()
    batch=inputs();indices=torch.tensor([1,2])
    with torch.no_grad():
        before=original(**batch,candidate_indices=indices)['action_logits']
        after=upgraded(**batch,candidate_indices=indices)['action_logits']
    assert not torch.allclose(before.softmax(-1),after.softmax(-1))
    head=upgraded.conserved_action_head
    masses=group_logsumexp(after,head.atom_group_id,head.num_groups)
    torch.testing.assert_close(masses,masses.new_full(masses.shape,-torch.log(torch.tensor(float(head.num_groups)))),
        atol=1e-6,rtol=0)


def test_grouped_head_retains_same_event_and_cross_event_geometry_and_decode_parity():
    torch.manual_seed(105)
    model=build_model(make_config(action_head_policy='conserved_grouped_v2',
        cross_event_relation_policy='relative_heads_v1'),ATOMS).eval()
    context=torch.randn(16);previous=torch.tensor([1.,0.,0.,0.,0.,0.,0.,0.])
    same_before=model.action_slot(context,1,[1])
    cross_before=model.action_slot(context,0,[],previous_heads=previous)
    with torch.no_grad():
        model.action_factors.relative_factors.weight.normal_(0,.25)
        model.cross_event_relative_factors.weight.normal_(0,.25)
    same_after=model.action_slot(context,1,[1])
    cross_after=model.action_slot(context,0,[],previous_heads=previous)
    assert not torch.allclose(same_before.softmax(-1),same_after.softmax(-1))
    assert not torch.allclose(cross_before.softmax(-1),cross_after.softmax(-1))
    batch=inputs();indices=torch.tensor([1,2]);targets=torch.tensor([[1,4,0],[4,1,0]])
    with torch.no_grad():
        output=model(**batch,candidate_indices=indices,action_targets=targets)
        for row in range(2):
            for slot in range(3):
                prefix=targets[row,:slot].tolist()
                actual=model.action_slot(output['action_context'][row],slot,prefix,
                    previous_heads=output['previous_heads'][row])
                reference=model.action_slot(output['audio_action_context'][row],slot,prefix)
                torch.testing.assert_close(actual,output['action_logits'][row,slot],atol=1e-5,rtol=1e-5)
                torch.testing.assert_close(reference,output['audio_action_logits'][row,slot],atol=1e-5,rtol=1e-5)
