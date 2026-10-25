"""Real Composer migrations and shared causal context, on a tiny CPU model."""
from fractions import Fraction as F

import numpy as np
import torch

from maichart.composer.checkpoint import migrate_audio_grounded
from maichart.composer.context_features import composer_action_state, motif_tensors
from maichart.composer.model import ComposerConfig, GROUNDED_PROFILE, build_model
from maichart.composer.motif_memory import ComposerMotifIndex, MOTIF_POLICY
from maichart.event_timing import EventPoint, EventTempo
from maichart.form.recurrence import Relation, SongRelations
from maichart.form.timing import MetricSpan
from maichart.generation.sustain_state import SustainStateIndex
from maichart.stream.grounded_inference import IncrementalState


ATOMS = ['1','2','1h[1:16]','1-4[8:1]*-6[4:1]','1-4[4:1]*-6[8:1]']


def _model():
    torch.set_num_threads(1)
    torch.manual_seed(731)
    return build_model(ComposerConfig(atom_vocab_size=len(ATOMS)+1,d_model=16,nhead=4,
        layers=1,dim_feedforward=32,dropout=0.,history_events=8,action_slots=3,
        architecture_profile=GROUNDED_PROFILE,head_timing_dim=99,lens_width=8,
        lens_chunk_size=3,history_dropout=0.,ledger_dropout=0.,symbolic_dropout=0.,
        history_anchor_policy='parsed_heads_v2',action_state_policy='committed_legacy_v1',
        slide_factor_policy='marginal_v1',motif_memory_policy='off'),ATOMS)


def _inputs():
    torch.manual_seed(913)
    x = torch.randn(2,6,141)
    x[:,:,120] = torch.tensor([.25,.5,1.,2.,6.,12.])
    return dict(candidate_features=x,history=torch.tensor([[[1,0],[3,0]],[[2,0],[4,0]]]),
        history_features=torch.randn(2,2,8),history_padding_mask=torch.zeros(2,2,dtype=torch.bool),
        condition=torch.randn(2,16),action_state=torch.randn(2,50),head_timing=torch.rand(2,6,99),
        candidate_mask=torch.zeros(2,6,dtype=torch.bool),workload_audio=torch.randn(2,8,104),
        ledger=torch.rand(2,6,6),interval_audio=torch.randn(2,6,16),trailing_audio=torch.randn(2,8,52),
        audio_lens=torch.randn(2,8,31,26),audio_lens_mask=torch.ones(2,8,31,dtype=torch.bool),
        planner_audio=torch.randn(2,4,104),planner_mask=torch.zeros(2,4,dtype=torch.bool),
        planner_indices=torch.tensor([[[0,1],[0,1],[1,2],[1,2],[2,3],[2,3]]]*2),
        planner_mix=torch.full((2,6),.5),candidate_indices=torch.tensor([1,2]),
        action_targets=torch.tensor([[4,1,0],[5,2,0]]))


def _motif():
    return dict(motif_ids=torch.tensor([[[1,0],[4,0]],[[2,0],[5,0]]]),
        motif_relative_beats=torch.tensor([[-1.,0.],[-.5,.25]]),
        motif_mask=torch.ones(2,2,dtype=torch.bool))


def test_real_migration_neutral_outputs_and_new_factor_gradients():
    old = _model().eval()
    with torch.no_grad():
        for factor in old.action_factors.factors.values():
            factor.weight.normal_(0,.05)
        old.action_factors.relative_factors.weight.normal_(0,.03)
    migrated = migrate_audio_grounded(old,ATOMS,motif_memory_policy=MOTIF_POLICY,
                                     slide_factor_policy='branch_joint_v2').eval()
    assert migrated.config.history_anchor_policy == old.config.history_anchor_policy
    assert migrated.config.action_state_policy == old.config.action_state_policy
    data = _inputs()
    with torch.no_grad():
        baseline = old(**data)
        upgraded = migrated(**data,**_motif())
    assert baseline.keys() == upgraded.keys()
    for key in baseline:
        torch.testing.assert_close(baseline[key],upgraded[key],rtol=0,atol=0,msg=key)
    migrated.train()
    output = migrated(**data,**_motif())
    loss = torch.nn.functional.cross_entropy(output['action_logits'].flatten(0,1),data['action_targets'].flatten())
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in migrated.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(),name
    for parameter in (migrated.motif_memory.output.weight,
                      migrated.action_factors.factors['slide_branch_joint'].weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


def test_candidate_state_shape_and_learned_motif_only_change_actions():
    model = migrate_audio_grounded(_model(),ATOMS,motif_memory_policy=MOTIF_POLICY,
                                  slide_factor_policy='branch_joint_v2').eval()
    with torch.no_grad():
        model.motif_memory.output.weight.normal_(0,.3)
    data = _inputs()
    vector_state = data['action_state']
    data['action_state'] = vector_state[:,None].expand(-1,6,-1).clone()
    with torch.no_grad():
        off = model(**data,**_motif(),motif_strength=0.)
        on = model(**data,**_motif(),motif_strength=1.)
        legacy_state = model(**{**data,'action_state':vector_state},**_motif(),motif_strength=0.)
    for key in off:
        # Linear kernels can round differently for [B,D] and [B,C,D] batches.
        torch.testing.assert_close(off[key],legacy_state[key],rtol=1e-5,atol=1e-6,msg=key)
    for key in ('candidate_logits','audio_candidate_logits','bin_logits','eos_logits','future',
                'audio_action_context','audio_action_logits','candidate_context','workload','plan'):
        torch.testing.assert_close(off[key],on[key],rtol=0,atol=0,msg=key)
    assert not torch.equal(off['action_logits'],on['action_logits'])
    assert not torch.equal(off['action_context'],on['action_context'])


def test_symbolic_dropout_hides_the_motif_branch_per_sample_and_its_gradients():
    model=migrate_audio_grounded(_model(),ATOMS,motif_memory_policy=MOTIF_POLICY).train()
    with torch.no_grad():model.motif_memory.output.weight.normal_(0,.3)
    data={**_inputs(),'symbolic_keep':torch.tensor([[0.],[1.]])}
    a=_motif();b={**a,'motif_ids':torch.ones_like(a['motif_ids'])}
    first=model(**data,**a);second=model(**data,**b)
    torch.testing.assert_close(first['action_logits'][0],second['action_logits'][0],rtol=0,atol=0)
    assert not torch.equal(first['action_logits'][1],second['action_logits'][1])
    torch.testing.assert_close(first['candidate_logits'],second['candidate_logits'],rtol=0,atol=0)
    first['action_logits'][0].square().sum().backward()
    for parameter in model.motif_memory.parameters():
        assert parameter.grad is None or not parameter.grad.any()
    model.zero_grad(set_to_none=True)
    second=model(**data,**a)
    second['action_logits'][1].square().sum().backward()
    assert model.motif_memory.output.weight.grad.abs().sum()>0


def test_unmasked_forward_matches_standalone_motif_decode():
    model=migrate_audio_grounded(_model(),ATOMS,motif_memory_policy=MOTIF_POLICY).eval()
    with torch.no_grad():model.motif_memory.output.weight.normal_(0,.3)
    data=_inputs()
    with torch.no_grad():
        output=model(**data,**_motif(),symbolic_keep=torch.ones(2,1))
        base=model(**data,decode_actions=False,symbolic_keep=torch.ones(2,1))
        standalone=model.decode_candidate(base['candidate_context'],data['candidate_indices'],
            action_targets=data['action_targets'],audio_context=base['audio_action_context'],**_motif())
    for key in standalone:torch.testing.assert_close(output[key],standalone[key],rtol=0,atol=0)


def test_shared_context_matches_real_incremental_state_and_ignores_future_suffix():
    tempo = [EventTempo(F(0),137.),EventTempo(F(18),193.)]
    prefix = [EventPoint(F(0),'1h[1:16]'),EventPoint(F(8),'2'),EventPoint(F(20),'1')]
    suffix = [EventPoint(F(36),'1-5[0##1]'),EventPoint(F(38),'2h[1:16]')]
    full = SustainStateIndex(prefix+suffix,tempo,-1.)
    altered = SustainStateIndex(prefix+[EventPoint(F(35),'7h[1:16]')],tempo,-1.)
    incremental = IncrementalState(tempo,-1.)
    for point in prefix:
        incremental.append(point)
    query = [F(33),F(36),F(40)]
    for policy in ('committed_causal_v2','candidate_prefix_v2'):
        expected = composer_action_state(incremental,len(prefix),query,policy)
        for index in (full,altered):
            actual = composer_action_state(index,len(prefix),query,policy)
            torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
    spans = tuple(MetricSpan(i,F(start),F(start+16),4,4,(),F(4),(),False,1.,'test')
                  for i,start in enumerate((0,16,32)))
    relations = SongRelations(spans,np.zeros((3,1)),[Relation(0,2,.99,.9,1.)])
    full_memory = ComposerMotifIndex(relations,[p.beat for p in prefix+suffix],
                                     np.array([[3,0],[2,0],[1,0],[4,0],[2,0]]))
    # Exercise the real append path used during generation, with empty archive.
    incremental_memory = ComposerMotifIndex(relations,[],np.zeros((0,2),np.int64))
    for point,ids in zip(incremental.points,[[3,0],[2,0],[1,0]]):
        incremental_memory.append(point.beat,ids)
    expected = motif_tensors(incremental_memory,F(36),3,1,4)
    assert expected['motif_mask'].any()
    for memory in (full_memory, ComposerMotifIndex(relations,[p.beat for p in prefix]+[F(35)],
                                                  np.array([[3,0],[2,0],[1,0],[5,5]]))):
        actual = motif_tensors(memory,F(36),3,1,4)
        for key in expected:
            torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
