"""Deployed music/type probabilities and unordered literal-set contracts."""
from dataclasses import replace

import pytest
import torch

from maichart.composer.stream_actions import mix_family_route_log_probs
from maichart.composer.stream_literal import literal_identity_support,permute_literal_targets,pack_literal_type_blocks
from maichart.composer.stream_model import StreamComposer
from maichart.composer.stream_model import StreamComposerConfig
from maichart.composer.event_set_codec import EventSetCodec
from maichart.composer.stream_timing import grouped_logsumexp
from maichart.composer.stream_train import _initialize_structured_heads,resolve_learning_policies
from test_stream_draft_integration import build
import test_stream_draft_deployment_alignment as alignment


def test_music_family_mixture_is_deployed_after_mode_and_preserves_within_type_odds():
    raw=torch.tensor([[2.,1.,-.5,1.,-.3]],requires_grad=True)
    families=torch.tensor([1,1,2,8,8])
    audio=torch.randn(1,31,requires_grad=True)
    routes=raw.log_softmax(-1)
    result=mix_family_route_log_probs(routes,audio,families,.25)
    old=grouped_logsumexp(routes,families-1,groups=31)
    supported=torch.isfinite(old)
    expected=.75*old.exp()+.25*audio.masked_fill(~supported,-1e4).softmax(-1)
    actual=grouped_logsumexp(result,families-1,groups=31).exp()
    torch.testing.assert_close(actual,expected,atol=2e-7,rtol=1e-5)
    torch.testing.assert_close(result[:,0]-result[:,1],routes[:,0]-routes[:,1])
    torch.testing.assert_close(result[:,3]-result[:,4],routes[:,3]-routes[:,4])
    torch.testing.assert_close(result.exp().sum(-1),torch.ones(1))
    (-result[:,3]).sum().backward()
    assert raw.grad.abs().sum()>0 and audio.grad.abs().sum()>0
    assert torch.isfinite(raw.grad).all() and torch.isfinite(audio.grad).all()


def test_literal_shuffle_is_local_reproducible_and_never_changes_set_or_stop():
    target=torch.tensor([[9,2,5,0,-1],[3,0,-1,-1,-1]])
    state=torch.random.get_rng_state().clone()
    first=permute_literal_targets(target,torch.Generator().manual_seed(43))
    same=permute_literal_targets(target,torch.Generator().manual_seed(43))
    assert torch.equal(first,same) and torch.equal(torch.random.get_rng_state(),state)
    assert torch.equal(first.sort(-1).values,target.sort(-1).values)
    assert first[0,3]==first[1,1]==0 and first[0,4]==-1


def test_literal_normalizer_uses_only_previous_heads_and_preserves_exception_targets():
    masks=torch.tensor([[False,False,False],[True,False,False],
        [False,True,False],[False,False,True]])
    bits=torch.tensor([0,1,2,4]);powers=torch.tensor([1,2,4])
    target=torch.tensor([[1,2,0]]);prefix=torch.tensor([[3]])
    local=torch.tensor([[0,1,0]]);ids=torch.tensor([1,2,3])
    support,conflicts=literal_identity_support(target,prefix,local,ids,masks,bits,powers)
    assert support[0,0].tolist()==[True,True,False]
    assert support[0,1].tolist()==[False,True,False]
    assert conflicts==0
    repeated=torch.tensor([[1,1,0]]);local=torch.zeros_like(repeated)
    support,conflicts=literal_identity_support(repeated,prefix,local,ids,masks,bits,powers)
    assert support[0,1,0] and conflicts==1


@pytest.mark.parametrize('crossing',[False,True])
@pytest.mark.parametrize('literal',[False,True])
def test_new_music_and_unordered_policy_teacher_generation_alignment(monkeypatch,crossing,literal):
    original=alignment.build
    def new_build(patch):
        ds,old,song,bank=original(patch)
        model=StreamComposer(replace(old.config,event_family_policy='shared_music_mixture_v2',
            literal_set_policy='permuted_unique_v1'),ds.codec)
        model.load_state_dict(old.state_dict(),strict=True)
        return ds,model,song,bank
    monkeypatch.setattr(alignment,'build',new_build)
    if literal:alignment.test_new_contract_literal_touch_and_stop_conditioning_agree(monkeypatch,crossing)
    else:alignment.test_new_contract_teacher_and_real_generation_readouts_agree(monkeypatch,crossing)


def test_actual_literal_training_keeps_global_rng_and_has_finite_supported_loss(monkeypatch):
    ds,old,_,_=alignment._touch_build(monkeypatch)
    model=StreamComposer(replace(old.config,event_family_policy='shared_music_mixture_v2',
        literal_set_policy='permuted_unique_v1'),ds.codec).train()
    event=ds.codec.encode('A1/B2/E3')
    target=torch.tensor([event['details']]);contexts=torch.randn(1,model.config.d_model,requires_grad=True)
    groups=pack_literal_type_blocks(target,model.semantic_atom_families.cpu())
    state=torch.random.get_rng_state().clone()
    a=model.literal_block_nll(contexts,target,groups=groups,permutation_seed=91,return_support=True)
    b=model.literal_block_nll(contexts,target,groups=groups,permutation_seed=91,return_support=True)
    assert torch.equal(torch.random.get_rng_state(),state)
    torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)
    assert torch.isfinite(a[0]).all() and a[-1]==0
    a[0].sum().backward();assert torch.isfinite(contexts.grad).all()


def test_fresh_detail_priors_are_learnable_and_stage_changes_do_not_reset_bias(monkeypatch):
    ds,model,_,_=build(monkeypatch)
    fresh=_initialize_structured_heads(model,ds)
    assert fresh['detail_prior_initialized']
    saved=replace(model.config)
    before={key:value.clone() for key,value in model.state_dict().items()
        if key.startswith(('detail_bias.','literal_head.bias'))}
    continued=_initialize_structured_heads(model,ds,saved,resume=True)
    fine=_initialize_structured_heads(model,ds,saved,resume=False)
    assert not continued['detail_prior_initialized'] and not fine['detail_prior_initialized']
    for key,value in before.items():torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)


def test_frequency_prior_keeps_rare_long_hold_legal_without_uniform_initial_odds():
    codec=EventSetCodec.fit(['1h[4:1]','1h[2:23]','B1','E2'])
    model=StreamComposer(StreamComposerConfig(d_model=32,nhead=4,layers=1,dim_feedforward=64,
        event_family_policy='semantic_mass_v1',plan_kind_policy='off',
        detail_representation_policy='typed_components_v2'),codec)
    counts=[[[0.]*size for size in sizes] for sizes in codec.detail_sizes]
    common=codec.encode('1h[4:1]');rare=codec.encode('1h[2:23]')
    counts[common['shape']][0][common['details'][0]]=100.
    atoms=[0.]*codec.num_atoms;atoms[codec._atoms['B1']]=100.;atoms[codec._atoms['E2']]=1.
    model.initialize_detail_prior(counts,atoms)
    prior=model.detail_bias[common['shape']][0].softmax(-1)
    assert prior[common['details'][0]]>prior[rare['details'][0]]>0
    assert model.detail_bias[common['shape']][0].requires_grad
    assert model.literal_head.bias[codec._atoms['B1']]>model.literal_head.bias[codec._atoms['E2']]
