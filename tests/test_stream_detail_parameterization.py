"""Exact migration and differentiable gauge fixing for detail mode offsets."""
from dataclasses import replace
import pytest
import torch
from torch.nn import functional as F
from maichart.composer.stream_detail_modes import ModeDetailControl
from maichart.composer.stream_model import StreamComposerConfig,StreamComposer
from maichart.composer.stream_train import upgrade_learning_stage
from maichart.composer.event_set_codec import EventSetCodec


def test_detail_mode_config_defaults_new_but_missing_checkpoint_field_stays_legacy():
    assert StreamComposerConfig().detail_mode_parameterization=='centered_v2'
    assert StreamComposerConfig.from_dict({}).detail_mode_parameterization=='legacy_v1'
    assert StreamComposerConfig.from_dict({'detail_mode_parameterization':'centered_v2'}).detail_mode_parameterization=='centered_v2'
    with pytest.raises(ValueError,match='parameterization'):
        StreamComposerConfig(detail_mode_parameterization='unknown')


def test_centered_modes_keep_contrasts_have_zero_mean_and_project_gradients():
    torch.manual_seed(14);control=ModeDetailControl(4,8,3)
    with torch.no_grad():control.choices.normal_(0,2);control.stop.normal_(0,2)
    candidates=torch.randn(7,8,requires_grad=True)
    before=control.bias(candidates);stop=control.stop_bias(2)
    assert torch.allclose(before.mean(0),torch.zeros(7),atol=1e-6)
    assert torch.allclose(stop.mean(),torch.tensor(0.),atol=1e-6)
    raw=control.choices@control.features(candidates).T
    assert torch.allclose(before[0]-before[1],raw[0]-raw[1],atol=2e-6)
    with torch.no_grad():control.choices.add_(torch.randn(1,3)*3);control.stop.add_(torch.randn(1,5)*3)
    assert torch.allclose(control.bias(candidates),before,atol=2e-6)
    assert torch.allclose(control.stop_bias(2),stop,atol=2e-6)
    loss=control.bias(candidates)[0].square().sum()+control.stop_bias(2)[0].square()
    loss.backward()
    assert torch.allclose(control.choices.grad.sum(0),torch.zeros(3),atol=2e-5)
    assert torch.allclose(control.stop.grad.sum(0),torch.zeros(5),atol=2e-6)
    assert candidates.grad.abs().sum()>0 and control.features.weight.grad.abs().sum()>0


def test_legacy_control_retains_original_uncentered_function_exactly():
    control=ModeDetailControl(4,8,3,parameterization='legacy_v1')
    with torch.no_grad():control.choices.normal_();control.stop.normal_()
    candidates=torch.randn(7,8)
    assert torch.equal(control.bias(candidates),control.choices@control.features(candidates).T)
    assert torch.equal(control.stop_bias(2),control.stop[:,2])


def _all_logits(model):
    bank=model.shared_details.encode_catalog()
    regular={}
    vector=torch.linspace(-.8,.9,32)
    for shape,slots in enumerate(model.codec.detail_sizes):
        for slot in range(len(slots)):
            weights=model.shared_details.candidate_weights(shape,slot,bank)
            base=F.linear(vector,weights,model.detail_bias[shape][slot])
            regular[shape,slot]=base[None]+model.detail_mode_control.bias(weights)
    hidden=torch.linspace(-.6,.7,model.config.d_model)
    literal=F.linear(hidden,model.literal_catalog_weights(bank),model.literal_head.bias)[None]
    literal=literal+model.literal_mode_bias(torch.arange(model.codec.num_atoms),detail_bank=bank)
    stop=F.linear(hidden,model.literal_stop_head.weight,model.literal_stop_head.bias)[None]
    stop=stop+torch.stack([model.detail_mode_control.stop_bias(i) for i in range(5)],-1)
    return regular,literal,stop


def test_new_stage_moves_common_bias_and_preserves_every_regular_literal_and_stop_mode_logit(monkeypatch):
    from test_stream_phrase_execution import execution_fixture
    _,baseline,_,_=execution_fixture(monkeypatch)
    codec=EventSetCodec.fit(['1','2/6','3h[4:1]','4/7h[8:1]','A1','A2h[4:1]','1>5[8:1]'])
    config=replace(baseline.config,phrase_execution_policy='component_modes_v2',
        detail_representation_policy='typed_components_v2',detail_mode_parameterization='legacy_v1')
    old=StreamComposer(config,codec)
    with torch.no_grad():old.detail_mode_control.choices.normal_(.8,.5);old.detail_mode_control.stop.normal_(.4,.3)
    before=_all_logits(old)
    source_bias=old.literal_head.bias.detach().clone()
    source_choices=old.detail_mode_control.choices.detach().clone()
    upgraded=upgrade_learning_stage(old,{})
    assert upgraded.config.detail_mode_parameterization=='centered_v2'
    after=_all_logits(upgraded)
    for key,value in before[0].items():assert torch.allclose(value,after[0][key],rtol=2e-6,atol=2e-6),key
    assert torch.allclose(before[1],after[1],rtol=2e-6,atol=2e-6)
    assert torch.allclose(before[2],after[2],rtol=2e-6,atol=2e-6)
    assert torch.allclose(upgraded.detail_mode_control.choices.mean(0),torch.zeros(8),atol=1e-7)
    assert torch.allclose(upgraded.detail_mode_control.stop.mean(0),torch.zeros(5),atol=1e-7)
    assert torch.equal(old.detail_mode_control.choices,source_choices)
    unsupported=~old.shared_details.literal_supported
    assert unsupported.any()
    assert torch.equal(upgraded.literal_head.bias[unsupported],source_bias[unsupported])
    legacy=upgrade_learning_stage(old,{'detail_mode_parameterization':'legacy_v1'})
    assert legacy.config.detail_mode_parameterization=='legacy_v1'
    assert all(torch.equal(value,legacy.state_dict()[key]) for key,value in old.state_dict().items())
