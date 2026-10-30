"""Decision paths and terminal likelihood for pattern-aware recovery learning."""
import pytest
import torch

from test_composer_stream_model import fixture
from maichart.composer.stream_model import NOTE,WAIT,EOS,StreamComposer


def test_pattern_state_has_direct_gradient_to_time_kind_and_configuration():
    model,batch=fixture()
    features=torch.ones(1,4,16)
    batch['continuation_features']=features
    result=model(batch)
    loss,_=model.loss(result,batch)
    loss.backward()
    for name in ('kind_continuation_projection','action_continuation_projection'):
        gradient=getattr(model,name).weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum()>0
    hidden=torch.zeros(1,1,model.config.d_model)
    previous=dict(history_delta=torch.tensor([[2]]),history_kind=torch.tensor([[NOTE]]),
                  continuation_features=torch.ones(1,1,16))
    allowed=torch.tensor([[[False,True,True,True,False]]])
    scores,_=model.repeat_delta_log_probs(hidden,previous,torch.zeros(1,1,5),allowed,{},torch.tensor([0]),symbolic_only=True)
    (-scores[0,0,3]).backward()
    assert model.gap_continuation_projection.weight.grad.abs().sum()>0


def test_motion_context_has_no_separate_additive_reward_and_learns_through_main_head():
    model,batch=fixture()
    assert model.config.motion_continuation_policy=='context_motion_v3'
    context=torch.randn(1,16,requires_grad=True)
    shape=torch.tensor([1]);reference=torch.tensor([0]);mask=torch.zeros(1,8,dtype=torch.bool);mask[:,0]=True
    motion=torch.zeros(1,3,8);motion[:,1,7]=1
    base=model.event_phase_logits(context,shape,reference,mask,motion)
    with torch.no_grad():
        model.motion_head.weight.fill_(1000);model.motion_head.bias.fill_(1000)
    torch.testing.assert_close(base,model.event_phase_logits(context,shape,reference,mask,motion))
    (-base.log_softmax(-1)[0,3]).backward()
    gradient=model.motion_context_projection.weight.grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
    assert model.motion_head.weight.grad is None


@pytest.mark.parametrize('middle',[WAIT,NOTE])
def test_motion_context_breaks_across_wait_or_literal_touch(middle):
    kinds=torch.tensor([[NOTE,NOTE,middle,NOTE]])
    shapes=torch.tensor([[1,1,0,1]])
    masks=torch.nn.functional.one_hot(torch.tensor([[0,7,0,0]]),8).bool()
    features=StreamComposer.motion_phase_features(kinds,shapes,masks,distinct=True,adjacent=True)
    assert features[0,1].sum()>0
    assert features[0,2:].sum()==0


def test_wait_and_eos_exit_supervise_only_defined_kind_likelihood():
    model,batch=fixture();out=model(batch)
    batch['transition_exit_mask']=batch['target_kind'].ne(NOTE)
    batch['transition_exit_labels']=batch['transition_exit_mask']
    total,metrics=model.loss(out,batch)
    with torch.no_grad():
        ce=torch.nn.functional.cross_entropy(out['kind_logits'].flatten(0,1),batch['target_kind'].flatten(),reduction='none')
        expected=(ce*batch['transition_exit_mask'].flatten()).sum()*.5/4
    torch.testing.assert_close(metrics['transition_exit_loss'],expected)
    assert metrics['exit_wait_count']==1 and metrics['exit_eos_count']==1
    assert torch.isfinite(total)


def test_incompatible_feature_width_is_never_silently_padded_or_truncated():
    model,batch=fixture();batch['continuation_features']=torch.zeros(1,4,8)
    with pytest.raises(ValueError,match='Continuation features'):
        model(batch)
