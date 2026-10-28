from types import SimpleNamespace

import pytest
import torch

from maichart.composer.recovery import recovery_episode_state
from maichart.composer.train import (_assert_model_finite, _forward_finite_flags,
                                    _model_finite_items, _window_update_health)


def test_device_flags_accept_large_finite_values_without_l2_overflow():
    loss=torch.tensor(1.,requires_grad=True)
    flags=_forward_finite_flags({'large':torch.full((8,),1e38),'empty':torch.empty(0),
                                 'metadata':'ignored'},loss,loss/3)
    assert flags.tolist()==[True,True,True]
    (loss/3).backward()
    torch.testing.assert_close(loss.grad,torch.tensor(1/3))


@pytest.mark.parametrize('bad_microbatch', [0, 1, 2])
def test_unused_nonfinite_output_poisoning_survives_finite_later_microbatches(bad_microbatch):
    parameter=torch.nn.Parameter(torch.tensor(1.))
    optimizer=torch.optim.SGD([parameter],lr=.1)
    flags=None
    for index in range(3):
        loss=parameter.square()/3
        output={'used':parameter,'unused':torch.tensor(float('nan') if index==bad_microbatch else 1.)}
        current=_forward_finite_flags(output,loss,loss)
        assert not current.requires_grad
        flags=current if flags is None else flags & current
        loss.backward()
    norm=torch.nn.utils.clip_grad_norm_([parameter],1.)
    health=_window_update_health(flags,norm,SimpleNamespace())
    assert health==[False,True,True,True]
    if all(health):optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert parameter.item()==1. and parameter.grad is None


@pytest.mark.parametrize('reason', ['output','loss','backward','gradient'])
def test_rejected_window_restores_recovery_episode_snapshot(reason):
    model=SimpleNamespace(_recovery_episodes=[{'sample_index':0,'events':[('0','1')]}])
    before=recovery_episode_state(model)
    model._recovery_episodes[0]['events'].append(('1','2'))
    flags=torch.ones(3,dtype=torch.bool)
    if reason!='gradient':flags[['output','loss','backward'].index(reason)]=False
    norm=torch.tensor(float('inf') if reason=='gradient' else 1.)
    assert not all(_window_update_health(flags,norm,model,before))
    assert recovery_episode_state(model)==before
    model._recovery_episodes[0]['events'].append(('2','3'))
    assert len(before['episodes'][0]['events'])==1


def test_healthy_window_keeps_new_recovery_episode_state():
    model=SimpleNamespace(_recovery_episodes=[])
    before=recovery_episode_state(model)
    model._recovery_episodes=[{'sample_index':1,'events':[]}]
    assert all(_window_update_health(torch.ones(3,dtype=torch.bool),torch.tensor(1.),model,before))
    assert model._recovery_episodes[0]['sample_index']==1


def test_post_update_overflow_is_detected_by_cached_parameters_and_buffers():
    model=torch.nn.Linear(1,1,bias=False)
    model.register_buffer('running',torch.ones(1))
    items=_model_finite_items(model)
    _assert_model_finite(model,items)
    # Finite gradients do not imply a finite update from an extreme optimizer state/lr.
    model.weight.data.fill_(1.)
    model.weight.grad=torch.full_like(model.weight,2.)
    optimizer=torch.optim.SGD(model.parameters(),lr=3e38)
    optimizer.step()
    with pytest.raises(FloatingPointError,match='weight'):_assert_model_finite(model,items)
    model.weight.data.fill_(1.)
    model.running.fill_(float('inf'))
    with pytest.raises(FloatingPointError,match='running'):_assert_model_finite(model,items)
    with pytest.raises(FloatingPointError,match='running'):_assert_model_finite(model)
