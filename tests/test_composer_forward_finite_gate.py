import pytest
import torch
from maichart.composer.train import _forward_finite_check


def test_forward_finite_check_marks_unused_nonfinite_and_ignores_other_values():
    output = {
        'used': torch.tensor([1.0, 2.0]),
        'unused_nan': torch.tensor([float('nan')]),
        'unused_inf': torch.tensor([float('inf')]),
        'bool': torch.tensor([True, False]),
        'integer': torch.tensor([1, 2]),
        'empty': torch.empty(0),
        'metadata': 'ignored',
    }
    snapshots = {key: value.clone() for key, value in output.items() if torch.is_tensor(value)}
    loss = torch.tensor(2.0, requires_grad=True)
    backward_loss = torch.tensor(1.0, requires_grad=True)
    rng = torch.random.get_rng_state().clone()

    result = _forward_finite_check(output, loss, backward_loss)

    assert result == (['unused_nan', 'unused_inf'], True, True)
    assert torch.equal(torch.random.get_rng_state(), rng)
    for key, before in snapshots.items():
        torch.testing.assert_close(output[key], before, equal_nan=True)
    assert loss.grad is None
    assert backward_loss.grad is None


@pytest.mark.parametrize('which,expected', [
    ('loss', ([], False, True)),
    ('backward_loss', ([], True, False)),
])
def test_forward_finite_check_reports_nonfinite_losses_independently(which, expected):
    output = {'finite': torch.tensor([1.0])}
    loss = torch.tensor(float('inf') if which == 'loss' else 1.0)
    backward_loss = torch.tensor(float('nan') if which == 'backward_loss' else 1.0)

    assert _forward_finite_check(output, loss, backward_loss) == expected


def test_forward_finite_check_preserves_normal_backward_gradients_and_rng():
    checked = torch.nn.Linear(3, 2)
    unchecked = torch.nn.Linear(3, 2)
    unchecked.load_state_dict(checked.state_dict())
    x = torch.tensor([[1.0, -2.0, 0.5], [0.25, 3.0, -1.0]])
    checked_loss = checked(x).square().sum()
    unchecked_loss = unchecked(x).square().sum()
    rng = torch.random.get_rng_state().clone()

    assert _forward_finite_check({'prediction': checked(x)}, checked_loss, checked_loss) == ([], True, True)
    assert torch.equal(torch.random.get_rng_state(), rng)
    checked_loss.backward()
    unchecked_loss.backward()

    for checked_parameter, unchecked_parameter in zip(checked.parameters(), unchecked.parameters()):
        assert torch.equal(checked_parameter.grad, unchecked_parameter.grad)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_gpu_gate_reports_mixed_device_failures_and_preserves_healthy_gradients():
    value=torch.tensor([1.,2.],device='cuda',requires_grad=True)
    loss=value.square().mean()
    assert _forward_finite_check({'output':value},loss,loss/2)==([],True,True)
    bad={'output':value,'unused_cuda':torch.tensor([float('nan')],device='cuda'),
         'unused_cpu':torch.tensor([float('inf')])}
    assert _forward_finite_check(bad,loss,loss/2)==(['unused_cuda','unused_cpu'],True,True)
    assert _forward_finite_check({'output':value},loss,torch.tensor(float('inf'),device='cuda'))==([],True,False)
    loss.backward()
    torch.testing.assert_close(value.grad,torch.tensor([1.,2.],device='cuda'))
