"""Fixed CPU oracle; the saved pre-change module also supports later GPU A/B."""
import copy
import torch
import pytest

from test_composer_stream_model import fixture


@pytest.mark.parametrize("valid_phases", [1, 2, 8])
def test_sparse_valid_phase_forward_loss_and_parameter_gradients_match_dense(valid_phases):
    model, batch = fixture()
    batch["target_phase_mask"][0, 0] = torch.arange(8) < valid_phases
    old = copy.deepcopy(model)
    packed = copy.deepcopy(batch)
    packed['_detail_phase_groups'] = []
    for shape, indices in batch['_detail_groups']:
        rows, phases = batch['target_phase_mask'].flatten(0, 1)[indices].nonzero(as_tuple=True)
        packed['_detail_phase_groups'].append((shape, indices, rows, phases))
    before, after = old(batch), model(packed)
    for key in before:
        torch.testing.assert_close(after[key], before[key], atol=2e-6, rtol=2e-5)
    loss_old, metrics_old = old.loss(before, batch)
    loss_new, metrics_new = model.loss(after, packed)
    for key in metrics_old:
        torch.testing.assert_close(metrics_new[key], metrics_old[key], atol=2e-6, rtol=2e-5)
    loss_old.backward()
    loss_new.backward()
    for (name, parameter), (old_name, original) in zip(model.named_parameters(), old.named_parameters()):
        assert name == old_name
        assert (parameter.grad is None) == (original.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, original.grad, atol=3e-6, rtol=3e-5, msg=name)
