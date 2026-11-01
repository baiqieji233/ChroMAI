import copy

import torch

from test_composer_stream_model import fixture


def test_time_only_recovery_cannot_train_any_configuration_head_through_exit_loss():
    model, batch = fixture(event_family_policy='shared_continuation_v1')
    batch['configuration_mask'] = torch.zeros_like(batch['loss_mask'], dtype=torch.bool)
    batch['transition_exit_mask'] = torch.ones_like(batch['target_kind'], dtype=torch.bool)
    batch['transition_exit_labels'] = torch.ones_like(batch['target_kind'], dtype=torch.bool)
    output = model(batch)
    loss, metrics = model.loss(output, batch)
    for name in ('configuration_loss', 'event_family_full_loss', 'event_family_audio_loss',
                 'shape_loss', 'phase_loss', 'detail_loss', 'literal_stop_loss'):
        assert metrics[name].item() == 0, name
    torch.testing.assert_close(metrics['transition_exit_loss'],
                               model.config.transition_exit_weight*metrics['time_loss'])
    assert metrics['timing_only_target_count'].item() == 2
    assert metrics['configuration_target_count'].item() == 0
    loss.backward()
    for name in ('event_family_head', 'audio_event_family_head', 'shape_head', 'phase_head',
                 'literal_head', 'literal_stop_head'):
        head = getattr(model, name)
        for parameter in head.parameters():
            assert parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0, name
    assert model.kind_head.weight.grad.abs().sum().item() > 0


def test_configuration_mask_preserves_all_authored_losses_exactly():
    model, batch = fixture(event_family_policy='shared_continuation_v1')
    output = model(batch)
    expected, metrics = model.loss(output, batch)
    explicit = dict(batch, configuration_mask=torch.ones_like(batch['loss_mask'], dtype=torch.bool))
    actual, observed = model.loss(output, explicit)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for name in ('time_loss', 'configuration_loss', 'plan_loss', 'detail_loss'):
        torch.testing.assert_close(observed[name], metrics[name], atol=0, rtol=0)


def test_uncertain_configuration_labels_do_not_change_timing_only_loss():
    model, batch = fixture(event_family_policy='shared_continuation_v1')
    batch['configuration_mask'] = torch.zeros_like(batch['loss_mask'], dtype=torch.bool)
    batch['_detail_groups'] = []
    batch['_literal_indices'] = torch.empty(0, dtype=torch.long)
    before, _ = model.loss(model(batch), batch)
    changed = copy.deepcopy(batch)
    changed['target_shape'].zero_()
    changed['target_phase_mask'].fill_(True)
    changed['target_details'].fill_(-1)
    after, _ = model.loss(model(changed), changed)
    torch.testing.assert_close(before, after, atol=0, rtol=0)
