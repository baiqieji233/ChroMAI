"""Multi-slot geometry contracts; execute after the implementation is complete."""
import copy

import pytest
import torch
from torch.nn import functional as F

from maichart.composer.model import ComposerConfig, GROUNDED_PROFILE, build_model


ATOMS = ['1', '2', '4', '5', '7', '8', '1h[4:1]', '5h[4:1]',
         '1-5[8:1]', '1-7[8:1]', '2-6[8:1]', '4-8[4:1]', 'A1', 'B6', 'C', 'E7']


def _model(head='conserved_grouped_v2', cross='relative_heads_v1'):
    return build_model(ComposerConfig(
        atom_vocab_size=len(ATOMS)+1, d_model=16, nhead=4, layers=1,
        dim_feedforward=32, dropout=0., history_events=8, action_slots=3,
        architecture_profile=GROUNDED_PROFILE, head_timing_dim=99,
        lens_width=8, lens_chunk_size=3, cross_event_relation_policy=cross,
        configuration_quality_policy='cooperative_v1', action_head_policy=head), ATOMS).eval()


def _id(atom):
    return ATOMS.index(atom)+1


def _inputs():
    targets = torch.tensor([[_id('1-5[8:1]'), _id('2-6[8:1]'), 0],
                            [_id('1'), _id('4-8[4:1]'), 0], [0, 0, 0]])
    previous = torch.tensor([[1, 0, 0, 0, 1, 0, 0, 0],
                             [0, 1, 0, 0, 0, 1, 0, 0], [0]*8], dtype=torch.bool)
    return torch.randn(3, 16), torch.randn(3, 16), previous, targets


def _activate(model):
    with torch.no_grad():
        for parameter in model.cooperative_geometry.parameters():
            parameter.normal_(std=.2)


@pytest.mark.parametrize('head', ['flat_v1', 'conserved_grouped_v2'])
def test_neutral_geometry_preserves_existing_decode_and_first_slot(head):
    torch.manual_seed(391)
    model = _model(head)
    old = copy.deepcopy(model)
    old.cooperative_geometry = None
    context, audio, previous, targets = _inputs()
    current = model.decode_candidate(context, audio_context=audio,
                                    previous_heads=previous, action_targets=targets)
    baseline = old.decode_candidate(context, audio_context=audio,
                                   previous_heads=previous, action_targets=targets)
    for key in ('action_logits', 'audio_action_logits'):
        torch.testing.assert_close(current[key], baseline[key], rtol=0, atol=0)
    _activate(model)
    # Even a trained later-slot branch must not rewrite the old first slot.
    torch.testing.assert_close(model.action_slot(context[0], 0, [], previous_heads=previous[0]),
                               old.action_slot(context[0], 0, [], previous_heads=previous[0]),
                               rtol=0, atol=0)


@pytest.mark.parametrize('head', ['flat_v1', 'conserved_grouped_v2'])
@pytest.mark.parametrize('cross', ['off', 'relative_heads_v1'])
def test_dense_packed_incremental_logits_and_gradients_match(head, cross):
    torch.manual_seed(392)
    model = _model(head, cross)
    _activate(model)
    packed_model = copy.deepcopy(model)
    context, audio, previous, targets = _inputs()
    dense = model.decode_candidate(context, audio_context=audio,
                                   previous_heads=previous, action_targets=targets)
    packed = packed_model.decode_candidate(context, audio_context=audio,
        previous_heads=previous, action_targets=targets, packed_actions=True)
    indices = packed['action_slot_indices']
    assert indices.tolist() == [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0]]
    rows, slots = indices.unbind(-1)
    for key in ('action_logits', 'audio_action_logits'):
        torch.testing.assert_close(packed[key], dense[key][rows, slots], atol=3e-6, rtol=3e-5)
    for row, slot in indices.tolist():
        chosen = targets[row, :slot].tolist()
        current = model.action_slot(context[row], slot, chosen, previous_heads=previous[row])
        audio_only = model.action_slot(audio[row], slot, chosen)
        torch.testing.assert_close(current, dense['action_logits'][row, slot], atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(audio_only, dense['audio_action_logits'][row, slot], atol=3e-6, rtol=3e-5)
    losses = [sum(F.cross_entropy(result[key] if result is packed else result[key][rows, slots],
                                 targets[rows, slots]) for key in ('action_logits', 'audio_action_logits'))
              for result in (dense, packed)]
    for loss in losses:
        loss.backward()
    left = {k: p.grad for k, p in model.named_parameters() if p.grad is not None}
    right = {k: p.grad for k, p in packed_model.named_parameters() if p.grad is not None}
    assert left.keys() == right.keys()
    for name in left:
        torch.testing.assert_close(left[name], right[name], atol=3e-6, rtol=3e-4, msg=name)
    assert left['cooperative_geometry.later_cross_factors.weight'].abs().sum() > 0
    assert left['cooperative_geometry.end_to_head_factors.weight'].abs().sum() > 0
    assert left['cooperative_geometry.end_to_end_factors.weight'].abs().sum() > 0
    assert left['cooperative_geometry.head_to_end_factors.weight'].abs().sum() > 0


def test_geometry_preserves_button_type_masses_and_touch_stop_logits():
    torch.manual_seed(393)
    model = _model()
    _activate(model)
    geometry = model.cooperative_geometry
    logits = torch.randn(2, len(ATOMS)+1)
    context = torch.randn(2, 16)
    previous = torch.eye(8)[[0, 4]]
    ends = geometry.prefix_end_lanes[torch.tensor([_id('1-5[8:1]'), _id('4-8[4:1]')])]
    actual = geometry(logits, context, previous_heads=previous,
                      prefix_heads=previous.roll(1,-1), prefix_ends=ends, slots=1)
    protected = torch.tensor([0, _id('A1'), _id('B6'), _id('C'), _id('E7')])
    assert torch.equal(actual[:, protected], logits[:, protected])
    for kind in range(3):
        ids = getattr(geometry, f'kind_{kind}')
        torch.testing.assert_close(actual[:, ids].logsumexp(-1), logits[:, ids].logsumexp(-1))
    assert not torch.allclose(actual, logits)


def test_terminal_geometry_uses_selected_slide_end_and_keeps_audio_history_free():
    torch.manual_seed(394)
    model = _model()
    geometry = model.cooperative_geometry
    with torch.no_grad():
        geometry.end_to_head_factors.weight.normal_()
        geometry.end_to_end_factors.weight.normal_()
    context, audio, previous, targets = _inputs()
    logits = torch.randn(len(ATOMS)+1)
    end5 = geometry.prefix_end_lanes[_id('1-5[8:1]')]
    end7 = geometry.prefix_end_lanes[_id('1-7[8:1]')]
    first = geometry(logits, context[0], prefix_ends=end5, slots=1)
    second = geometry(logits, context[0], prefix_ends=end7, slots=1)
    assert not torch.allclose(first, second)
    assert not geometry.prefix_end_lanes[_id('1')].any()
    assert not geometry.prefix_end_lanes[_id('1h[4:1]')].any()
    _activate(model)
    original = model.decode_candidate(context, audio_context=audio,
        previous_heads=previous, action_targets=targets)
    changed = model.decode_candidate(context, audio_context=audio,
        previous_heads=previous.roll(3, -1), action_targets=targets)
    torch.testing.assert_close(original['audio_action_logits'], changed['audio_action_logits'], rtol=0, atol=0)
    assert not torch.allclose(original['action_logits'][0, 1], changed['action_logits'][0, 1])


def test_future_slots_do_not_leak_into_prefix_or_activate_past_stop():
    torch.manual_seed(395)
    model = _model()
    _activate(model)
    context, audio, previous, targets = _inputs()
    first = model.decode_candidate(context, previous_heads=previous, action_targets=targets)
    changed = targets.clone()
    changed[:, 1:] = torch.tensor([_id('1-7[8:1]'), _id('8')])
    second = model.decode_candidate(context, previous_heads=previous, action_targets=changed)
    torch.testing.assert_close(first['action_logits'][:, :2], second['action_logits'][:, :2])
    # Geometry itself is neutral once an earlier STOP has ended the event.
    geometry = model.cooperative_geometry
    logits = torch.randn(2, len(ATOMS)+1)
    actual = geometry(logits, context[:2], previous_heads=previous[:2],
        prefix_heads=previous[:2],prefix_ends=geometry.prefix_end_lanes[targets[:2, 0]], slots=2,
        active=torch.zeros(2, dtype=torch.bool))
    torch.testing.assert_close(actual, logits, rtol=0, atol=0)


def test_all_candidate_geometry_broadcasts_same_as_selected_decode():
    torch.manual_seed(396)
    model = _model()
    _activate(model)
    context, _, previous, targets = _inputs()
    candidates = torch.stack((context, context+.2), dim=1)
    all_targets = targets[:, None, :].expand(-1, 2, -1)
    all_result = model.decode_candidate(candidates, previous_heads=previous, action_targets=all_targets)
    chosen = torch.tensor([0, 1, 0])
    selected = model.decode_candidate(candidates, chosen,
        previous_heads=previous, action_targets=targets)
    torch.testing.assert_close(selected['action_logits'],
        all_result['action_logits'][torch.arange(3), chosen], atol=3e-6, rtol=3e-5)


def test_tap_first_prefix_learns_slide_destination_without_changing_type_mass():
    """Same Slide head, different ends: prior Tap position must be observable."""
    torch.manual_seed(397)
    model=_model()
    geometry=model.cooperative_geometry
    context=torch.ones(16)
    logits=torch.randn(len(ATOMS)+1)
    heads=model.action_factors.atom_head_lanes[_id('1')]
    assert not geometry.prefix_end_lanes[_id('1')].any()
    before=geometry(logits,context,prefix_heads=heads,slots=1)
    torch.testing.assert_close(before,logits,rtol=0,atol=0)
    # The zero-initialized factor is directly trainable, with no zero gate.
    (-before.log_softmax(-1)[_id('1-5[8:1]')]).backward()
    assert geometry.head_to_end_factors.weight.grad.abs().sum()>0
    with torch.no_grad():
        geometry.head_to_end_factors.weight.zero_()
        geometry.head_to_end_factors.weight[4].fill_(.25)
    actual=geometry(logits,context,prefix_heads=heads,slots=1)
    slide_ids=geometry.kind_2
    torch.testing.assert_close(actual[slide_ids].logsumexp(-1),logits[slide_ids].logsumexp(-1))
    protected=torch.ones(len(ATOMS)+1,dtype=torch.bool);protected[slide_ids]=False
    assert torch.equal(actual[protected],logits[protected])
    relative_gain=(actual[_id('1-5[8:1]')]-actual[_id('1-7[8:1]')])-(
        logits[_id('1-5[8:1]')]-logits[_id('1-7[8:1]')])
    torch.testing.assert_close(relative_gain,torch.tensor(4.))
    changed=geometry(logits,context,prefix_heads=heads.roll(1),slots=1)
    assert not torch.allclose(actual[slide_ids],changed[slide_ids])
