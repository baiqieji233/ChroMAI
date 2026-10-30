import torch

from maichart.composer.model import ComposerConfig, GROUNDED_PROFILE, build_model


ATOMS = ['1', '2', '3h[4:1]', '4h[4:1]', '1-5[4:1]', '2-6[4:1]', 'A1']


def make_model(policy):
    config = ComposerConfig(atom_vocab_size=len(ATOMS)+1, d_model=16, nhead=4,
        layers=1, dim_feedforward=32, dropout=0., history_events=8, action_slots=3,
        architecture_profile=GROUNDED_PROFILE, head_timing_dim=99, lens_width=8,
        configuration_quality_policy=policy)
    return build_model(config, ATOMS).eval()


def test_large_same_event_bias_preserves_stop_touch_and_type_mass():
    model = make_model('cooperative_v2')
    with torch.no_grad():
        model.action_factors.relative_factors.weight.fill_(5.)
    context = torch.ones(16)
    prefix = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.])
    base = torch.zeros(len(ATOMS)+1)
    delta = model.action_factors.relative_logits(context, prefix)
    assert delta[1] == 80
    old = (base + delta).softmax(-1)
    assert old[0] < 1e-30 and old[-1] < 1e-30
    result = model._same_event_logits(base, context, prefix)
    probability = result.softmax(-1)
    torch.testing.assert_close(probability[[0, -1]], base.softmax(-1)[[0, -1]])
    for kind in range(3):
        ids = getattr(model.cooperative_geometry, f'kind_{kind}')
        torch.testing.assert_close(result[ids].logsumexp(-1), base[ids].logsumexp(-1))


def test_same_event_movement_keeps_learnable_gradient():
    model = make_model('cooperative_v2')
    base = torch.zeros(len(ATOMS)+1)
    prefix = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.])
    context = torch.ones(16)
    result = model._same_event_logits(base, context, prefix)
    result[1].backward()
    assert model.action_factors.relative_factors.weight.grad.abs().sum() > 0


def test_dense_packed_and_generation_match_for_v2():
    torch.manual_seed(42)
    model = make_model('cooperative_v2')
    with torch.no_grad():
        model.action_factors.relative_factors.weight.normal_(0, .3)
        for table in (model.cooperative_geometry.later_cross_factors,
                      model.cooperative_geometry.end_to_head_factors,
                      model.cooperative_geometry.end_to_end_factors,
                      model.cooperative_geometry.head_to_end_factors):
            table.weight.normal_(0, .1)
    context = torch.randn(2, 16)
    audio = torch.randn(2, 16)
    targets = torch.tensor([[1, 2, 0], [5, 3, 0]])
    previous = torch.tensor([[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0]], dtype=torch.bool)
    dense = model.decode_candidate(context, action_targets=targets, audio_context=audio,
        project_audio_actions=True, previous_heads=previous)
    packed = model.decode_candidate(context, action_targets=targets, audio_context=audio,
        project_audio_actions=True, previous_heads=previous, packed_actions=True)
    rows, slots = packed['action_slot_indices'].unbind(-1)
    for key in ('action_logits', 'audio_action_logits'):
        torch.testing.assert_close(packed[key], dense[key][rows, slots])
    for row in range(2):
        for slot in range(3):
            prefix = targets[row, :slot].tolist()
            actual = model.action_slot(context[row], slot, prefix, previous_heads=previous[row])
            torch.testing.assert_close(actual, dense['action_logits'][row, slot])
            audio_actual = model.action_slot(audio[row], slot, prefix)
            torch.testing.assert_close(audio_actual, dense['audio_action_logits'][row, slot])


def test_v1_and_off_retain_raw_same_event_addition():
    context = torch.randn(2, 16)
    prefix = torch.tensor([[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0]])
    base = torch.randn(2, len(ATOMS)+1)
    for policy in ('off', 'cooperative_v1'):
        model = make_model(policy)
        expected = base + model.action_factors.relative_logits(context, prefix)
        torch.testing.assert_close(model._same_event_logits(base, context, prefix), expected, rtol=0, atol=0)
