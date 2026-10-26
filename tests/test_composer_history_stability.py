import pytest
import torch

from maichart.composer.checkpoint import load, migrate_audio_grounded, save, validate
from maichart.composer.model import build_model, resolve_composer_policy
from maichart.composer.stability import bounded_history_memory, history_attention_context
from test_composer_audio_grounded_v3 import ATOMS, config, inputs


def test_memory_bound_is_per_event_preserves_small_vectors_and_has_finite_zero_gradients():
    x = torch.tensor([[[0., 0., 0., 0.], [1., -1., 1., -1.], [30., -40., 50., -60.]],
                      [[1e30, -1e30, 1e30, -1e30], [3., 4., 5., 6.], [7., 8., 9., 10.]]], requires_grad=True)
    padding = torch.tensor([[False, False, False], [False, True, True]])
    y = bounded_history_memory(x, padding, 8.)
    assert torch.isfinite(y).all()
    assert (y.square().mean(-1).sqrt() <= 8.00001).all()
    torch.testing.assert_close(y[0, :2], x[0, :2], rtol=0, atol=0)
    assert torch.count_nonzero(y[padding]) == 0
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(x.grad[0, 0], torch.ones(4))
    assert torch.count_nonzero(x.grad[padding]) == 0


@pytest.mark.parametrize('value', [0., .99, 64.1, float('nan'), float('inf'), True])
def test_invalid_memory_bound_rejected(value):
    with pytest.raises(ValueError, match='history_memory_rms_bound'):
        config(history_memory_rms_bound=value)


def test_legacy_checkpoint_missing_stability_fields_keeps_exact_outputs_and_state_keys(tmp_path):
    torch.manual_seed(12)
    parent = build_model(config(), ATOMS).eval()
    path = tmp_path / 'old.pt'
    payload = save(path, parent, None, epoch=1, args={}, manifest='unused', atom_catalog=ATOMS)
    for key in ('history_memory_policy', 'history_memory_rms_bound', 'history_attention_policy'):
        payload['model_config'].pop(key)
    torch.save(payload, path)
    restored, _ = load(path)
    restored.eval()
    assert restored.config.history_memory_policy == 'unbounded_v1'
    assert restored.config.history_attention_policy == 'auto_v1'
    assert restored.state_dict().keys() == parent.state_dict().keys()
    data = inputs()
    with torch.no_grad():
        before, after = parent(**data), restored(**data)
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0, msg=key)


def test_stability_policy_resolution_and_explicit_migration(tmp_path):
    assert resolve_composer_policy('history_memory_policy') == 'rms_cap_v1'
    assert resolve_composer_policy('history_attention_policy') == 'math_v1'
    assert resolve_composer_policy('history_memory_policy', architecture_profile='music_context_v2') == 'unbounded_v1'
    assert resolve_composer_policy('history_attention_policy', architecture_profile='music_context_v2') == 'auto_v1'
    parent = build_model(config(), ATOMS)
    for key, value in [('history_memory_policy', 'rms_cap_v1'), ('history_attention_policy', 'math_v1')]:
        with pytest.raises(ValueError, match='finetune'):
            resolve_composer_policy(key, value, parent.config.to_dict(), resume=True)
    migrated = migrate_audio_grounded(parent, ATOMS, history_memory_policy='rms_cap_v1',
                                     history_memory_rms_bound=12., history_attention_policy='math_v1')
    assert parent.config.history_memory_policy == 'unbounded_v1'
    assert migrated.config.history_memory_policy == 'rms_cap_v1'
    for key, value in parent.state_dict().items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, migrated.state_dict()[key], rtol=0, atol=0)
    path = tmp_path / 'migrated.pt'
    payload = save(path, migrated, None, epoch=0, args={}, manifest='unused', atom_catalog=ATOMS)
    restored, _ = load(path)
    assert restored.config == migrated.config
    assert validate(payload).history_memory_rms_bound == 12.


def test_actual_memory_interface_bounds_valid_vectors_zeros_masked_rows_and_retains_gradients():
    torch.manual_seed(2)
    model = build_model(config(history_memory_policy='rms_cap_v1', history_attention_policy='math_v1'), ATOMS).eval()
    data = inputs()
    data['history_padding_mask'][1] = True  # the dummy-key protection must remain finite
    seen = []
    model.history_encoder.register_forward_hook(lambda _m, _i, out: out * 1000.)
    def inspect(_m, args, kwargs):
        memory = args[1]
        seen.append(memory)
        assert memory.float().square().mean(-1).sqrt().max() <= 8.00001
        assert torch.count_nonzero(memory[1]) == 0
        assert kwargs['key_padding_mask'][1, 0] == False
    model.history_attention.register_forward_pre_hook(inspect, with_kwargs=True)
    out = model(**data, candidate_indices=torch.tensor([1, 2]), action_targets=torch.tensor([[1, 0], [2, 0]]))
    assert seen
    out['action_logits'].square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.history_encoder.layers[0].linear1.weight.grad.abs().sum() > 0


def test_padding_output_growth_cannot_change_visible_legacy_predictions():
    torch.manual_seed(6)
    model = build_model(config(), ATOMS).eval()
    data = inputs()
    data['history_padding_mask'][1, 1] = True
    data['history'][1, 1] = 0
    with torch.no_grad():
        before = model(**data)
        hook = model.history_encoder.register_forward_hook(
            lambda _m, _i, out: out.masked_fill(data['history_padding_mask'][..., None], 1000.))
        try:
            after = model(**data)
        finally:
            hook.remove()
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0, msg=key)


def test_math_context_is_local_and_legacy_context_does_not_change_flags():
    if not hasattr(torch.nn, 'attention'):
        pytest.skip('Modern SDPA context unavailable')
    from torch.nn.attention import SDPBackend, sdpa_kernel
    def flags():
        return (torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled(),
                torch.backends.cuda.math_sdp_enabled(), torch.backends.cuda.cudnn_sdp_enabled())
    initial = flags()
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        outer = flags()
        with pytest.raises(RuntimeError, match='probe failure'):
            with history_attention_context('math_v1', 'cuda'):
                assert flags() == (False, False, True, False)
                raise RuntimeError('probe failure')
        assert flags() == outer
        with history_attention_context('auto_v1', 'cuda'):
            assert flags() == outer
    assert flags() == initial


def test_math_policy_restores_reduced_precision_setting_on_error():
    getter = getattr(torch.backends.cuda, 'fp16_bf16_reduction_math_sdp_allowed', None)
    setter = getattr(torch.backends.cuda, 'allow_fp16_bf16_reduction_math_sdp', None)
    if getter is None or setter is None:
        pytest.skip('Math reduction control unavailable')
    original = getter()
    try:
        setter(True)
        with pytest.raises(RuntimeError):
            with history_attention_context('math_v1', 'cuda'):
                assert getter() is False
                raise RuntimeError('test')
        assert getter() is True
    finally:
        setter(original)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA BF16 numerical regression')
@pytest.mark.parametrize('single_valid_key', [False, True])
def test_math_policy_avoids_spurious_bf16_dropout_gradient_for_one_hot_attention(single_valid_key):
    if not torch.cuda.is_bf16_supported():
        pytest.skip('BF16 unavailable')
    from torch.nn import functional as F
    q = torch.zeros(1, 1, 16, 48, device='cuda', dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    if single_valid_key:
        q.normal_(); k.normal_()
    else:
        q[..., 0] = 32.
        k[..., 0, 0] = 32.
    q.requires_grad_(); k.requires_grad_()
    v = torch.full_like(k, 2., requires_grad=True)
    mask = torch.zeros(1, 1, 1, 16, device='cuda', dtype=q.dtype)
    mask[..., 1 if single_valid_key else 12:] = float('-inf')
    with history_attention_context('math_v1', 'cuda'):
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=.1)
    result.backward(torch.ones_like(result))
    # Scores select the first key exactly at these precisions. For ANY
    # dropout mask, the local Q/K derivative must vanish; comparing seeds
    # across different fused backends would not fix their dropout masks.
    assert q.grad.abs().max() < 1e-6
    assert k.grad.abs().max() < 1e-6
    assert torch.isfinite(v.grad).all()
    assert torch.any(result != 2.)  # dropout was retained, not disabled
