"""Semantic and failure-boundary regressions for the training efficiency audit."""
import copy
from fractions import Fraction

import pytest
import torch

from maichart.event_timing import EventPoint
from maichart.form.codec import RationalCodec
from maichart.form.model import FormConfig, FormTransformer
from maichart.form.training_batch import add_memory_packing, transfer_layout_batch
from maichart.form.training_runtime import validate_cpu_batch
from maichart.form.rhythm_geometry import numeric_timing_loss
from maichart.form.repetition_penalty import (
    compute_pattern_diversity_loss, apply_repetition_penalty_sampling,
)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('profile', ['none', 'exact_fraction_v1', 'exact_fraction_v2'])
@pytest.mark.parametrize('quality', ['legacy', 'musical_evidence_v1'])
@pytest.mark.parametrize('curve_mask_dims', [2, 3])
def test_packing_preserves_logits_auxiliaries_and_gradients(profile, quality, curve_mask_dims):
    from test_form_model import make_batch
    torch.manual_seed(891)
    config = FormConfig(vocab_size=19, d_model=32, nhead=4, layers=2,
        dim_feedforward=64, dropout=0, audio_dim=6, descriptor_dim=5, condition_dim=3,
        max_tokens=16, max_audio_tokens=8, max_memory_tokens=16,
        rhythm_geometry_profile=profile, quality_profile=quality)
    dense = FormTransformer(config).train()
    compact = copy.deepcopy(dense)
    batch = make_batch()
    batch['task'][1] = 1
    batch['audio_padding_mask'][1, 2:] = True
    batch['memory_strength'][0] = 0
    batch['draft'] = torch.tensor([[0]*7, [1, 2, 3, 4, 0, 0, 0]])
    batch['right'] = torch.tensor([[0]*8, [3, 4, 0, 0, 0, 0, 0, 0]])
    batch['skeleton'] = torch.tensor([[0, 0], [3, 0]])
    batch['action_state_padding_mask'][:] = True
    batch['load_curve'] = torch.rand(2, 4, 4)
    batch['load_curve_padding_mask'] = torch.tensor([[False, False, True, True], [False, False, False, True]])
    visible = torch.tensor([[True, False, True, False], [True, True, False, True]])
    batch['load_curve_condition_mask'] = (~visible if curve_mask_dims == 2
                                         else visible[..., None].expand(-1, -1, 4))
    if quality != 'legacy':
        batch['span_bounds'] = torch.rand(2, 2)
    if profile != 'none':
        for branch in ('audio', 'history', 'motif', 'draft', 'right', 'load_curve'):
            batch[branch+'_geometry'] = torch.randn(*batch[branch].shape[:2], config.geometry_dim)
        batch['decoder_geometry'] = torch.randn(*batch['decoder_input'].shape, config.geometry_dim)
    before = torch.get_rng_state().clone()
    packed = add_memory_packing(dict(batch))
    assert torch.equal(before, torch.get_rng_state())
    assert packed['memory_packing']['packed_tokens'] < packed['memory_packing']['dense_tokens']
    a, b = dense(batch), compact(packed)
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=1e-5, atol=2e-6)
    for output in (a, b):
        (output['logits'].square().mean() + output['relation_logits'].square().mean()).backward()
    for (name, expected), (_, actual) in zip(dense.named_parameters(), compact.named_parameters()):
        if expected.grad is None:
            assert actual.grad is None, name
        else:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-4, atol=2e-6, msg=name)


def test_all_masked_memories_keep_three_global_tokens_and_relation_position():
    from test_form_model import make_model, make_batch
    model, batch = make_model(), make_batch(1)
    for name in ('audio', 'form', 'action_state'):
        batch[name+'_padding_mask'] = torch.ones(batch[name].shape[:2], dtype=torch.bool)
    for name in ('history', 'motif', 'draft', 'right'):
        batch[name].zero_()
    with torch.no_grad():
        expected = model(batch)
        packed = add_memory_packing(dict(batch))
        assert packed['memory_gather_index'].shape == (1, 3)
        actual = model(packed)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=2e-6)


def test_packing_mismatch_fails_before_attention():
    from test_form_model import make_model, make_batch
    batch = add_memory_packing(make_batch())
    batch['memory_source_length'] += 1
    with pytest.raises(ValueError, match='branch layout'):
        make_model()(batch)


def test_cpu_checks_reject_nonfinite_invalid_tokens_and_leaked_future():
    from test_form_model import make_batch
    batch = make_batch()
    validate_cpu_batch(batch, vocab_size=19)
    broken = copy.deepcopy(batch)
    broken['audio'][0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite training input: audio'):
        validate_cpu_batch(broken, vocab_size=19)
    broken = copy.deepcopy(batch)
    broken['history'][0, 0] = 19
    with pytest.raises(ValueError, match='outside vocabulary'):
        validate_cpu_batch(broken, vocab_size=19)
    broken = copy.deepcopy(batch)
    broken['right'][0, 0] = 3
    with pytest.raises(ValueError, match='draft task input'):
        validate_cpu_batch(broken, vocab_size=19)


def test_numeric_loss_ignores_masked_nan_and_preserves_valid_gradient():
    prediction = torch.tensor([[[.5, .2, .8], [float('nan')]*3]], requires_grad=True)
    labels = torch.tensor([[[.3, 0., 1.], [float('nan')]*3]])
    mask = torch.tensor([[True, False]])
    result = numeric_timing_loss(prediction, labels, mask)
    expected = torch.nn.functional.smooth_l1_loss(prediction[:, :1], labels[:, :1])
    torch.testing.assert_close(result, expected)
    result.backward()
    assert torch.isfinite(prediction.grad).all()
    assert not prediction.grad[:, 1:].any()
    empty = numeric_timing_loss(prediction, labels, torch.zeros_like(mask))
    assert empty == 0


def test_diversity_padding_nan_cannot_contaminate_valid_rows():
    hidden = torch.randn(2, 24, 8)
    mask = torch.arange(24)[None] < torch.tensor([16, 0])[:, None]
    expected = compute_pattern_diversity_loss(hidden[:1, :16])
    hidden[~mask] = float('nan')
    hidden.requires_grad_(True)
    result = compute_pattern_diversity_loss(hidden, valid_mask=mask, reduction='none')
    torch.testing.assert_close(result[0], expected)
    assert result[1] == 0
    result.sum().backward()
    assert torch.isfinite(hidden.grad).all()
    assert not hidden.grad[~mask].any()


def test_repetition_penalty_never_increases_negative_logit():
    logits = torch.tensor([-2., 2., 0., -1.])
    actual = apply_repetition_penalty_sampling(logits, [0, 1, 0, 1, 2, 2], penalty_strength=1.2)
    assert actual[0] < logits[0] and actual[1] < logits[1]
    assert actual[2] == 0 and actual[3] == logits[3]


def test_complete_configuration_metric_detects_one_wrong_path_character():
    from maichart.form.layout_metrics import layout_diagnostics
    codec = RationalCodec()
    tokens = codec.encode([EventPoint(Fraction(0), '1-5[8:1]'), EventPoint(Fraction(1), '2/7')])
    target = torch.tensor([tokens[1:]])
    prefix = torch.tensor([tokens[:-1]])
    logits = torch.full((1, target.size(1), len(codec)), -10.)
    logits.scatter_(2, target[..., None], 10.)
    wrong = int((target[0] == codec.ids['N:5']).nonzero()[0])
    logits[0, wrong, codec.ids['N:5']] = -10.
    logits[0, wrong, codec.ids['N:4']] = 10.
    ce = torch.nn.functional.cross_entropy(logits.transpose(1, 2), target, reduction='none')
    values, counts = layout_diagnostics(logits, target, prefix, torch.tensor([0]), ce)
    assert counts['draft_configuration_exact_accuracy'] == 2
    assert values['draft_configuration_exact_accuracy'] == .5
    assert values['draft_note_configuration_accuracy'] > .9
    assert values['draft_position_fraction_exact_accuracy'] == 1


def test_cpu_metadata_is_not_copied_by_layout_transfer():
    batch = {'audio': torch.randn(1, 2, 6), 'audio_times_sec': torch.tensor([[.1, .2]])}
    moved = transfer_layout_batch(batch, 'cpu')
    assert moved['audio_times_sec'] is batch['audio_times_sec']
