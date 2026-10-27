from collections import Counter
from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch

from maichart.composer.checkpoint import migrate_audio_grounded
from maichart.composer.events import (event_metadata, head_pad_indices_for_policy,
    merge_separate_head_timing)
from maichart.composer.generate import _sample_actions
from maichart.composer.guard import first_rejected_prefix
from maichart.composer.model import ComposerConfig, GROUNDED_PROFILE, build_model
from test_composer_audio_grounded_v3 import inputs


ATOMS = ['1', '2', '3h[4:1]', '4h[4:1]', '5-8[4:1]', '6-8[4:1]', 'A1', 'B2', 'C']


def config(**overrides):
    values = dict(atom_vocab_size=len(ATOMS)+1, d_model=16, nhead=4, layers=1,
        dim_feedforward=32, dropout=0., history_events=8, action_slots=3,
        architecture_profile=GROUNDED_PROFILE, head_timing_dim=99, lens_width=8,
        lens_chunk_size=3)
    values.update(overrides)
    return ComposerConfig(**values)


def test_separate_timing_distinguishes_button_and_touch_and_merges_legacy():
    button = head_pad_indices_for_policy(event_metadata('1'), 'separate_pads_v2')[0]
    touch = head_pad_indices_for_policy(event_metadata('A1'), 'separate_pads_v2')[0]
    assert button != touch
    assert head_pad_indices_for_policy(event_metadata('1')) == head_pad_indices_for_policy(event_metadata('A1'))
    timing = torch.zeros(1, 123)
    timing[0, button], timing[0, touch] = 1, 1
    timing[0, 41+button], timing[0, 41+touch] = 2, 1
    timing[0, 82+button], timing[0, 82+touch] = 4, 3
    old = merge_separate_head_timing(timing)
    assert old.shape == (1, 99)
    assert old[0, 0] == 1 and old[0, 33] == 1 and old[0, 66] == 3
    timing[0, touch] = 0
    old = merge_separate_head_timing(timing)
    assert old[0, 33] == 2 and old[0, 66] == 4


def test_migration_initially_preserves_outputs_with_equivalent_timing():
    torch.manual_seed(11)
    old = build_model(config(), ATOMS).eval()
    new = migrate_audio_grounded(old, ATOMS, head_timing_policy='separate_pads_v2',
        cross_event_relation_policy='relative_heads_v1').eval()
    assert torch.count_nonzero(new.head_timing_detail_projection.weight) == 0
    assert torch.count_nonzero(new.cross_event_relative_factors.weight) == 0
    data = inputs()
    timing = torch.zeros(2, 6, 123)
    timing[..., :8] = 1
    timing[..., 41:49] = torch.rand(2, 6, 8)
    timing[..., 82:90] = torch.rand(2, 6, 8)
    new_data = {**data, 'head_timing': timing}
    old_data = {**data, 'head_timing': merge_separate_head_timing(timing)}
    with torch.no_grad():
        a = old(**old_data, candidate_indices=torch.tensor([1, 2]))
        b = new(**new_data, candidate_indices=torch.tensor([1, 2]))
    for key in ('candidate_logits', 'action_logits', 'future', 'eos_logits'):
        torch.testing.assert_close(a[key], b[key], rtol=1e-5, atol=1e-6)


def test_cross_event_first_slot_gradient_direction_and_category_mass():
    torch.manual_seed(12)
    model = build_model(config(cross_event_relation_policy='relative_heads_v1'), ATOMS)
    context = torch.randn(1, 16)
    heads = torch.zeros(1, 8); heads[0, 0] = 1
    result = model.decode_candidate(context, previous_heads=heads)
    result['action_logits'][0, 0, 1].backward()
    assert model.cross_event_relative_factors.weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.cross_event_relative_factors.weight.normal_()
        logits = torch.randn(1, len(ATOMS)+1)
        one = model._cross_event_logits(logits, context, heads)
        other_heads = torch.roll(heads, 1, -1)
        two = model._cross_event_logits(logits, context, other_heads)
    assert not torch.allclose(one, two)
    for kind in range(3):
        ids = getattr(model, f'cross_event_kind_{kind}')
        torch.testing.assert_close(logits[:, ids].logsumexp(-1), one[:, ids].logsumexp(-1))
    torch.testing.assert_close(one[:, [0, 7, 8, 9]], logits[:, [0, 7, 8, 9]])


def test_decode_requires_batch_matched_heads_and_symbolic_dropout_hides_relation():
    model = build_model(config(cross_event_relation_policy='relative_heads_v1'), ATOMS).eval()
    context = torch.randn(2, 16)
    with pytest.raises(ValueError, match='previous_heads'):
        model.decode_candidate(context)
    with pytest.raises(ValueError, match='previous_heads'):
        model.decode_candidate(context, previous_heads=torch.zeros(1, 8))
    with torch.no_grad():
        model.cross_event_relative_factors.weight.normal_()
        data = inputs()
        changed = {**data, 'history': torch.flip(data['history'], [0, 1])}
        a = model(**data, candidate_indices=torch.tensor([1, 2]), symbolic_keep=torch.zeros(2, 1))
        b = model(**changed, candidate_indices=torch.tensor([1, 2]), symbolic_keep=torch.zeros(2, 1))
    assert torch.count_nonzero(a['previous_heads']) == 0
    torch.testing.assert_close(a['action_logits'], b['action_logits'])


class PrefixGuard:
    def __init__(self, rejected):
        self.rejected = rejected
        self.checks, self.rejections = 5, 2
        self.reason_counts = Counter({'original': 2})
        self.last_reason = 'original'

    def accepts(self, event, **kwargs):
        self.checks += 1
        return not any(tuple(event.note_text.split('/'))[:len(p)] == p for p in self.rejected)


def test_rejected_prefix_localization_and_alternative_combination():
    beat = Fraction(1)
    guard = PrefixGuard({('1',)})
    assert first_rejected_prefix(guard, beat, ['1', '2', '3']) == ('1',)
    assert (guard.checks, guard.rejections, guard.reason_counts) == (5, 2, Counter({'original': 2}))
    guard = PrefixGuard({('1', '2')})
    assert first_rejected_prefix(guard, beat, ['1', '2', '3']) == ('1', '2')
    atoms = ['1', '2', '3']
    grammar = SimpleNamespace(heads={a: (a,) for a in atoms})
    logits = torch.tensor([[-10., 9., 8., 7.], [-10., 0., 9., 8.], [10., 0., 0., 0.]])
    chosen = _sample_actions(logits, atoms, grammar, 0., None, blocked_prefixes=[('1', '2')])
    assert chosen == ['1', '3']
