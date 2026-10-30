"""Exact-query sharing must retain labels, multiplicity, gradients and chunk bounds."""
import copy
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from maichart.composer.audio_lens import AudioLens, LensBank
from maichart.composer.acoustics import AcousticContext
from maichart.composer.data import collate_composer
from maichart.composer.device_audio import AudioQueryMaterializer


def context(seed):
    times = np.asarray([0., .071, .2, .47, .71, 1.01], dtype=np.float64)
    values = np.random.default_rng(seed).normal(size=(len(times), 120)).astype(np.float32)
    bank = AcousticContext(values, times, 1.01)
    bank.lens_bank = LensBank(values, times, 1.01)
    return bank


def item(bank, queries, compact):
    return {'candidate_features': torch.zeros(2, 141), 'candidate_target': torch.tensor(0),
            'action_target': torch.tensor([1, 0]), 'action_state': torch.zeros(50),
            'future_target': torch.zeros(8), 'condition': torch.zeros(16),
            'history': torch.ones(1, 2, dtype=torch.long), 'history_features': torch.zeros(1, 8),
            'candidate_mask': torch.zeros(2, dtype=torch.bool), 'song_index': torch.tensor(0),
            'event_index': torch.tensor(1), 'eos_target': torch.tensor(0.),
            'audio_query_seconds': torch.tensor(queries, dtype=torch.float64),
            'audio_previous_seconds': torch.tensor(.03, dtype=torch.float64),
            '_audio_bank': bank, '_audio_grounded': True, '_audio_music': True,
            '_compact_audio_queries': compact}


@pytest.mark.parametrize('interleaved', [False, True])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_compact_queries_expand_to_identical_inputs_without_merging_songs(interleaved, device):
    a, b = context(11), context(99)
    banks = (a, b, a) if interleaved else (a, a, b)
    queries = [[.1, .2, -.1, 1.1], [.1, .2, .47, .7], [.1, .2 + 1e-12, .47, .7]]
    rows = [item(bank, q, False) for bank, q in zip(banks, queries)]
    for row, previous in zip(rows, (.01, .07, .18)):
        row['audio_previous_seconds'] = torch.tensor(previous, dtype=torch.float64)
    dense = collate_composer(rows)
    shared = collate_composer([{**row, '_compact_audio_queries': True} for row in rows])
    materializer = AudioQueryMaterializer()
    def materialize(batch):
        values = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        return materializer.materialize(batch, values)
    before, after = materialize(dense), materialize(shared)
    inverse = after['audio_lens_inverse']
    for key in ('workload_audio', 'trailing_audio', 'interval_audio'):
        torch.testing.assert_close(after[key], before[key], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(after['audio_lens'][inverse], before['audio_lens'], atol=2e-6, rtol=2e-6)
    assert torch.equal(after['audio_lens_mask'][inverse], before['audio_lens_mask'])
    assert after['audio_lens'].size(0) < inverse.numel()
    assert inverse[0, 0] != inverse[1 if interleaved else 2, 0]
    # Nearby times on the same source are not rounded together.
    if interleaved:
        assert inverse[0, 1] != inverse[2, 1]
    for key in ('action_target', 'candidate_target', 'history', 'condition'):
        assert torch.equal(before[key], after[key])


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_lens_sharing_keeps_weighted_gradient_multiplicity_and_rng(device, monkeypatch):
    # Strict FP32 equivalence, rather than shape-dependent TF32 kernels.
    # Production mixed-precision comparison is measured separately.
    monkeypatch.setattr(torch.backends.cudnn, 'allow_tf32', False)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    torch.manual_seed(122)
    dense = AudioLens(width=16, chunk_size=2).to(device).train()
    shared = copy.deepcopy(dense)
    inverse = torch.tensor([[0, 1, 1, 2, 3, 0], [3, 2, 4, 4, 4, 1]], device=device)
    base = torch.randn(5, 31, 26, device=device)
    mask = torch.rand(5, 31, device=device) > .2
    mask[2] = False
    a, b = base.clone().requires_grad_(), base.clone().requires_grad_()
    weight = torch.randn(2, 6, 16, device=device)
    state = torch.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state() if device == 'cuda' else None
    old = dense(a[inverse], mask[inverse])
    new = shared(b, mask, inverse=inverse)
    assert torch.equal(state, torch.get_rng_state())
    if cuda_state is not None:
        assert torch.equal(cuda_state, torch.cuda.get_rng_state())
    torch.testing.assert_close(new, old, atol=2e-6, rtol=2e-5)
    (old * weight).sum().backward(); (new * weight).sum().backward()
    torch.testing.assert_close(a.grad, b.grad, atol=5e-6, rtol=2e-4)
    for (_, left), (_, right) in zip(dense.named_parameters(), shared.named_parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=2e-5, rtol=2e-4)


def test_unique_lens_does_not_exceed_existing_convolution_row_budget():
    model = AudioLens(width=8, chunk_size=2)
    inverse = torch.arange(21).reshape(3, 7)
    rows = []
    handle = model.input_projection.register_forward_pre_hook(lambda _, args: rows.append(args[0].size(0)))
    try:
        model(torch.randn(21, 31, 26), torch.ones(21, 31, dtype=torch.bool), inverse=inverse)
    finally:
        handle.remove()
    assert max(rows) <= 3 * 2 and sum(rows) == 21


def test_grounded_model_keeps_logits_and_gradients_with_compact_lens():
    from maichart.composer.model import build_model
    from test_composer_audio_grounded_v3 import config, inputs, ATOMS
    torch.manual_seed(83)
    dense = build_model(config(history_dropout=0., ledger_dropout=0., symbolic_dropout=0.), ATOMS).train()
    shared = copy.deepcopy(dense)
    data = inputs()
    inverse = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 0, 1, 2, 3]])
    patches = data['audio_lens'][0].clone()
    valid = data['audio_lens_mask'][0].clone()
    data['audio_lens'], data['audio_lens_mask'] = patches[inverse], valid[inverse]
    targets = torch.tensor([[1, 0, 0], [2, 3, 0]])
    common = dict(candidate_indices=torch.tensor([1, 2]), action_targets=targets, packed_actions=True)
    outputs = []
    for model, kwargs in [(dense, data), (shared, {**data, 'audio_lens': patches,
                           'audio_lens_mask': valid, 'audio_lens_inverse': inverse})]:
        torch.manual_seed(421)
        output = model(**kwargs, **common)
        loss = sum(output[key].square().mean() for key in ('candidate_logits', 'action_logits', 'audio_action_logits'))
        loss.backward(); outputs.append(output)
    for key in ('candidate_logits', 'action_logits', 'audio_action_logits', 'placement_logits'):
        torch.testing.assert_close(outputs[0][key], outputs[1][key], atol=3e-6, rtol=3e-5)
    assert dense.state_dict().keys() == shared.state_dict().keys()
    for (_, left), (_, right) in zip(dense.named_parameters(), shared.named_parameters()):
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, atol=2e-5, rtol=3e-4)


def test_prewarm_same_identity_in_two_cache_directories_builds_both(tmp_path, monkeypatch):
    from maichart.composer import train, structure, structure_backend
    song = SimpleNamespace(audio=SimpleNamespace(store=object(), path='song.wav'))
    built = []
    monkeypatch.setattr(structure, '_identity', lambda *a: {'schema': structure.ENHANCED_SCHEMA})
    monkeypatch.setattr(structure, 'structure_fingerprint', lambda *a, **kw: 'shared_identity')
    monkeypatch.setattr(structure, '_bank', lambda song, cache, *a, **kw: built.append(str(cache)))
    monkeypatch.setattr(structure_backend, '_models', {})
    def dataset(cache):
        return SimpleNamespace(cache_dir=cache, structure_policy='metrical_context_v2', structure_backend='native',
            base=SimpleNamespace(songs=[(song,), (song,)]), _song_feature_key=lambda i: 'audio',
            _audio_bank_path=lambda *a: None, _source_song=lambda *a, **kw: song)
    train._prewarm_missing_song_banks(dataset(tmp_path/'train'), dataset(tmp_path/'validation'),
                                    SimpleNamespace(cache_dir=tmp_path, max_steps=0), torch.device('cpu'))
    assert built == [str(tmp_path/'train'), str(tmp_path/'validation')]
