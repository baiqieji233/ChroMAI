import copy

import pytest
import torch

from maichart.composer.action_factors import CompositionalActions
from maichart.composer.model import ComposerConfig, build_model, GROUNDED_PROFILE
from maichart.composer.checkpoint import save, load, inspect_payload


def test_parsed_heads_preserve_touch_angles_without_duration_digits():
    atoms = ['1', '3', '4', '5', '6', 'C', 'Ch[4:1]', 'Ch[8:3]', 'B5', 'D3', 'E6', '15']
    factors = CompositionalActions(atoms, 3, 'parsed_heads_v2')
    embedding = torch.nn.Embedding(len(atoms) + 1, 3)
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(embedding.weight.numel()).reshape_as(embedding.weight))
    def anchor(atom):
        return factors.history_anchor_embedding(torch.tensor(atoms.index(atom) + 1), embedding)
    for atom in ['C', 'Ch[4:1]', 'Ch[8:3]']:
        torch.testing.assert_close(anchor(atom), torch.zeros(3))
    for atom, tap in [('B5', '5'), ('D3', '3'), ('E6', '6')]:
        torch.testing.assert_close(anchor(atom), anchor(tap))
    torch.testing.assert_close(anchor('15'), (anchor('1') + anchor('5')) / 2)
    missing = CompositionalActions(['4', 'B5', '15'], 3, 'parsed_heads_v2')
    torch.testing.assert_close(missing.history_anchor_embedding(torch.tensor([2, 3]),
        torch.nn.Embedding(4, 3)), torch.zeros(2, 3))


def test_legacy_anchor_is_exact_original_digit_algorithm():
    atoms = ['1', '4', '5', 'C', 'Ch[8:3]', 'B5', '15']
    factors = CompositionalActions(atoms, 3)
    embedding = torch.nn.Embedding(len(atoms) + 1, 3)
    mapping = {atom: i + 1 for i, atom in enumerate(atoms)}
    expected = [0]
    for index, atom in enumerate(atoms, 1):
        digit = next((c for c in atom if c in '12345678'), None)
        expected.append(mapping.get(digit, mapping.get('4', index)) if digit is not None else mapping.get('4', index))
    ids = torch.arange(len(atoms) + 1)
    assert factors._lane_anchor_ids.tolist() == expected
    assert torch.equal(factors.history_anchor_embedding(ids, embedding), embedding(torch.tensor(expected)))
    state = copy.deepcopy(factors.state_dict())
    factors.set_history_anchor_policy('parsed_heads_v2')
    assert list(state) == list(factors.state_dict())
    factors.load_state_dict(state)


def test_checkpoint_policy_roundtrip_and_old_resume(tmp_path):
    atoms = ['1', '4', 'Ch[4:1]']
    model = build_model(ComposerConfig(atom_vocab_size=4, d_model=8, nhead=2,
        layers=1, dim_feedforward=16, architecture_profile=GROUNDED_PROFILE,
        head_timing_dim=99, lens_width=8), atoms)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / 'new.pt'
    payload = save(path, model, optimizer, epoch=0, args={}, manifest=tmp_path / 'manifest.json', atom_catalog=atoms)
    loaded, _ = load(path, include_optimizer=True)
    assert loaded.config.history_anchor_policy == 'parsed_geometry_v3'
    assert loaded.action_factors.history_anchor_policy == 'parsed_geometry_v3'
    payload['model_config'].pop('history_anchor_policy')
    old_path = tmp_path / 'old.pt'
    torch.save(payload, old_path)
    assert inspect_payload(payload)['can_resume']
    old, _ = load(old_path, include_optimizer=True)
    assert old.config.history_anchor_policy == 'legacy_digit_v1'
    assert old.action_factors.history_anchor_policy == 'legacy_digit_v1'
    legacy_path = tmp_path / 'legacy_saved.pt'
    save(legacy_path, old, optimizer, epoch=1, args={}, manifest=tmp_path / 'manifest.json', atom_catalog=atoms)
    reloaded, metadata = load(legacy_path, include_optimizer=True)
    assert reloaded.config.history_anchor_policy == 'legacy_digit_v1'
    assert inspect_payload(metadata)['can_resume']


def test_unknown_policy_rejected():
    with pytest.raises(ValueError, match='history_anchor_policy'):
        ComposerConfig(atom_vocab_size=2, history_anchor_policy='unknown')
    factors = CompositionalActions(['1'], 2)
    with pytest.raises(ValueError, match='history_anchor_policy'):
        factors.set_history_anchor_policy('unknown')
    with pytest.raises(ValueError, match='history_anchor_policy'):
        CompositionalActions(['1'], 2, 'unknown')
