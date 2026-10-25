import copy
import hashlib
import json

import pytest
import torch

from maichart.composer.action_factors import CompositionalActions, FACTOR_CONTRACT, FACTOR_FIELDS


def test_geometry_wrap_half_lanes_and_per_note_mixed_weights():
    model = CompositionalActions(['1','2','3','4','5','6','7','8',
        'D1','E8','A3','B2','C','1/D1','1/A1/B3','C/D1'], 8, 'parsed_geometry_v3')
    weights = model._parsed_anchor_weights
    torch.testing.assert_close(weights[9], torch.tensor([.5,0,0,0,0,0,0,.5]))
    torch.testing.assert_close(weights[10], torch.tensor([0,0,0,0,0,0,.5,.5]))
    assert weights[11,2] == 1 and weights[12,1] == 1
    assert not weights[13].any()
    torch.testing.assert_close(weights[14], torch.tensor([.75,0,0,0,0,0,0,.25]))
    torch.testing.assert_close(weights[15], torch.tensor([2/3,0,1/3,0,0,0,0,0]))
    torch.testing.assert_close(weights[16],weights[9])
    embedding = torch.nn.Embedding(17,8)
    with torch.no_grad():
        embedding.weight.zero_(); embedding.weight[1:9] = torch.eye(8)
    torch.testing.assert_close(model.history_anchor_embedding(torch.arange(17),embedding), weights)


def test_legacy_geometry_policies_unchanged_and_factor_identity_stable():
    atoms = ['1','4','D1','E8','1/A1/B3','C']
    old = CompositionalActions(atoms,4)
    parsed = CompositionalActions(atoms,4,'parsed_heads_v2')
    new = CompositionalActions(atoms,4,'parsed_geometry_v3')
    assert parsed._parsed_anchor_weights[3,0] == 1
    assert parsed._parsed_anchor_weights[4,7] == 1
    assert parsed._parsed_anchor_weights[5,0] == .5
    assert parsed._parsed_anchor_weights[5,2] == .5
    # Policies alter only nonpersistent derived geometry; old strict states load.
    for target in (parsed,new):
        target.load_state_dict(copy.deepcopy(old.state_dict()),strict=True)
        assert target.get_extra_state() == old.get_extra_state()
    expected = hashlib.sha256(json.dumps({'contract':FACTOR_CONTRACT,'atoms':tuple(atoms),
        'fields':old.factor_values},ensure_ascii=True,separators=(',',':'),sort_keys=True).encode('ascii')).digest()
    assert bytes(old.catalog_fingerprint.tolist()) == expected
    assert tuple(old.factors) == FACTOR_FIELDS


def test_joint_branch_binding_distinguishes_swapped_durations_and_modifiers():
    atoms = ['1-4[8:1]*-6[4:1]', '1-4[4:1]*-6[8:1]',
             '1-6[4:1]*-4[8:1]', '1-4b[8:1]*-6[4:1]',
             '1-4[8:1]*-6b[4:1]']
    old = CompositionalActions(atoms,4)
    new = CompositionalActions(atoms,4,'legacy_digit_v1','branch_joint_v2')
    for field in FACTOR_FIELDS:
        torch.testing.assert_close(getattr(old,'atom_'+field),getattr(new,'atom_'+field))
    assert new.atom_slide_branch_joint[1] != new.atom_slide_branch_joint[2]
    assert new.atom_slide_branch_joint[1] == new.atom_slide_branch_joint[3]
    assert new.atom_slide_branch_joint[4] != new.atom_slide_branch_joint[5]
    # Swapping duration associations is invisible to all old marginal fields.
    for field in FACTOR_FIELDS:
        assert getattr(old,'atom_'+field)[1] == getattr(old,'atom_'+field)[2]
    with torch.no_grad():
        for name in FACTOR_FIELDS:
            old.factors[name].weight.normal_()
            new.factors[name].weight.copy_(old.factors[name].weight)
    torch.testing.assert_close(new.composed_weight(),old.composed_weight(),rtol=0,atol=0)
    ids = torch.arange(len(atoms)+1)
    torch.testing.assert_close(new.embed(ids),old.embed(ids),rtol=0,atol=0)
    assert new.get_extra_state()['contract'] == 'parsed_legal_action_factors_v2'
    restored = CompositionalActions(atoms,4,'parsed_geometry_v3','branch_joint_v2')
    restored.load_state_dict(copy.deepcopy(new.state_dict()),strict=True)
    torch.testing.assert_close(new.composed_weight(),restored.composed_weight())
    # A raw strict=False restore is unsafe; integration must migrate parameters.
    with pytest.raises(ValueError,match='catalog identity'):
        new.load_state_dict(old.state_dict(),strict=False)
    new(torch.ones(1,4)).sum().backward()
    assert new.factors['slide_branch_joint'].weight.grad.abs().sum() > 0
