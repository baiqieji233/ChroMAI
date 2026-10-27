from maichart.composer.contacts import ContactBudget, contact_decision
from maichart.generation.event_cli import _NoteGrammar
import pytest


def grammar():
    return _NoteGrammar.from_atoms(['1', '2', '5', '15', 'A1', 'B1', 'C', 'Ch[4:1]', '1h[4:1]', '1-5[4:1]'])


def test_touch_combination_is_unknown_and_permitted():
    g = grammar()
    d = contact_decision(['1', '2'], 'C', g)
    assert d['verdict'] == 'unknown' and d['allowed']
    assert d['minimum_contacts'] is None
    assert contact_decision(['A1', 'B1'], 'C', g)['allowed']


def test_same_head_and_hold_occupancy():
    g = grammar()
    assert contact_decision(['1-5[4:1]'], '1', g)['reason'] == 'same_head'
    assert not contact_decision([], '1', g, occupied=['1'])['allowed']
    assert not contact_decision([], '1h[4:1]', g, blocked=['1'])['allowed']
    assert contact_decision([], 'A1', g, occupied=['1'])['allowed']


def test_diagnostic_and_off_never_enforce():
    g = grammar()
    d = contact_decision(['1'], '1', g, policy='diagnostic_v1')
    assert d['verdict'] == 'reject' and d['allowed']
    assert contact_decision(['1'], '1', g, policy='off')['allowed']
    with pytest.raises(ValueError):
        contact_decision([], '1', g, policy='other')


def test_unknown_atom_and_compact_chords():
    budget = ContactBudget(grammar())
    assert budget.decision([], 'invalid')['verdict'] == 'unknown'
    assert not budget.decision(['15'], '5')['allowed']
    # A contact lower bound must not be manufactured from head count.
    assert budget.decision(['15'], '2')['minimum_contacts'] == 2


def test_vectorized_budget_and_real_sampler_reject_center_after_far_buttons():
    import numpy as np
    import torch
    from maichart.composer.generate import _sample_actions
    g=grammar();atoms=['1','5','C','A1','Ch[4:1]']
    budget=ContactBudget(g)
    mask=budget.mask(atoms,['1','5'])
    assert not mask[3] and not mask[5] and mask[4] and mask[0]
    for selected,occupied in (([],['1','5']),(['1'],['5']),(['C'],[])):
        m=budget.mask(atoms,selected,occupied=occupied)
        for i,atom in enumerate(atoms,1):
            decision=budget.decision(selected,atom,occupied=occupied)
            if decision['reason']=='far_buttons_center_assumed_three_contacts':assert not m[i]
    rows=torch.full((3,6),-100.)
    rows[0,1]=10.;rows[1,2]=10.;rows[2,3]=10.;rows[2,0]=0.
    chosen=_sample_actions(rows,atoms,g,0.,torch.Generator(),contact_budget=budget,
                           contact_policy='contact_groups_v1')
    assert chosen==['1','5']


def test_far_buttons_center_template_and_held_sources():
    g = grammar()
    d = contact_decision(['1','5'], 'C', g)
    assert not d['allowed'] and d['minimum_contacts'] == 3
    assert d['assumption'] == 'uncalibrated_human_reach_template'
    assert not contact_decision(['1','5'], 'Ch[4:1]', g)['allowed']
    diagnostic = contact_decision([], 'C', g, occupied=['1','5'], policy='diagnostic_v1')
    assert diagnostic['verdict'] == 'reject' and diagnostic['allowed']
    assert contact_decision(['1','5','B1'], 'C', g)['verdict'] == 'unknown'
