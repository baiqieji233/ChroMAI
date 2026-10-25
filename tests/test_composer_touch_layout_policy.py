import gc
import weakref
from fractions import Fraction

import torch
from maichart.composer.generate import _sample_actions, _active_constraints
from maichart.composer.guard import EventGuard, _cached_markers
from maichart.event_timing import EventTempo, EventPoint
from maichart.generation.event_cli import _NoteGrammar


def sample(atoms, *, occupied=(), policy='open'):
    rows=torch.tensor([[-1000.]+[100.-i for i in range(len(atoms))]]).repeat(len(atoms),1)
    return _sample_actions(rows,atoms,_NoteGrammar.from_atoms(atoms),0.,None,
                           occupied=occupied,touch_layout_policy=policy)


def test_touch_chords_do_not_consume_button_budget():
    assert sample(['B1','C','E5','1','2','3'])==['B1','C','E5','1','2']
    assert sample(['B1','C','E5'],policy='legacy_two_heads')==['B1','C']
    assert sample(['A1','1','2'],occupied={'C'})==['A1','1','2']


def test_touch_and_button_heads_have_independent_occupancy():
    assert sample(['A1','1','2'],occupied={'A1'})==['1','2']
    assert sample(['A1','1','2'],occupied={'1'})==['A1','2']
    assert sample(['C','B1'],occupied={'1','2'})==['C','B1']
    state=type('State',(),{'markers':[{'type':'touch_hold','second':0.,'end_second':2.,'touch_pad':'C'},
        {'type':'hold','second':0.,'end_second':2.,'lane':1}]})()
    heads,blocked,_=_active_constraints(state,Fraction(1),[EventTempo(Fraction(0),120.)],0.,touch_layout_policy='open')
    assert heads==blocked=={'C','1'}


def test_marker_cache_is_shared_without_retaining_guard_or_mutable_results():
    _cached_markers.cache_clear()
    tempo=[EventTempo(Fraction(0),120.)]
    first=EventGuard(tempo,0.,8.)
    original=first._markers('1h[4:1]',120.)
    reference=weakref.ref(first)
    del first
    gc.collect()
    assert reference() is None
    second=EventGuard(tempo,0.,8.)
    original[0][0]['type']='corrupted'
    assert second._markers('1h[4:1]',120.)[0][0]['type']=='hold'
    assert _cached_markers.cache_info().hits==1


def test_cooldown_keeps_touch_and_button_distinct():
    guard=EventGuard([EventTempo(Fraction(0),120.)],0.,8.,touch_layout_policy='open')
    guard.commit(EventPoint(Fraction(0),'1'))
    assert guard.last_head_seconds=={'1':0.}
    guard.commit(EventPoint(Fraction(1),'A1'))
    assert guard.last_head_seconds=={'1':0.,'A1':.5}

def test_legacy_preserves_button_touch_alias_and_cooldown():
    assert sample(['A1','1','2'],occupied={'A1'},policy='legacy_two_heads')==['2']
    guard=EventGuard([EventTempo(Fraction(0),120.)],0.,8.)
    guard.commit(EventPoint(Fraction(0),'1'))
    assert guard.last_head_seconds=={'A1':0.}
    assert not guard.accepts(EventPoint(Fraction(1,100),'A1'))
    assert guard.last_reason=='repeated_head_too_soon'


def test_candidate_capacity_preserves_touch_options_during_double_button_hold():
    from maichart.composer.generate import _candidate_has_capacity
    held={'1','2'}
    assert not _candidate_has_capacity(held,set(),'open')  # button-only vocabulary
    touch_actions=(frozenset({'C'}),frozenset({'B1','B2'}))
    assert _candidate_has_capacity(held,set(),'open',touch_actions)
    assert _candidate_has_capacity(held,{'C','B1'},'open',(frozenset({'E8'}),))
    assert not _candidate_has_capacity(held,{'C','B1'},'open',touch_actions)
    assert not _candidate_has_capacity(held,set(),'legacy_two_heads',touch_actions)
    assert _candidate_has_capacity({'C','B1'},set(),'open')
    assert not _candidate_has_capacity({'C','B1'},set(),'legacy_two_heads')


def test_recovery_guard_inherits_open_policy_with_distinct_cooldown_keys():
    from types import SimpleNamespace
    from maichart.composer.recovery import RecoveryBuilder
    song=SimpleNamespace(tempo=[EventTempo(Fraction(0),120.)],first=0.)
    builder=RecoveryBuilder(SimpleNamespace(touch_layout_policy='open'))
    guard=builder._guard(song,SimpleNamespace(end_sec=8.),[EventPoint(Fraction(0),'1')],1)
    assert guard.touch_layout_policy=='open'
    assert guard.last_head_seconds=={'1':0.}
    guard.commit(EventPoint(Fraction(1,100),'A1'))
    assert guard.last_head_seconds=={'1':0.,'A1':.005}


def test_recovery_guard_defaults_to_legacy_policy_for_old_dataset():
    from types import SimpleNamespace
    from maichart.composer.recovery import RecoveryBuilder
    song=SimpleNamespace(tempo=[EventTempo(Fraction(0),120.)],first=0.)
    builder=RecoveryBuilder(SimpleNamespace())
    guard=builder._guard(song,SimpleNamespace(end_sec=8.),[EventPoint(Fraction(0),'1')],1)
    assert guard.touch_layout_policy=='legacy_two_heads'
    assert guard.last_head_seconds=={'A1':0.}
    assert not guard.accepts(EventPoint(Fraction(1,100),'A1'))
    assert guard.last_reason=='repeated_head_too_soon'
