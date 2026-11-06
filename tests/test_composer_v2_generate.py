"""Sampling and transaction contracts; no checkpoint or MERT download needed."""
from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch

from maichart.composer_v2.codec import ComponentCodec
from maichart.composer_v2.generate import WindowSampler, _check_window, _sample, _tempo_map, _rollback_committed, _sampling_policy
from maichart.composer.event_set_codec import _atoms
from maichart.event_timing import EventPoint, EventTempo
from maichart.tempo_clock import TempoClock


def test_empty_window_eos_is_a_valid_joint_result():
    sampler = WindowSampler.__new__(WindowSampler)
    sampler.codec = ComponentCodec.fit(['1'])
    sampler.cache = None
    sampler.next = lambda target, allowed: sampler.codec.eos_id
    ids = sampler.joint(Fraction(16))
    assert sampler.codec.decode_points(ids) == []
    # EOS finishes this call only; another window can independently be sampled.
    assert sampler.joint(Fraction(16)) == ids


def test_greedy_policy_has_one_attempt_and_no_identical_rollback():
    args = SimpleNamespace(temperature=0., candidates=4, window_retries=16, rollback_windows=2)
    policy = _sampling_policy(args)
    assert policy['effective_candidates'] == policy['effective_retries'] == 1
    assert policy['effective_rollback_budget'] == 0
    args.temperature = .9
    policy = _sampling_policy(args)
    assert policy['effective_candidates'] == 4
    assert policy['effective_retries'] == 16
    assert policy['effective_rollback_budget'] == 2


def test_new_rollouts_clear_symbolic_kv_and_keep_audio_memory():
    sampler = WindowSampler.__new__(WindowSampler)
    sampler.codec = ComponentCodec.fit(['1'])
    audio_memory = object()
    sampler.cache = {'audio': audio_memory, 'decode_state': {'length': 999}}
    class Model:
        @staticmethod
        def reset_decode_cache(cache):
            cache.pop('decode_state', None)
    sampler.model = Model()
    def next_token(target, allowed):
        assert 'decode_state' not in sampler.cache
        assert sampler.cache['audio'] is audio_memory
        return sampler.codec.eos_id
    sampler.next = next_token
    sampler.joint(Fraction(16))
    sampler.cache['decode_state'] = {'length': 123}
    sampler.joint(Fraction(16))


def test_alternatives_preserve_exact_rhythm_and_atom_counts():
    sampler = WindowSampler.__new__(WindowSampler)
    sampler.codec = ComponentCodec.fit(['1', '2', '3'])
    sampler.cache = None
    sampler.next = lambda target, allowed: min(allowed)
    origin = Fraction(16)
    baseline = [EventPoint(origin + Fraction(1, 7), '1/2'),
                EventPoint(origin + 7, '3')]
    result = sampler.codec.decode_points(sampler.scaffold(baseline, origin), origin)
    assert [(p.beat, len(_atoms(p.note_text))) for p in result] == [
        (p.beat, len(_atoms(p.note_text))) for p in baseline]


class Guard:
    def __init__(self):
        self.prefix = [EventPoint(Fraction(0), '1h[1:8]')]
        self.last_reason = None

    def accepts(self, point):
        if point.note_text == '1':
            self.last_reason = 'overlaps_previous_long_hold'
            return False
        return True

    def commit(self, point):
        self.prefix.append(point)


def test_rejected_transaction_retains_committed_prefix():
    guard = Guard()
    trial, reason = _check_window(guard, [EventPoint(Fraction(16), '2'),
                                         EventPoint(Fraction(17), '1')])
    assert trial is None
    assert reason == 'overlaps_previous_long_hold'
    assert len(guard.prefix) == 1
    trial, reason = _check_window(guard, [EventPoint(Fraction(16), '2')])
    assert len(trial.prefix) == 2
    assert len(guard.prefix) == 1


def test_onset_retry_preserves_valid_prefix_and_rechecks_only_failed_onset():
    sampler=WindowSampler.__new__(WindowSampler)
    sampler.codec=ComponentCodec.fit(['1','2','3']);sampler.cache=None
    first=sampler.codec.encode_points([EventPoint(Fraction(16),'2'),EventPoint(Fraction(17),'1')],16)
    script=iter([*first,sampler.codec.eos_id,*sampler.codec.encode_atom('3'),sampler.codec.eos_id])
    calls=[]
    def next_token(target,allowed):
        calls.append(list(target));token=next(script);assert token in allowed;return token
    sampler.next=next_token
    original=Guard()
    ids=sampler.joint(16,guard=original,origin=16,event_retries=3)
    points=sampler.codec.decode_points(ids,origin=16)
    assert points==[EventPoint(Fraction(16),'2'),EventPoint(Fraction(17),'3')]
    assert len(original.prefix)==1
    assert sampler.checked_guard.prefix[1:]==points
    assert len(sampler.repair_report)==1 and sampler.repair_report[0]['beat']=='17'
    assert calls[-4]==first[:len(sampler.codec.encode_points(points[:1],16))+1]


def test_duplicate_head_mask_distinguishes_head_from_slide_endpoint():
    sampler=WindowSampler.__new__(WindowSampler)
    c=sampler.codec=ComponentCodec.fit(['1','2','1-5[8:1]','A1'])
    allowed=[c.ids['P:1'],c.ids['P:2'],c.ids['P:A1']]
    assert c.ids['P:1'] not in sampler._head_choices(allowed,[c.ids['N:tap']],{'1'})
    assert c.ids['P:A1'] in sampler._head_choices(allowed,[c.ids['N:touch']],{'1'})
    # A Slide may land on the same numbered key as another note's head.
    assert sampler._head_choices(allowed,[c.ids['N:slide'],c.ids['P:2'],c.ids['S:-']],{'1'})==allowed


def test_grammar_mask_and_zero_temperature():
    logits = torch.tensor([100., 1., 2.])
    generator = torch.Generator().manual_seed(2)
    assert _sample(logits, [1, 2], generator, 0., .95) == 2
    with pytest.raises(ValueError, match='no legal'):
        _sample(logits, [], generator, 0., .95)


def test_tempo_map_requires_explicit_beat_zero():
    with pytest.raises(ValueError, match='beat 0'):
        _tempo_map([{'beat': 4, 'bpm': 120}], 120.)
    tempo = _tempo_map([{'beat': 0, 'bpm': 120}, {'beat': 8, 'bpm': 90}], 120.)
    assert [row.beat for row in tempo] == [Fraction(0), Fraction(8)]


def test_rollback_revokes_source_window_suffix_and_replays_earlier_guard():
    tempo = [EventTempo(Fraction(0), 120.)]
    clock = TempoClock(tempo, .25)
    points = [EventPoint(Fraction(1), '2'), EventPoint(Fraction(9), '3h[1:8]'),
              EventPoint(Fraction(17), '4')]
    records = [dict(start_beat=str(start)) for start in (0, 8, 16)]
    class State:
        def __init__(self, *args):
            pass
        def _candidate_markers(self, second):
            return [dict(source_beat=Fraction(9), end_second=30., type='hold')]
    class ReplayGuard:
        def __init__(self, *args):
            self.prefix = []
        def accepts(self, point):
            return True
        def commit(self, point):
            self.prefix.append(point)
    kept, kept_records, guard, rewind, report = _rollback_committed(
        points, records, tempo, clock, Fraction(24), 100., 8,
        state_factory=State, guard_factory=ReplayGuard)
    assert rewind == 8
    assert kept == points[:1] == guard.prefix
    assert kept_records == records[:1]
    assert report['source_beat'] == '9'
    assert report['revoked_windows'] == 2
    # The caller's committed state remains intact until accepting this rollback.
    assert len(points) == 3 and len(records) == 3


def test_rollback_does_not_rewind_when_old_actions_have_ended():
    tempo = [EventTempo(Fraction(0), 120.)]
    class State:
        def __init__(self, *args):
            pass
        def _candidate_markers(self, second):
            return [dict(source_beat=Fraction(1), end_second=1., type='hold')]
    assert _rollback_committed([EventPoint(Fraction(1), '1')], [], tempo,
        TempoClock(tempo), Fraction(16), 100., 8, state_factory=State) is None
