from fractions import Fraction
from collections import OrderedDict
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from maichart.composer.candidates import MUSIC_CANDIDATE_POLICY_V6
from maichart.composer.event_views import EventViews
from maichart.composer.recovery import RecoveryBuilder
from maichart.event_timing import EventPoint, EventTempo
from maichart.tempo_clock import TempoClock


def point(beat, text='1'):
    return EventPoint(Fraction(beat), text)


class RecordingGuard:
    def __init__(self, *args):
        self.points = []

    def accepts(self, candidate):
        return candidate.note_text != '8'

    def commit(self, candidate):
        self.points.append(candidate)


def builder(*, excluded=(), source=None):
    tempo = [EventTempo(Fraction(0), 120.)]
    source = source or [point(i, str(i % 4 + 1)) for i in range(12)]
    views = EventViews.build(source, tempo, 0., 8., excluded=excluded)
    song = SimpleNamespace(tempo=tempo, first=0., audio=SimpleNamespace(duration_sec=8.))
    values = np.zeros((161, 120), np.float32)
    table = {'clock': TempoClock(tempo, 0.), 'condition': torch.zeros(16)}
    dataset = SimpleNamespace(
        samples=[SimpleNamespace(song_index=0, event_index=1, weight=.7)],
        base=SimpleNamespace(songs=[(song,)]), event_views={0: views},
        candidate_count=96, candidate_policy=MUSIC_CANDIDATE_POLICY_V6,
        architecture_profile='audio_grounded_v3',
        history_events=8, action_slots=3, codec_atoms=['1', '2', '3', '4', '8'],
        atom_to_id={text: i+1 for i, text in enumerate(['1', '2', '3', '4', '8'])},
        _song_table=lambda index: table,
        _song_features=lambda index: (values, np.linspace(0., 8., 161)))
    return RecoveryBuilder(dataset, guard_factory=RecordingGuard)


def test_recovery_rebuilds_generated_history_and_masks_every_nonpointer_label():
    recovery = builder()
    prefix = [point(0), point(1, '4'), point(2, '4')]
    item, reason = recovery.build(prefix, 0, generated_count=2)
    assert reason == 'accepted'
    assert item['history'][:, 0].tolist() == [1, 4, 4]
    assert item['event_index'].item() == 3
    assert item['head_timing'][:, 3].all()  # A4 was actually generated.
    assert not item['head_timing'][:, 1].any()  # Authored A2 never happened.
    for key in ('action_weight', 'future_weight', 'eos_weight', 'workload_weight', 'trailing_weight'):
        assert not torch.count_nonzero(item[key])
    assert item['pointer_weight'].item() == 1.
    assert item['sample_weight'].item() == pytest.approx(.7)


def test_recovery_never_crosses_unknown_reference_intervals():
    recovery = builder(excluded=[(.7, .8)])
    item, reason = recovery.build([point(0), point(1, '4'), point(2, '4')], 0, generated_count=2)
    assert item is None
    assert reason == 'unknown_reference_span'


def test_recovery_requires_original_configuration_to_remain_legal():
    source = [point(i, '8' if i == 3 else '1') for i in range(12)]
    recovery = builder(source=source)
    item, reason = recovery.build([point(0), point(1), point(2)], 0, generated_count=2)
    assert item is None
    assert reason == 'illegal_reference_continuation'


def test_recovery_rejects_unreachable_reference_time_without_inserting_it():
    recovery = builder(source=[point(0), point(1), point(2), point(Fraction(3001, 1000))])
    item, reason = recovery.build([point(0), point(1), point(2)], 0, generated_count=2)
    assert item is None
    assert reason == 'unreachable_reference_time'


def test_recovery_callback_sees_its_preceding_actual_proposal():
    recovery = builder()
    seen = []

    def propose(item, index):
        assert len(item['_candidate_beats']) == recovery.dataset.candidate_count
        assert all(isinstance(beat, Fraction) for beat in item['_candidate_beats'])
        assert tuple(p.beat for p in item['_prefix_points']) == tuple(map(Fraction, range(len(seen)+1)))
        seen.append(item['history'][:, 0].tolist())
        return point(len(seen), '4')

    batch, diagnostics = recovery.batch_from_indices([0], propose=propose)
    assert seen == [[1], [1, 4]]
    assert diagnostics == {'accepted': 1}
    assert batch['history'][0, :, 0].tolist() == [1, 4, 4]
    assert batch['action_weight'].tolist() == [0.]
    assert not any(key.startswith('_') for key in batch)
    with pytest.raises(ValueError, match='actual model'):
        recovery.rollout(0)


def test_recovery_does_not_reuse_new_helper_auxiliary_targets():
    recovery = builder()
    recovery.dataset._grounded_features = lambda *a, **kw: {
        'placement_target': torch.ones(96, 5), 'placement_weight': torch.ones(96, 5),
        'audio_lens': torch.ones(96, 31, 26)}
    item, reason = recovery.build([point(0), point(1), point(2)], 0, generated_count=2)
    assert reason == 'accepted'
    assert item['audio_lens'].all()
    assert not item['placement_target'].any()
    assert not item['placement_weight'].any()


def test_recovery_history_uses_canonical_atom_order_for_new_profile():
    recovery = builder()
    item, _ = recovery._inputs([point(0, '4/1')], 0)
    assert item['history'][0].tolist() == [1, 4, 0]


def test_recovery_reuses_dataset_audio_context_cache(monkeypatch):
    from maichart.composer import recovery as module

    recovery = builder()
    dataset = recovery.dataset
    dataset._context_cache = OrderedDict()
    dataset._context_cache_limit = 4
    dataset._feature_lock = threading.RLock()
    dataset._song_feature_key = lambda index: 'fixture'
    created = []
    original = module.AcousticContext

    def context(*args, **kwargs):
        created.append(original(*args, **kwargs))
        return created[-1]

    monkeypatch.setattr(module, 'AcousticContext', context)
    recovery._inputs([point(0)], 0)
    recovery._inputs([point(0), point(1, '4')], 0)
    assert len(created) == 1
    assert dataset._context_cache['fixture'] is created[0]
