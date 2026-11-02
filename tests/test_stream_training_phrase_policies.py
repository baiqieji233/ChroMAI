from types import SimpleNamespace

import pytest

from maichart.composer.stream_train import (
    LEARNING_POLICIES, LEGACY_LEARNING_POLICIES, _LocalBatches, parser, resolve_learning_policies,
)
from maichart.composer.stream_evaluation_settings import validate


def test_fresh_training_defaults_and_new_learning_contract():
    args = parser().parse_args(['--manifest', 'data.json', '--checkpoint-dir', 'fresh'])
    assert (args.epochs, args.lr, args.lr_schedule, args.warmup_steps) == (50, .0005, 'constant', 0)
    policies = resolve_learning_policies(args)
    assert policies['training_window_policy'] == 'phrase_complete_v1'
    assert policies['phrase_execution_policy'] == 'persistent_modes_v1'
    assert policies['detail_representation_policy'] == 'shared_components_v1'
    assert policies['duration_feature_policy'] == 'log_time_v2'
    assert policies['max_phrase_events'] == 64


def test_resume_inherits_legacy_and_rejects_changed_numeric_contract():
    parent = SimpleNamespace(**LEGACY_LEARNING_POLICIES, max_phrase_events=64,
        phrase_seam_events=2, phrase_modes=4, phrase_relation_weight=.1)
    policies = resolve_learning_policies(SimpleNamespace(), parent, resume=True)
    assert all(policies[key] == value for key, value in LEGACY_LEARNING_POLICIES.items())
    with pytest.raises(ValueError, match='Resume inherits max_phrase_events'):
        resolve_learning_policies(SimpleNamespace(max_phrase_events=32), parent, resume=True)


def test_new_execution_rejects_incompatible_learning_inputs():
    with pytest.raises(ValueError, match='requires aligned_phrase'):
        resolve_learning_policies(SimpleNamespace(phrase_context_policy='off'))


def test_song_sortish_sampler_keeps_every_sample_and_resume_suffix():
    samples = [(song, 0, length, 1.) for song in range(3) for length in range(1, 42)]
    ds = SimpleNamespace(samples=samples, training_window_policy='phrase_complete_v1')
    class Dataset:
        def __len__(self): return len(samples)
    ds = Dataset()
    ds.samples = samples
    ds.training_window_policy = 'phrase_complete_v1'
    batches = list(_LocalBatches(ds, 8, 17, 3))
    assert sorted(index for batch in batches for index in batch) == list(range(len(samples)))
    assert list(_LocalBatches(ds, 8, 17, 3, skip=4)) == batches[4:]
    assert list(_LocalBatches(ds, 8, 17, 4)) != batches


def test_phrase_mode_eval_temperature_includes_greedy():
    assert validate({})['phrase_mode_temperature'] == 1.
    assert validate({'phrase_mode_temperature': 0.})['phrase_mode_temperature'] == 0.
    with pytest.raises(ValueError): validate({'phrase_mode_temperature': -1.})
