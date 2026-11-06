from types import SimpleNamespace

import pytest

from maichart.composer.stream_train import (
    LEARNING_POLICIES, LEGACY_LEARNING_POLICIES, _LocalBatches, parser, resolve_learning_policies,
)
from maichart.composer.stream_evaluation_settings import validate


def test_fresh_training_defaults_and_new_learning_contract():
    args = parser().parse_args(['--manifest', 'data.json', '--checkpoint-dir', 'fresh'])
    assert (args.epochs, args.lr, args.lr_schedule, args.warmup_steps) == (50, .0005, 'constant', 0)
    assert (args.batch_size,args.grad_accumulation,args.audio_cache_gb)==(16,4,1.)
    assert args.generation_eval_songs == 0
    policies = resolve_learning_policies(args)
    assert policies['training_window_policy'] == 'phrase_complete_v1'
    assert policies['phrase_execution_policy'] == 'component_modes_v2'
    assert policies['detail_representation_policy'] == 'typed_components_v2'
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


def test_song_style_fresh_defaults_and_legacy_resume_contract():
    fresh = resolve_learning_policies(SimpleNamespace())
    assert fresh['song_profile_policy'] == 'masked_deployment_joint_v3'
    assert fresh['song_response_policy'] == 'unified_profile_response_v2'
    assert (fresh['song_profile_components'], fresh['song_profile_weight']) == (4, .1)
    assert (fresh['song_response_modes'], fresh['song_response_group_size']) == (4, 4)
    parent = SimpleNamespace(**{k: v for k, v in LEGACY_LEARNING_POLICIES.items()
                                if not k.startswith('song_')})
    resumed = resolve_learning_policies(SimpleNamespace(), parent, resume=True)
    assert resumed['song_profile_policy'] == resumed['song_response_policy'] == 'off'
    with pytest.raises(ValueError, match='Resume inherits song_profile_policy'):
        resolve_learning_policies(SimpleNamespace(song_profile_policy='masked_joint_v1'), parent, resume=True)
    with pytest.raises(ValueError, match='music_response_v1 requires'):
        resolve_learning_policies(SimpleNamespace(song_response_policy='music_response_v1', phrase_execution_policy='off'))


def test_cli_marks_explicit_resume_batch_options():
    base=['--manifest','data.json','--checkpoint-dir','out','--resume','old.pt']
    omitted=parser().parse_args(base)
    assert omitted._explicit_batch_controls==()
    from maichart.composer.stream_train import _inherit_resume_batch_defaults
    _inherit_resume_batch_defaults(omitted,{'args':{'batch_size':32,'grad_accumulation':2,'lr':.0001,'audio_cache_gb':3.}})
    assert (omitted.batch_size,omitted.grad_accumulation,omitted.lr,omitted.audio_cache_gb)==(32,2,.0001,3.)
    explicit=parser().parse_args(base+['--batch-size=16','--grad-accumulation','4'])
    assert explicit._explicit_batch_controls==('batch_size','grad_accumulation')
    _inherit_resume_batch_defaults(explicit,{'args':{'batch_size':32,'grad_accumulation':2}})
    assert (explicit.batch_size,explicit.grad_accumulation)==(16,4)
