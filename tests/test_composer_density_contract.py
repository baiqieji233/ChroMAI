import pytest
import torch

from maichart.composer import CHECKPOINT_SCHEMA, FAMILY, INPUT_CONTRACT, TRAINING_SEMANTICS
from maichart.composer.candidates import MUSIC_CANDIDATE_POLICY, MUSIC_CANDIDATE_POLICY_V6
from maichart.composer.checkpoint import inspect_payload, migrate_music_context, save
from maichart.composer.events import HEAD_TIMING_DIM
from maichart.composer.model import ChartComposer, ComposerConfig


def _model(*, profile='music_context_v2', head_timing_dim=0):
    return ChartComposer(ComposerConfig(
        atom_vocab_size=5, d_model=16, nhead=4, layers=1, dim_feedforward=32,
        history_events=4, action_slots=2, dropout=0., architecture_profile=profile,
        head_timing_dim=head_timing_dim))


def _payload(config, *, semantics=None, candidate_policy=None, audio_policy=None,
             optimizer_state=None):
    return {
        'schema': CHECKPOINT_SCHEMA, 'family': FAMILY, 'input_contract': INPUT_CONTRACT,
        'model_config': config.to_dict(),
        'atom_catalog': [str(index) for index in range(config.atom_vocab_size - 1)],
        'model_state_dict': {}, 'optimizer_state_dict': optimizer_state,
        **({'training_semantics': semantics} if semantics is not None else {}),
        **({'candidate_policy': candidate_policy} if candidate_policy is not None else {}),
        **({'audio_feature_policy': audio_policy} if audio_policy is not None else {}),
    }


def test_head_timing_config_is_optional_but_strict():
    assert ComposerConfig(atom_vocab_size=3).head_timing_dim == 0
    assert ComposerConfig(atom_vocab_size=3, architecture_profile='music_context_v2',
                          head_timing_dim=HEAD_TIMING_DIM).head_timing_dim == HEAD_TIMING_DIM
    with pytest.raises(ValueError):
        ComposerConfig(atom_vocab_size=3, architecture_profile='audio_workload_v1',
                       head_timing_dim=HEAD_TIMING_DIM)
    with pytest.raises(ValueError):
        ComposerConfig(atom_vocab_size=3, architecture_profile='music_context_v2',
                       head_timing_dim=HEAD_TIMING_DIM - 1)


def test_head_timing_projection_is_zero_initialized_and_required_when_enabled():
    model = _model(head_timing_dim=HEAD_TIMING_DIM).eval()
    assert torch.count_nonzero(model.head_timing_projection.weight) == 0
    assert torch.count_nonzero(model.head_timing_projection.bias) == 0
    features = torch.randn(1, 3, 141)
    common = dict(candidate_features=features, history=torch.zeros(1, 0, 2, dtype=torch.long),
                  condition=torch.randn(1, 16), action_state=torch.randn(1, 50),
                  candidate_mask=torch.zeros(1, 3, dtype=torch.bool), decode_actions=False)
    with pytest.raises(ValueError, match='head_timing is required'):
        model(**common)
    result = model(**common, head_timing=torch.randn(1, 3, HEAD_TIMING_DIM))
    assert result['candidate_logits'].shape == (1, 3)


def test_old_music_context_defaults_to_v5_and_cannot_resume():
    config = _model().config
    payload = _payload(config, optimizer_state={})
    inspected = inspect_payload(payload)
    assert inspected['valid'] and inspected['can_generate']
    assert not inspected['can_resume']
    assert payload['candidate_policy'] == MUSIC_CANDIDATE_POLICY


def test_phase_safe_intermediate_checkpoint_stays_generatable_but_not_resumable():
    config = _model().config
    payload = _payload(config, semantics='chart_composer_training_v6_phase_safe_density_guard',
                      candidate_policy=MUSIC_CANDIDATE_POLICY_V6, optimizer_state={})
    inspected = inspect_payload(payload)
    assert inspected['valid'] and inspected['can_generate']
    assert not inspected['can_resume']


def test_current_music_context_requires_timing_contract_for_resume():
    config = _model(head_timing_dim=HEAD_TIMING_DIM).config
    payload = _payload(config, semantics=TRAINING_SEMANTICS,
                      candidate_policy=MUSIC_CANDIDATE_POLICY_V6,
                      optimizer_state={})
    assert inspect_payload(payload)['can_resume']
    old_shape = _payload(_model().config, semantics=TRAINING_SEMANTICS,
                         candidate_policy=MUSIC_CANDIDATE_POLICY_V6,
                         optimizer_state={})
    assert not inspect_payload(old_shape)['valid']


def test_explicit_migration_adds_zero_head_timing_projection():
    old_music = _model()
    upgraded_music = migrate_music_context(old_music)
    assert upgraded_music.config.head_timing_dim == HEAD_TIMING_DIM
    assert torch.count_nonzero(upgraded_music.head_timing_projection.weight) == 0
    torch.testing.assert_close(upgraded_music.action_head.weight, old_music.action_head.weight)
    torch.testing.assert_close(upgraded_music.action_head.bias, old_music.action_head.bias)
    old_workload = _model(profile='audio_workload_v1')
    upgraded_workload = migrate_music_context(old_workload)
    assert upgraded_workload.config.architecture_profile == 'music_context_v2'
    assert upgraded_workload.config.head_timing_dim == HEAD_TIMING_DIM


def test_save_keeps_dim0_music_checkpoints_on_the_old_contract(tmp_path):
    model = _model()
    payload = save(tmp_path / 'old.pt', model, None, epoch=1, args={},
                   manifest=tmp_path / 'manifest.json', atom_catalog=['1', '2', '3', '4'])
    assert payload['training_semantics'] == 'chart_composer_training_v5_music_context_shared_slots'
    assert payload['candidate_policy'] == MUSIC_CANDIDATE_POLICY
