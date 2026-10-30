"""Song-paired policy integration contracts; no training process is started."""
import pytest
import torch
from pathlib import Path
from maichart.composer.model import build_model, resolve_composer_policy
from maichart.composer.checkpoint import save, load, migrate_audio_grounded, quality_training_compatible
from maichart.composer.quality_inputs import QUALITY_INPUT_POLICY, SONG_QUALITY_INPUT_POLICY
from test_composer_audio_grounded_v3 import ATOMS, config


def test_song_policy_defaults_and_resume_inheritance():
    assert resolve_composer_policy('configuration_quality_policy') == 'song_paired_v3'
    parent = {'configuration_quality_policy': 'cooperative_v2'}
    assert resolve_composer_policy('configuration_quality_policy', parent_config=parent, resume=True) == 'cooperative_v2'
    assert resolve_composer_policy('configuration_quality_policy', parent_config={}, architecture_profile='music_context_v2') == 'off'
    with pytest.raises(ValueError, match='finetune'):
        resolve_composer_policy('configuration_quality_policy', 'song_paired_v3', parent, resume=True)


@pytest.mark.parametrize('policy,contract', [('cooperative_v2', QUALITY_INPUT_POLICY), ('song_paired_v3', SONG_QUALITY_INPUT_POLICY)])
def test_song_checkpoint_preserves_input_contract(tmp_path, policy, contract):
    model = build_model(config(configuration_quality_policy=policy), ATOMS)
    path = tmp_path / 'checkpoint.pt'
    save(path, model, None, epoch=0, args={}, manifest=tmp_path/'unused.json', atom_catalog=ATOMS)
    restored, payload = load(path)
    assert restored.config.configuration_quality_policy == policy
    assert payload['configuration_quality_input_policy'] == contract
    assert quality_training_compatible(payload)
    if policy == 'song_paired_v3':
        recipe=payload['training_objective_recipe']['configuration_quality']
        assert recipe['fine_rhythm_target']=='existing_event_ctx_fine_note_ratio'
        assert not recipe['extra_delta_loss']
    for name, value in model.state_dict().items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    payload['configuration_quality_input_policy'] = 'invalid'
    assert not quality_training_compatible(payload)


def test_song_migration_reports_new_and_dropped_weights():
    old = build_model(config(configuration_quality_policy='cooperative_v2'), ATOMS)
    old_state = {k:v.clone() if torch.is_tensor(v) else v for k,v in old.state_dict().items()}
    new = migrate_audio_grounded(old, ATOMS, configuration_quality_policy='song_paired_v3')
    report = new.migration_report
    assert report['target_policy'] == 'song_paired_v3'
    assert report['dropped'] and report['new_or_reinitialized']
    assert not any('anchor' in key for key in new.state_dict())
    for name in report['copied_same_shape']:
        torch.testing.assert_close(old_state[name], new.state_dict()[name], rtol=0, atol=0)
    for name,value in old_state.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, old.state_dict()[name], rtol=0, atol=0)


def test_both_training_pages_offer_song_policy():
    static = Path(__file__).resolve().parents[1]/'src/maichart/webui/static'
    for name in ('index.html', 'composer.html'):
        text = (static/name).read_text(encoding='utf-8')
        assert 'value="song_paired_v3"' in text
        assert '全曲音乐与历史配对（新版）' in text
