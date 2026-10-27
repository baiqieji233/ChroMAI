import copy
from types import SimpleNamespace

import pytest
import torch

from maichart.composer.checkpoint import save
from maichart.composer.data import collate_composer
from maichart.composer.model import ComposerConfig, GROUNDED_PROFILE, build_model
from maichart.composer.train import _loss
from maichart.webui import composer_api


ATOMS = ['1', '2', '3h[4:1]', '4-8[4:1]']


def _model():
    return build_model(ComposerConfig(atom_vocab_size=5, d_model=16, nhead=4,
        layers=1, dim_feedforward=32, dropout=0., action_slots=33,
        history_events=8, candidate_count=16, architecture_profile=GROUNDED_PROFILE,
        head_timing_dim=99, lens_width=8), ATOMS)


def _item(note_count):
    target = torch.zeros(33, dtype=torch.long)
    target[:note_count] = 1
    return {'candidate_features': torch.zeros(2, 141), 'candidate_target': torch.tensor(0),
        'action_target': target, 'action_state': torch.zeros(50),
        'future_target': torch.zeros(8), 'condition': torch.zeros(16),
        'history': torch.zeros(2, 33, dtype=torch.long), 'history_features': torch.zeros(2, 8),
        'candidate_mask': torch.zeros(2, dtype=torch.bool), 'song_index': torch.tensor(0),
        'event_index': torch.tensor(0), 'eos_target': torch.tensor(0.),
        'audio_lens': torch.zeros(2, 31, 26)}


@pytest.mark.parametrize('notes', [0, 1, 2, 32, 33])
def test_grounded_collation_keeps_atoms_and_first_stop_including_full_capacity(notes):
    item = _item(notes)
    batch = collate_composer([item])
    expected = min(33, notes + 1)
    assert batch['action_target'].shape == (1, expected)
    torch.testing.assert_close(batch['action_target'][0], item['action_target'][:expected])
    assert batch['history'].shape == (1, 2, 1)
    # Older profiles retain their existing collation/decoder shape.
    item.pop('audio_lens')
    assert collate_composer([item])['action_target'].shape == (1, 33)


@pytest.mark.parametrize('width', [1, 4, 33])
def test_history_compaction_keeps_last_occupied_column_and_source_capacity(width):
    item = _item(1)
    item['history'][0, width-1] = 2
    batch = collate_composer([item, _item(2)])
    assert batch['history'].shape == (2, 2, width)
    torch.testing.assert_close(batch['history'][0], item['history'][..., :width])
    assert item['history'].shape == (2, 33)
    item.pop('audio_lens')
    assert collate_composer([item])['history'].shape == (1, 2, 33)


def test_empty_history_compaction_preserves_empty_event_axis():
    item = _item(1)
    item['history'] = item['history'][:0]
    item['history_features'] = item['history_features'][:0]
    batch = collate_composer([item])
    assert batch['history'].shape == (1, 0, 1)
    assert batch['history_padding_mask'].shape == (1, 0)


def test_padding_elision_preserves_v9_losses_and_every_decoder_gradient():
    torch.manual_seed(37)
    dense = _model().eval()
    compact = copy.deepcopy(dense)
    targets = torch.zeros(2, 33, dtype=torch.long)
    targets[0, 0], targets[1, :2] = 1, torch.tensor([2, 3])
    context = torch.randn(2, 16)
    gradients = []
    losses = []
    for model, labels in ((dense, targets), (compact, targets[:, :3])):
        output = model.decode_candidate(context, action_targets=labels, audio_context=context)
        output['candidate_logits'] = torch.zeros(2, 2)
        batch = {'candidate_target': torch.tensor([0, 1]), 'action_target': labels,
                 'future_target': torch.zeros(2, 8), 'eos_target': torch.zeros(2)}
        loss, metrics = _loss(output, batch, 5)
        assert metrics['audio_action_loss'] > 0
        loss.backward()
        losses.append(loss.detach())
        gradients.append({name: parameter.grad.clone() for name, parameter in model.named_parameters()
                          if parameter.grad is not None})
    torch.testing.assert_close(losses[0], losses[1])
    assert gradients[0].keys() == gradients[1].keys()
    for name, gradient in gradients[0].items():
        torch.testing.assert_close(gradient, gradients[1][name], rtol=1e-4, atol=1e-6,
                                   msg=lambda detail: f'{name}: {detail}')


def test_training_modes_are_mutually_exclusive_before_loading_any_checkpoint():
    from maichart.composer.train import parser
    with pytest.raises(SystemExit):
        parser().parse_args(['--manifest','unused','--checkpoint-dir','unused',
                             '--resume','parent1.pt','--init-checkpoint','parent2.pt'])


@pytest.mark.parametrize('filename',['composer_last.pt','composer_inference.pt'])
def test_fresh_training_preserves_existing_training_and_inference_files(tmp_path,filename):
    from maichart.composer.train import main
    checkpoint=tmp_path/filename
    checkpoint.write_bytes(b'preserve original checkpoint')
    with pytest.raises(ValueError,match=f'already contains {filename}'):
        main(['--manifest','unused','--checkpoint-dir',str(tmp_path),'--device','cpu'])
    assert checkpoint.read_bytes()==b'preserve original checkpoint'


@pytest.mark.parametrize('flags',[['--epochs','0'],['--lr','0'],['--lr','nan']])
def test_training_rejects_noop_or_nonfinite_settings_before_dataset_prepare(tmp_path,flags):
    from maichart.composer.train import main
    with pytest.raises(ValueError):
        main(['--manifest','unused','--checkpoint-dir',str(tmp_path),'--device','cpu',*flags])


def _api_parent(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    parent = tmp_path / 'parent.pt'
    model = _model()
    save(parent, model, torch.optim.AdamW(model.parameters(), lr=1e-4),
         epoch=1, args={'lr': 1e-4}, manifest=manifest, atom_catalog=ATOMS)
    return {'manifest': str(manifest), 'checkpoint_dir': str(tmp_path / 'next'),
            'cache_dir': str(tmp_path / 'cache'), 'init_checkpoint': str(parent)}


def _request(payload):
    calls = []
    runtime = SimpleNamespace(
        _create_task=lambda kind: SimpleNamespace(task_id='test', kind=kind, result={}),
        _run_subprocess_task=lambda task, args, **kwargs: calls.append(args))
    composer_api.start_train(SimpleNamespace(_json_response=lambda _: None), payload, runtime=runtime)
    return calls[0]


def test_finetune_api_inherits_parent_backbone_without_replacing_destination_profile(tmp_path):
    payload = _api_parent(tmp_path)
    payload.update(mode='finetune', architecture_profile=GROUNDED_PROFILE)
    args = _request(payload)
    for key, expected in {'d-model': 16, 'layers': 1, 'nhead': 4, 'dim-feedforward': 32,
                          'history-events': 8, 'candidate-count': 16, 'action-slots': 33}.items():
        assert args[args.index('--' + key) + 1] == str(expected)
    assert args[args.index('--architecture-profile') + 1] == GROUNDED_PROFILE


def test_resume_api_leaves_optimizer_learning_rate_untouched_unless_requested(tmp_path):
    payload = _api_parent(tmp_path)
    payload.update(mode='resume', resume=payload.pop('init_checkpoint'))
    assert '--lr' not in _request(payload)
    payload['lr'] = 3e-5
    args = _request(payload)
    assert args[args.index('--lr') + 1] == '3e-05'


def test_api_forwards_optional_schedule_and_rejects_schedule_drift_on_old_resume(tmp_path):
    payload=_api_parent(tmp_path)
    payload.update(mode='finetune',lr_schedule='cosine',warmup_steps=12,min_lr_ratio=.2)
    args=_request(payload)
    assert args[args.index('--lr-schedule')+1]=='cosine'
    assert args[args.index('--warmup-steps')+1]=='12'
    assert args[args.index('--min-lr-ratio')+1]=='0.2'
    payload.update(mode='resume',resume=payload.pop('init_checkpoint'))
    with pytest.raises(ValueError,match='lr_schedule'):
        _request(payload)


def test_api_accepts_empty_optional_fields_from_standalone_form(tmp_path):
    payload=_api_parent(tmp_path)
    payload.update(mode='finetune',lr_schedule='cosine',warmup_steps='',min_lr_ratio='')
    args=_request(payload)
    assert '--warmup-steps' not in args and '--min-lr-ratio' not in args
