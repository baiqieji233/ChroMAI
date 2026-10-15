"""Training-loop host synchronization and safety checks after the speed fixes."""
from fractions import Fraction as F
import json
import random

import pytest
import torch

from maichart.event_timing import EventPoint
from maichart.form.codec import RationalCodec
from maichart.form.layout_metrics import _token_tables, layout_diagnostics
from maichart.form.model import FormConfig, FormTransformer
from maichart.form.rational_objectives import event_decision_loss
from maichart.form.train import layout_loss
from maichart.form.training_input_perturbation import perturb_onset_skeleton
from maichart.generation.simai_compile import TempoChange

CODEC = RationalCodec()


def test_token_tables_match_vocabulary_membership_and_are_built_once():
    note, position, fraction = _token_tables(CODEC, torch.device('cpu'))
    ids = torch.arange(len(CODEC.tokens))
    note_ids = torch.tensor([i for i, t in enumerate(CODEC.tokens) if t.startswith('N:')])
    position_ids = torch.tensor([i for i, t in enumerate(CODEC.tokens)
                                 if t in ('<at>', '<note>') or t.startswith('P:')])
    fraction_ids = torch.tensor([i for i, t in enumerate(CODEC.tokens) if t.startswith('P:')])
    for table, members in ((note, note_ids), (position, position_ids), (fraction, fraction_ids)):
        assert torch.equal(table[ids], (ids[:, None] == members).any(-1))
    assert _token_tables(CODEC, torch.device('cpu'))[0] is note


def test_event_decision_loss_matches_tuple_indexing_reference():
    torch.manual_seed(3)
    logits = torch.randn(3, 9, len(CODEC.tokens))
    decoder_input = torch.tensor([[CODEC.bos_id, CODEC.end_id, 7, CODEC.end_id, 0, 0, 0, 0, 0]] * 3)
    target = torch.tensor([[CODEC.at_id, CODEC.eos_id, 8, CODEC.at_id, 0, 0, 0, 0, 0]] * 3)
    batch = {'decoder_input': decoder_input, 'target': target}
    values = logits.float()[..., (CODEC.at_id, CODEC.eos_id)]
    valid = ((decoder_input == CODEC.bos_id) | (decoder_input == CODEC.end_id)) & (target != 0)
    valid &= (target == CODEC.at_id) | (target == CODEC.eos_id)
    labels = (target != CODEC.at_id).long()
    reference = torch.nn.functional.cross_entropy(values.transpose(1, 2), labels, reduction='none')
    reference = (reference * valid).sum(1) / valid.sum(1).clamp_min(1)
    torch.testing.assert_close(event_decision_loss(logits, batch), reference, rtol=0, atol=0)


def _gpu_model_and_batch(device):
    torch.manual_seed(5)
    model = FormTransformer(FormConfig(vocab_size=len(CODEC), d_model=32, nhead=4, layers=1,
        dim_feedforward=64, dropout=0, audio_dim=6, descriptor_dim=5, condition_dim=3,
        max_tokens=16, max_audio_tokens=8, max_memory_tokens=16)).to(device).train()
    model.validate_inputs = False
    tokens = torch.tensor([[CODEC.bos_id, CODEC.at_id, 8, CODEC.note_id, 20, CODEC.end_id, CODEC.eos_id],
                           [CODEC.bos_id, CODEC.at_id, 9, CODEC.note_id, 21, CODEC.end_id, 0]])
    batch = {'audio': torch.randn(2, 4, 6), 'form': torch.randn(2, 2, 8),
             'history': torch.tensor([[1, 2, 0], [3, 0, 0]]), 'motif': torch.tensor([[4, 5], [6, 0]]),
             'draft': torch.zeros(2, 1, dtype=torch.long), 'right': torch.zeros(2, 1, dtype=torch.long),
             'condition': torch.randn(2, 3), 'condition_null': torch.tensor([False, True]),
             'task': torch.zeros(2, dtype=torch.long), 'relation': torch.randn(2, 4),
             'memory_strength': torch.ones(2), 'action_state': torch.randn(2, 3, 50),
             'action_state_padding_mask': torch.zeros(2, 3, dtype=torch.bool),
             'decoder_input': tokens[:, :-1], 'target': tokens[:, 1:], 'weight': torch.ones(2),
             'relation_target': torch.rand(2, 2), 'relation_valid': torch.tensor([True, False])}
    return model, {key: value.to(device) for key, value in batch.items()}


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA to observe host synchronization')
def test_layout_step_queues_forward_loss_and_backward_without_host_sync():
    device = torch.device('cuda')
    model, batch = _gpu_model_and_batch(device)

    def step():
        prediction = model(batch)
        loss_sum, parts = layout_loss(prediction, batch, .1, event_decision_weight=.25)
        finite = torch.isfinite(loss_sum.detach())
        loss_sum.backward()
        return finite, parts

    step()  # first call builds per-device lookup tables
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode('error')
    try:
        finite, _ = step()
        # Sanity check that the debug mode reports what the old loop did.
        with pytest.raises(RuntimeError):
            bool(finite)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert bool(finite)


def _fixture_training(tmp_path, monkeypatch):
    from maichart.form import train, data
    from test_form_pipeline import fixture_audio
    audio = fixture_audio(5)
    folder = tmp_path/'song'; folder.mkdir()
    (folder/'track.wav').write_bytes(b'fixture')
    (folder/'maidata.txt').write_text('&title=fixture\n&wholebpm=120\n&lv_5=13\n&inote_5=(120){4}1,2,3,4,5,6,7,8,E\n',
                                      encoding='utf-8')
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'songs': [{'song_id': 'one', 'maidata_path': 'song/maidata.txt',
        'audio_path': 'song/track.wav', 'difficulties': [{'difficulty_index': 5, 'level_raw': '13'}]}]}))
    original_dataset = data.FormDataset
    monkeypatch.setattr(train, 'FormDataset',
                        lambda *a, **kw: original_dataset(*a, **kw, audio_loader=lambda *a, **kw: audio))
    return train, ['--manifest', str(manifest), '--cache-dir', str(tmp_path/'cache'), '--epochs', '1',
                   '--batch-size', '1', '--grad-accumulation', '2', '--d-model', '16', '--nhead', '2',
                   '--layers', '1', '--max-tokens', '256', '--max-memory-tokens', '64',
                   '--max-region-seconds', '2', '--checkpoint-interval-steps', '1', '--precision', 'fp32',
                   '--seed', '33', '--max-steps', '4']


def test_nonfinite_layout_loss_still_stops_before_any_layout_update(tmp_path, monkeypatch):
    train, common = _fixture_training(tmp_path, monkeypatch)
    original = train.layout_loss
    monkeypatch.setattr(train, 'layout_loss',
                        lambda *a, **kw: (lambda loss, parts: (loss * float('nan'), parts))(*original(*a, **kw)))
    with pytest.raises(ValueError, match='Nonfinite layout loss'):
        train.main(common + ['--checkpoint-dir', str(tmp_path/'out')])
    saved = torch.load(tmp_path/'out/form_last.pt', weights_only=False)
    assert saved['optimizer_steps'] == 0
    assert saved['training_position']['phase'] == 'layout_revision'


def test_repetition_penalty_weight_is_rejected_because_it_has_no_gradient(tmp_path):
    from maichart.form import train
    options = train.parser().parse_args(['--manifest', 'fixture.json', '--checkpoint-dir', str(tmp_path),
                                         '--repetition-penalty-weight', '.1'])
    with pytest.raises(ValueError, match='repetition_penalty_weight'):
        train._validate(options)


def test_shifted_and_added_skeleton_onsets_use_the_generation_grid():
    tempo = [TempoChange(F(0), 184.0)]
    points = [EventPoint(F(k, 4), '1') for k in range(64)]
    allowed = {1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48}
    noisy = perturb_onset_skeleton(points, tempo, -.5, shift_sigma_ms=30, add_prob=.5, rng=random.Random(4))
    assert len(noisy) > len(points)
    assert all(point.beat >= 0 and point.beat.denominator in allowed for point in noisy)
    dropped = perturb_onset_skeleton(points, tempo, -.5, drop_prob=.5, rng=random.Random(4))
    assert dropped and {p.beat for p in dropped} < {p.beat for p in points}
    assert all(any(p is q for q in points) for p in dropped)
