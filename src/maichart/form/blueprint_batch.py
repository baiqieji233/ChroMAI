"""Pad whole-song Planner inputs without changing each chart's loss clock."""
from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_blueprint(samples):
    inputs, targets, weights = zip(*samples)
    audio_lengths = torch.tensor([item['audio'].shape[1] for item in inputs])
    form_lengths = torch.tensor([item['form'].shape[1] for item in inputs])
    if torch.any(audio_lengths < 1) or torch.any(form_lengths < 1):
        raise ValueError('Blueprint audio and form inputs must be nonempty')
    batch = {
        'audio': pad_sequence([item['audio'][0] for item in inputs], batch_first=True),
        'form': pad_sequence([item['form'][0] for item in inputs], batch_first=True),
        'condition': torch.cat([item['condition'] for item in inputs]),
    }
    if any('audio_times_sec' in item for item in inputs):
        if not all('audio_times_sec' in item for item in inputs):
            raise ValueError('Cannot mix Planner input clocks within one batch')
        batch['audio_times_sec'] = pad_sequence(
            [item['audio_times_sec'][0] for item in inputs], batch_first=True)
    batch['audio_padding_mask'] = torch.arange(batch['audio'].shape[1])[None] >= audio_lengths[:, None]
    batch['form_padding_mask'] = torch.arange(batch['form'].shape[1])[None] >= form_lengths[:, None]
    target = {
        key: pad_sequence([item[key][0] for item in targets], batch_first=True)
        for key in targets[0]
    }
    chart_weights = torch.tensor(weights, dtype=torch.float32)
    trainable = (chart_weights > 0) & (target['frame_weight'].sum(1) > 0)
    return batch, target, chart_weights, trainable
