"""Optional audio-only evidence for an already selected integer beat.

This branch changes only conditional phase scores. It neither proposes events
nor changes the LM's total phase-token probability mass, actions, or EOS score.
Raw 100 Hz mel frames are shared across all phase queries for an integer beat.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .onset_policy import POLICY, CHOICES, resolve_policy, effective_strength
RADIUS = 14  # 5-wide convolutions with dilation 1, 2, 4.


def phase_values(codec):
    return np.array([float(codec.phases[i]) for i in sorted(codec.phases)], dtype=np.float64)


def audio_tiles(song, anchors, phases):
    """Only music/clock/codec enter this function; no target or history notes."""
    from .audio import _interpolate
    anchors = np.asarray(anchors, dtype=np.float64)
    first = np.floor(song.clock.at_floats(anchors) * 100).astype(np.int64) - RADIUS - 1
    last = np.ceil(song.clock.at_floats(anchors + 1) * 100).astype(np.int64) + RADIUS + 1
    lengths = last - first + 1
    grid = first[:, None] + np.arange(int(lengths.max()))[None, :]
    valid = np.arange(grid.shape[1])[None, :] < lengths[:, None]
    unique, inverse = np.unique(grid, return_inverse=True)
    # Reads the existing native-rate bank; never asks MERT for extra frames.
    frames = _interpolate(song.audio.mel, song.audio.mel_times_sec, unique / 100.)
    frames = frames[inverse].reshape(*grid.shape, 98)
    valid &= (grid >= 0) & (grid / 100. <= song.audio.duration_sec)
    positions = (song.clock.at_floats(anchors[:, None] + phases[None, :]) * 100 - first[:, None])
    return dict(onset_audio=torch.as_tensor(frames, dtype=torch.float32),
                onset_mask=torch.as_tensor(valid),
                onset_positions=torch.as_tensor(positions, dtype=torch.float32))


def collate_tiles(items):
    width = max(x['onset_audio'].size(1) for x in items)
    return dict(onset_audio=torch.cat([F.pad(x['onset_audio'], (0, 0, 0, width-x['onset_audio'].size(1))) for x in items]),
                onset_mask=torch.cat([F.pad(x['onset_mask'], (0, width-x['onset_mask'].size(1))) for x in items]),
                onset_positions=torch.cat([x['onset_positions'] for x in items]))


def query_rows(ids, tile_ids, phase_id, *, row=0):
    return [(row, p, tile_ids[p]) for p, token in enumerate(ids) if token == phase_id]


class PhaseAudioReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(98))
        self.register_buffer('std', torch.ones(98))
        self.encoder = nn.Sequential(nn.Conv1d(98, 32, 5, padding=2), nn.GELU(),
            nn.Conv1d(32, 32, 5, padding=4, dilation=2), nn.GELU(),
            nn.Conv1d(32, 32, 5, padding=8, dilation=4), nn.GELU())
        self.score = nn.Conv1d(32, 1, 1, bias=False)
        nn.init.zeros_(self.score.weight)

    @torch.no_grad()
    def fit_normalization(self, songs):
        # One bounded training-only read, persisted in the checkpoint. No batch
        # statistics or validation data are used during forward/inference.
        indices = np.unique(np.linspace(0, len(songs)-1, min(32, len(songs)), dtype=int))
        chunks = []
        for index in indices:
            mel = songs[index].audio.mel
            frames = np.unique(np.linspace(0, len(mel)-1, min(512, len(mel)), dtype=int))
            chunks.append(np.asarray(mel[frames], dtype=np.float32))
        values = np.concatenate(chunks)
        self.mean.copy_(torch.as_tensor(values.mean(0), device=self.mean.device))
        self.std.copy_(torch.as_tensor(np.maximum(values.std(0), .01), device=self.std.device))

    def forward(self, batch):
        x = (batch['onset_audio'].float() - self.mean) / self.std
        x = x * batch['onset_mask'][..., None]
        scores = self.score(self.encoder(x.transpose(1, 2))).squeeze(1).float()
        pos = batch['onset_positions'].clamp(0, scores.size(1)-1)
        left = pos.long()
        right = (left+1).clamp_max(scores.size(1)-1)
        fraction = pos-left
        return scores.gather(1,left)*(1-fraction)+scores.gather(1,right)*fraction


def apply_phase_evidence(logits, scores, queries, token_ids, *, start=0, strength=1.):
    if strength == 0:
        return logits
    selected = [(r, p-start, tile) for r, p, tile in queries if start <= p < start+logits.size(1)]
    if not selected:
        return logits
    rows, positions, tiles = torch.tensor(selected, device=logits.device, dtype=torch.long).unbind(1)
    flat_positions = rows * logits.size(1) + positions
    selected_logits = logits.flatten(0, 1).index_select(0, flat_positions).float()
    old = selected_logits.index_select(1, token_ids)
    change = scores.index_select(0, tiles) * strength
    shifted = old + change
    # Subtract the CHANGE in group mass. In particular, zero initialization is
    # exactly zero; do not subtract and re-add the full LSE to each score.
    shift = torch.logsumexp(shifted, -1) - torch.logsumexp(old, -1)
    updated = shifted - shift[:, None]
    selected_logits = selected_logits.index_copy(1, token_ids, updated)
    return logits.float().flatten(0, 1).index_copy(0, flat_positions, selected_logits).reshape_as(logits)
