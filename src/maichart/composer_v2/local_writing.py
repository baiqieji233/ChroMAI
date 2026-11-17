"""Audio-predicted local writing tendencies shared by training and generation."""
from bisect import bisect_left

import numpy as np
import torch
from torch import nn

from .ordered_configuration import (
    BEAT_INTERVALS, FEATURES, KINDS, _event_atoms,
)

POLICY = 'audio_writing_v2'
SOURCE_RANGES = ((-4, 0), (0, 2), (2, 4), (4, 8))
BASE_FEATURES = tuple(name for name in FEATURES if not name.startswith('position_'))
FIELDS = (*BASE_FEATURES, *(f'direction_{i}' for i in range(8)), 'chord')
WIDTH = len(FIELDS)
HORIZONS = len(BEAT_INTERVALS)
_BASE_COLUMNS = tuple(FEATURES.index(name) for name in BASE_FEATURES)
_KIND_END = 1 + len(KINDS)
_STEP_END = _KIND_END + 5
GROUPS = ((0, 1), (1, _KIND_END), (_KIND_END, _STEP_END),
          (_STEP_END, len(BASE_FEATURES)), (len(BASE_FEATURES), WIDTH - 1), (WIDTH - 1, WIDTH))
# Native profile uses 256 semantic values and 12 onset bins.
INPUT_WIDTH = (256 + 12 + 12 + 1) * len(SOURCE_RANGES) + 1


def audio_rows(profile, clock):
    """One compact row per integer beat, computed from the existing audio cache."""
    semantic = np.pad(profile.semantic, ((0, 0), (0, max(0, 256 - profile.semantic.shape[1]))))
    block = np.concatenate((semantic, profile.onset, np.log1p(profile.onset_raw)), axis=1)
    if block.shape[1] != 280:
        raise ValueError('Local writing requires the 256-dimensional beat music profile')
    count = len(block)
    sums = np.concatenate((np.zeros((1, block.shape[1])), block.cumsum(0, dtype=np.float64)))
    anchors = np.arange(count + 1)
    parts = []
    for left, right in SOURCE_RANGES:
        a = np.clip(anchors + left, 0, count)
        b = np.clip(anchors + right, 0, count)
        parts.append((sums[b] - sums[a]) / np.maximum(1, b - a)[:, None])
        parts.append(((b - a) / (right - left))[:, None])
    seconds = clock.at_floats(anchors.astype(np.float64))
    beat_seconds = clock.at_floats(anchors.astype(np.float64) + 1) - seconds
    parts.append(np.log(np.maximum(beat_seconds, 1e-6))[:, None])
    return np.concatenate(parts, axis=1).astype(np.float32)


def inputs(rows, song, anchors):
    anchors = np.asarray(anchors, dtype=np.int64)
    indices = np.clip(anchors, 0, len(rows) - 1)
    values = rows[indices].copy()
    values[(anchors < 0) | (anchors >= len(rows))] = 0
    memory = np.asarray(song.summary['beats'], dtype=np.float64)
    right = np.searchsorted(memory, anchors).clip(0, len(memory) - 1)
    left = np.maximum(0, right - 1)
    mix = np.clip((anchors - memory[left]) / np.maximum(memory[right] - memory[left], 1e-6), 0, 1)
    return dict(writing_audio=torch.from_numpy(values), writing_left=torch.from_numpy(left),
                writing_right=torch.from_numpy(right), writing_mix=torch.as_tensor(mix, dtype=torch.float32))


def _motion_rows(configuration, song):
    cached = getattr(configuration, '_writing_motion', None)
    if cached is not None:
        return cached
    movement = np.zeros((len(song.points), 9), dtype=np.float32)
    available = np.zeros(len(song.points), dtype=np.bool_)
    previous = ()
    from .generate import _atom_heads
    for index, point in enumerate(song.points):
        _, _, buttons = _event_atoms(point.note_text)
        movement[index, -1] = len(set(_atom_heads(point.note_text))) > 1
        if len(previous) == len(buttons) == 1 and index and configuration.event_weights[index - 1, 1] > 0:
            movement[index, (buttons[0] - previous[0]) % 8] = 1
            available[index] = True
        previous = buttons
    configuration._writing_motion = movement, available
    return movement, available


def add_targets(item, configuration, song):
    """Future notes are read here only and never enter planner inputs."""
    mass = float(item['token_weights'].sum() * item['sample_weight'])
    anchors = item['anchors']
    targets = np.zeros((len(anchors), HORIZONS, WIDTH), dtype=np.float32)
    weights = np.zeros_like(targets)
    supervised = {int(i) for i, weight in zip(item['tile_ids'], item['token_weights']) if weight > 0}
    movement, available = _motion_rows(configuration, song)
    cached = getattr(configuration, '_writing_targets', None)
    if cached is None:
        cached = configuration._writing_targets = {}
    for index in supervised:
        beat = anchors[index]
        if beat not in cached:
            first = bisect_left(configuration.beats, beat)
            base, trust = configuration.target_at(first, beat)
            values = np.zeros((HORIZONS, WIDTH), dtype=np.float32)
            valid = np.zeros_like(values)
            values[:, :len(BASE_FEATURES)] = base[2:][:, _BASE_COLUMNS]
            valid[:, :len(BASE_FEATURES)] = trust[2:][:, _BASE_COLUMNS]
            for row, (left, right) in enumerate(BEAT_INTERVALS):
                a = bisect_left(configuration.beats, beat + left)
                b = bisect_left(configuration.beats, beat + right)
                quality = configuration.quality.covered(*song.clock.at_floats(np.array([beat + left, beat + right])))
                if b > a and quality > 0:
                    use = available[a:b]
                    if use.any():
                        values[row, len(BASE_FEATURES):-1] = movement[a:b][use, :-1].mean(0)
                        valid[row, len(BASE_FEATURES):-1] = quality
                    values[row, -1] = movement[a:b, -1].mean()
                    valid[row, -1] = quality
            for left, right in GROUPS:
                valid[:, left:right] /= (right - left) * len(GROUPS)
            cached[beat] = values, valid
        targets[index], weights[index] = cached[beat]
    # Preserve quality attenuation and exact accumulation across unequal batches.
    weights *= mass / max(1, len(supervised) * HORIZONS)
    item['writing_targets'] = torch.from_numpy(targets)
    item['writing_weights'] = torch.from_numpy(weights)
    item['curve_teacher_allowed'] = False


def collate(items, out):
    if 'writing_audio' not in items[0]:
        return
    for name in ('writing_audio', 'writing_left', 'writing_right', 'writing_mix'):
        out[name] = torch.cat([item[name] for item in items])
    out['writing_owner'] = torch.cat([
        torch.full((len(item['anchors']),), index, dtype=torch.long) for index, item in enumerate(items)])
    if 'writing_targets' in items[0]:
        out['future_configuration_targets'] = torch.cat([item['writing_targets'] for item in items])
        out['future_configuration_weights'] = torch.cat([item['writing_weights'] for item in items])


class LocalWriting(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        width = min(128, dimension)
        self.audio = nn.Sequential(nn.LayerNorm(INPUT_WIDTH), nn.Linear(INPUT_WIDTH, width), nn.SiLU())
        self.song = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, width, bias=False))
        self.level = nn.Linear(1, width, bias=False)
        self.predictor = nn.Sequential(nn.SiLU(), nn.Linear(width, HORIZONS * WIDTH))
        self.projection = nn.Linear(HORIZONS * WIDTH, dimension, bias=False)
        nn.init.zeros_(self.projection.weight)

    def plan(self, memory, batch):
        owner = batch['writing_owner']
        left = memory[owner, batch['writing_left']]
        right = memory[owner, batch['writing_right']]
        mix = batch['writing_mix'][:, None]
        song = left * (1 - mix) + right * mix
        level = batch['condition'][owner, :1].float()
        hidden = self.audio(batch['writing_audio'].float()) + self.song(song) + self.level(level)
        logits = self.predictor(hidden).reshape(-1, HORIZONS, WIDTH)
        # The same predicted distribution enters both training and generation.
        context = self.projection(logits.float().sigmoid().flatten(1))
        return dict(writing_logits=logits, writing_context=context)

    def read(self, context, batch, start, value, role_mask):
        indices = torch.as_tensor([row[start:] for row in batch['tile_ids_cpu']],
                                  dtype=torch.long, device=context.device)
        result = context[indices.clamp_min(0)]
        return result * (role_mask[value] & indices.ge(0))[..., None]
