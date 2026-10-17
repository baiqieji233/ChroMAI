"""CPU preparation of validated, compact attention batches.

Packing removes padding keys only, after each branch has received its original
position/branch/geometry embedding. It neither truncates authored context nor
changes checkpoint shapes. The final three global tokens retain their slots at
the end, including the relation-head token.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from maichart.form.training_runtime import validate_cpu_batch


def add_memory_packing(batch):
    """Build gather indices on CPU, avoiding dynamic CUDA nonzero/sort syncs."""
    size = len(batch['task'])
    masks = []

    def array(value):
        return value.detach().numpy()

    def add(key, mask=None):
        value = batch.get(key)
        if value is None or value.shape[1] == 0:
            return
        padding = batch.get(key + '_padding_mask') if mask is None else mask
        masks.append(np.zeros(value.shape[:2], dtype=bool) if padding is None
                     else np.asarray(array(padding) if torch.is_tensor(padding) else padding, dtype=bool))

    add('audio'); add('form')
    tasks = array(batch['task'])
    for key in ('history', 'motif', 'draft', 'right'):
        if key not in batch:
            continue
        padding = batch.get(key + '_padding_mask')
        padding = (array(batch[key]) == 0 if padding is None else array(padding).copy())
        if key in ('draft', 'right'):
            padding = padding | (tasks == 0)[:, None]
        if key == 'motif' and 'memory_strength' in batch:
            padding = padding | (array(batch['memory_strength']).reshape(size, 1) == 0)
        add(key, padding)
    add('action_state')
    if 'load_curve' in batch:
        curve = batch['load_curve']
        visible = batch.get('load_curve_condition_mask')
        if visible is None:
            visible = np.ones(curve.shape, dtype=bool)
        elif visible.ndim == 2:
            visible = np.broadcast_to(~array(visible)[..., None], curve.shape)
        else:
            visible = array(visible)
        padding = batch.get('load_curve_padding_mask',
                            batch.get('load_curve_condition_mask_padding_mask'))
        padding = (np.zeros(curve.shape[:2], dtype=bool) if padding is None else array(padding))
        add('load_curve', padding | ~visible.any(-1))
    if 'skeleton' in batch:
        add('skeleton', array(batch['skeleton']) == 0)
    if 'span_bounds' in batch:
        masks.append(np.zeros((size, 1), dtype=bool))
    local = np.concatenate(masks, axis=1) if masks else np.zeros((size, 0), dtype=bool)
    lengths = (~local).sum(1)
    width = int(lengths.max(initial=0))
    indices = np.zeros((size, width + 3), dtype=np.int64)
    padding = np.ones(indices.shape, dtype=bool)
    for row in range(size):
        kept = np.flatnonzero(~local[row])
        indices[row, :len(kept)] = kept
        padding[row, :len(kept)] = False
    indices[:, -3:] = np.arange(local.shape[1], local.shape[1] + 3)
    padding[:, -3:] = False
    batch['memory_gather_index'] = torch.from_numpy(indices)
    batch['memory_gather_padding_mask'] = torch.from_numpy(padding)
    batch['memory_source_length'] = local.shape[1] + 3
    batch['memory_packing'] = {'dense_tokens': local.shape[1] + 3,
                               'packed_tokens': width + 3,
                               'valid_tokens': int(lengths.sum()) + size * 3}
    return batch


@dataclass(frozen=True)
class TrainingCollator:
    """Spawn-safe collation; validation runs in the prefetch/loader producer."""

    vocab_size: int
    attention_padding: str = 'compact'

    def __call__(self, items):
        from maichart.form.data import collate_form
        batch = collate_form(items)
        validate_cpu_batch(batch, vocab_size=self.vocab_size)
        if self.attention_padding == 'compact':
            add_memory_packing(batch)
        elif self.attention_padding != 'dense':
            raise ValueError('attention_padding must be compact or dense')
        return batch


# These clocks have already constructed labels/geometry on CPU. The layout
# model consumes the resulting geometry, not the raw timestamps or duplicate
# padding masks generated for them. Planner transfer is deliberately separate.
_CPU_ONLY = frozenset(('audio_times_sec', 'audio_times_sec_padding_mask',
    'load_curve_times_sec', 'load_curve_times_sec_padding_mask',
    'audio_geometry_padding_mask', 'history_geometry_padding_mask',
    'load_curve_geometry_padding_mask'))


def transfer_layout_batch(batch, device):
    device = torch.device(device)
    return {key: value.to(device, non_blocking=device.type == 'cuda')
            if torch.is_tensor(value) and key not in _CPU_ONLY else value
            for key, value in batch.items()}
