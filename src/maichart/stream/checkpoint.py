"""Independent chart_stream_v1 checkpoint schema."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from dataclasses import asdict

import torch

from . import CHECKPOINT_SCHEMA, FAMILY, INPUT_CONTRACT
from .model import ChartStreamTransformer, StreamConfig
from maichart.form.codec import RationalCodec
from .runtime import resolve_device


def _identity(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {'path': str(path), 'size': path.stat().st_size,
            'mtime_ns': path.stat().st_mtime_ns,
            'sha256': hashlib.sha256(data).hexdigest()}


def save(path, model, optimizer, *, epoch, args, manifest, metrics=None, atom_catalog=(),
         optimizer_steps=0, training_position=None):
    payload = {
        'schema': CHECKPOINT_SCHEMA, 'family': FAMILY, 'input_contract': INPUT_CONTRACT,
        'model_config': asdict(model.config), 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'epoch': int(epoch), 'optimizer_steps': int(optimizer_steps),
        'training_position': training_position or {}, 'args': dict(args), 'manifest': _identity(manifest),
        'codec': {'schema': 'rational-events-v1', 'tokens': RationalCodec().tokens},
        'atom_catalog': sorted(set(atom_catalog)),
        'metrics': metrics or {}, 'architecture': model.architecture,
        'metrics_aggregation': 'batch_mean',
    }
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temp); temp.replace(path)
    return payload


def load(path, device='cpu'):
    device = resolve_device(device)
    # A training checkpoint includes two Adam moments per parameter. Loading
    # the entire payload on CUDA wastes VRAM during inference. Restore on CPU
    # and transfer only the model's live parameters to the selected device.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('schema') != CHECKPOINT_SCHEMA or payload.get('family') != FAMILY:
        raise ValueError('Not a chart_stream_v1 checkpoint')
    if payload.get('input_contract') != INPUT_CONTRACT:
        raise ValueError('Chart-stream checkpoint 输入合同不兼容')
    config = StreamConfig(**payload['model_config'])
    model = ChartStreamTransformer(config)
    obsolete = {'memory_projection.weight', 'memory_projection.bias',
                'plan_projection.weight', 'plan_projection.bias',
                'event_gate.weight', 'event_gate.bias'}
    state = {key: value for key, value in payload['model_state_dict'].items() if key not in obsolete}
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f'Chart-stream checkpoint 参数不兼容：{exc}') from exc
    model.to(device)
    return model, payload

