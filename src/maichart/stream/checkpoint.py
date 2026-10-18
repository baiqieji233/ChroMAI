"""Independent chart_stream_v1 checkpoint schema."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import shutil
from pathlib import Path
from dataclasses import asdict,replace

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
         optimizer_steps=0, training_position=None, scaler=None, dataset_fingerprint=None,aliases=()):
    payload = {
        'schema': CHECKPOINT_SCHEMA, 'family': FAMILY, 'input_contract': INPUT_CONTRACT,
        'model_config': asdict(model.config), 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'dataset_fingerprint': dataset_fingerprint,
        'epoch': int(epoch), 'optimizer_steps': int(optimizer_steps),
        'training_position': training_position or {}, 'args': dict(args), 'manifest': _identity(manifest),
        'codec': {'schema': 'rational-events-v1', 'tokens': RationalCodec().tokens},
        'atom_catalog': sorted(set(atom_catalog)),
        'metrics': metrics or {}, 'architecture': model.architecture,
        'metrics_aggregation': 'batch_mean',
        'grounding_training': {'profile':model.config.grounding_profile,
            'optimizer_steps':int(optimizer_steps),
            'alignment_supervised':bool(model.config.grounding_profile!='none' and optimizer_steps>0
                and args.get('alignment_loss_weight',0)>0 and (metrics or {}).get('alignment_targets',0)>0)},
    }
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    temp = Path(name)
    try:
        torch.save(payload, temp)
        temp.replace(path)
        # Step and last snapshots have identical contents. Serialize CUDA
        # tensors once; publish independent immutable names atomically.
        for alias in aliases:
            alias=Path(alias)
            if alias.resolve()==path.resolve():continue
            alias.parent.mkdir(parents=True,exist_ok=True)
            fd,name=tempfile.mkstemp(prefix=alias.name+'.',suffix='.tmp',dir=alias.parent)
            os.close(fd);alias_temp=Path(name);alias_temp.unlink()
            try:
                try:os.link(path,alias_temp)
                except OSError:shutil.copyfile(path,alias_temp)
                alias_temp.replace(alias)
            finally:alias_temp.unlink(missing_ok=True)
    finally:
        temp.unlink(missing_ok=True)
    return payload


def validate_contract(payload):
    if payload.get('schema') != CHECKPOINT_SCHEMA or payload.get('family') != FAMILY:
        raise ValueError('Not a chart_stream_v1 checkpoint')
    if payload.get('input_contract') != INPUT_CONTRACT:
        raise ValueError('Chart-stream checkpoint 输入合同不兼容')
    expected = RationalCodec().to_dict()
    codec = payload.get('codec') or {}
    if codec.get('schema') != expected['schema'] or list(codec.get('tokens', ())) != expected['tokens']:
        raise ValueError('Chart-stream checkpoint codec token 顺序不兼容')
    config = StreamConfig(**payload['model_config'])
    if (config.audio_dim, config.condition_dim, config.action_dim, config.plan_dim) != (120, 16, 50, 8):
        raise ValueError('Chart-stream checkpoint 输入/规划维度不兼容')
    if config.vocab_size != len(expected['tokens']):
        raise ValueError('Chart-stream checkpoint codec vocabulary size 不兼容')
    return config


def load(path, device='cpu'):
    device = resolve_device(device)
    # Keep saved optimizer tensors on CPU during inference.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    config = validate_contract(payload)
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


def upgrade_grounding(model,**settings):
    """Explicit finetune migration; no old checkpoint is rewritten."""
    target=ChartStreamTransformer(replace(model.config,**settings))
    result=target.load_state_dict(model.state_dict(),strict=False)
    if result.unexpected_keys or any(not key.startswith('grounding.') for key in result.missing_keys):
        raise ValueError('Unsupported stream migration outside grounding branches')
    return target

