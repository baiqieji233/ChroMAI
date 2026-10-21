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

TRAINING_OBJECTIVE_CONTRACT='audio_fidelity_position_mask_v1'


LEGACY_PLAN_DIM = 8
CURRENT_PLAN_DIM = 10


def _migrate_plan_state(state, config):
    """Expand the legacy planner heads without changing their trained prefix."""
    if config.plan_dim != LEGACY_PLAN_DIM:
        raise ValueError('Plan migration source must have plan_dim=8')
    width = config.d_model
    expected = {
        'plan_head.weight': (LEGACY_PLAN_DIM, width),
        'plan_head.bias': (LEGACY_PLAN_DIM,),
    }
    if config.grounding_profile != 'none':
        expected['grounding.intent_projection.weight'] = (width, LEGACY_PLAN_DIM)
    migrated = dict(state)
    for name, shape in expected.items():
        value = migrated.get(name)
        if value is None or tuple(value.shape) != shape:
            raise ValueError(f'Chart-stream plan migration parameter {name} has an unexpected shape')
        if name.endswith('intent_projection.weight'):
            target_shape = (width, CURRENT_PLAN_DIM)
            expanded = value.new_zeros(target_shape)
            expanded[:, :LEGACY_PLAN_DIM] = value
        else:
            target_shape = ((CURRENT_PLAN_DIM, width) if name.endswith('weight')
                            else (CURRENT_PLAN_DIM,))
            expanded = value.new_zeros(target_shape)
            expanded[:LEGACY_PLAN_DIM] = value
        migrated[name] = expanded
    return migrated


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
        'training_objective_contract':(TRAINING_OBJECTIVE_CONTRACT if str(args.get('training_semantics','')).startswith(
            ('chart_stream_training_v14_','chart_stream_training_v15_')) else None),
        'grounding_training': {'profile':model.config.grounding_profile,
            'motor_profile':model.config.motor_profile,
            'hand_features_trained':bool(model.config.grounding_profile!='none'
                and model.config.motor_profile=='causal_hands_v1' and optimizer_steps>0),
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
    if (config.audio_dim, config.condition_dim, config.action_dim) != (120, 16, 50) or config.plan_dim not in (8, 10):
        raise ValueError('Chart-stream checkpoint 输入/规划维度不兼容')
    if config.vocab_size != len(expected['tokens']):
        raise ValueError('Chart-stream checkpoint codec vocabulary size 不兼容')
    if config.grounding_profile!='none' and config.motor_profile=='causal_hands_v1':
        from .hand_tracking import HAND_DIM
        weight=(payload.get('model_state_dict') or {}).get('grounding.hand_projection.weight')
        if not torch.is_tensor(weight) or tuple(weight.shape)!=(config.d_model,HAND_DIM):
            raise ValueError('Checkpoint declares causal hands but its hand_projection is missing or incompatible')
    return config


def load(path, device='cpu'):
    device = resolve_device(device)
    # Keep saved optimizer tensors on CPU during inference.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    config = validate_contract(payload)
    plan_migration = config.plan_dim == LEGACY_PLAN_DIM
    model_config = replace(config, plan_dim=CURRENT_PLAN_DIM) if plan_migration else config
    model = ChartStreamTransformer(model_config)
    obsolete = {'memory_projection.weight', 'memory_projection.bias',
                'plan_projection.weight', 'plan_projection.bias',
                'event_gate.weight', 'event_gate.bias',
                'grounding.blueprint_head.weight', 'grounding.blueprint_head.bias',
                'grounding.blueprint_gate.0.weight', 'grounding.blueprint_gate.0.bias',
                'grounding.blueprint_projection.weight'}
    state = {key: value for key, value in payload['model_state_dict'].items() if key not in obsolete}
    if plan_migration:
        state = _migrate_plan_state(state, config)
    kinematic_key = 'grounding.kinematic_projection.weight'
    if kinematic_key in state and hasattr(model, 'grounding'):
        target_weight = model.grounding.kinematic_projection.weight
        saved_weight = state[kinematic_key]
        if saved_weight.ndim == 2 and saved_weight.size(0) == target_weight.size(0) and saved_weight.size(1) < target_weight.size(1):
            padded = torch.zeros_like(target_weight, dtype=saved_weight.dtype, device=saved_weight.device)
            padded[:, :saved_weight.size(1)] = saved_weight
            state[kinematic_key] = padded
    try:
        result = model.load_state_dict(state, strict=False)
    except RuntimeError as exc:
        raise ValueError(f'Chart-stream checkpoint 参数不兼容：{exc}') from exc
    from .grounded_model import UPGRADE_PARAMETERS
    missing = [key for key in result.missing_keys if key not in UPGRADE_PARAMETERS]
    if missing or result.unexpected_keys:
        raise ValueError(f'Chart-stream checkpoint 参数不兼容：missing={missing[:8]} unexpected={result.unexpected_keys[:8]}')
    # Registered upgrade branches are zero-initialised: an older checkpoint
    # keeps exactly its trained function until it is finetuned.
    payload['upgraded_parameters'] = sorted(set(result.missing_keys) & set(UPGRADE_PARAMETERS))
    if plan_migration:
        payload['plan_dim_migration'] = {
            'from': LEGACY_PLAN_DIM, 'to': CURRENT_PLAN_DIM,
            'zero_initialized': ('plan_head.weight[8:10]', 'plan_head.bias[8:10]',
                                 'grounding.intent_projection.weight[:,8:10]'
                                 if config.grounding_profile != 'none' else None),
        }
    model.to(device)
    return model, payload


def upgrade_grounding(model,**settings):
    """Explicit finetune migration; no old checkpoint is rewritten."""
    target=ChartStreamTransformer(replace(model.config,**settings))
    if model.config.motor_profile=='causal_hands_v1' and target.config.motor_profile=='legacy_v13':
        raise ValueError('Cannot silently remove trained hand features during finetune')
    result=target.load_state_dict(model.state_dict(),strict=False)
    if result.unexpected_keys or any(not key.startswith('grounding.') for key in result.missing_keys):
        raise ValueError('Unsupported stream migration outside grounding branches')
    if (model.config.motor_profile!=target.config.motor_profile
            and hasattr(target,'grounding')):
        # Channel 23 changes from legacy run-dependent Touch decay to a true
        # known buttonless-event indicator. Do not reuse its old calibration.
        with torch.no_grad():target.grounding.kinematic_projection.weight[:,23].zero_()
    return target

