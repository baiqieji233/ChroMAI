"""Checkpoint contract for chart_composer_v1."""
from __future__ import annotations
from pathlib import Path
import os, tempfile
import torch
from . import CHECKPOINT_SCHEMA,FAMILY,INPUT_CONTRACT,TRAINING_SEMANTICS
from .model import ChartComposer,ComposerConfig


def save(path, model, optimizer, *, epoch, args, manifest, metrics=None, optimizer_steps=0, atom_catalog=()):
    payload={'schema':CHECKPOINT_SCHEMA,'family':FAMILY,'input_contract':INPUT_CONTRACT,
             'model_config':model.config.to_dict(),'model_state_dict':model.state_dict(),
             'optimizer_state_dict':optimizer.state_dict() if optimizer is not None else None,
             'epoch':int(epoch),'optimizer_steps':int(optimizer_steps),'args':dict(args),
             'manifest':str(Path(manifest).resolve()),'metrics':metrics or {},
             'training_semantics':TRAINING_SEMANTICS,'atom_catalog':sorted(set(atom_catalog))}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent);os.close(fd)
    temp=Path(name)
    try: torch.save(payload,temp);temp.replace(path)
    finally: temp.unlink(missing_ok=True)
    return payload


def validate(payload):
    if payload.get('schema')!=CHECKPOINT_SCHEMA or payload.get('family')!=FAMILY:
        raise ValueError('Not a chart_composer_v1 checkpoint')
    if payload.get('input_contract')!=INPUT_CONTRACT:raise ValueError('Composer input contract mismatch')
    return ComposerConfig(**payload['model_config'])


def load(path,device='cpu'):
    payload=torch.load(path,map_location='cpu',weights_only=False);config=validate(payload)
    model=ChartComposer(config);model.load_state_dict(payload['model_state_dict']);model.to(device)
    return model,payload


def inspect_payload(payload):
    try: config=validate(payload);valid=True;error=None
    except (ValueError,KeyError,TypeError) as exc: config=None;valid=False;error=str(exc)
    return {'valid':valid,'family':payload.get('family'),'schema':payload.get('schema'),
            'model_config':config.to_dict() if config else payload.get('model_config'),
            'epoch':payload.get('epoch'),'optimizer_steps':payload.get('optimizer_steps',0),
            'training_semantics':payload.get('training_semantics'),'error':error}
