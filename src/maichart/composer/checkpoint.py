"""Checkpoint contract for chart_composer_v1."""
from __future__ import annotations
from pathlib import Path
import os, tempfile
import torch
from . import CHECKPOINT_SCHEMA,FAMILY,INPUT_CONTRACT,TRAINING_SEMANTICS
from .model import ChartComposer,ComposerConfig


def save(path, model, optimizer, *, epoch, args, manifest, metrics=None, optimizer_steps=0,
         atom_catalog=(), dataset_fingerprint=None):
    payload={'schema':CHECKPOINT_SCHEMA,'family':FAMILY,'input_contract':INPUT_CONTRACT,
             'model_config':model.config.to_dict(),'model_state_dict':model.state_dict(),
             'optimizer_state_dict':optimizer.state_dict() if optimizer is not None else None,
             'epoch':int(epoch),'optimizer_steps':int(optimizer_steps),'args':dict(args),
             'manifest':str(Path(manifest).resolve()),'metrics':metrics or {},
             'training_semantics':TRAINING_SEMANTICS,'atom_catalog':sorted(set(atom_catalog)),
             'dataset_fingerprint':dataset_fingerprint}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent);os.close(fd)
    temp=Path(name)
    try: torch.save(payload,temp);temp.replace(path)
    finally: temp.unlink(missing_ok=True)
    return payload


def validate(payload):
    if payload.get('schema')!=CHECKPOINT_SCHEMA or payload.get('family')!=FAMILY:
        raise ValueError('Not a chart_composer_v2 checkpoint')
    if payload.get('input_contract')!=INPUT_CONTRACT:raise ValueError('Composer input contract mismatch')
    if payload.get('training_semantics') not in {None, TRAINING_SEMANTICS}:
        raise ValueError('Composer training semantics mismatch')
    config=ComposerConfig(**payload['model_config'])
    expected={'audio_dim':120,'condition_dim':16,'action_dim':50,'candidate_feature_dim':125,
              'history_feature_dim':8,'future_dim':8}
    for key,value in expected.items():
        if getattr(config,key)!=value:
            raise ValueError(f'Composer input contract dimension mismatch: {key}={getattr(config,key)!r}')
    catalog=payload.get('atom_catalog')
    if not isinstance(catalog,list) or len(catalog)+1!=config.atom_vocab_size:
        raise ValueError('Composer checkpoint atom catalog does not match atom_vocab_size')
    return config


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
