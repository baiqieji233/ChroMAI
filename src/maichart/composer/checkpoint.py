"""Checkpoint contract for chart_composer_v1."""
from __future__ import annotations
from pathlib import Path
import copy,os,tempfile
import torch
from . import CHECKPOINT_SCHEMA,FAMILY,INPUT_CONTRACT,TRAINING_SEMANTICS
from .model import ChartComposer,ComposerConfig


def _cpu_snapshot(value,memo=None):
    """Keep CUDA storage copying out of torch.save's storage serializer.

    Bounded tensor copies avoid a single large storage.cpu() operation. The
    training thread is paused during this snapshot, so parameters and optimizer
    states represent the same update. CPU-only callers do not copy tensors.
    """
    memo={} if memo is None else memo
    if torch.is_tensor(value):
        if value.device.type=='cpu':return value
        if id(value) in memo:return memo[id(value)]
        source=value.detach().contiguous().reshape(-1)
        result=torch.empty(value.shape,dtype=value.dtype,device='cpu')
        target=result.reshape(-1)
        chunk=max(1,(8*1024**2)//value.element_size())
        for start in range(0,source.numel(),chunk):
            target[start:start+chunk].copy_(source[start:start+chunk],non_blocking=False)
        memo[id(value)]=result
        return result
    if isinstance(value,dict):
        result=copy.copy(value)
        for key,item in value.items():result[key]=_cpu_snapshot(item,memo)
        return result
    if isinstance(value,list):return [_cpu_snapshot(item,memo) for item in value]
    if isinstance(value,tuple):return tuple(_cpu_snapshot(item,memo) for item in value)
    return value


def _write_payload(path, payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    # Complete device transfers before opening the output file. A blocked
    # transfer cannot leave a misleading partially written new checkpoint.
    payload=_cpu_snapshot(payload)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent);os.close(fd)
    temp=Path(name)
    try: torch.save(payload,temp);temp.replace(path)
    finally: temp.unlink(missing_ok=True)


def save(path, model, optimizer, *, epoch, args, manifest, metrics=None, optimizer_steps=0,
         atom_catalog=(), dataset_fingerprint=None, training_state=None):
    payload={'schema':CHECKPOINT_SCHEMA,'family':FAMILY,'input_contract':INPUT_CONTRACT,
             'model_config':model.config.to_dict(),'model_state_dict':model.state_dict(),
             'optimizer_state_dict':optimizer.state_dict() if optimizer is not None else None,
             'epoch':int(epoch),'optimizer_steps':int(optimizer_steps),'args':dict(args),
             'manifest':str(Path(manifest).resolve()),'metrics':metrics or {},
             'training_semantics':TRAINING_SEMANTICS,'atom_catalog':sorted(set(atom_catalog)),
             'dataset_fingerprint':dataset_fingerprint,
             'training_state':(training_state or {}) if optimizer is not None else {},
             'checkpoint_kind':'training' if optimizer is not None else 'inference'}
    _write_payload(path,payload)
    return payload


def export_inference(source, destination):
    """Lossless FP32 weights export; retain the complete resumable source."""
    if Path(source).resolve()==Path(destination).resolve():
        raise ValueError('Inference export must not overwrite the training checkpoint')
    payload=read_payload(source)
    validate(payload)
    payload.pop('optimizer_state_dict',None)
    payload.pop('training_state',None)
    payload['checkpoint_kind']='inference'
    _write_payload(destination,payload)
    return payload


def read_payload(path):
    # Lazy CPU mappings keep optimizer tensors out of RAM for inspect/generate.
    # Torch's old non-ZIP serialization cannot be mapped; support those too.
    try:
        return torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    except RuntimeError as error:
        if 'mmap' not in str(error).lower(): raise
        return torch.load(path,map_location='cpu',weights_only=False)


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


def load(path,device='cpu',*,include_optimizer=False):
    # A resumable optimizer owns and mutates CPU tensors for the entire run.
    # Materialize those rather than retaining a mapping of a file that Windows
    # must later replace at the same checkpoint path.
    payload=(torch.load(path,map_location='cpu',weights_only=False) if include_optimizer else read_payload(path))
    config=validate(payload)
    if not include_optimizer:
        payload.pop('optimizer_state_dict',None)
        payload.pop('training_state',None)
    model=ChartComposer(config);model.load_state_dict(payload['model_state_dict']);model.to(device)
    return model,payload


def inspect_payload(payload):
    try: config=validate(payload);valid=True;error=None
    except (ValueError,KeyError,TypeError) as exc: config=None;valid=False;error=str(exc)
    return {'valid':valid,'family':payload.get('family'),'schema':payload.get('schema'),
            'model_config':config.to_dict() if config else payload.get('model_config'),
            'epoch':payload.get('epoch'),'optimizer_steps':payload.get('optimizer_steps',0),
            'training_semantics':payload.get('training_semantics'),'error':error,
            'args':payload.get('args',{}),
            'can_resume':bool(valid and payload.get('optimizer_state_dict') is not None),
            'can_init':valid,'can_train':valid,'can_generate':valid,
            'parameters':sum(value.numel() for value in payload.get('model_state_dict',{}).values()
                             if torch.is_tensor(value)),
            'checkpoint_kind':payload.get('checkpoint_kind','training'),
            'model_bytes':sum(value.numel()*value.element_size()
                              for value in payload.get('model_state_dict',{}).values() if torch.is_tensor(value))}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Export lossless Composer inference weights')
    parser.add_argument('source');parser.add_argument('destination')
    options=parser.parse_args()
    export_inference(options.source,options.destination)
    print(str(Path(options.destination).resolve()),flush=True)
