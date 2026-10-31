"""Isolated checkpoint contract for causal whole-event Composer models."""
from dataclasses import asdict
from pathlib import Path
import os
import torch

SCHEMA='maichart-composer-event-set-checkpoint-v1'
FAMILY='chart_composer_v1'
PROFILE='event_set_stream_v1'

def read(path):
    try:return torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    except RuntimeError:return torch.load(path,map_location='cpu',weights_only=False)

def save(path,model,optimizer=None,*,epoch=0,optimizer_steps=0,args=None,state=None,metrics=None,delta_vocab=None,condition_defaults=None):
    payload=dict(schema=SCHEMA,family=FAMILY,model_config=model.config.to_dict(),
        model_state_dict=model.state_dict(),event_set_codec=model.codec.to_dict(),
        delta_vocab=delta_vocab.to_dict(),epoch=int(epoch),optimizer_steps=int(optimizer_steps),
        args=dict(args or {}),training_state=state,metrics=dict(metrics or {}),
        condition_defaults=condition_defaults,checkpoint_kind='full' if optimizer is not None else 'inference')
    if optimizer is not None:payload['optimizer_state_dict']=optimizer.state_dict()
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    try:torch.save(payload,temporary);os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()
    return payload

def load(path,device='cpu',*,include_optimizer=False):
    from .event_set_codec import EventSetCodec
    from .stream_model import StreamComposer,StreamComposerConfig
    payload=(torch.load(path,map_location='cpu',weights_only=False) if include_optimizer else read(path))
    if payload.get('schema')!=SCHEMA or payload.get('family')!=FAMILY:
        raise ValueError('Not an event-set stream checkpoint; start fresh for the new architecture')
    codec=EventSetCodec.from_dict(payload['event_set_codec'])
    model=StreamComposer(StreamComposerConfig.from_dict(payload['model_config']),codec).to(device)
    model.load_state_dict(payload['model_state_dict'],strict=True)
    return model,payload

def inspect_payload(payload):
    from .event_set_codec import EventSetCodec
    from .stream_model import StreamComposerConfig
    from .stream_continuation import continuation_dim
    if payload.get('schema')!=SCHEMA:raise ValueError('Unsupported event-set checkpoint schema')
    config=StreamComposerConfig.from_dict(payload['model_config'])
    codec=EventSetCodec.from_dict(payload['event_set_codec'])
    return dict(valid=True,family=FAMILY,schema=SCHEMA,model_config=config.to_dict(),
        epoch=payload.get('epoch',0),optimizer_steps=payload.get('optimizer_steps',0),
        args=payload.get('args',{}),can_resume=bool(payload.get('optimizer_state_dict') and payload.get('training_state')),
        can_init=True,checkpoint_kind=payload.get('checkpoint_kind'),
        atom_vocab_size=codec.num_atoms,event_shapes=codec.num_shapes,codec_coverage=codec.coverage,
        generation_capabilities=dict(composer_generation=True,event_set_stream=True,rational_event_positions=True,
            absolute_onset_prediction=getattr(config,'timing_output_policy','relative_delta_v1')=='absolute_grid_v1',
            semantic_event_families=getattr(config,'event_family_policy','legacy_v1')!='legacy_v1',
            audio_grounded_event_families=getattr(config,'event_family_policy','legacy_v1') in ('audio_semantic_v2','audio_prior_transition_v3'),
            music_conditioned_transition_exit=getattr(config,'transition_policy','off') in ('music_exit_v1','music_exit_v2','music_exit_v3'),
            pattern_aware_continuation=config.continuation_policy=='pattern_runs_v2'),
        continuation_feature_dim=continuation_dim(config.continuation_policy),
        training_semantics='causal_event_set_stream_v1',condition_defaults=payload.get('condition_defaults'))
