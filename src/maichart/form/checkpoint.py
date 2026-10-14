"""Strict new-family loading; never reinterpret an old event checkpoint."""
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import torch

from maichart.form import FAMILY, CHECKPOINT_SCHEMA, RATIONAL_FAMILY, RATIONAL_CHECKPOINT_SCHEMA
from maichart.form.codec import RationalCodec
from maichart.form.model import FormConfig, FormTransformer

INPUT_CONTRACT = {"version":"song_form_inputs_v1", "time":"exact_quarter_fraction",
    "meter":"explicit_or_unknown", "segmentation":"audio_content_resource_caps_v1",
    "audio":"event_acoustic_with_explicit_meter_120_v1", "recurrence":"audio_only_causal_chart_v1",
    "decoder":"rational_events_v1", "revision":"masked_reconstruction_v1",
    "skeleton":"whole_song_onsets_masked_layout_v1",
    "condition":"form_condition_v2",
    "load_curve":"load_curve_v1"}
LEGACY_INPUT_CONTRACT = {"version":"song_form_inputs_v1", "time":"exact_quarter_fraction",
    "meter":"explicit_or_unknown", "segmentation":"audio_content_resource_caps_v1",
    "audio":"event_acoustic_with_explicit_meter_120_v1", "recurrence":"audio_only_causal_chart_v1",
    "decoder":"rational_events_v1", "revision":"masked_reconstruction_v1",
    "skeleton":"whole_song_onsets_masked_layout_v1"}
RATIONAL_INPUT_CONTRACT = {**INPUT_CONTRACT, 'version':'song_form_inputs_v2',
    'audio':'rational_audio_queries_v1', 'skeleton':'absent_joint_exact_events_v1',
    'planner':'musical_query_workload_v1', 'decoder':'joint_rational_events_v2'}


def load(path, *, device='cpu', inference=False, task='draft', record_identity=False):
    if record_identity:
        # Hash the same open file used for loading, even if form_last.pt is
        # atomically replaced by a training checkpoint during generation.
        with open(path, 'rb') as source:
            payload=torch.load(source,map_location=device,weights_only=False)
            source.seek(0)
            checksum=hashlib.sha256()
            for block in iter(lambda:source.read(1024*1024),b''):
                checksum.update(block)
            stat=os.fstat(source.fileno())
        payload['source_checkpoint_identity']={'path':str(Path(path).resolve()),'size':stat.st_size,
                                              'mtime_ns':str(stat.st_mtime_ns),'sha256':checksum.hexdigest()}
    else:
        payload=torch.load(path,map_location=device,weights_only=False)
    rational = payload.get('schema')==RATIONAL_CHECKPOINT_SCHEMA and payload.get('family')==RATIONAL_FAMILY
    if not rational and (payload.get('schema')!=CHECKPOINT_SCHEMA or payload.get('family')!=FAMILY):
        raise ValueError('This is not a song_form_v1 model; old event checkpoints use their original generator.')
    contract=payload.get('input_contract')
    legacy_contract=contract==LEGACY_INPUT_CONTRACT
    if contract!=(RATIONAL_INPUT_CONTRACT if rational else INPUT_CONTRACT) and not legacy_contract:
        raise ValueError('Song-form input contract mismatch')
    codec=RationalCodec.from_dict(payload['codec'])
    config=FormConfig(**payload['model_config'])
    if config.vocab_size!=len(codec): raise ValueError('Vocabulary shape mismatch')
    model=FormTransformer(config).to(device)
    missing,unexpected=model.load_state_dict(payload['model_state_dict'],strict=False)
    allowed_missing = ({'condition_null_embedding', 'load_curve_projection.weight',
                        'load_curve_projection.bias'} if legacy_contract else set())
    if unexpected or any(name not in allowed_missing for name in missing):
        raise ValueError('Model checkpoint parameter contract mismatch')
    caps=payload.get('capabilities',{})
    task_cap=caps.get(task,{})
    if inference and (payload.get('optimizer_steps',0)<=0 or payload.get('blueprint_steps',0)<=0
                      or payload.get('task_training',{}).get(task,0)<=0
                      or task_cap.get('steps', payload.get('task_training',{}).get(task,0))<=0
                      or (not legacy_contract and task_cap.get('contract', contract['version']) not in (contract['version'], 'form_condition_v2'))):
        raise ValueError(f'{task} and whole-song blueprint must both receive effective training before use')
    if rational:
        from maichart.form.rational_planner import RationalPlannerConfig, RationalPlanner
        planner=RationalPlanner(RationalPlannerConfig(**payload['blueprint_config'])).to(device)
    else:
        from maichart.form.blueprint import BlueprintConfig, SongBlueprint
        planner=SongBlueprint(BlueprintConfig(**payload['blueprint_config'])).to(device)
    missing,unexpected=planner.load_state_dict(payload['blueprint_state_dict'],strict=False)
    allowed_planner_missing = ({'load_curve_head.weight', 'load_curve_head.bias'}
                               if legacy_contract else set())
    if unexpected or any(name not in allowed_planner_missing for name in missing):
        raise ValueError('Blueprint checkpoint parameter contract mismatch')
    model.eval(); planner.eval()
    return model,planner,codec,payload


def payload(model,planner,codec,**metadata):
    from maichart.form.rational_planner import RationalPlanner
    rational=isinstance(planner,RationalPlanner)
    return {'schema':RATIONAL_CHECKPOINT_SCHEMA if rational else CHECKPOINT_SCHEMA,
        'family':RATIONAL_FAMILY if rational else FAMILY,
        'input_contract':RATIONAL_INPUT_CONTRACT if rational else INPUT_CONTRACT,
        'model_config':asdict(model.config),'model_state_dict':model.state_dict(),
        'blueprint_config':asdict(planner.config),'blueprint_state_dict':planner.state_dict(),
        'codec':codec.to_dict(),**metadata}
