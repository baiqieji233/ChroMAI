"""Strict new-family loading; never reinterpret an old event checkpoint."""
from dataclasses import asdict
import torch

from maichart.form import FAMILY, CHECKPOINT_SCHEMA
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


def load(path, *, device='cpu', inference=False, task='draft'):
    payload=torch.load(path,map_location=device,weights_only=False)
    if payload.get('schema')!=CHECKPOINT_SCHEMA or payload.get('family')!=FAMILY:
        raise ValueError('This is not a song_form_v1 model; old event checkpoints use their original generator.')
    contract=payload.get('input_contract')
    legacy_contract=contract==LEGACY_INPUT_CONTRACT
    if contract!=INPUT_CONTRACT and not legacy_contract:
        raise ValueError('Song-form input contract mismatch')
    codec=RationalCodec.from_dict(payload['codec'])
    config=FormConfig(**payload['model_config'])
    if config.vocab_size!=len(codec): raise ValueError('Vocabulary shape mismatch')
    model=FormTransformer(config).to(device)
    missing,unexpected=model.load_state_dict(payload['model_state_dict'],strict=False)
    allowed_missing={'condition_null_embedding','load_curve_projection.weight','load_curve_projection.bias'}
    if unexpected or any(name not in allowed_missing for name in missing):
        raise ValueError('Model checkpoint parameter contract mismatch')
    caps=payload.get('capabilities',{})
    task_cap=caps.get(task,{})
    if inference and (payload.get('optimizer_steps',0)<=0 or payload.get('blueprint_steps',0)<=0
                      or payload.get('task_training',{}).get(task,0)<=0
                      or task_cap.get('steps', payload.get('task_training',{}).get(task,0))<=0
                      or (not legacy_contract and task_cap.get('contract', INPUT_CONTRACT['version']) not in (INPUT_CONTRACT['version'], 'form_condition_v2'))):
        raise ValueError(f'{task} and whole-song blueprint must both receive effective training before use')
    from maichart.form.blueprint import BlueprintConfig, SongBlueprint
    planner=SongBlueprint(BlueprintConfig(**payload['blueprint_config'])).to(device)
    missing,unexpected=planner.load_state_dict(payload['blueprint_state_dict'],strict=False)
    if unexpected or any(name != 'load_curve_head.weight' and name != 'load_curve_head.bias' for name in missing):
        raise ValueError('Blueprint checkpoint parameter contract mismatch')
    model.eval(); planner.eval()
    return model,planner,codec,payload


def payload(model,planner,codec,**metadata):
    return {'schema':CHECKPOINT_SCHEMA,'family':FAMILY,'input_contract':INPUT_CONTRACT,
        'model_config':asdict(model.config),'model_state_dict':model.state_dict(),
        'blueprint_config':asdict(planner.config),'blueprint_state_dict':planner.state_dict(),
        'codec':codec.to_dict(),**metadata}
