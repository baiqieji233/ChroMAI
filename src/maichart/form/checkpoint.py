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
RATIONAL_GEOMETRY_INPUT_CONTRACT = {**RATIONAL_INPUT_CONTRACT,
    'version':'song_form_inputs_v3',
    'rhythm_geometry':'exact_fraction_v1',
    'audio_feature_policy':'native_measured_v2',
    'decoder':'joint_rational_events_v3_exact_geometry'}


def planner_is_ready(payload):
    """Reject known untrained rational Planners while preserving old evidence.

    Early v2 trainers counted zero-objective optimizer calls as blueprint
    updates. A counter alone is therefore insufficient when saved arguments
    explicitly show that workload supervision was disabled. Legacy payloads
    without that metadata keep their historical readiness behavior.
    """
    if payload.get('blueprint_steps', 0) <= 0:
        return False
    if payload.get('family') != RATIONAL_FAMILY:
        return True
    if 'planner_effective_steps' in payload:
        return payload['planner_effective_steps'] > 0
    capability = (payload.get('capabilities') or {}).get('load_curve') or {}
    if capability.get('steps', 0) > 0 and capability.get('contract') == 'load_curve_v1':
        return True
    args = payload.get('args') or {}
    return (args.get('load_curve_loss_weight', 1.) > 0
            and args.get('blueprint_loss_weight', 1.) > 0)


def task_is_ready(payload, task='draft'):
    """One readiness predicate for inspection and the real inference loader."""
    contract=payload.get('input_contract') or {}
    trained=payload.get('task_training') or {}
    capability=(payload.get('capabilities') or {}).get(task) or {}
    legacy=contract==LEGACY_INPUT_CONTRACT
    return (payload.get('optimizer_steps',0)>0 and planner_is_ready(payload)
            and trained.get(task,0)>0 and capability.get('steps',trained.get(task,0))>0
            and (legacy or capability.get('contract',contract.get('version'))
                 in (contract.get('version'),'form_condition_v2')))


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
    expected_rational = (RATIONAL_GEOMETRY_INPUT_CONTRACT
                         if payload.get('model_config',{}).get('rhythm_geometry_profile') == 'exact_fraction_v1'
                         else RATIONAL_INPUT_CONTRACT)
    if contract!=(expected_rational if rational else INPUT_CONTRACT) and not legacy_contract:
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
    if inference and not task_is_ready(payload,task):
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


def load_finetune_model(path, config: FormConfig, *, device='cpu') -> FormTransformer:
    """Load a none -> exact-geometry finetune migration without altering old weights.

    The ordinary loader first validates the source checkpoint against its own
    contract.  Only the three newly introduced geometry modules may be absent
    from the source state; every pre-existing parameter remains strict.
    """
    source, _planner, _codec, payload = load(path, device='cpu')
    old_profile = getattr(source.config, 'rhythm_geometry_profile', 'none')
    if old_profile != 'none' or config.rhythm_geometry_profile != 'exact_fraction_v1':
        raise ValueError('Only none -> exact_fraction_v1 finetune migration is supported')
    model = FormTransformer(config).to(device)
    missing, unexpected = model.load_state_dict(payload['model_state_dict'], strict=False)
    allowed = {
        'time_geometry_projection.weight', 'decoder_geometry_projection.weight',
        'interval_head.weight', 'interval_head.bias',
    }
    if unexpected or set(missing) != allowed:
        raise ValueError('Geometry finetune migration parameter contract mismatch')
    return model


def payload(model,planner,codec,**metadata):
    from maichart.form.rational_planner import RationalPlanner
    rational=isinstance(planner,RationalPlanner)
    return {'schema':RATIONAL_CHECKPOINT_SCHEMA if rational else CHECKPOINT_SCHEMA,
        'family':RATIONAL_FAMILY if rational else FAMILY,
        'input_contract':(RATIONAL_GEOMETRY_INPUT_CONTRACT if rational
                           and getattr(model.config,'rhythm_geometry_profile','none') == 'exact_fraction_v1'
                           else RATIONAL_INPUT_CONTRACT if rational else INPUT_CONTRACT),
        'model_config':asdict(model.config),'model_state_dict':model.state_dict(),
        'blueprint_config':asdict(planner.config),'blueprint_state_dict':planner.state_dict(),
        'codec':codec.to_dict(),**metadata}
