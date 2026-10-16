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
# v2 replaces the span-anchored four-beat harmonic with bar/group phase taken
# from the song's meter map, so the same musical position receives the same
# coordinate in every span. The decoder contract changes because the geometry
# width changes: a v1 model cannot consume v2 coordinates without migration.
RATIONAL_METER_GEOMETRY_INPUT_CONTRACT = {**RATIONAL_INPUT_CONTRACT,
    'version':'song_form_inputs_v4',
    'rhythm_geometry':'exact_fraction_v2',
    'audio_feature_policy':'native_measured_v2',
    'decoder':'joint_rational_events_v4_meter_geometry'}


def _geometry_contract(profile):
    if profile == 'exact_fraction_v2':
        return RATIONAL_METER_GEOMETRY_INPUT_CONTRACT
    if profile == 'exact_fraction_v1':
        return RATIONAL_GEOMETRY_INPUT_CONTRACT
    return RATIONAL_INPUT_CONTRACT


# ``exact_fraction_v1`` -> ``exact_fraction_v2`` channel correspondence. Only
# channels that mean the same thing are carried across; the new bar/group phase
# channels start at zero so finetuning learns them without disturbing the
# transferable sub-beat behaviour.
V1_TO_V2_CHANNELS = ((0, 0), (1, 1), (3, 2), (4, 3), (5, 4))


def migrate_geometry_projection(state, source_profile, target_profile, width):
    """Return a v2 geometry projection initialised from a v1 checkpoint.

    ``time_geometry_projection`` and ``decoder_geometry_projection`` are
    ``Linear(dim, width, bias=False)``, so migrating means copying the weight
    rows whose channel semantics are unchanged and leaving the rest at zero.
    """
    from maichart.form.rhythm_geometry import geometry_dim
    source_dim, target_dim = geometry_dim(source_profile), geometry_dim(target_profile)
    migrated = {}
    for name in ('time_geometry_projection.weight', 'decoder_geometry_projection.weight'):
        weight = state.get(name)
        if weight is None or tuple(weight.shape) != (width, source_dim):
            raise ValueError(f'Checkpoint geometry parameter {name} has an unexpected shape')
        target = weight.new_zeros((width, target_dim))
        for source_channel, target_channel in V1_TO_V2_CHANNELS:
            target[:, target_channel] = weight[:, source_channel]
        migrated[name] = target
    return migrated


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
    expected_rational = _geometry_contract(
        payload.get('model_config',{}).get('rhythm_geometry_profile', 'none'))
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
    """Load a geometry finetune migration without altering transferable weights.

    The ordinary loader first validates the source checkpoint against its own
    contract. Only the geometry modules introduced by the target profile may
    be absent, and only their channels may differ:

    ``none`` -> ``exact_fraction_v1``/``exact_fraction_v2``
        Every geometry parameter is new and starts at its zero/random init.
    ``exact_fraction_v1`` -> ``exact_fraction_v2``
        The projections keep their learned sub-beat channels and gain zero
        bar/group phase channels, so a trained checkpoint transfers rather
        than restarting.
    """
    from maichart.form.rhythm_geometry import geometry_dim
    source, _planner, _codec, payload = load(path, device='cpu')
    old_profile = getattr(source.config, 'rhythm_geometry_profile', 'none')
    new_profile = config.rhythm_geometry_profile
    if old_profile == new_profile:
        raise ValueError('Finetune migration requires a different rhythm geometry profile')
    if new_profile not in ('exact_fraction_v1', 'exact_fraction_v2'):
        raise ValueError('Finetune migration targets exact_fraction_v1 or exact_fraction_v2')
    if geometry_dim(new_profile) < geometry_dim(old_profile):
        raise ValueError('Finetune migration cannot shrink a geometry contract')
    state = dict(payload['model_state_dict'])
    if old_profile == 'none':
        allowed_missing = {
            'time_geometry_projection.weight', 'decoder_geometry_projection.weight',
            'interval_head.weight', 'interval_head.bias',
        }
    else:
        state.update(migrate_geometry_projection(state, old_profile, new_profile, config.d_model))
        allowed_missing = set()
    model = FormTransformer(config).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or set(missing) != allowed_missing:
        raise ValueError('Geometry finetune migration parameter contract mismatch')
    return model


def payload(model,planner,codec,**metadata):
    from maichart.form.rational_planner import RationalPlanner
    rational=isinstance(planner,RationalPlanner)
    return {'schema':RATIONAL_CHECKPOINT_SCHEMA if rational else CHECKPOINT_SCHEMA,
        'family':RATIONAL_FAMILY if rational else FAMILY,
        'input_contract':(_geometry_contract(getattr(model.config,'rhythm_geometry_profile','none'))
                          if rational else INPUT_CONTRACT),
        'model_config':asdict(model.config),'model_state_dict':model.state_dict(),
        'blueprint_config':asdict(planner.config),'blueprint_state_dict':planner.state_dict(),
        'codec':codec.to_dict(),**metadata}
