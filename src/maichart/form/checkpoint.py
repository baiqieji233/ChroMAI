"""Strict new-family loading; never reinterpret an old event checkpoint."""
from dataclasses import asdict
import hashlib
import math
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


def _geometry_contract(profile, quality_profile='legacy', *, curve_bin_seconds=None,
                       planner_context_seconds=None, position_alignment_tolerance_ms=None):
    from maichart.form.quality_profile import (
        validate_quality_profile, evidence_profile, density_profile, duration_semantics,
    )
    validate_quality_profile(quality_profile)
    if quality_profile != 'legacy':
        base = _geometry_contract(profile)
        return {**base, 'version': 'song_form_inputs_v5_musical_evidence',
                'quality_profile': quality_profile, 'descriptor_dim': 512,
                'form_descriptor': 'structure_shape_and_local_level_v2',
                'audio_feature_policy': 'native_measured_v2',
                'audio_evidence_profile': evidence_profile(quality_profile),
                'rhythm_geometry': profile,
                'decoder': 'joint_rational_events_v5_musical_evidence',
                'branches': 'separate_curve_skeleton_and_span_v2',
                'span_bounds': 'duration_beats_div64_seconds_div20_v1',
                'load_curve_density': density_profile(quality_profile),
                'duration_semantics': duration_semantics(quality_profile),
                'curve_bin_seconds': float(1. if curve_bin_seconds is None else curve_bin_seconds),
                'position_alignment': 'authored_audio_queries_v1',
                'position_alignment_tolerance_ms': float(15. if position_alignment_tolerance_ms is None
                                                         else position_alignment_tolerance_ms),
                'planner': 'musical_query_workload_temporal_v2',
                'planner_context_seconds': float(1. if planner_context_seconds is None else planner_context_seconds)}
    if profile == 'exact_fraction_v2':
        return RATIONAL_METER_GEOMETRY_INPUT_CONTRACT
    if profile == 'exact_fraction_v1':
        return RATIONAL_GEOMETRY_INPUT_CONTRACT
    return RATIONAL_INPUT_CONTRACT


def input_contract_for_configs(model_config, blueprint_config=None, *, rational=True):
    """Validate saved semantic dimensions without constructing neural modules."""
    from maichart.form.quality_profile import validate_quality_profile
    from maichart.form.rhythm_geometry import GEOMETRY_PROFILES

    config = model_config or {}
    planner = blueprint_config or {}
    profile = config.get('rhythm_geometry_profile', 'none')
    if profile not in GEOMETRY_PROFILES or (not rational and profile != 'none'):
        raise ValueError('Song-form rhythm geometry contract mismatch')
    quality = validate_quality_profile(config.get('quality_profile', 'legacy'))
    planner_quality = validate_quality_profile(planner.get('quality_profile', 'legacy'))
    if quality != planner_quality or (not rational and quality != 'legacy'):
        raise ValueError('Model and Planner musical-evidence profiles must match')
    if quality == 'legacy':
        return _geometry_contract(profile) if rational else INPUT_CONTRACT
    missing = ({'descriptor_dim', 'curve_bin_seconds', 'position_alignment_tolerance_ms'} - config.keys()
               | {f'planner.{key}' for key in {'descriptor_dim', 'context_seconds'} - planner.keys()})
    if missing:
        raise ValueError('Musical-evidence input contract lacks metadata: ' + ', '.join(sorted(missing)))
    if config.get('descriptor_dim') != 512 or planner.get('descriptor_dim') != 512:
        raise ValueError('Musical-evidence descriptor_dim must be 512 in model and Planner')
    options = {}
    for name, value, minimum, maximum in (
        ('curve_bin_seconds', config.get('curve_bin_seconds', 1.), .01, 30.),
        ('planner_context_seconds', planner.get('context_seconds', 1.), 0., 30.),
        ('position_alignment_tolerance_ms', config.get('position_alignment_tolerance_ms', 15.), 1., 100.),
    ):
        if isinstance(value, bool):
            raise ValueError(f'{name} must be numeric')
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{name} must be numeric') from exc
        if not math.isfinite(number) or not minimum <= number <= maximum:
            raise ValueError(f'{name} out of range')
        options[name] = number
    return _geometry_contract(profile, quality, **options)


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
    enhanced = ((payload.get('model_config') or {}).get('quality_profile') == 'musical_evidence_v1'
                or contract.get('quality_profile') == 'musical_evidence_v1')
    if enhanced:
        # Migration inherits useful weights and historical counters, not proof
        # that the expanded layout has been trained. A Blueprint-only save
        # must never silently acquire draft/revision capability from v4 counts.
        return bool(payload.get('optimizer_steps', 0) > 0 and planner_is_ready(payload)
                    and trained.get(task, 0) > 0
                    and contract.get('version') == 'song_form_inputs_v5_musical_evidence'
                    and capability.get('steps', 0) > 0
                    and capability.get('contract') in (contract['version'], 'form_condition_v2'))
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
    legacy_contract=not rational and contract==LEGACY_INPUT_CONTRACT
    expected_contract = input_contract_for_configs(payload.get('model_config'),
                                                   payload.get('blueprint_config'), rational=rational)
    if contract != expected_contract and not legacy_contract:
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
    """Explicitly migrate geometry and musical evidence, preserving shared weights."""
    from maichart.form.rhythm_geometry import geometry_dim
    source, _planner, _codec, payload = load(path, device='cpu')
    old_profile = getattr(source.config, 'rhythm_geometry_profile', 'none')
    new_profile = config.rhythm_geometry_profile
    old_quality, new_quality = source.config.quality_profile, config.quality_profile
    if old_quality != 'legacy' and new_quality != old_quality:
        raise ValueError('Finetune migration cannot downgrade musical-evidence quality profile')
    if old_profile == new_profile and asdict(config) == asdict(source.config):
        raise ValueError('Finetune migration requires a different rhythm geometry profile or musical-evidence contract')
    if old_quality != new_quality and payload['family'] != RATIONAL_FAMILY:
        raise ValueError('Musical-evidence migration requires song_form_v2')
    if old_quality != new_quality and (source.config.descriptor_dim != 504 or config.descriptor_dim != 512):
        raise ValueError('Musical-evidence migration requires descriptor_dim 504 -> 512')
    if old_quality == new_quality and source.config.descriptor_dim != config.descriptor_dim:
        raise ValueError('Descriptor width may change only during musical-evidence migration')
    state = dict(payload['model_state_dict'])
    allowed_missing = set()
    if old_profile != new_profile:
        if new_profile not in ('exact_fraction_v1', 'exact_fraction_v2'):
            raise ValueError('Finetune migration targets exact_fraction_v1 or exact_fraction_v2')
        if geometry_dim(new_profile) < geometry_dim(old_profile):
            raise ValueError('Finetune migration cannot shrink a geometry contract')
        if old_profile == 'none':
            allowed_missing.update({'time_geometry_projection.weight', 'decoder_geometry_projection.weight',
                                    'interval_head.weight', 'interval_head.bias'})
        else:
            state.update(migrate_geometry_projection(state, old_profile, new_profile, config.d_model))
    model = FormTransformer(config).to(device)
    if old_quality != new_quality:
        state['form_projection.weight'] = migrate_descriptor_projection(
            state['form_projection.weight'], config.d_model)
        branch = state['branch_embedding.weight']
        if tuple(branch.shape) != (11, config.d_model):
            raise ValueError('Musical-evidence migration branch shape mismatch')
        expanded = model.branch_embedding.weight.detach().clone()
        expanded[:11] = branch.to(expanded)
        state['branch_embedding.weight'] = expanded
        allowed_missing.update(name for name in model.state_dict() if name.startswith(
            ('span_projection.', 'alignment_query.', 'alignment_key.', 'position_alignment_head.')))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or set(missing) != allowed_missing:
        raise ValueError('Geometry/quality finetune migration parameter contract mismatch')
    return model


def migrate_descriptor_projection(weight, width):
    """Append eight descriptor channels before the three relative coordinates."""
    if tuple(weight.shape) != (width, 507):
        raise ValueError('Musical-evidence migration descriptor projection shape mismatch')
    expanded = weight.new_zeros((width, 515))
    expanded[:, :504] = weight[:, :504]
    expanded[:, -3:] = weight[:, -3:]
    return expanded


def load_finetune_planner(path, config, *, device='cpu'):
    """Transfer a validated rational Planner into its explicit evidence contract."""
    from maichart.form.rational_planner import RationalPlanner
    _model, source, _codec, saved = load(path, device='cpu')
    if saved['family'] != RATIONAL_FAMILY:
        raise ValueError('Musical-evidence Planner migration requires song_form_v2')
    old_quality, new_quality = source.config.quality_profile, config.quality_profile
    if old_quality != 'legacy' and new_quality != old_quality:
        raise ValueError('Finetune migration cannot downgrade musical-evidence quality profile')
    state = dict(saved['blueprint_state_dict'])
    planner = RationalPlanner(config).to(device)
    allowed_missing = set()
    if old_quality != new_quality:
        if source.config.descriptor_dim != 504 or config.descriptor_dim != 512:
            raise ValueError('Musical-evidence Planner migration requires descriptor_dim 504 -> 512')
        state['form_projection.weight'] = migrate_descriptor_projection(
            state['form_projection.weight'], config.width)
        allowed_missing.add('temporal_projection.weight')
    elif source.config.descriptor_dim != config.descriptor_dim:
        raise ValueError('Descriptor width may change only during musical-evidence migration')
    missing, unexpected = planner.load_state_dict(state, strict=False)
    if unexpected or set(missing) != allowed_missing:
        raise ValueError('Planner finetune migration parameter contract mismatch')
    return planner


def payload(model,planner,codec,**metadata):
    from maichart.form.rational_planner import RationalPlanner
    rational=isinstance(planner,RationalPlanner)
    config, planner_config = asdict(model.config), asdict(planner.config)
    contract = input_contract_for_configs(config, planner_config, rational=rational)
    return {'schema':RATIONAL_CHECKPOINT_SCHEMA if rational else CHECKPOINT_SCHEMA,
        'family':RATIONAL_FAMILY if rational else FAMILY,
        'input_contract':contract,
        'model_config':config,'model_state_dict':model.state_dict(),
        'blueprint_config':planner_config,'blueprint_state_dict':planner.state_dict(),
        'codec':codec.to_dict(),**metadata}
