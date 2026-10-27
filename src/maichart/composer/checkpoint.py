"""Checkpoint contract for chart_composer_v1."""
from __future__ import annotations
from pathlib import Path
import copy,os,tempfile
import torch
from . import CHECKPOINT_SCHEMA,FAMILY,INPUT_CONTRACT,TRAINING_SEMANTICS,GROUNDED_TRAINING_SEMANTICS,ACTION_SAMPLING_POLICY
from .model import ChartComposer,ComposerConfig,WORKLOAD_PROFILES,GROUNDED_PROFILE,build_model
from .candidates import COMPOSER_AUDIO_FEATURE_POLICY, MUSIC_CANDIDATE_POLICY, MUSIC_CANDIDATE_POLICY_V6
from .events import HEAD_TIMING_DIM
from . import MOTIF_TRAINING_POLICY
from maichart.form.recurrence import RECURRENCE_DESCRIPTOR_POLICY,LEGACY_RECURRENCE_DESCRIPTOR_POLICY
from .objectives import (LEGACY_ACTION_OBJECTIVE, JOINT_TRAINING_SEMANTICS, RECOVERY_TRAINING_SEMANTICS,
                         resolve_action_objective, training_semantics_for_objective)


def motif_training_compatible(payload):
    return ((payload.get('model_config') or {}).get('motif_memory_policy','off')=='off'
            or (payload.get('motif_training_policy')==MOTIF_TRAINING_POLICY
                and payload.get('motif_input_policy')==RECURRENCE_DESCRIPTOR_POLICY))


def generation_motif_input_policy(payload):
    """Keep the audio-relation interpretation that the checkpoint was trained on."""
    if (payload.get('model_config') or {}).get('motif_memory_policy','off')=='off':
        return RECURRENCE_DESCRIPTOR_POLICY
    policy=payload.get('motif_input_policy',LEGACY_RECURRENCE_DESCRIPTOR_POLICY)
    if policy not in {RECURRENCE_DESCRIPTOR_POLICY,LEGACY_RECURRENCE_DESCRIPTOR_POLICY}:
        raise ValueError('Unknown Composer motif input policy; update the runtime before generation')
    return policy


_LEGACY_SEMANTICS = 'chart_composer_training_v2_audio_only_candidate_actions'
_WORKLOAD_SEMANTICS = 'chart_composer_training_v4_audio_workload_slots'
_OLD_MUSIC_SEMANTICS = 'chart_composer_training_v5_music_context_shared_slots'
_PHASE_SAFE_SEMANTICS = 'chart_composer_training_v6_phase_safe_density_guard'
_GROUNDED_V7_SEMANTICS = 'chart_composer_training_v7_audio_grounded_recovery'
_GROUNDED_V8_SEMANTICS = 'chart_composer_training_v8_boundary_plans_recovery'
_GROUNDED_V9_SEMANTICS = 'chart_composer_training_v9_active_slots_phrase_tail'
_GROUNDED_V10_SEMANTICS = 'chart_composer_training_v10_touch_ring_recovery'

# Policy fields were added after the first workload checkpoints. Missing
# fields therefore get a profile-specific historical default, while explicit
# mismatches are rejected before model weights are materialized.
_PROFILE_POLICIES = {
    GROUNDED_PROFILE: {'audio_feature_policy':COMPOSER_AUDIO_FEATURE_POLICY,
                      'candidate_policy':MUSIC_CANDIDATE_POLICY_V6,
                      'training_semantics':GROUNDED_TRAINING_SEMANTICS},
    'legacy_v2': {'audio_feature_policy': 'legacy', 'candidate_policy': 'relative_v1',
                  'training_semantics': _LEGACY_SEMANTICS},
    'audio_workload_v1': {'audio_feature_policy': COMPOSER_AUDIO_FEATURE_POLICY,
                          'candidate_policy': 'absolute_recovery_v4',
                          'training_semantics': _WORKLOAD_SEMANTICS},
    'music_context_v2': {'audio_feature_policy': COMPOSER_AUDIO_FEATURE_POLICY,
                         'candidate_policy': MUSIC_CANDIDATE_POLICY_V6,
                         'training_semantics': TRAINING_SEMANTICS},
}


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
    objective=resolve_action_objective(args.get('action_objective_policy',LEGACY_ACTION_OBJECTIVE),
                                      architecture_profile=model.config.architecture_profile)
    current_music = (model.config.architecture_profile == 'music_context_v2'
                     and model.config.head_timing_dim == HEAD_TIMING_DIM)
    music_semantics = TRAINING_SEMANTICS if current_music else _OLD_MUSIC_SEMANTICS
    music_candidate_policy = MUSIC_CANDIDATE_POLICY_V6 if current_music else MUSIC_CANDIDATE_POLICY
    payload={'schema':CHECKPOINT_SCHEMA,'family':FAMILY,'input_contract':INPUT_CONTRACT,
             'model_config':model.config.to_dict(),'model_state_dict':model.state_dict(),
             'optimizer_state_dict':optimizer.state_dict() if optimizer is not None else None,
             'epoch':int(epoch),'optimizer_steps':int(optimizer_steps),'args':dict(args),
             'manifest':str(Path(manifest).resolve()),'metrics':metrics or {},
             'training_semantics':(music_semantics if model.config.architecture_profile=='music_context_v2'
                else 'chart_composer_training_v4_audio_workload_slots' if model.config.architecture_profile=='audio_workload_v1'
                else 'chart_composer_training_v2_audio_only_candidate_actions'),
             'atom_catalog':sorted(set(atom_catalog)),
             'audio_feature_policy':(COMPOSER_AUDIO_FEATURE_POLICY
                 if model.config.architecture_profile in WORKLOAD_PROFILES else 'legacy'),
             'candidate_policy':(music_candidate_policy if model.config.architecture_profile=='music_context_v2'
                 else 'absolute_recovery_v4' if model.config.architecture_profile=='audio_workload_v1' else 'relative_v1'),
             'dataset_fingerprint':dataset_fingerprint,
             'training_state':(training_state or {}) if optimizer is not None else {},
             'checkpoint_kind':'training' if optimizer is not None else 'inference'}
    if model.config.architecture_profile==GROUNDED_PROFILE:
        if model.config.history_anchor_policy != model.action_factors.history_anchor_policy:
            raise ValueError('Save history_anchor_policy differs from the effective model representation')
        if model.config.slide_factor_policy != model.action_factors.slide_factor_policy:
            raise ValueError('Save slide_factor_policy differs from the effective model representation')
        if list(atom_catalog)!=list(model.action_factors.atom_catalog):
            raise ValueError('Save catalog order differs from the compositional action mapping')
        payload.update(training_semantics=training_semantics_for_objective(GROUNDED_PROFILE,objective),candidate_policy=MUSIC_CANDIDATE_POLICY_V6,
            action_sampling_policy=ACTION_SAMPLING_POLICY,
            atom_catalog=list(atom_catalog),
            architecture_features='ordered_lens_interpolated_plan_banded_pointer_compositional_actions_v1')
        if model.config.motif_memory_policy!='off':
            payload['motif_training_policy']=MOTIF_TRAINING_POLICY
            payload['motif_input_policy']=RECURRENCE_DESCRIPTOR_POLICY
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
    values=dict(payload['model_config'])
    values.setdefault('architecture_profile','legacy_v2')
    values.setdefault('head_timing_dim',0)
    values.setdefault('history_anchor_policy','legacy_digit_v1')
    values.setdefault('action_state_policy','committed_legacy_v1')
    values.setdefault('motif_memory_policy','off')
    values.setdefault('slide_factor_policy','marginal_v1')
    values.setdefault('history_memory_policy','unbounded_v1')
    values.setdefault('history_memory_rms_bound',8.)
    values.setdefault('history_attention_policy','auto_v1')
    try:
        config=ComposerConfig(**values)
    except TypeError as exc:
        if 'unexpected keyword argument' not in str(exc):raise
        raise ValueError('Composer checkpoint 与当前进程加载的模型代码不兼容：'
            f'{exc}。请更新本工作区代码并重启正在使用的 WebUI 后端，再重新检查 checkpoint。') from exc
    from .model import validate_supported_context_policies
    validate_supported_context_policies(config)
    generation_motif_input_policy(payload)
    objective=resolve_action_objective((payload.get('args') or {}).get('action_objective_policy',LEGACY_ACTION_OBJECTIVE),
                                      architecture_profile=config.architecture_profile)
    if config.architecture_profile not in {'legacy_v2',*WORKLOAD_PROFILES}:
        raise ValueError('Unknown Composer architecture profile')
    contract=_PROFILE_POLICIES[config.architecture_profile]
    semantics=payload.get('training_semantics')
    allowed_semantics={contract['training_semantics']}
    if config.architecture_profile==GROUNDED_PROFILE:
        allowed_semantics.update({_GROUNDED_V7_SEMANTICS, _GROUNDED_V8_SEMANTICS, _GROUNDED_V9_SEMANTICS,
                                  _GROUNDED_V10_SEMANTICS,JOINT_TRAINING_SEMANTICS,RECOVERY_TRAINING_SEMANTICS})
        if semantics not in allowed_semantics:
            raise ValueError('audio_grounded_v3 requires its explicit training semantics')
        new_semantics={JOINT_TRAINING_SEMANTICS,RECOVERY_TRAINING_SEMANTICS}
        if ((semantics in new_semantics or objective!=LEGACY_ACTION_OBJECTIVE)
                and semantics!=training_semantics_for_objective(GROUNDED_PROFILE,objective)):
            raise ValueError('Composer action objective/training semantics mismatch')
        if semantics in {_GROUNDED_V10_SEMANTICS,GROUNDED_TRAINING_SEMANTICS,*new_semantics} and payload.get('action_sampling_policy')!=ACTION_SAMPLING_POLICY:
            raise ValueError('Current Composer recovery action sampling policy mismatch')
    if config.architecture_profile=='music_context_v2':
        # Existing v5 music-context weights remain valid for generation and
        # explicit finetune after a newer training contract is introduced.
        allowed_semantics.update({_OLD_MUSIC_SEMANTICS, _PHASE_SAFE_SEMANTICS})
        if semantics == TRAINING_SEMANTICS and config.head_timing_dim != HEAD_TIMING_DIM:
            raise ValueError('Current Composer training contract requires head_timing_dim=99')
        if semantics in {_OLD_MUSIC_SEMANTICS, _PHASE_SAFE_SEMANTICS} and config.head_timing_dim != 0:
            raise ValueError('Legacy music-context contract requires head_timing_dim=0')
        if semantics is None and config.head_timing_dim != 0:
            raise ValueError('Missing training semantics cannot describe a head-timing checkpoint')
    if semantics is not None and semantics not in allowed_semantics:
        raise ValueError('Composer training semantics/profile mismatch')
    for field in ('audio_feature_policy','candidate_policy'):
        # Missing metadata predates the strict contract. Resolve it to the
        # documented profile default so generation remains backward compatible.
        legacy_music = (config.architecture_profile == 'music_context_v2'
                        and semantics in {_OLD_MUSIC_SEMANTICS, None})
        phase_safe_music = (config.architecture_profile == 'music_context_v2'
                            and semantics == _PHASE_SAFE_SEMANTICS)
        default = (COMPOSER_AUDIO_FEATURE_POLICY if field == 'audio_feature_policy'
                   and config.architecture_profile in WORKLOAD_PROFILES else
                   'legacy' if field == 'audio_feature_policy' else
                   'absolute_recovery_v4' if config.architecture_profile == 'audio_workload_v1' else
                   MUSIC_CANDIDATE_POLICY if legacy_music else contract[field])
        actual=payload.get(field,default)
        allowed={contract[field]}
        if legacy_music and field=='candidate_policy':
            allowed.add(MUSIC_CANDIDATE_POLICY)
        if phase_safe_music and field == 'candidate_policy':
            allowed.add(MUSIC_CANDIDATE_POLICY_V6)
        if actual not in allowed:
            raise ValueError(f'Composer {field}/profile mismatch: {actual!r}')
        payload.setdefault(field,actual)
    expected={'audio_dim':120,'condition_dim':16,'action_dim':50,
              'candidate_feature_dim':141 if config.architecture_profile in WORKLOAD_PROFILES else 125,
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
    # Generation calls load under inference_mode. Parameters still need normal
    # version counters for derived-weight caches and later explicit finetuning.
    with torch.inference_mode(False):
        model=build_model(config,payload['atom_catalog'])
        model.load_state_dict(payload['model_state_dict'])
        model.to(device)
    return model,payload


def inspect_payload(payload):
    try: config=validate(payload);valid=True;error=None
    except (ValueError,KeyError,TypeError) as exc: config=None;valid=False;error=str(exc)
    migration = None
    current_semantics=None
    if config is not None:
        current_semantics=training_semantics_for_objective(config.architecture_profile,
            (payload.get('args') or {}).get('action_objective_policy',LEGACY_ACTION_OBJECTIVE))
        if config.architecture_profile == GROUNDED_PROFILE and payload.get('training_semantics') != current_semantics:
            migration = 'Older grounded planning/recovery contracts require explicit finetune with a new output directory'
        elif not motif_training_compatible(payload):
            migration = 'Motif input/dropout contract changed; use explicit finetune with a new output directory'
        elif config.architecture_profile == 'music_context_v2' and payload.get('training_semantics') != TRAINING_SEMANTICS:
            migration = 'Existing music-context contracts require explicit finetune under the current training contract'
        elif config.architecture_profile not in {'music_context_v2',GROUNDED_PROFILE}:
            migration = 'Workload v1 supports explicit finetune migration; legacy models retain generation only'
    return {'valid':valid,'family':payload.get('family'),'schema':payload.get('schema'),
            'model_config':config.to_dict() if config else payload.get('model_config'),
            'epoch':payload.get('epoch'),'optimizer_steps':payload.get('optimizer_steps',0),
            'training_semantics':payload.get('training_semantics'),'error':error,
            'motif_training_policy':payload.get('motif_training_policy'),
            'motif_input_policy':payload.get('motif_input_policy'),
            'audio_feature_policy':payload.get('audio_feature_policy','legacy'),
            'workload_curve_trained':bool(config and config.architecture_profile in WORKLOAD_PROFILES
                                         and payload.get('optimizer_steps',0)>0),
            'args':payload.get('args',{}),
            'candidate_policy':payload.get('candidate_policy'),
            'can_resume':bool(valid and config.architecture_profile in {'music_context_v2',GROUNDED_PROFILE}
                              and config.head_timing_dim==HEAD_TIMING_DIM
                              and payload.get('training_semantics')==current_semantics
                              and payload.get('optimizer_state_dict') is not None and motif_training_compatible(payload)),
            'can_init':bool(valid and config.architecture_profile in WORKLOAD_PROFILES),
            'can_train':bool(valid and config.architecture_profile in WORKLOAD_PROFILES),'can_generate':valid,
            'training_migration':migration,
            'parameters':sum(value.numel() for key,value in payload.get('model_state_dict',{}).items()
                             if torch.is_tensor(value) and not key.startswith('action_factors.atom_')
                             and key!='action_factors.catalog_fingerprint'),
            'checkpoint_kind':payload.get('checkpoint_kind','training'),
            'model_bytes':sum(value.numel()*value.element_size()
                              for value in payload.get('model_state_dict',{}).values() if torch.is_tensor(value))}


def _expanded_slot_embedding(value, target):
    if target.shape[0] < value.shape[0]:
        raise ValueError('Finetune migration cannot reduce action_slots')
    result = value.mean(0, keepdim=True).expand_as(target).clone()
    result[:value.shape[0]] = value
    return result


def migrate_music_context(model, head_timing_dim=HEAD_TIMING_DIM, *, action_slots=None, **policy_options):
    """Explicit approximate finetune migration, never an exact resume.

    Shared layers retain weights. Eight slot classifiers become one mean
    classifier; slot identity and new evidence/attention require training.
    This is intentionally not advertised as equivalent old-model generation.
    """
    if int(head_timing_dim) != HEAD_TIMING_DIM:
        raise ValueError(f'Explicit music-context migration requires head_timing_dim={HEAD_TIMING_DIM}')
    slots = model.config.action_slots if action_slots is None else int(action_slots)
    if slots < model.config.action_slots:
        raise ValueError('Finetune migration cannot reduce action_slots')
    allowed={'history_anchor_policy','action_state_policy','motif_memory_policy','motif_max_events','slide_factor_policy',
             'history_memory_policy','history_memory_rms_bound','history_attention_policy'}
    if set(policy_options)-allowed:raise ValueError('Unknown music-context migration option')
    config=ComposerConfig(**{**model.config.to_dict(),**policy_options,'architecture_profile':'music_context_v2',
                             'head_timing_dim':HEAD_TIMING_DIM,'action_slots':slots})
    from .model import validate_supported_context_policies
    validate_supported_context_policies(config)
    if model.config.to_dict()==config.to_dict():
        return model
    if model.config.architecture_profile not in {'audio_workload_v1','music_context_v2'}:
        raise ValueError('Only workload v1 can migrate to music-context v2')
    upgraded=ChartComposer(config)
    old=model.state_dict(); new=upgraded.state_dict()
    for key,value in old.items():
        if key.startswith('action_head.') and model.config.architecture_profile=='audio_workload_v1':
            new[key]=value.reshape(model.config.action_slots,config.atom_vocab_size,*value.shape[1:]).mean(0)
        elif key == 'slot_embedding.weight':
            new[key] = _expanded_slot_embedding(value, new[key])
        elif key in new and new[key].shape==value.shape:
            new[key]=value
    upgraded.load_state_dict(new)
    return upgraded


def migrate_audio_grounded(model,atom_catalog,**options):
    """Explicit partial transfer; new listening/planning/pointer paths need learning."""
    if int(options.get('action_slots', model.config.action_slots)) < model.config.action_slots:
        raise ValueError('Finetune migration cannot reduce action_slots')
    if model.config.architecture_profile==GROUNDED_PROFILE:
        if list(model.action_factors.atom_catalog)!=list(atom_catalog):
            raise ValueError('Migration catalog order differs from existing action mapping')
        if all(getattr(model.config,key)==value for key,value in options.items()):return model
    if model.config.architecture_profile not in WORKLOAD_PROFILES:
        raise ValueError('Only Composer workload/music models support this migration')
    config=ComposerConfig(**{**model.config.to_dict(),**options,
        'architecture_profile':GROUNDED_PROFILE,'head_timing_dim':HEAD_TIMING_DIM})
    upgraded=build_model(config,atom_catalog)
    current=upgraded.state_dict()
    for key,value in model.state_dict().items():
        if key.startswith('action_head.') and model.config.architecture_profile=='audio_workload_v1':
            current[key]=value.reshape(model.config.action_slots,config.atom_vocab_size,*value.shape[1:]).mean(0)
        elif key == 'slot_embedding.weight':
            current[key] = _expanded_slot_embedding(value, current[key])
        elif key.startswith('action_factors.') and (key.endswith('catalog_fingerprint') or '.atom_' in key):
            # The target policy owns its derived maps. Learned old factor rows
            # transfer below; a new joint factor starts at zero.
            continue
        elif key in current and torch.is_tensor(value) and torch.is_tensor(current[key]) and value.shape==current[key].shape:
            current[key]=value
    upgraded.load_state_dict(current)
    return upgraded


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Export lossless Composer inference weights')
    parser.add_argument('source');parser.add_argument('destination')
    options=parser.parse_args()
    export_inference(options.source,options.destination)
    print(str(Path(options.destination).resolve()),flush=True)
