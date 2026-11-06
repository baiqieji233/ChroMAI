"""Isolated checkpoint contract for causal whole-event Composer models."""
from dataclasses import asdict
from pathlib import Path
import os
import torch

SCHEMA='maichart-composer-event-set-checkpoint-v1'
FAMILY='chart_composer_v1'
PROFILE='event_set_stream_v1'
UNIFIED_OBJECTIVE_VERSION='stream_causal_choice_recovery_v14'
CONFIGURATION_SEQUENCE_OBJECTIVE_VERSION='stream_configuration_sequence_v13'
LEGACY_UNIFIED_OBJECTIVE_VERSION='stream_ordered_draft_first_arrival_v12'


def objective_version_for_config(config):
    value=config if isinstance(config,dict) else vars(config)
    if (value.get('plan_kind_policy')=='unified_first_arrival_v6' or
            value.get('recovery_policy')=='first_step_phrase_v6' or
            value.get('song_choice_objective_policy')=='event_credit_v1'):
        return UNIFIED_OBJECTIVE_VERSION
    new=(value.get('draft_objective_policy')=='sequence_evidence_v2' or
         value.get('recovery_policy')=='certified_phrase_v5' or
         value.get('event_family_policy')=='shared_music_mixture_v2' or
         value.get('literal_set_policy')=='permuted_unique_v1' or
         value.get('duration_objective_policy')=='hold_crps_v2' or
         value.get('deployment_condition_fraction',1.)<1.)
    return CONFIGURATION_SEQUENCE_OBJECTIVE_VERSION if new else LEGACY_UNIFIED_OBJECTIVE_VERSION


def training_objective_compatible(payload):
    config=payload.get('model_config') or {}
    current=(config.get('draft_policy')=='beat_ordered_draft_v1' or
             config.get('plan_kind_policy') in ('unified_first_arrival_v5','unified_first_arrival_v6'))
    if not current and config.get('song_response_policy')!='unified_profile_response_v2':
        return True
    recipe=(payload.get('args') or {}).get('training_objective_recipe') or {}
    return recipe.get('version')==objective_version_for_config(config)

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
    training_state=payload.get('training_state') or {}
    content_identity=training_state.get('dataset_content_identity') if isinstance(training_state,dict) else None
    content_verified=(isinstance(content_identity,dict) and
        content_identity.get('schema')=='stream_content_identity_v1' and bool(content_identity.get('fingerprint')))
    objective_verified=training_objective_compatible(payload)
    preview_updates=int(payload.get('model_state_dict',{}).get('curve_preview_trained_updates',0))
    return dict(valid=True,family=FAMILY,schema=SCHEMA,model_config=config.to_dict(),
        epoch=payload.get('epoch',0),optimizer_steps=payload.get('optimizer_steps',0),
        args=payload.get('args',{}),can_resume=bool(payload.get('optimizer_state_dict') and payload.get('training_state') and content_verified and objective_verified),
        dataset_content_identity=content_identity,
        training_migration=('训练目标已更新为有序短段草稿与统一出音分布；请在新目录从头训练（fresh），旧检查点仍可按原策略生成。' if not objective_verified else
            None if content_verified else '旧检查点缺少数据内容指纹，不能直接续训；请在新目录使用 finetune。检查点仍可用于生成。'),
        can_init=True,checkpoint_kind=payload.get('checkpoint_kind'),
        event_objective_scope=('conditional_on_deployed_selected_song_component' if getattr(config,'draft_policy','off')=='beat_ordered_draft_v1'
            else 'checkpoint_defined_hierarchical_event_objective'),
        profile_objective_scope=('proper_mixture_likelihood_once_per_chart' if config.song_response_policy=='unified_profile_response_v2' else 'checkpoint_defined_profile_objective'),
        curve_preview_policy=config.curve_preview_policy,curve_preview_trained_updates=preview_updates,
        deployment_condition_fraction=config.deployment_condition_fraction,
        song_choice_objective_policy=config.song_choice_objective_policy,
        effective_loss_weights=dict(plan=0. if config.plan_kind_policy in ('unified_first_arrival_v5','unified_first_arrival_v6') else config.plan_weight,
            phrase=0. if getattr(config,'draft_policy','off')=='beat_ordered_draft_v1' else config.phrase_loss_weight,
            relation=0. if getattr(config,'draft_policy','off')=='beat_ordered_draft_v1' else config.phrase_relation_weight,
            draft=getattr(config,'draft_loss_weight',0.) if getattr(config,'draft_policy','off')=='beat_ordered_draft_v1' else 0.,
            curve_preview=config.curve_preview_weight if config.curve_preview_policy!='off' else 0.),
        atom_vocab_size=codec.num_atoms,event_shapes=codec.num_shapes,codec_coverage=codec.coverage,
        generation_capabilities=dict(composer_generation=True,event_set_stream=True,rational_event_positions=True,
            learned_curve_preview=config.curve_preview_policy=='audio_counts_v1' and preview_updates>0,
            curve_preview_policy=config.curve_preview_policy,curve_preview_trained_updates=preview_updates,
            phrase_search=True,phrase_candidates=(1,2,4),transactional_sustain_repair=True,
            song_profile=getattr(config,'song_profile_policy','off') in ('masked_joint_v1','masked_ordered_joint_v2','masked_deployment_joint_v3'),
            song_response=getattr(config,'song_response_policy','off') in ('music_response_v1','unified_profile_response_v2'),
            persistent_phrase_modes=getattr(config,'phrase_execution_policy','off') in ('persistent_modes_v1','component_modes_v2'),
            ordered_draft=getattr(config,'draft_policy','off')=='beat_ordered_draft_v1',
            draft_policy=getattr(config,'draft_policy','off'),
            draft_width=96,draft_max_anchors=16,
            adjustable_draft=getattr(config,'draft_policy','off')=='beat_ordered_draft_v1',
            unified_first_arrival=getattr(config,'plan_kind_policy','off') in ('unified_first_arrival_v5','unified_first_arrival_v6'),
            shared_detail_components=getattr(config,'detail_representation_policy','private_v1') in ('shared_components_v1','typed_components_v2'),
            logarithmic_duration_features=getattr(config,'duration_feature_policy','legacy_v1')=='log_time_v2',
            learned_phrase_context=getattr(config,'phrase_context_policy','off')!='off',
            ordered_phrase_memory=getattr(config,'phrase_context_policy','off') in ('ordered_phrase_v2','aligned_phrase_v3','shared_ordered_v4'),
            aligned_phrase_context=getattr(config,'phrase_context_policy','off') in ('aligned_phrase_v3','shared_ordered_v4'),
            duration_interval_audio=getattr(config,'duration_audio_policy','off')!='off',
            optional_audio_tracking=getattr(config,'phrase_context_policy','off')!='off',audio_tracking_default=0.,
            absolute_onset_prediction=getattr(config,'timing_output_policy','relative_delta_v1')=='absolute_grid_v1',
            semantic_event_families=getattr(config,'event_family_policy','legacy_v1')!='legacy_v1',
            audio_grounded_event_families=getattr(config,'event_family_policy','legacy_v1') in ('audio_semantic_v2','audio_prior_transition_v3','audio_state_phrase_v4','shared_continuation_v1','shared_music_mixture_v2'),
            shared_continuation_event_families=getattr(config,'event_family_policy','legacy_v1') in ('shared_continuation_v1','shared_music_mixture_v2'),
            deployed_audio_family_mixture=config.event_family_policy=='shared_music_mixture_v2',
            literal_set_policy=config.literal_set_policy,
            hold_duration_calibration=config.duration_objective_policy=='hold_crps_v2' and config.duration_calibration_weight>0,
            duration_objective_policy=config.duration_objective_policy,
            duration_calibration_scope='fully_parsed_regular_hold_only',
            state_phrase_event_families=getattr(config,'event_family_policy','legacy_v1')=='audio_state_phrase_v4',
            family_preserving_template=getattr(config,'template_policy','cross_family_v1')=='family_preserving_v1',
            selected_native_state=getattr(config,'candidate_state_policy','cursor_v1')=='selected_native_v1',
            music_conditioned_transition_exit=getattr(config,'transition_policy','off') in ('music_exit_v1','music_exit_v2','music_exit_v3'),
            pattern_aware_continuation=config.continuation_policy=='pattern_runs_v2'),
        continuation_feature_dim=continuation_dim(config.continuation_policy),
        training_semantics='causal_event_set_stream_v1',condition_defaults=payload.get('condition_defaults'))
