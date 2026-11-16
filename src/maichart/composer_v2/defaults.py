"""Release defaults shared by the CLI and main-workbench adapters.

Checkpoint configuration remains authoritative for loading and exact resume.
"""

import math

PROFILE = 'composer_v2_phrase_music_20261109'
TRAIN_DEFAULTS = dict(epochs=50, batch_size=8, grad_accumulation=8, lr=.0002,
    lr_schedule='cosine', warmup_steps=200, min_lr_ratio=.1, d_model=384, layers=8,
    nhead=8, dim_feedforward=1536, dropout=.1, window_beats=16, history_beats=16,
    future_beats=4, ticks_per_beat=48, max_tokens=2048, audio_backend='mert',
    prepare_device='auto', device='auto', precision='auto', validation_max_batches=0,
    checkpoint_interval_steps=500, save_every=5, early_stopping_patience=5, log_interval=20, seed=0, max_steps=0,
    limit_charts=0, cpu_threads=2, cuda_memory_budget_gb=0., audio_cache_gb=.25,
    activation_checkpointing='auto', attention_projection_packing='auto', decoder_flash_attention='off')
ENCODER_DEFAULTS = dict(audio_layers=4,song_layers=2)
MUSICAL_CONTEXT_POLICY = 'music_context_views_v1'
MUSICAL_CONTEXT_CHOICES = ('auto', 'off', MUSICAL_CONTEXT_POLICY)
MUSICAL_CONTEXT_PROBABILITY = .25
MUSICAL_RELATION_POLICY = 'musical_relations_v1'
MUSICAL_RELATION_CHOICES = ('auto', 'off', MUSICAL_RELATION_POLICY)
MUSICAL_RELATION_FRACTION = .3
FUTURE_CONFIGURATION_POLICY = 'ordered_future_v1'
FUTURE_CONFIGURATION_CHOICES = ('auto', 'off', FUTURE_CONFIGURATION_POLICY)
FUTURE_CONFIGURATION_WEIGHT = .05
FUTURE_CONFIGURATION_TRAIN_VIEW_FRACTION = .25
FINETUNE_SCOPE_CHOICES = ('auto', 'all', 'music_reading')


def resolve_musical_context(requested, probability=None, *, sequence_policy,
                            parent_contract=None, resume=False):
    """Resolve the training-only policy before preparing data or caches."""
    if requested not in MUSICAL_CONTEXT_CHOICES:
        raise ValueError('未知的按音乐组织配置策略')
    saved = parent_contract or {}
    saved_policy = saved.get('musical_context_policy', 'off')
    saved_probability = saved.get('musical_context_probability', MUSICAL_CONTEXT_PROBABILITY)
    continuous = sequence_policy == 'continuous_events_v1'
    policy = requested
    if requested == 'auto':
        policy = saved_policy if resume else MUSICAL_CONTEXT_POLICY if continuous else 'off'
    if policy not in ('off', MUSICAL_CONTEXT_POLICY):
        raise ValueError('未知的按音乐组织配置策略')
    if probability is None:
        probability = saved_probability if resume else MUSICAL_CONTEXT_PROBABILITY
    if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
        raise ValueError('音乐独立练习比例须在 0 至 1 之间')
    if resume and (policy != saved_policy or probability != saved_probability):
        raise ValueError('更改按音乐组织配置或音乐独立练习比例需要使用新目录微调')
    if policy != 'off' and not continuous:
        raise ValueError('按音乐组织配置仅适用于 Composer v2 连续事件训练')
    return policy, float(probability)


def resolve_auxiliary_policy(requested, value, *, feature, choices, default,
                             sequence_policy, saved_config, resume):
    """Resolve an optional continuous-event training feature."""
    if requested not in choices:
        raise ValueError(f'Unknown {feature} policy')
    value_key = 'fraction' if feature == 'musical_relation' else 'weight'
    saved_policy = saved_config.get(f'{feature}_policy', 'off')
    saved_value = saved_config.get(f'{feature}_{value_key}', default)
    continuous = sequence_policy == 'continuous_events_v1'
    policy = requested
    if policy == 'auto':
        policy = saved_policy if resume else choices[-1] if continuous else 'off'
    if policy not in choices[1:]:
        raise ValueError(f'Unknown saved {feature} policy')
    if value is None:
        value = saved_value if resume else default
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{feature}_{value_key} must be finite and nonnegative')
    if value_key == 'fraction' and value >= 1:
        raise ValueError(f'{feature}_{value_key} must be less than 1')
    if value_key == 'weight' and value > 1:
        raise ValueError(f'{feature}_{value_key} must be at most 1')
    if policy == 'off':
        value = default
    if resume and (policy != saved_policy or value != saved_value):
        raise ValueError(f'Changing {feature} requires --init-checkpoint in a new directory')
    if policy != 'off' and not continuous:
        raise ValueError(f'{feature} requires continuous event training')
    return policy, float(value)


def resolve_finetune_scope(requested, parent_contract, *, resume, initialized):
    if requested not in FINETUNE_SCOPE_CHOICES:
        raise ValueError('Unknown fine-tuning scope')
    saved = parent_contract.get('finetune_scope', 'all')
    scope = (saved if resume else 'all') if requested == 'auto' else requested
    if scope not in FINETUNE_SCOPE_CHOICES[1:]:
        raise ValueError('Unknown saved fine-tuning scope')
    if resume and scope != saved:
        raise ValueError('Changing fine-tuning scope requires --init-checkpoint in a new directory')
    if scope == 'music_reading' and not (initialized or resume):
        raise ValueError('Music-reading fine-tuning requires --init-checkpoint')
    return scope


GENERATION_DEFAULTS = dict(temperature=.9, top_p=.95, action_temperature=.9,
    action_top_p=.95, candidates=2, intercept_tap_on_slide=True,
    tap_on_slide_threshold_ms=200., intercept_slide_head_tap=False,
    touch_reachability_policy='palm_span_v1', touch_cohold_radius=180.,
    forward_feasibility='balanced_v1', forward_tap_on_slide_ms=150.,
    self_refine='off',
    count=1,parallel=1,cpu_threads=0,phrase_search='layout',phrase_rank_policy='comfort_v1',phrase_beats=4.,phrase_commit_beats=2.,tempo_octave_policy='off',downbeat_alignment='off',record_candidate_probabilities=False)
EVALUATION_DEFAULTS=dict(limit_charts=2,max_batches=8,timing_rollouts=2,cpu_threads=2,
    window_sampling='spread',check_training_overlap=False)
MAX_BATCH_COUNT = 32
