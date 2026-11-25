"""Versioned, opt-in contract for cross-phrase arrangement learning."""

POLICY = 'phrase_intent_v1'
CHOICES = ('auto', 'off', POLICY)
METRICS = ('intent_loss', 'intent_teacher_coverage', 'intent_teacher_route_coverage',
           'intent_execution_strength', 'intent_execution_coverage', 'intent_logit_shift',
           'intent_prior_entropy', 'intent_teacher_entropy', 'intent_selection_evidence',
           'intent_reconstruction_loss', 'intent_audio_only_loss', 'intent_shuffled_support_loss',
           'intent_teacher_variant_0', 'intent_teacher_variant_1',
           'intent_teacher_variant_2', 'intent_teacher_variant_3')


def resolve(requested, *, saved, resume, sequence):
    previous = (saved or {}).get('intent_policy', 'off')
    if requested not in CHOICES or previous not in CHOICES[1:]:
        raise ValueError('未知的编排构思设置')
    selected = previous if requested == 'auto' else requested
    if resume and previous != selected:
        raise ValueError('断点续训须沿用原编排构思设置，更改请在新目录微调')
    if selected != 'off' and sequence != 'continuous_events_v1':
        raise ValueError('编排构思学习需要 Composer v2 连续事件模型')
    return selected


def fixed_backbone_options(requested, *, saved, scope, resume):
    """Keep the pretrained modules and audio inputs of an intent-only finetune."""
    if scope != 'intent_only' or resume:
        return {}

    from .conditions import policy_from_config
    from .defaults import FUTURE_CONFIGURATION_WEIGHT

    options = (
        ('condition_policy', policy_from_config(saved), '等级与 Radar / Style 条件'),
        ('future_configuration_policy', saved.get('future_configuration_policy', 'off'), '局部写法学习'),
        ('future_configuration_weight', saved.get('future_configuration_weight', FUTURE_CONFIGURATION_WEIGHT), '写法学习权重'),
        ('music_context', saved.get('music_context_policy', 'off'), '音乐上下文'),
        ('rhythm_structure', saved.get('structure_policy', 'off'), '节拍结构与对齐参考'),
        ('onset_readout', saved.get('onset_readout_policy', 'off'), '独立声学起音参考'),
        ('intensity_curve', saved.get('intensity_policy', 'off'), '学习强度曲线'),
        ('spatial_policy', saved.get('spatial_policy', 'off'), '盘面走势学习'),
        ('audio_layers', saved.get('audio_layers', 2), '音频编码层数'),
        ('local_audio_block_beats', saved.get('local_audio_block_beats', 1), '局部音频编码'),
    )
    for key, previous, label in options:
        choice = requested.get(key)
        if choice in (None, 'auto'):
            continue
        if not isinstance(choice, bool) and (choice == previous or str(choice) == str(previous)):
            continue
        raise ValueError(f'仅训练编排构思须沿用检查点的「{label}」。如需更改，请选择可训练对应模块的微调范围。')
    return {key: previous for key, previous, _ in options}
