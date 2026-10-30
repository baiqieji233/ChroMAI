"""Event-level chart composer model family.

The composer predicts the next authored event as a structured decision rather
than continuing Fraction and note characters token by token.  It intentionally
shares the repository's audio/query, exact timing, action-state and exporter
modules while keeping a new checkpoint family and training contract.
"""
FAMILY = 'chart_composer_v1'
CHECKPOINT_SCHEMA = 'maichart-chart-composer-checkpoint-v2'
INPUT_CONTRACT = 'chart_composer_inputs_v2_audio_only_history_state'
RELATIONAL_INPUT_CONTRACT = 'chart_composer_inputs_v3_separate_pads_relative_heads'
TRAINING_SEMANTICS = 'chart_composer_training_v6_unified_events_head_timing'
GROUNDED_TRAINING_SEMANTICS = 'chart_composer_training_v11_history_relation_mask'
LEGACY_ACTION_SAMPLING_POLICY = 'calibrated_kind_touch_rings_v2'
PREVIOUS_ACTION_SAMPLING_POLICY = 'generation_matched_recovery_prefix_retry_v3'
MASKED_TOUCH_ACTION_SAMPLING_POLICY = 'generation_matched_recovery_masked_touch_v4'
# v5 = v4 plus the generation samplers that keep learned rhythm-grid mass
# and tempers only binary/ternary switches (grid_family_coherent_v2), and damp
# repetition only past official run tails (official_tail_v2).
ACTION_SAMPLING_POLICY = 'generation_matched_recovery_musical_tails_v5'
MOTIF_TRAINING_POLICY = 'symbolic_masked_archive_v1'

__all__ = ['FAMILY','CHECKPOINT_SCHEMA','INPUT_CONTRACT','RELATIONAL_INPUT_CONTRACT',
           'TRAINING_SEMANTICS','GROUNDED_TRAINING_SEMANTICS','ACTION_SAMPLING_POLICY',
           'LEGACY_ACTION_SAMPLING_POLICY','PREVIOUS_ACTION_SAMPLING_POLICY','MASKED_TOUCH_ACTION_SAMPLING_POLICY',
           'MOTIF_TRAINING_POLICY']
