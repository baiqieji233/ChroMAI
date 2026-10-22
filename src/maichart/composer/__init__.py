"""Event-level chart composer model family.

The composer predicts the next authored event as a structured decision rather
than continuing Fraction and note characters token by token.  It intentionally
shares the repository's audio/query, exact timing, action-state and exporter
modules while keeping a new checkpoint family and training contract.
"""
FAMILY = 'chart_composer_v1'
CHECKPOINT_SCHEMA = 'maichart-chart-composer-checkpoint-v2'
INPUT_CONTRACT = 'chart_composer_inputs_v2_audio_only_history_state'
TRAINING_SEMANTICS = 'chart_composer_training_v2_audio_only_candidate_actions'

__all__ = ['FAMILY','CHECKPOINT_SCHEMA','INPUT_CONTRACT','TRAINING_SEMANTICS']
