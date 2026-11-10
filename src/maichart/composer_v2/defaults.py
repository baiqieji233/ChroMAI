"""Release defaults shared by the CLI and main-workbench adapters.

Checkpoint configuration remains authoritative for loading and exact resume.
"""

PROFILE = 'composer_v2_phrase_music_20261109'
TRAIN_DEFAULTS = dict(epochs=50, batch_size=8, grad_accumulation=8, lr=.0002,
    lr_schedule='cosine', warmup_steps=200, min_lr_ratio=.1, d_model=384, layers=8,
    nhead=8, dim_feedforward=1536, dropout=.1, window_beats=16, history_beats=16,
    future_beats=4, ticks_per_beat=48, max_tokens=2048, audio_backend='mert',
    prepare_device='auto', device='auto', precision='auto', validation_max_batches=0,
    checkpoint_interval_steps=500, save_every=5, early_stopping_patience=5, log_interval=20, seed=0, max_steps=0,
    limit_charts=0, cpu_threads=2, cuda_memory_budget_gb=0., audio_cache_gb=.25)
ENCODER_DEFAULTS = dict(audio_layers=4,song_layers=2)
GENERATION_DEFAULTS = dict(temperature=.9, top_p=.95, action_temperature=.9,
    action_top_p=.95, candidates=2, intercept_tap_on_slide=True,
    tap_on_slide_threshold_ms=200., count=1,parallel=1,cpu_threads=0,phrase_search='layout',phrase_beats=4.,phrase_commit_beats=2.,record_candidate_probabilities=False)
EVALUATION_DEFAULTS=dict(limit_charts=2,max_batches=8,timing_rollouts=2,cpu_threads=2,
    window_sampling='spread',check_training_overlap=False)
MAX_BATCH_COUNT = 32
