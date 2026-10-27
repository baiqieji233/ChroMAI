"""Operational defaults for a new Composer training stage.

These describe the training recipe, not missing fields in old checkpoints.
Keep checkpoint/model legacy defaults independent from this fresh recipe.
"""

TRAINING_DEFAULTS = {
    'epochs': 20,
    'batch_size': 64,
    'grad_accumulation': 4,
    'lr': 2e-4,
    'd_model': 384,
    'layers': 6,
    'nhead': 8,
    'dim_feedforward': 1536,
    'cpu_threads': 2,
    'sample_workers': 1,
    'recovery_interval': 25,
}
