"""Unclipped beat distances for the committed history window."""
import numpy as np


def history_beat_ages(beats):
    """Subtract in float64 before returning small relative float32 distances."""
    values = np.asarray(beats, dtype=np.float64)
    if not len(values):
        return np.empty(0, dtype=np.float32)
    return (values[-1] - values).astype(np.float32)
