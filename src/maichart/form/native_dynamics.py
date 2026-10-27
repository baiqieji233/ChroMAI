"""Shared waveform dynamics on explicit source times; no FFT or chart labels."""
from __future__ import annotations
import numpy as np


def native_dynamics(audio, times):
    from maichart.form.waveform_cache import mapped_waveform
    from maichart.form.rational_audio import waveform
    from pathlib import Path

    source = getattr(audio, 'source', '')
    if Path(source).is_file():
        signal, sr = mapped_waveform(source, audio.metadata.get('audio_sha256'), audio.cache_dir)
    else:
        signal, sr = waveform(source, audio.metadata.get('audio_sha256'))
    times = np.asarray(times, dtype=np.float64)
    result = np.zeros((len(times), 4), dtype=np.float32)
    short = max(16, round(sr * .006))
    centers = np.rint(times * sr).astype(np.int64)
    groups = np.floor(times / 8.).astype(np.int64)
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        padding=max(short,1024+round(.04*sr))+1
        lo = max(0, int(centers[rows].min()) - padding)
        hi = min(len(signal), int(centers[rows].max()) + padding)
        block = np.asarray(signal[lo:hi], dtype=np.float64)
        power = np.r_[0., np.cumsum(block * block)]
        changes = np.r_[0., np.cumsum(np.diff(block) ** 2)]

        def energy(left, right):
            a = np.clip(left - lo, 0, len(block))
            b = np.clip(right - lo, 0, len(block))
            return np.maximum(0., (power[b] - power[a]) / np.maximum(right - left, 1))

        c = centers[rows]
        result[rows, 0] = np.sqrt(np.maximum(0., energy(c, c + short) - energy(c - short, c)))
        result[rows, 1] = np.sqrt(energy(c - 1024, c + 1024))
        a = np.clip(c - short - lo, 0, max(0, len(block) - 1))
        b = np.clip(c + short - 1 - lo, 0, max(0, len(block) - 1))
        result[rows, 2] = np.sqrt(np.maximum(0., (changes[b] - changes[a]) / (2 * short - 1)))
        before, after = c - round(.04 * sr), c + round(.04 * sr)
        result[rows, 3] = (np.sqrt(energy(after - 1024, after + 1024))
                          - np.sqrt(energy(before - 1024, before + 1024))) / .08
    scale = np.asarray(audio.metadata['normalization_scale'], dtype=np.float64)
    result[:, :3] /= np.maximum(scale[[92, 93, 94]], 1e-8)
    result[:, 3] = np.clip(result[:, 3] / max(scale[93], 1e-8), -8., 8.)
    result[(times < 0.) | (times >= audio.duration_sec)] = 0.
    # A malformed legacy cache must never poison a whole training run.  Keep
    # the measured values when valid and replace only non-finite entries.
    return np.nan_to_num(result, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32, copy=False)
