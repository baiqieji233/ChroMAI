"""Native short-window dynamics and reusable real-time acoustic context."""
from __future__ import annotations

import numpy as np

FEATURE_POLICY = 'native_dynamics_context_v3'
CONTEXT_DIM = 104
INTERVAL_DIM = 16
TRAILING_CONTEXT_DIM = 52
_COLUMNS = (92, 93, 94, 95, 96, 97)


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


class AcousticContext:
    """Time-weighted .5/2/8-second views plus a whole-audio baseline.

    Only immutable audio is cached. No authored notes or predicted chart state
    enter this bank, so planning remains identical across candidate windows.
    """
    def __init__(self, values, times, duration, *, validated_bank=False):
        # Macroscopic pooling does not need the 5ms transient grid. Retain
        # native candidate features separately and bound this prefix's memory.
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        values = np.asarray(values, dtype=np.float32)
        if len(times) != len(values):
            raise ValueError('AcousticContext values and times must have the same length')
        if not validated_bank:
            finite = np.isfinite(times)
            if len(values):
                finite &= np.isfinite(values).all(axis=1)
            if not finite.all():
                times, values = times[finite], values[finite]
        if len(times) and not validated_bank:
            order = np.argsort(times, kind='stable')
            times, values = times[order], values[order]
            # A legacy/native bank may contain repeated timestamps.  Collapse
            # them before building a piecewise-linear prefix so no zero-width
            # interval can create a NaN slope.
            unique, first, counts = np.unique(times, return_index=True, return_counts=True)
            if len(unique) != len(times):
                sums = np.add.reduceat(values.astype(np.float64), first, axis=0)
                values = (sums / counts[:, None]).astype(np.float32)
                times = unique
        if not len(times):
            self.times=times;self.duration=max(float(duration),1e-6)
            self.data=np.zeros((0,26));self.prefix=np.zeros((1,26));self.global_mean=np.zeros(26)
            return
        # Clip before deduplicating.  Clipping after ``unique`` can turn both
        # an out-of-range search result and the final valid index into the same
        # last row, recreating a zero-width interval at the tail.
        take = np.unique(np.clip(
            np.r_[np.searchsorted(times, np.arange(0., duration, .05)), len(times) - 1],
            0, max(0, len(times) - 1)))
        self.times = times[take]
        self.duration = max(float(duration), 1e-6)
        source = np.nan_to_num(np.asarray(values[take], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        dynamics = source[:, _COLUMNS]
        dynamics = np.sign(dynamics) * np.log1p(np.abs(dynamics))
        bands = source[:, :80].reshape(-1, 8, 10).mean(-1)
        data = np.concatenate((dynamics, bands, source[:, 80:92]), axis=-1)
        self.data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        intervals = np.diff(self.times)
        self.prefix = np.vstack((np.zeros((1, 26)),
            np.cumsum((self.data[:-1] + self.data[1:]) * .5 * intervals[:, None], axis=0)))
        self.global_mean = np.nan_to_num(self.prefix[-1] / self.duration if len(data) else np.zeros(26), nan=0.0)

    def _integral(self, seconds):
        if len(self.times) < 2:
            return np.zeros((len(seconds), 26), dtype=np.float64)
        seconds = np.clip(seconds, self.times[0], self.times[-1])
        left = np.clip(np.searchsorted(self.times, seconds, side='right') - 1, 0, len(self.times) - 2)
        base_time=self.times[left]
        base=self.data[left]
        delta = seconds - base_time
        denom = np.maximum(self.times[left + 1] - base_time, 1e-9)
        slope = ((self.data[left + 1] - base) / denom[:, None])
        return self.prefix[left] + base * delta[:, None] + .5 * slope * delta[:, None] ** 2

    def at(self, seconds):
        seconds = np.asarray(seconds, dtype=np.float64).reshape(-1)
        widths=np.array([.5,2.,8.])[:,None]
        lo=np.clip(seconds[None]-widths/2.,0.,self.duration)
        hi=np.clip(seconds[None]+widths/2.,0.,self.duration)
        # One boundary lookup replaces six small searches, retaining the
        # original per-window arithmetic and [short, medium, long, global] order.
        integrals=self._integral(np.concatenate((hi.reshape(-1),lo.reshape(-1)))).reshape(2,3,len(seconds),26)
        means=(integrals[0]-integrals[1])/np.maximum(hi-lo,1e-6)[...,None]
        local=means.transpose(1,0,2).reshape(len(seconds),78)
        global_rows=np.broadcast_to(self.global_mean,(len(seconds),26))
        return np.nan_to_num(np.concatenate((local,global_rows),axis=1),nan=0.,posinf=0.,neginf=0.).astype(np.float32)

    def trailing(self, seconds):
        """Audio-only past 1s/4s views, aligned with the ledger's windows."""
        seconds = np.asarray(seconds, dtype=np.float64).reshape(-1)
        integrals = self._integral(np.concatenate((seconds, seconds - 1., seconds - 4.)))
        end, one, four = np.split(integrals, 3)
        return np.concatenate([
            (end - before) / width
            for before, width in ((one, 1.), (four, 4.))
        ], axis=-1).astype(np.float32)

    def interval(self, committed_second, candidate_seconds):
        """Describe skipped music; never convert acoustic strength to notes.

        The start is the last committed event, NOT a decoder search cursor.
        Means retain six dynamics and eight spectral bands; attack area and
        endpoint harmonic change distinguish longer gaps from local probes.
        """
        seconds = np.asarray(candidate_seconds, dtype=np.float64).reshape(-1)
        start = np.clip(float(committed_second), 0., self.duration)
        stop = np.clip(seconds, start, self.duration)
        boundaries=np.r_[stop,start]
        integrals = self._integral(boundaries)
        area = integrals[:-1] - integrals[-1:]
        width = np.maximum(stop - start, 1e-6)
        mean = area[:, :14] / width[:, None]
        if len(self.times):
            endpoints = np.column_stack([
                np.interp(boundaries, self.times, self.data[:, c]) for c in range(14, 26)])
            begin, end = endpoints[-1], endpoints[:-1]
            norm = np.linalg.norm(end, axis=-1) * np.linalg.norm(begin)
            novelty = np.where(norm > 1e-9, 1. - (end * begin).sum(-1) / np.maximum(norm, 1e-9), 0.)
        else:
            novelty = np.zeros(len(stop))
        return np.column_stack((mean, np.log1p(np.maximum(area[:, 0], 0.)), novelty)).astype(np.float32)
