"""Conservative exact digital-silence bounds and an explicit authored end."""
import math

import numpy as np
from maichart.form.waveform_cache import mapped_waveform


def exact_signal_end(signal, sample_rate):
    # Search backward in bounded views; do not copy the full song or define
    # silence by loudness. Quiet nonzero samples always remain in range.
    for stop in range(len(signal),0,-65536):
        start = max(0,stop-65536)
        indices = np.flatnonzero(signal[start:stop] != 0)
        if len(indices): return (start+int(indices[-1])+1)/sample_rate
    return 0.


def generation_bounds(audio, *, requested_end=None, trim_exact_silence=True):
    duration = float(audio.duration_sec)
    if requested_end is not None and (not math.isfinite(requested_end) or not 0 < requested_end <= duration):
        raise ValueError(f'生成结束时间必须大于 0 且不超过音频时长 {duration:.3f} 秒')
    digital_end = duration
    if trim_exact_silence:
        signal,sr = mapped_waveform(audio.source,audio.metadata['audio_sha256'],audio.cache_dir)
        measured = exact_signal_end(signal,sr)
        # A substantial exact-zero suffix avoids treating an isolated sample
        # crossing at EOF as a musical boundary. This is never an RMS gate.
        if duration-measured >= .05:
            digital_end = measured
    end = min(digital_end,requested_end if requested_end is not None else duration)
    return {'file_duration_sec':duration, 'exact_nonzero_end_sec':digital_end,
        'requested_end_sec':requested_end, 'effective_end_sec':end,
        'trim_exact_silence':bool(trim_exact_silence), 'amplitude_threshold':0.,
        'scope':'note_heads_and_action_ends',
        'limitations':['A nonzero fade or reverb is retained. Musical ending must be learned or specified.']}
