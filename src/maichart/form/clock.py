"""Explicit chart-relative model clocks and absolute editor/audio clocks."""
from __future__ import annotations

import numpy as np


CHART_CURVE_AXIS = 'chart_relative_v1'
AUDIO_CURVE_AXIS = 'audio_absolute_v2'


def curve_axis(profile):
    axis = profile.get('time_axis', CHART_CURVE_AXIS)
    if axis not in (CHART_CURVE_AXIS, AUDIO_CURVE_AXIS):
        raise ValueError(f'Unsupported song-form curve time axis: {axis}')
    return axis


def profile_seconds(profile, audio_seconds, first):
    values = np.asarray(audio_seconds)
    return values if curve_axis(profile) == AUDIO_CURVE_AXIS else values-float(first)


def chart_curve_to_audio(times, first):
    return np.asarray(times,dtype=np.float64)+float(first)


def clock_report(first):
    return {'chart_beat_zero_audio_sec':float(first),
            'virtual_leading_silence_sec':max(0.,-float(first)),
            'audio_before_chart_sec':max(0.,float(first)),
            'audio_modified':False,
            'relation':'audio_second = first + integrated_chart_seconds',
            'blank_chart_cells_count_separately':True}
