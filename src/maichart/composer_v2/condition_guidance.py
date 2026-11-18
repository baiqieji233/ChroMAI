"""Opt-in sampling preferences for measurable Radar and Style controls."""
from __future__ import annotations

from collections import OrderedDict
import math

import numpy as np
import torch

from maichart.generation.conditions import GENERATION_DEFAULTS as REFERENCE
from maichart.generation.event_quality import PEAK_WINDOW_SEC

POLICY = 'music_soft_v1'
STYLE_KINDS = {
    'tap_bias': ('tap',),
    'slide_bias': ('slide',),
    'hold_bias': ('hold', 'touch_hold'),
    'touch_bias': ('touch', 'touch_hold'),
}
KEYS = ('peak', *STYLE_KINDS)
MAX_KIND_LOG_ODDS = 1.8
MAX_PEAK_LOG_ODDS = 3.0


def validate(enabled, sequence_policy, condition_policy):
    if enabled and (sequence_policy != 'continuous_events_v1' or condition_policy != 'radar_style_v1'):
        raise ValueError('增强手动 Radar / Style 需要支持雷达与风格条件的 Composer v2 连续事件检查点。')


def _log_odds(value):
    value = min(.99, max(.01, value))
    return math.log(value / (1 - value))


def _style_bias(value, key):
    return float(np.clip(_log_odds(value) - _log_odds(REFERENCE[key]), -MAX_KIND_LOG_ODDS, MAX_KIND_LOG_ODDS))


class ConditionGuidance:
    def __init__(self, overrides, song, codec, device):
        self.codec = codec
        self.manual = {key: float(overrides[key]) for key in KEYS if overrides.get(key) is not None}
        self.kind_offsets = {}
        for key, kinds in STYLE_KINDS.items():
            if key not in self.manual:
                continue
            delta = _style_bias(self.manual[key], key)
            for kind in kinds:
                token = codec.ids.get('N:' + kind)
                if token is not None:
                    self.kind_offsets[token] = self.kind_offsets.get(token, 0.) + delta
        self.kind_bias = None
        if any(self.kind_offsets.values()):
            offsets = torch.zeros(codec.vocab_size, dtype=torch.float32)
            for token, value in self.kind_offsets.items():
                offsets[token] = min(MAX_KIND_LOG_ODDS, max(-MAX_KIND_LOG_ODDS, value))
            self.kind_bias = offsets.to(device)
        self.peak = None
        if 'peak' in self.manual:
            baseline = REFERENCE['peak']
            delta = self.manual['peak'] - baseline
            gain = MAX_PEAK_LOG_ODDS * delta / ((1 - baseline) if delta > 0 else baseline)
            if gain:
                self.peak = PeakTiming(song, codec, gain)
        self.last_query = None
        self.last_timing = None

    def timing(self, after, end):
        if self.peak is None or after < 0:
            return {}
        key = (after, end)
        if key != self.last_query:
            self.last_query = key
            self.last_timing = self.peak.nearby(after, end)
        return self.last_timing

    def report(self):
        return dict(policy=POLICY, manual=dict(self.manual),
                    kind_logit_bias={self.codec.tokens[key]: min(MAX_KIND_LOG_ODDS, max(-MAX_KIND_LOG_ODDS,value))
                                     for key,value in self.kind_offsets.items() if value},
                    peak_active=bool(self.peak is not None and self.peak.times.size))

    def apply(self, logits, choices, prefix, *, action, temperature, nearby=None):
        if action:
            if self.kind_bias is None or not prefix or not (
                prefix[-1] in self.codec.phases or prefix[-1] == self.codec.end_note_id
            ):
                return logits
            if not any(token in self.kind_offsets for token in choices):
                return logits
            guided = logits.float() + self.kind_bias
            if self.codec.event_end_id in choices:
                # Preserve the odds of finishing the onset. Style chooses
                # the type conditional on writing another note there.
                indices = [token for token in choices if token in self.codec.kind_ids]
                original = logits[indices].float()
                updated = guided[indices]
                if temperature == 0:
                    shift = updated.max() - original.max()
                else:
                    shift = temperature*(torch.logsumexp(updated/temperature,0)
                                         - torch.logsumexp(original/temperature,0))
                guided[indices] = updated - shift
            return guided
        if self.peak is None:
            return logits
        offsets = self.peak.offsets(nearby, prefix)
        if not offsets:
            return logits
        guided = logits.float().clone()
        guided[list(offsets)] += torch.as_tensor(list(offsets.values()),device=guided.device)
        return guided


class PeakTiming:
    """Apply a bounded timing preference near separate, supported audio attacks."""

    def __init__(self, song, codec, gain):
        self.clock = song.clock
        self.codec = codec
        self.gain = gain
        self.duration = song.audio.duration_sec
        self.phase_ids = tuple(codec.phases)
        self.phases = np.array([float(codec.phases[key]) for key in self.phase_ids], dtype=np.float64)
        self.times, self.strength = self._audio_attacks(song.audio)
        self.rows = OrderedDict()

    @staticmethod
    def _audio_attacks(audio):
        times = np.asarray(audio.mel_times_sec, dtype=np.float64)
        flux = np.maximum(0., np.asarray(audio.mel[:, 96], dtype=np.float64))
        if times.size < 3:
            return times[:0], flux[:0]
        flux = np.nan_to_num(flux, nan=0., posinf=0., neginf=0.)
        positive = flux[flux > 1e-8]
        if not positive.size:
            return times[:0], flux[:0]
        scale = float(np.quantile(positive, .85))
        energy = np.minimum(flux / scale, 1.)
        # Compare one-second flux activity within this song. Isolated attacks
        # and quiet passages keep the automatic timing preference.
        half = PEAK_WINDOW_SEC / 2
        lo = np.searchsorted(times, times-half)
        hi = np.searchsorted(times, times+half, side='right')
        summed = np.concatenate(([0.],np.cumsum(energy)))
        activity = (summed[hi] - summed[lo]) / np.maximum(1, hi-lo)
        low, high = np.quantile(activity, (.45,.9))
        if high <= low + 1e-6:
            return times[:0], flux[:0]
        active = np.clip((activity-low)/(high-low),0.,1.)
        peaks = np.flatnonzero((flux[1:-1] >= flux[:-2]) & (flux[1:-1] > flux[2:]))+1
        strengths = active[peaks] * np.clip((energy[peaks]-.25)/.75, 0., 1.)
        selected = strengths > .05
        return times[peaks][selected], strengths[selected]

    def _row(self, whole):
        if whole not in self.rows:
            seconds = self.clock.at_floats(whole + self.phases)
            right = np.searchsorted(self.times, seconds).clip(0,len(self.times)-1)
            left = np.maximum(0,right-1)
            nearest = np.where(np.abs(seconds-self.times[left]) <= np.abs(seconds-self.times[right]),left,right)
            match = np.exp(-.5*((seconds-self.times[nearest])/.012)**2)*self.strength[nearest]
            self.rows[whole] = seconds, self.times[nearest], match
            if len(self.rows) > 24:
                self.rows.popitem(last=False)
        self.rows.move_to_end(whole)
        return self.rows[whole]

    def nearby(self, after, end):
        if not self.times.size:
            return {}
        from .data import _beats_at_seconds
        start = self.clock.at(after)
        stop = min(self.clock.at(end),self.duration,start+PEAK_WINDOW_SEC)
        hi = math.floor(_beats_at_seconds(self.clock,[stop])[0])
        result = {}
        for whole in range(math.floor(after),hi+1):
            seconds, attack_time, support = self._row(whole)
            gap = seconds-start
            valid = (gap > 0) & (seconds < stop) & (attack_time > start+.025)
            value = np.where(valid, support * np.exp(-np.maximum(gap,0.)/.32), 0.)
            if value.max(initial=0.) > .04:
                result[whole] = value
        return result

    def offsets(self, nearby, prefix):
        if not nearby or not prefix:
            # [AT] versus song end is left to the checkpoint.
            return {}
        if prefix[-1] == self.codec.phase_id:
            if self.gain < 0:
                # Reducing peaks changes coarse spacing while the original
                # phase distribution continues to place notes on the music.
                return {}
            whole = int(''.join(self.codec.digits[token] for token in prefix[1:-1]))
            values = nearby.get(whole)
            return {} if values is None else {
                self.phase_ids[i]: self.gain*.35*float(values[i]) for i in np.flatnonzero(values>.04)
            }
        digits = ''.join(self.codec.digits[token] for token in prefix[1:])
        offsets = {}
        for whole, values in nearby.items():
            text = str(whole)
            if not text.startswith(digits):
                continue
            token = self.codec.phase_id if len(text)==len(digits) else self.codec.ids['NUM:'+text[len(digits)]]
            offsets[token] = max(offsets.get(token,0.), float(values.max()))
        return {token: self.gain * value for token,value in offsets.items()}
