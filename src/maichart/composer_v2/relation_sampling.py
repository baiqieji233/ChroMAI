"""Score genuine training windows against the chart and its own cached audio."""
from bisect import bisect_left

import numpy as np

POLICY = 'musical_relations_v1'
STRATA = ('stable', 'gradual', 'joint_change', 'rest_or_end')
SPAN = 4


def _audio_features(summary):
    mel = np.asarray(summary['mel'], dtype=np.float32)
    blocks = [mel[:, :96]]
    detail = summary.get('detail')
    if detail is not None:
        blocks.append(np.asarray(detail, dtype=np.float32))
    semantic = np.asarray(summary['semantic'], dtype=np.float32)
    if semantic.ndim == 2 and semantic.shape[1] and np.all(summary['semantic_valid']):
        blocks.append(semantic)
    normalized = []
    for block in blocks:
        deviations = block.std(axis=0)
        floor = max(1e-5, float(deviations.max()) * .1)
        normalized.append((block - block.mean(axis=0)) / np.maximum(deviations, floor)
                          / np.sqrt(block.shape[1] * len(blocks)))
    return np.concatenate(normalized, axis=1)


def _deltas(features):
    return np.linalg.norm(np.diff(features, axis=0), axis=1)


def _relative(movements):
    movements = np.asarray(movements)
    positive = movements[movements > 1e-6]
    scale = np.quantile(positive, .75) if positive.size else 1.
    return np.clip(movements / max(float(scale), 1e-6) / 2., 0., 1.)


def source_scores(song, chart, windows):
    """Return a bounded score per ordinary window and the chosen diagnostic stratum."""
    audio = _audio_features(song.summary)
    summary_beats = np.asarray(song.summary['beats'])
    mel = np.asarray(song.summary['mel'])
    energy = mel[:, 97] if mel.shape[1] > 97 else np.linalg.norm(mel, axis=1)
    positive_energy = energy[energy > 0]
    typical_energy = float(np.median(positive_energy)) if len(positive_energy) else 1.
    if len(audio) < 3 or not len(windows):
        return {}
    rows = []
    for index, begin, end in windows:
        if begin == end:
            continue
        center = float(chart.beats[(begin + end - 1) // 2])
        bounds = center + np.arange(-1.5, 2., 1.) * SPAN
        seconds = chart.clock.at_floats(bounds)
        audio_cuts = np.searchsorted(summary_beats, bounds)
        chart_cuts = [bisect_left(chart.beats, beat) for beat in bounds]
        rest = 0.
        for split in range(begin, end):
            onset = chart.seconds[split]
            prior = chart.seconds[split - 1] if split else 0.
            if chart.quality.covered(prior, onset) <= 0:
                continue
            prior_beat = chart.beats[split - 1] if split else 0
            if chart.beats[split] - prior_beat >= 2 and not chart.has_sustain(prior, onset):
                lo, hi = np.searchsorted(summary_beats, [float(prior_beat), float(chart.beats[split])])
                if lo < hi:
                    rest = max(rest, min(1., float(energy[lo:hi].mean()) / typical_energy))
        final = chart.seconds[end - 1]
        if end == len(chart.beats) and chart.quality.covered(final, chart.quality.duration) > 0:
            rest = max(rest, min(1., (chart.quality.duration - final) / 2))
        plausible = (np.all(np.diff(audio_cuts) > 0)
                     and chart.quality.covered(seconds[0], seconds[-1]) > 0)
        if plausible:
            audio_sections = [audio[lo:hi].mean(0) for lo, hi in zip(audio_cuts, audio_cuts[1:])]
            chart_sections = []
            for lo, hi in zip(chart_cuts, chart_cuts[1:]):
                # Preserve absolute positions while avoiding a density-only score.
                values = chart.events[lo:hi]
                pooled = values.mean(0) if len(values) else np.zeros(chart.events.shape[1])
                chart_sections.append(np.r_[pooled, min(1., len(values) / (4 * SPAN))])
            audio_motion, chart_motion = _deltas(audio_sections), _deltas(chart_sections)
            audio_direction = np.diff(audio_sections, axis=0)
            chart_direction = np.diff(chart_sections, axis=0)
            directions = (
                np.dot(*audio_direction) / max(1e-6, np.prod(audio_motion)),
                np.dot(*chart_direction) / max(1e-6, np.prod(chart_motion)),
            )
            populated = np.all(np.diff(chart_cuts) > 0)
        else:
            audio_motion = chart_motion = (0., 0.)
            directions, populated = (0., 0.), False
        rows.append((index, audio_motion, chart_motion, directions, plausible, populated, rest))

    if not rows:
        return {}
    indices, audio_motion, chart_motion, align, plausible, populated, rest = zip(*rows)
    audio_change, chart_change = _relative(audio_motion), _relative(chart_motion)
    joint_min = np.minimum(audio_change, chart_change).min(1)
    joint_peak = np.minimum(audio_change, chart_change).max(1)
    stable = (1. - np.maximum(audio_change, chart_change).max(1)) * populated
    gradual = np.sqrt(joint_min) * np.maximum(0., np.min(align, axis=1))
    change = np.sqrt(joint_peak)
    scores = np.clip(np.stack((stable * plausible, gradual * plausible, change * plausible, rest), axis=1), 0., 1.)

    # Reserve each stratum's strongest sources before filling by strength,
    # leaving ordinary supervision in the remaining windows.
    chosen = {}
    capacity = max(1, (len(windows) + 1) // 2)
    ranked = np.argsort(-scores, axis=0, kind='stable')
    for rank in range(len(indices)):
        for col, name in enumerate(STRATA):
            row = ranked[rank, col]
            score = float(scores[row, col])
            if score > 0 and indices[row] not in chosen:
                chosen[indices[row]] = (score, name)
                if len(chosen) >= capacity:
                    return chosen
    return chosen
