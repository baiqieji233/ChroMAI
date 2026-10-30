"""Chart-structure metrics comparable to official Master charts.

These are diagnostics, not training targets. They measure the properties that
separated v23 output from authored charts on 2026-10-29:

* bar echo: a 4-beat group repeating an earlier group's rhythm, and its button
  layout exactly, as a left-right mirror, or as a 180-degree rotation;
* rhythm grid usage, especially ternary (third/sixth-beat) positions;
* repetition patterns: AA repeats, ABAB ping-pong, chained double taps.

Bars are fixed 4-beat groups of the chart's beat axis. No meter inference
exists in this project, so on non-4/4 songs the echo numbers are an
approximation; compare runs on the same song instead of across meters.

Usage: ``python -m maichart.composer.structure_metrics chart.txt [...]`` or
``--manifest train_manifest.json --limit 400`` for an official reference.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from fractions import Fraction

from .events import event_metadata

# Same definitions measured on 400 official Master charts from the v22 data
# split (2026-10-29). Recompute with ``--manifest`` after data changes.
# 341 parsable charts, 24,415 bars, 192,250 events. v23 (6 runs) for contrast:
# layout echo .002, ternary .004, aa .092, ping-pong .005, chained doubles .085.
OFFICIAL_MASTER_REFERENCE = {
    'rhythm_echo_rate': .556,
    'layout_echo_rate': .152,
    'layout_echo_by_transform': {'identity': .048, 'mirror_lr': .092, 'rotate_180': .012},
    'ternary_rate': .043,
    'ternary_isolated_rate': .062,
    'aa_rate': .157,
    'pingpong_rate': .045,
    'double_chain_rate': .425,
    'echo_evidence_profile': {
        'share_by_E': [.110,.377,.244,.106,.163],
        'copy_rate_by_E': [.068,.149,.249,.439,.791],
        'source': '341 official Master charts; full-prefix strongest evidence, 2026-10-29',
    },
}

_MIRROR_LR = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
_GRID_NAMES = {1: 'integer', 2: 'half', 4: 'quarter', 8: 'eighth', 3: 'third', 6: 'sixth'}


def _lanes(note_text) -> frozenset[int]:
    try:
        meta = event_metadata(str(note_text))
    except Exception:
        return frozenset()
    return frozenset(int(head) for head in meta.heads if len(head) == 1 and head.isdigit())


def _transform(lanes: frozenset[int], kind: str) -> frozenset[int]:
    if kind == 'mirror_lr':
        return frozenset(_MIRROR_LR[x] for x in lanes)
    if kind == 'rotate_180':
        return frozenset((x + 3) % 8 + 1 for x in lanes)
    return lanes


def _bar_echo(events):
    bars = defaultdict(list)
    for beat, lanes in events:
        if not lanes:
            continue
        index = int(beat // 4)
        bars[index].append((beat - 4 * index, lanes))
    keys = sorted(k for k in bars if len(bars[k]) >= 4)
    counts = Counter()
    for i, key in enumerate(keys):
        current = bars[key]
        rhythm = tuple(offset for offset, _ in current)
        same_rhythm = [bars[k] for k in keys[:i] if tuple(o for o, _ in bars[k]) == rhythm]
        counts['rhythm'] += bool(same_rhythm)
        for kind in ('identity', 'mirror_lr', 'rotate_180'):
            if any(all(_transform(pl, kind) == cl for (_, pl), (_, cl) in zip(prev, current)) for prev in same_rhythm):
                counts[kind] += 1
                break
    return len(keys), counts


def structure_metrics(points) -> dict:
    """Metrics for ``points`` (objects with ``beat``/``note_text`` or (beat, text) pairs)."""
    from .echo_metrics import echo_evidence_profile
    points = list(points)
    profile = echo_evidence_profile(points)
    raw = []
    for point in points:
        beat, text = (point.beat, point.note_text) if hasattr(point, 'beat') else point
        raw.append((Fraction(beat), _lanes(text)))
    raw.sort(key=lambda item: item[0])
    if not raw:
        return {'events': 0, 'reference': OFFICIAL_MASTER_REFERENCE, 'echo_evidence_profile': profile}
    beats = sorted({beat for beat, _ in raw})
    grid = Counter(_GRID_NAMES.get(beat.denominator, 'other') for beat in beats)
    quarter=Fraction(1,4)
    offsets=Counter()
    longest=count=0;previous=None;run_start=None;longest_span=None
    for beat in beats:
        offset=beat%quarter
        if offset>quarter/2:offset-=quarter
        offsets[str(offset)]+=1
        if offset and offset==previous:
            count+=1
        elif offset:
            count=1;run_start=beat
        else:count=0
        if count>longest:
            longest=count;longest_span=dict(offset_beats=str(offset),start_beat=str(run_start),end_beat=str(beat))
        previous=offset
    # Isolated ternary onset: no other third/sixth-beat onset within one beat.
    ternary = [beat for beat in beats if beat.denominator in (3, 6)]
    isolated = sum(1 for i, beat in enumerate(ternary)
                   if not any(abs(other - beat) <= 1 for j, other in enumerate(ternary) if j != i))
    bars, echo = _bar_echo(raw)
    # AA / ABAB on events with button lanes, decisions within 2 beats of the previous event.
    buttons = [(b, l) for b, l in raw if l]
    decisions = aa = pingpong = 0
    for i in range(3, len(buttons)):
        beat, current = buttons[i]
        history = [buttons[i - k] for k in range(1, 5) if i - k >= 0]
        if beat - history[0][0] > 2:
            continue
        decisions += 1
        aa += current == history[0][1]
        if (history[0][1] == history[2][1] and history[1][1] != history[0][1]
                and current == history[1][1] and beat - history[2][0] <= 3):
            pingpong += 1
    # Chained double taps: after A, A, B (B single lane, within 2 beats), how often B again.
    chain_contexts = chain_hits = 0
    for i in range(3, len(raw)):
        a2, a1, b0, current = raw[i - 3][1], raw[i - 2][1], raw[i - 1][1], raw[i][1]
        if a2 and a2 == a1 and b0 and b0 != a1 and len(b0) == 1 and raw[i][0] - raw[i - 3][0] <= 2:
            chain_contexts += 1
            chain_hits += current == b0
    layout = echo['identity'] + echo['mirror_lr'] + echo['rotate_180']
    return {
        'events': len(raw),
        'echo_evidence_profile': profile,
        'bars': bars,
        'rhythm_echo_rate': echo['rhythm'] / bars if bars else 0.,
        'layout_echo_rate': layout / bars if bars else 0.,
        'layout_echo_by_transform': {kind: (echo[kind] / bars if bars else 0.)
                                     for kind in ('identity', 'mirror_lr', 'rotate_180')},
        'ternary_rate': (grid['third'] + grid['sixth']) / len(beats),
        'ternary_isolated_rate': isolated / len(ternary) if ternary else None,
        'ternary_onsets': len(ternary),
        'grid_level_distribution': {name: grid[name] / len(beats)
                                    for name in ('integer', 'half', 'quarter', 'eighth', 'third', 'sixth', 'other')},
        'absolute_grid_observation': {
            'scope':'exact_onset_phases; nonquarter does not itself mean musically incorrect',
            'quarter_aligned_rate':offsets['0']/len(beats),
            'signed_quarter_offsets':dict(offsets),
            'beat_denominators':dict(Counter(str(beat.denominator) for beat in beats)),
            'longest_same_nonquarter_phase_run':longest,
            'longest_same_nonquarter_phase_span':longest_span},
        'aa_rate': aa / decisions if decisions else 0.,
        'pingpong_rate': pingpong / decisions if decisions else 0.,
        'double_chain_rate': chain_hits / chain_contexts if chain_contexts else 0.,
        'double_chain_contexts': chain_contexts,
        'pattern_decisions': decisions,
        'onsets': len(beats),
        'reference': OFFICIAL_MASTER_REFERENCE,
    }


def _merge(results):
    """Pool per-chart counts into one corpus-level result (weights by bars/events)."""
    total = Counter()
    for r in results:
        if not r.get('events'):
            continue
        bars = r['bars']
        total['bars'] += bars
        total['events'] += r['events']
        for key in ('rhythm_echo_rate', 'layout_echo_rate'):
            total[key] += r[key] * bars
        for kind, value in r['layout_echo_by_transform'].items():
            total['echo_' + kind] += value * bars
        total['ternary_rate'] += r['ternary_rate'] * r['onsets']
        if r['ternary_onsets']:
            total['ternary_isolated'] += r['ternary_isolated_rate'] * r['ternary_onsets']
            total['ternary_onsets'] += r['ternary_onsets']
        total['onsets'] += r['onsets']
        for key in ('aa_rate', 'pingpong_rate'):
            total[key] += r[key] * r['pattern_decisions']
        total['decisions'] += r['pattern_decisions']
        total['chain_hits'] += r['double_chain_rate'] * r['double_chain_contexts']
        total['chain_contexts'] += r['double_chain_contexts']
    bars = max(total['bars'], 1)
    return {
        'charts': sum(1 for r in results if r.get('events')), 'events': total['events'], 'bars': total['bars'],
        'rhythm_echo_rate': total['rhythm_echo_rate'] / bars,
        'layout_echo_rate': total['layout_echo_rate'] / bars,
        'layout_echo_by_transform': {k: total['echo_' + k] / bars for k in ('identity', 'mirror_lr', 'rotate_180')},
        'ternary_rate': total['ternary_rate'] / max(total['onsets'], 1),
        'ternary_isolated_rate': (total['ternary_isolated'] / total['ternary_onsets']) if total['ternary_onsets'] else None,
        'aa_rate': total['aa_rate'] / max(total['decisions'], 1),
        'pingpong_rate': total['pingpong_rate'] / max(total['decisions'], 1),
        'double_chain_rate': total['chain_hits'] / max(total['chain_contexts'], 1),
        'reference': OFFICIAL_MASTER_REFERENCE,
    }


def _chart_metrics(path, difficulty):
    from maichart.event_timing import parse_event_chart
    chart = parse_event_chart(path, difficulty, strict_notes=False)
    return structure_metrics(chart.points)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('charts', nargs='*', help='maidata.txt files (generated output uses difficulty 5)')
    parser.add_argument('--difficulty', type=int, default=5)
    parser.add_argument('--manifest', help='training manifest JSON: measure its official charts')
    parser.add_argument('--limit', type=int, default=400)
    args = parser.parse_args(argv)
    paths = list(args.charts)
    if args.manifest:
        with open(args.manifest, encoding='utf-8') as handle:
            manifest = json.load(handle)
        paths += [song['maidata_path'] for song in manifest['songs'][:args.limit]
                  if any(d['difficulty_index'] == args.difficulty for d in song['difficulties'])]
    results = []
    for path in paths:
        try:
            results.append(_chart_metrics(path, args.difficulty))
        except Exception as exc:  # report and continue: one bad chart must not hide the corpus
            print(json.dumps({'path': str(path), 'error': str(exc)}, ensure_ascii=False))
    print(json.dumps(_merge(results), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
