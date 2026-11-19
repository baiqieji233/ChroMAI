"""Compare two charts of the same recording around a chosen musical boundary."""
import argparse
from bisect import bisect_left
import json
import math
from pathlib import Path

from maichart.event_timing import parse_event_chart
from maichart.tempo_clock import TempoClock
from .gesture_pairs import rim_family
from .motion_writing import CONTACT, EVENT_FIELDS, contact


def _chart(path, slot):
    chart = parse_event_chart(path, slot, strict_notes=True)
    clock = TempoClock(chart.tempo_changes, chart.audio_first)
    return chart.points, [clock.at(p.beat) for p in chart.points]


def describe(chart, start, stop):
    points, seconds = chart
    left, right = bisect_left(seconds, start), bisect_left(seconds, stop)
    selected = points[left:right]
    contacts = [contact(p.note_text) for p in selected]
    keys = [key for _, key in contacts]
    # Overlapping four-contact runs can cover the same music at different subdivisions.
    runs = []
    interrupted = 0
    for i in range(len(keys)-3):
        if any(b.beat-a.beat>1 for a,b in zip(selected[i:i+3],selected[i+1:i+4])):
            interrupted += 1
        else:
            runs.append(rim_family(keys[i:i+4]))
    names = {family: runs.count(family) for family in ('sweep', 'alternation', 'jack') if family in runs}
    actions = {field: sum(EVENT_FIELDS.index(field) in flags for flags,_ in contacts)
               for field in CONTACT[1:]}
    return dict(head_events=len(selected), four_contact_runs=names,
                unclassified_runs=runs.count(None), interrupted_runs=interrupted,
                contacts={k:v for k,v in actions.items() if v},
                first_head_sec=seconds[left] if selected else None,
                beat_gaps=[str(b.beat-a.beat) for a,b in zip(selected,selected[1:])],
                interval_ms=[round((b-a)*1000,1) for a,b in zip(seconds[left:right],seconds[left+1:right])])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--generated', required=True, type=Path)
    parser.add_argument('--at-sec', required=True, type=float, nargs='+')
    parser.add_argument('--reference-slot', type=int, default=5)
    parser.add_argument('--generated-slot', type=int, default=5)
    args = parser.parse_args(argv)
    reference = _chart(args.reference, args.reference_slot)
    generated = _chart(args.generated, args.generated_slot)
    times = reference[1]
    results = []
    for sec in args.at_sec:
        if not math.isfinite(sec):
            parser.error('检查时间必须为有限秒数')
        i = bisect_left(times, sec)
        if i and (i == len(times) or sec-times[i-1] <= times[i]-sec):
            i -= 1
        if i < 4 or i+4 >= len(times) or sec<times[0] or sec>times[-1]:
            results.append(dict(requested_sec=sec, status='insufficient_reference_context'))
            continue
        # Centre boundaries between heads so that slight onset jitter stays in the same phrase.
        left = (times[i-4]+times[i-5])/2 if i>4 else times[0]-(times[1]-times[0])/2
        mid = (times[i]+times[i-1])/2
        right = (times[i+3]+times[i+4])/2
        results.append(dict(requested_sec=sec, reference_cue_sec=times[i], bounds_sec=[left,mid,right],
                            reference=[describe(reference,left,mid),describe(reference,mid,right)],
                            generated=[describe(generated,left,mid),describe(generated,mid,right)]))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
