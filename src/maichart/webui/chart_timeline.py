"""Read-only note timelines using the same timing rules as MiaCode playback."""
from __future__ import annotations

from bisect import bisect_right
from fractions import Fraction
from functools import lru_cache
import math
from pathlib import Path
import re

from maichart.event_timing import EventTempo, parse_event_chart
from maichart.generation.muri.parser import _parse_one, _reference_data
from maichart.notes import BASIC_COMPACT_CHORD_RE, _split_top_level_slashes
from maichart.tempo_clock import TempoClock


def _beat_at_second(clock, second):
    """Invert the actual tempo map, rather than extending the head's BPM."""
    if not math.isfinite(second) or second < clock.seconds[0]:
        return None
    index=max(0,bisect_right(clock.seconds,second)-1)
    delta=Fraction(str(second))-Fraction(str(clock.seconds[index]))
    beat=clock.beats[index]+delta*Fraction(str(clock.bpms[index]))/60
    # Native marker times are doubles. Remove only their arithmetic roundoff.
    concise=beat.limit_denominator(1_000_000)
    return str(concise if abs(clock.at(concise)-second)<1e-9 else beat)


def _unknown_head(raw, beat, second, source_id, message):
    """Keep a recognisable head visible, without inventing a sustain tail."""
    match=re.match(r'^([ABCDE][1-8]?|[1-8])',raw)
    if match is None:return None
    head=match.group(1)
    if head.startswith('C'):head='C'
    outside=re.sub(r'\[[^\]]*\]','',raw)
    touch=head[0] in 'ABCDE'
    kind=('touch_hold' if 'h' in outside.lower() else 'touch') if touch else (
        'slide' if any(symbol in outside[1:] for symbol in '-^v<>VpqszwW') else
        'hold' if 'h' in outside.lower() else 'tap')
    return dict(id=source_id,beat=str(beat),start_sec=second,kind=kind,head=head,
                raw=raw,has_head=True,unsupported=True,parse_message=message)


def _marker_note(marker, beat, raw, source_id, branch, clock):
    kind='slide' if marker['type']=='wifi' else marker['type']
    start=float(marker['second'])
    head=str(marker['touch_pad'] or marker['lane'])
    note=dict(id=f'{source_id}:{branch}',beat=str(beat),start_sec=start,kind=kind,
              head=head,raw=raw,has_head=bool(marker['has_head_star']) if kind=='slide' else True,
              is_break=bool(marker['is_break']),is_ex=bool(marker['is_ex']))
    if kind=='slide' and branch:note['has_head']=False
    end=float(marker['end_second'])
    if kind in {'hold','touch_hold','slide'} and math.isfinite(end) and end>=start:
        note['end_sec']=end
        end_beat=_beat_at_second(clock,end)
        if end_beat is not None:note['end_beat']=end_beat
    if kind=='slide':
        note['end_head']=str(marker['end_lane'])
        note['trace_start_sec']=float(marker['slide_trace_second'])
        note['path']=' / '.join(marker['slide_segment_keys'])
        segments=[]
        for key,shoot,duration in zip(marker['slide_segment_keys'],
                                      marker['slide_segment_shoot_seconds'],
                                      marker['slide_segment_durations']):
            segment=dict(start_sec=float(shoot),end_sec=float(shoot)+max(.001,float(duration)),
                         head=key[0],end_head=key[-1],path=key)
            end_beat=_beat_at_second(clock,segment['end_sec'])
            if end_beat is not None:segment['end_beat']=end_beat
            segments.append(segment)
        note['segments']=segments
    return note


@lru_cache(maxsize=24)
def _cached(path_text, modified_ns, size, difficulty_index):
    path=Path(path_text)
    chart=parse_event_chart(path,difficulty_index,strict_notes=False)
    # Multiple BPM directives at one location have one effective final BPM.
    changes={change.beat:change.bpm for change in chart.tempo_changes}
    tempo=[EventTempo(beat,bpm) for beat,bpm in sorted(changes.items())]
    clock=TempoClock(tempo,chart.audio_first)
    notes=[]
    warnings=[]
    try:refs=_reference_data()
    except (OSError,ValueError,RuntimeError) as exc:
        refs={}
        warnings.append(dict(code='slide_reference_unavailable',message=str(exc)))
    for point_index,point in enumerate(chart.points):
        second=clock.at(point.beat)
        for atom_index,raw in enumerate(_split_top_level_slashes(point.note_text)):
            if not raw:continue
            compact=BASIC_COMPACT_CHORD_RE.fullmatch(raw)
            atoms=[button+compact['modifiers'] for button in compact['positions']] if compact else [raw]
            for child_index,atom in enumerate(atoms):
                source_id=f'{point_index}:{atom_index}:{child_index}'
                try:
                    markers=_parse_one(atom,point_index+1,atom_index+1,second,0,clock.bpm(point.beat),refs)
                    if not markers:raise ValueError('Unsupported note syntax')
                except (ValueError,TypeError,KeyError,IndexError,OverflowError) as exc:
                    message=str(exc)
                    fallback=_unknown_head(raw,point.beat,second,source_id,message)
                    if fallback is not None:notes.append(fallback)
                    warnings.append(dict(code='unsupported_note',beat=str(point.beat),raw=raw,message=message))
                    continue
                for branch,marker in enumerate(markers):
                    notes.append(_marker_note(marker,point.beat,raw,source_id,branch,clock))
    chart_end=clock.at(chart.end_beat)
    duration=max(0.,chart_end,*(note.get('end_sec',note['start_sec']) for note in notes))
    # These are quarter-note beats, with no assumption about time signature.
    # Very long documents can use coarser integer-beat ruler marks explicitly.
    last_beat=max(chart.end_beat,Fraction(_beat_at_second(clock,duration) or '0'))
    step=max(1,math.ceil(float(last_beat)/20000))
    beats=[dict(beat=str(beat),time_sec=clock.at(Fraction(beat)))
           for beat in range(0,math.floor(last_beat)+1,step)]
    current=path.stat()
    if (current.st_mtime_ns,current.st_size)!=(modified_ns,size):
        raise ValueError('谱面文件已改变，请重新载入后查看时间轴')
    return dict(schema='maichart-preview-timeline-v1',difficulty_index=difficulty_index,
                first_sec=chart.audio_first,end_beat=str(chart.end_beat),chart_end_sec=chart_end,
                duration_sec=duration,notes=notes,beats=beats,beat_step=step,
                tempo_map=[dict(beat=str(change.beat),time_sec=clock.at(change.beat),bpm=change.bpm) for change in tempo],
                warnings=warnings)


def timeline(path, difficulty_index, *, audio_duration_sec=None):
    path=Path(path).resolve()
    stat=path.stat()
    value=dict(_cached(str(path),stat.st_mtime_ns,stat.st_size,int(difficulty_index)))
    if audio_duration_sec is not None:
        duration=float(audio_duration_sec)
        if math.isfinite(duration) and duration>0:
            value['audio_duration_sec']=duration
            value['duration_sec']=max(value['duration_sec'],duration)
    return value
