"""Optional, bounded approximation of selected onsets to common Simai grids."""
from bisect import bisect_right
from fractions import Fraction
import math

from maichart.event_timing import seconds_at_beat


COMMON_DIVISIONS = (4, 8, 12, 16, 24, 32)


def normalize_divisions(values):
    if isinstance(values, str):
        parts = [part.strip() for part in values.split(',') if part.strip()]
        if any(not part.isascii() or not part.isdecimal() for part in parts):
            raise ValueError('Rhythm divisions must be comma-separated positive integers')
        values = [int(part) for part in parts]
    if not isinstance(values,(list,tuple)) or not values:
        raise ValueError('Select between 1 and 32 rhythm divisions')
    if any(isinstance(value,bool) or not isinstance(value,int) or not 1 <= value <= 768 for value in values):
        raise ValueError('Rhythm divisions must be integers in [1, 768]')
    unique = tuple(sorted(set(values)))
    if len(unique) > 32:
        raise ValueError('Select between 1 and 32 rhythm divisions')
    return unique


def validate_snap_ms(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 50:
        raise ValueError('rhythm_snap_ms must be in [0, 50]')
    return value


def approximate_onsets(records, tempo, first, *, tolerance_ms=15., min_interval_ms=0.,
                       end_beat=None, duration_sec=None, rest_regions=(),divisions=COMMON_DIVISIONS,
                       mode='point'):
    """Move only safe, nearby points; never combine cells or replace note text.

    The grids refer to 4/division quarter-note beats, independent of meter.
    Every move is checked against the exact tempo clock, neighboring onsets,
    known tempo boundaries, silence regions, and the original frame timestamp.
    """
    if mode not in {'point','sequence'}:
        raise ValueError('rhythm_snap_mode must be point or sequence')
    if mode == 'sequence':
        return _sequence_onsets(records,tempo,first,tolerance_ms=tolerance_ms,
            min_interval_ms=min_interval_ms,end_beat=end_beat,duration_sec=duration_sec,
            rest_regions=rest_regions,divisions=divisions)
    tolerance = validate_snap_ms(tolerance_ms)/1000
    divisions = normalize_divisions(divisions)
    result = [dict(item) for item in records]
    beats = [Fraction(item['beat']) for item in result]
    if beats != sorted(set(beats)):
        raise ValueError('Rhythm approximation requires sorted, unique beats')
    tempo_boundaries = sorted(Fraction(item.beat) for item in tempo)
    stats = dict(enabled=True, divisions=list(divisions), tolerance_ms=float(tolerance_ms),
                 changed=0, already_common=0, outside_tolerance=0, blocked=0, maximum_shift_ms=0.)
    occupied = set(beats)
    for index, item in enumerate(result):
        beat = beats[index]
        stamp = float(seconds_at_beat(beat,tempo,first))
        if any((beat*division/4).denominator == 1 for division in divisions):
            stats['already_common'] += 1
            continue
        if tolerance == 0:
            stats['outside_tolerance'] += 1
            continue
        candidates = {}
        for division in divisions:
            step = Fraction(4,division)
            cell = beat // step
            for position in (cell*step, (cell+1)*step):
                if position >= 0:
                    candidates[position] = min(division,candidates.get(position,division))
        ranked = sorted(((abs(seconds_at_beat(b,tempo,first)-stamp),d,b) for b,d in candidates.items()))
        nearby = [(error,division,position) for error,division,position in ranked if error <= tolerance+1e-9]
        if not nearby:
            stats['outside_tolerance'] += 1
            continue
        changed = False
        for error, division, position in nearby:
            second = float(seconds_at_beat(position,tempo,first))
            if position in occupied or position in tempo_boundaries or beat in tempo_boundaries:
                continue
            if bisect_right(tempo_boundaries,position) != bisect_right(tempo_boundaries,beat):
                continue
            if end_beat is not None and position >= end_beat:
                continue
            if second < 0 or (duration_sec is not None and second >= duration_sec):
                continue
            if any(float(region['start_sec']) <= second < float(region['end_sec']) for region in rest_regions):
                continue
            if 'source_second' in item and abs(second-float(item['source_second'])) > tolerance+1e-9:
                continue
            previous = result[index-1] if index else None
            following = result[index+1] if index+1 < len(result) else None
            if previous is not None and position <= Fraction(previous['beat']):
                continue
            if following is not None and position >= Fraction(following['beat']):
                continue
            minimum = float(min_interval_ms)/1000
            if previous is not None and second-float(previous['second']) < minimum-1e-9:
                continue
            if following is not None and float(following['second'])-second < minimum-1e-9:
                continue
            occupied.remove(beat)
            occupied.add(position)
            item.update(pre_snap_beat=str(beat),pre_snap_second=stamp,beat=str(position),second=second,
                        rhythm_snap_shift_ms=(second-stamp)*1000,rhythm_snap_division=division)
            stats['changed'] += 1
            stats['maximum_shift_ms'] = max(stats['maximum_shift_ms'],error*1000)
            changed = True
            break
        if not changed:
            stats['blocked'] += 1
    stats['count_before'] = stats['count_after'] = len(result)
    return result, stats


def _sequence_onsets(records,tempo,first,*,tolerance_ms,min_interval_ms,
                     end_beat,duration_sec,rest_regions,divisions):
    """Reconsider locally regular runs, including already-common wrong positions."""
    from maichart.generation.event_audio import _time_to_beat,_normalize_tempo_map

    original=[dict(item) for item in records]
    result,point_stats=approximate_onsets(original,tempo,first,tolerance_ms=tolerance_ms,
        min_interval_ms=min_interval_ms,end_beat=end_beat,duration_sec=duration_sec,
        rest_regions=rest_regions,divisions=divisions,mode='point')
    divisions=normalize_divisions(divisions)
    tolerance=validate_snap_ms(tolerance_ms)/1000
    minimum=float(min_interval_ms)/1000
    beats=[Fraction(item['beat']) for item in original]
    boundaries=[Fraction(item.beat) for item in tempo]
    clock=_normalize_tempo_map(tempo,tempo[0].bpm)
    raw=[float(item.get('source_second',item['second'])) for item in original]
    if any(not math.isfinite(value) for value in raw):
        raise ValueError('Onset source times must be finite')
    raw_beats=[_time_to_beat(second-first,clock) for second in raw]
    stamps={}
    def stamp(beat):
        if beat not in stamps:stamps[beat]=float(seconds_at_beat(beat,tempo,first))
        return stamps[beat]
    def safe(index,position):
        before=beats[index]
        second=stamp(position)
        bpm=float(tempo[max(0,bisect_right(boundaries,before)-1)].bpm)
        # Less than the 16th/triplet grid separation protects exact triplets;
        # the audio-frame cap only supplies a modest jitter allowance.
        raw_cap=min(tolerance,.020,.075*60/bpm)
        return (position >= 0 and abs(second-raw[index]) <= raw_cap+1e-9
            and abs(second-float(original[index]['second'])) <= tolerance+1e-9
            and (position==before or position not in boundaries and before not in boundaries)
            and bisect_right(boundaries,position)==bisect_right(boundaries,before)
            and (end_beat is None or position < end_beat)
            and second >= 0 and (duration_sec is None or second < duration_sec)
            and not any(float(region['start_sec']) <= second < float(region['end_sec'])
                        for region in rest_regions))
    runs=[]
    gapped_run_count=ungapped_run_count=0
    gapped_run_changes=ungapped_run_changes=0
    index=0
    while tolerance > 0 and index+4 <= len(original):
        chosen=None
        for length in range(min(8,len(original)-index),3,-1):
            stop=index+length
            source=raw[index:stop]
            gaps=[b-a for a,b in zip(source,source[1:])]
            if any(gap <= 0 for gap in gaps):continue
            mean=sum(gaps)/len(gaps)
            variation=math.sqrt(sum((gap-mean)**2 for gap in gaps)/len(gaps))/mean
            if any(beats[index] < boundary < beats[stop-1] for boundary in boundaries):continue
            if any(source[0] < float(region['end_sec']) and source[-1] > float(region['start_sec'])
                   for region in rest_regions):continue
            prior_gaps=[b-a for a,b in zip(beats[index:stop],beats[index+1:stop])]
            if len(set(prior_gaps)) == 1:
                continue
            candidates=[]
            for division in divisions:
                step=Fraction(4,division)
                positions=[round(Fraction(str(raw_beats[i]))/step)*step for i in range(index,stop)]
                intervals=[b-a for a,b in zip(positions,positions[1:])]
                grid_steps=[interval/step for interval in intervals]
                uniform_grid=bool(intervals) and intervals[0] > 0 and len(set(intervals)) == 1
                if not uniform_grid:
                    if any(value.denominator != 1 or not 1 <= value <= 2 for value in grid_steps):
                        continue
                    # Nonuniform runs may include a single missed grid onset,
                    # but most intervals must retain the primary spacing.
                    if sum(value == 1 for value in grid_steps) < max(2,math.ceil(len(intervals)*.5)):
                        continue
                if any('rhythm_grid_division' in result[i] and position != Fraction(result[i]['beat'])
                       for i,position in zip(range(index,stop),positions)):continue
                if not all(safe(i,position) for i,position in zip(range(index,stop),positions)):continue
                seconds=[stamp(position) for position in positions]
                if any(b-a < minimum-1e-9 for a,b in zip(seconds,seconds[1:])):continue
                if index and (positions[0] <= Fraction(result[index-1]['beat'])
                    or seconds[0]-float(result[index-1]['second']) < minimum-1e-9):continue
                if stop < len(result) and (positions[-1] >= Fraction(result[stop]['beat'])
                    or float(result[stop]['second'])-seconds[-1] < minimum-1e-9):continue
                cost=sum((a-b)**2 for a,b in zip(seconds,source))/length
                source_rms_ms=math.sqrt(cost)*1000
                current_cost=sum((stamp(Fraction(result[i]['beat']))-raw[i])**2
                                  for i in range(index,stop))/length
                changed_positions=sum(position != Fraction(result[i]['beat'])
                                     for i,position in zip(range(index,stop),positions))
                has_missing_onset=(not uniform_grid and any(value == 2 for value in grid_steps))
                if (not has_missing_onset and variation > .35):
                    continue
                # Missing-note gaps raise raw interval variation by design.
                # Admit those runs only when the projected grid fits the frame
                # timestamps tightly and both ends agree with existing anchors.
                if (has_missing_onset and (source_rms_ms > 15 or changed_positions < 2
                        or cost >= current_cost-1e-10
                        or positions[0] != Fraction(result[index]['beat'])
                        or positions[-1] != Fraction(result[stop-1]['beat']))):
                    continue
                missed_steps=sum(value == 2 for value in grid_steps) if has_missing_onset else 0
                candidates.append((cost,division,positions,seconds,grid_steps,
                                   changed_positions,missed_steps,variation))
            if candidates:
                chosen=(stop,min(candidates,key=lambda item:(item[0],item[1])))
                break
        if chosen is None:
            index+=1;continue
        stop,(cost,division,positions,seconds,grid_steps,changed_positions,
              missed_steps,variation)=chosen
        for i,position,second in zip(range(index,stop),positions,seconds):
            item=dict(original[i])
            item['rhythm_grid_division']=division
            if position != beats[i]:
                item.update(pre_snap_beat=str(beats[i]),pre_snap_second=float(original[i]['second']),
                    beat=str(position),second=second,
                    rhythm_snap_shift_ms=(second-float(original[i]['second']))*1000,
                    rhythm_snap_division=division)
            result[i]=item
        if missed_steps:
            gapped_run_count+=1
            gapped_run_changes+=changed_positions
        else:
            ungapped_run_count+=1
            ungapped_run_changes+=changed_positions
        runs.append({'start_index':index,'end_index':stop-1,'division':division,
                     'missed_grid_steps':missed_steps,
                     'source_gap_cv':variation,'source_rms_ms':math.sqrt(cost)*1000})
        index=stop-1
    changed=[i for i,item in enumerate(result) if Fraction(item['beat']) != beats[i]]
    common=lambda beat:any((beat*division/4).denominator==1 for division in divisions)
    ambiguous_phases=_ambiguous_uniform_phases(
        original,tempo,first,divisions,enabled=tolerance > 0,rest_regions=rest_regions)
    stats={'enabled':True,'mode':'sequence','algorithm':'bounded_local_grid_inference_v2',
        'divisions':list(divisions),'tolerance_ms':float(tolerance_ms),'changed':len(changed),
        'already_common':sum(common(beat) for beat in beats),
        'common_reconsidered':sum(common(beats[i]) for i in changed),
        'maximum_shift_ms':max((abs(float(result[i]['second'])-float(original[i]['second']))*1000
                                for i in changed),default=0.),
        'uniform_runs_count':len(runs),'uniform_runs':runs[:32],
        'gapped_runs_count':gapped_run_count,'gapped_run_changed_onsets':gapped_run_changes,
        'ungapped_runs_count':ungapped_run_count,'ungapped_run_changed_onsets':ungapped_run_changes,
        'run_records_truncated':len(runs)>32,'ambiguous_phase_runs':ambiguous_phases[:32],
        'ambiguous_phase_runs_truncated':len(ambiguous_phases)>32,'point_pass':point_stats,
        'count_before':len(original),'count_after':len(result)}
    return result,stats


def _ambiguous_uniform_phases(records,tempo,first,divisions,*,enabled=True,rest_regions=()):
    """Report coherent off-grid phases without deciding whether they are errors."""
    from maichart.generation.event_audio import _time_to_beat,_normalize_tempo_map

    if not enabled or len(records) < 5:
        return []
    clock=_normalize_tempo_map(tempo,tempo[0].bpm)
    raw=[float(item.get('source_second',item['second'])) for item in records]
    raw_beats=[float(_time_to_beat(second-first,clock)) for second in raw]
    spans=[]
    index=0
    while index+5 <= len(records):
        found=None
        for stop in range(min(len(records),index+16),index+4,-1):
            gaps=[b-a for a,b in zip(raw[index:stop],raw[index+1:stop])]
            if not gaps or any(gap <= 0 for gap in gaps):
                continue
            raw_start,raw_end=raw[index],raw[stop-1]
            beat_start,beat_end=Fraction(str(raw_beats[index])),Fraction(str(raw_beats[stop-1]))
            if any(beat_start < Fraction(item.beat) < beat_end for item in tempo):
                continue
            if any(raw_start < float(region['end_sec']) and raw_end > float(region['start_sec'])
                   for region in rest_regions):
                continue
            mean=sum(gaps)/len(gaps)
            variation=math.sqrt(sum((gap-mean)**2 for gap in gaps)/len(gaps))/mean
            if variation > .08:
                continue
            candidates=[]
            for division in divisions:
                step=Fraction(4,division)
                beats=raw_beats[index:stop]
                residuals=[float(Fraction(str(beat)) % step) for beat in beats]
                # A single coherent nonzero phase is intrinsically ambiguous:
                # it can be a delayed model peak or authored syncopation.
                wrapped=[min(value,float(step)-value) for value in residuals]
                phase=residuals[0]
                distance=[min(abs(value-phase),float(step)-abs(value-phase)) for value in residuals]
                if max(wrapped) < .0125 or max(distance) > .0125:
                    continue
                signed_phase=phase if phase <= float(step)/2 else phase-float(step)
                candidates.append((variation,division,signed_phase))
            if candidates:
                found=(stop,min(candidates,key=lambda item:(item[0],item[1],item[2])))
                break
        if found is None:
            index+=1
            continue
        stop,(variation,division,phase)=found
        anchor=Fraction(str(raw_beats[index]))
        offset=seconds_at_beat(anchor+Fraction(str(phase)),tempo,first)-seconds_at_beat(anchor,tempo,first)
        spans.append({'start_index':index,'end_index':stop-1,'division':division,
                      'phase_offset_beats':phase,'phase_offset_ms':float(offset)*1000,
                      'gap_cv':variation})
        index=stop-1
    return spans
