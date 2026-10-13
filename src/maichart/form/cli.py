"""Analyze, generate and revise with the independent song-form architecture."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path

from maichart.event_timing import EventTempo, parse_event_chart, seconds_at_beat
from maichart.form.data import organize_song
from maichart.generation.conditions import GenerationCondition, GENERATION_DEFAULTS
from maichart.generation.event_audio import extract_event_audio
from maichart.generation.event_compile import compile_event_points
from maichart.generation.timeline import normalize_tempo_map


_EXPORT_TAIL_TICKS_PER_BEAT = 1920


def _export_end_timing(audio_end_beat, tempo, first, *, existing_end=None):
    """Normalize only the audio EOF anchor; preserve a later source-chart end."""
    audio_end = Fraction(audio_end_beat)
    ticks = (audio_end * _EXPORT_TAIL_TICKS_PER_BEAT).__ceil__()
    normalized_audio_end = Fraction(ticks, _EXPORT_TAIL_TICKS_PER_BEAT)
    export_end = max(normalized_audio_end, Fraction(existing_end)) if existing_end is not None else normalized_audio_end
    extension_sec = float(seconds_at_beat(export_end, tempo, first) - seconds_at_beat(audio_end, tempo, first))
    return export_end, {
        'audio_end_beat': str(audio_end),
        'export_end_beat': str(export_end),
        'end_extension_sec': extension_sec,
        'end_only_normalized': export_end == normalized_audio_end and normalized_audio_end != audio_end,
        'notes_unchanged': True,
    }


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('analyze','generate','revise'))
    p.add_argument('--audio',required=True);p.add_argument('--output',required=True)
    p.add_argument('--checkpoint');p.add_argument('--maidata');p.add_argument('--difficulty',type=int,default=5)
    p.add_argument('--start');p.add_argument('--end')
    p.add_argument('--bpm',type=float);p.add_argument('--first',type=float,default=0.)
    p.add_argument('--meter-map');p.add_argument('--tempo-map');p.add_argument('--structure-file')
    p.add_argument('--cache-dir',default='cache/song_form');p.add_argument('--max-region-seconds',type=float,default=14.)
    p.add_argument('--device',default='cpu');p.add_argument('--level',type=float,default=13.3)
    p.add_argument('--memory-strength',type=float,default=1.);p.add_argument('--onset-threshold',type=float,default=.5)
    p.add_argument('--onset-merge-ms',type=float,default=40.,
                   help='keep the strongest onset within this many milliseconds; 0 disables merging')
    p.add_argument('--rhythm-snap',action='store_true',help='approximate nearby onsets to common musical grids')
    p.add_argument('--rhythm-snap-ms',type=float,default=15.,help='maximum approximation shift in milliseconds')
    p.add_argument('--rhythm-snap-divisions',default='4,8,12,16,24,32',help='comma-separated allowed musical divisions')
    p.add_argument('--rhythm-snap-mode',choices=('sequence','point'),default='sequence',
                   help='infer locally consistent grids or use independent nearest-point approximation')
    p.add_argument('--temperature',type=float,default=.7);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--planner-type-strength',type=float,default=1.,help='bounded soft planner type guidance; 0 disables it')
    p.add_argument('--style-balance-strength',type=float,default=1.,help='bounded note-head share feedback; 0 disables it')
    p.add_argument('--condition-file',help='JSON GenerationCondition for Radar/Style controls')
    p.add_argument('--intensity-curve-file',help='Saved song-form intensity curve JSON')
    p.add_argument('--intensity-curve-strength',type=float,default=1.0)
    p.add_argument('--intensity-curve-span',type=float,default=.5)
    p.add_argument('--intensity-curve-mode',choices=('follow','hard'),default='follow')
    p.add_argument('--repair',action='store_true');p.add_argument('--repair-checks',type=int,default=24)
    p.add_argument('--repair-edits',type=int,default=4);p.add_argument('--repair-seconds',type=float,default=10.)
    return p


def _json(path): return json.loads(Path(path).read_text(encoding='utf-8-sig')) if path else None


def analysis_report(song):
    from maichart.form.clock import clock_report
    return {'schema':'maichart-song-form-analysis-v1','audio_duration':song.audio.duration_sec,
            'first':song.first,'tempo_map':[{'beat':str(t.beat),'bpm':t.bpm} for t in song.tempo],
            'timing_alignment':clock_report(song.first),
            'meter_map':[m.to_dict() for m in song.meters],
            'spans':[s.to_dict() for s in song.spans], 'segmentation':song.audit,
            'recurrence':{'links':[asdict(link) for link in song.relations.links],**song.relations.audit},
            'limitations':['Audio recurrence is provisional evidence, not confirmed section naming.',
                          'Compute resource boundaries are not new musical phrases.']}


def main(argv=None):
    args=parser().parse_args(argv)
    from maichart.form.type_guidance import validate_strength
    validate_strength(args.planner_type_strength,'planner_type_strength')
    validate_strength(args.style_balance_strength,'style_balance_strength')
    from maichart.form.onset_selection import merge_close_onsets
    merge_close_onsets([], args.onset_merge_ms)
    from maichart.form.rhythm_snap import validate_snap_ms, normalize_divisions
    validate_snap_ms(args.rhythm_snap_ms)
    rhythm_snap_divisions = normalize_divisions(args.rhythm_snap_divisions)
    if args.rhythm_snap and args.operation != 'generate':
        raise ValueError('Rhythm approximation applies only to automatic generation, not source-chart revision')
    if args.intensity_curve_file and args.operation != 'generate':
        raise ValueError('Song-form intensity curves apply only to generate; revise uses the source chart clock')
    audio=extract_event_audio(args.audio,cache_dir=args.cache_dir)
    structure=_json(args.structure_file) or {}
    meters=_json(args.meter_map) or structure.get('meter_map')
    explicit_tempo=_json(args.tempo_map) or structure.get('tempo_map')
    if explicit_tempo:
        tempo=normalize_tempo_map(explicit_tempo)
    else:
        from maichart.generation.event_cli import _musical_bpm
        resolved_bpm = args.bpm if args.bpm is not None else _musical_bpm(audio.tempo_bpm)
        tempo=[EventTempo(Fraction(0),resolved_bpm)]
    first=args.first
    chart=None
    if args.operation=='revise':
        if not args.maidata or args.start is None or args.end is None: raise ValueError('Revision needs maidata and exact --start/--end beats')
        chart=parse_event_chart(args.maidata,args.difficulty,strict_notes=True)
        tempo,first=chart.tempo_changes,chart.audio_first
    song=organize_song(audio,tempo,first,meters=meters,max_seconds=args.max_region_seconds)
    if not song.spans: raise ValueError('No audio interval available')
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    report=analysis_report(song)
    if args.operation=='analyze':
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'form_analysis regions={len(song.audit.get("musical_regions",[]))} compute_spans={len(song.spans)} links={len(song.relations.links)} report={output}',flush=True)
        return 0
    if not args.checkpoint: raise ValueError('Generate/revise requires a trained song_form_v1 checkpoint')
    from maichart.form.checkpoint import load
    from maichart.form.generate import generate_chart,revise_chart
    model,planner,codec,payload=load(args.checkpoint,device=args.device,inference=True,
                                   task='revision' if args.operation=='revise' else 'draft',record_identity=True)
    report['source_checkpoint'] = {**payload['source_checkpoint_identity'],
                                  'family':payload['family'],'completed_epoch':payload.get('epoch'),
                                  'training_position':payload.get('training_position'),
                                  'optimizer_steps':payload.get('optimizer_steps'),
                                  'blueprint_steps':payload.get('blueprint_steps'),
                                  'model_config':payload['model_config']}
    condition=GenerationCondition.from_dict(_json(args.condition_file)) if args.condition_file else GenerationCondition(level=args.level,radar=GENERATION_DEFAULTS,style=GENERATION_DEFAULTS)
    report['generation_condition'] = condition.to_dict()
    from maichart.generation.conditions import CONDITION_KEYS
    report['effective_generation_settings'] = {
        'operation':args.operation,'level':condition.level,'temperature':args.temperature,'seed':args.seed,
        'device':args.device,'onset_threshold':args.onset_threshold,'onset_merge_ms':args.onset_merge_ms,
        'memory_strength':args.memory_strength,'max_region_seconds':args.max_region_seconds,
        'planner_type_strength':args.planner_type_strength if args.operation=='generate' else 0.,
        'style_balance_strength':args.style_balance_strength if args.operation=='generate' else 0.,
        'rhythm_snap_enabled':args.rhythm_snap,'rhythm_snap_ms':args.rhythm_snap_ms,
        'rhythm_snap_divisions':list(rhythm_snap_divisions),
        'rhythm_snap_mode':args.rhythm_snap_mode if args.rhythm_snap else 'disabled',
        'intensity_curve_enabled':bool(args.intensity_curve_file),
        'intensity_curve_strength':args.intensity_curve_strength,'intensity_curve_span':args.intensity_curve_span,
        'intensity_curve_mode':args.intensity_curve_mode,'repair':args.repair,
        'repair_checks':args.repair_checks,'repair_edits':args.repair_edits,'repair_seconds':args.repair_seconds,
        'model_max_tokens':model.config.max_tokens,'model_max_memory_tokens':model.config.max_memory_tokens,
        'condition_vector_keys':list(CONDITION_KEYS),'condition_vector':condition.to_vector().tolist(),
        'condition_vector_dtype':'float32','condition_vector_level_divisor':15.0,
        'condition_kind':'learned_soft_condition_not_a_hard_quota',
        'first':first,'tempo_map':report['tempo_map']}
    intensity_curve = None
    curve_condition_warnings = []
    if args.intensity_curve_file:
        from maichart.generation.music_intensity import load_intensity_profile
        intensity_curve = load_intensity_profile(args.intensity_curve_file)
        if intensity_curve.get('method') != 'learned_song_form_v1':
            raise ValueError('song_form CLI requires a learned_song_form_v1 intensity curve')
        from maichart.form.curve_validation import validate_saved_curve
        curve_condition_warnings = validate_saved_curve(
            intensity_curve, audio_path=args.audio, checkpoint_path=args.checkpoint,
            checkpoint_identity=payload['source_checkpoint_identity'], first=first,
            tempo_map=explicit_tempo, bpm_input=args.bpm, condition=condition.to_dict(),
            resolved_tempo_map=tempo)
        if curve_condition_warnings:
            report['curve_condition_warnings'] = curve_condition_warnings
        if not 0.0 <= args.intensity_curve_strength <= 1.0:
            raise ValueError('intensity curve strength must be between 0 and 1')
        if not 0.0 <= args.intensity_curve_span <= 1.0:
            raise ValueError('intensity curve span must be between 0 and 1')
    if args.operation=='generate':
        points,generated=generate_chart(model,planner,codec,song,condition,payload['atom_catalog'],
            threshold=args.onset_threshold,memory_strength=args.memory_strength,onset_merge_ms=args.onset_merge_ms,
            rhythm_snap_enabled=args.rhythm_snap,rhythm_snap_ms=args.rhythm_snap_ms,
            rhythm_snap_divisions=rhythm_snap_divisions,
            rhythm_snap_mode=args.rhythm_snap_mode,
            planner_type_strength=args.planner_type_strength,style_balance_strength=args.style_balance_strength,
            temperature=args.temperature,seed=args.seed,device=args.device,
            intensity_curve=intensity_curve,
            intensity_curve_strength=args.intensity_curve_strength,
            intensity_curve_span=args.intensity_curve_span,
            intensity_curve_mode=args.intensity_curve_mode)
        report.update(generated)
    else:
        points,revision=revise_chart(model,codec,song,chart.points,condition,payload['atom_catalog'],
            start=args.start,end=args.end,temperature=args.temperature,seed=args.seed,device=args.device,
            memory_strength=args.memory_strength)
        report['revision']=revision
    if args.repair:
        from maichart.form.repair import repair_chart,RepairBudget
        from maichart.generation.post_process import PostProcessConfig
        result=repair_chart(points,tempo,first,
            config=PostProcessConfig(enable_conflict_detection=True,conflict_engine='miacode',conflict_repair_mode='local',note_repair_mode='edit'),
            budget=RepairBudget(args.repair_checks,args.repair_edits,args.repair_seconds))
        points=list(result.points);report['repair']=result.to_dict()
        print('form_repair '+json.dumps({k:report['repair'][k] for k in ('checks','cache_hits','elapsed_secs','remaining_actionable_hard','status')}),flush=True)
    from maichart.generation.event_quality import assess_event_chart
    if 'quality' in report: report['draft_quality']=report.pop('quality')
    report['quality']=assess_event_chart(points,tempo,first=first,duration_sec=audio.duration_sec,target_condition=condition)
    export_end, export_timing = _export_end_timing(
        song.spans[-1].end, tempo, first, existing_end=chart.end_beat if chart else None)
    report['export_timing'] = export_timing
    text,_=compile_event_points(points,tempo_changes=tempo,first=first,end_beat=export_end,
                                title=chart.title if chart else Path(args.audio).stem,
                                level=chart.level if chart else str(condition.level),difficulty=args.difficulty)
    output.write_text(text,encoding='utf-8')
    report_path=output.with_suffix('.form.json')
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=lambda value:list(value) if isinstance(value,(set,frozenset)) else str(value)),encoding='utf-8')
    print(f'form_complete notes={len(points)} output={output} report={report_path}',flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
