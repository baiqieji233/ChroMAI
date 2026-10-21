"""Generate a structured event chart with chart_composer_v1."""
from __future__ import annotations
import argparse,json,random
from fractions import Fraction
from pathlib import Path
import numpy as np
import torch

from maichart.event_timing import EventTempo,EventPoint,seconds_at_beat
from maichart.form.codec import RationalCodec
from maichart.form.data import organize_song
from maichart.form.rational_audio import extract_rational_audio,planner_audio_queries
from maichart.form.curve import resample_load_curve
from maichart.generation.conditions import GenerationCondition,GENERATION_DEFAULTS
from maichart.generation.event_compile import compile_event_points
from maichart.generation.timeline import normalize_tempo_map
from maichart.stream.audio_bounds import generation_bounds
from maichart.stream.grounded_model import phrase_features
from .checkpoint import load
from .candidates import candidate_beats,candidate_features
from .model import ChartComposer


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--bpm',type=float);p.add_argument('--first',type=float,default=0.);p.add_argument('--level',type=float,default=13.)
    p.add_argument('--temperature',type=float,default=.8);p.add_argument('--seed',type=int,default=0);p.add_argument('--device',default='auto')
    p.add_argument('--max-events',type=int,default=0);p.add_argument('--condition-file');p.add_argument('--tempo-map')
    return p


def _device(value):return torch.device('cuda' if value=='auto' and torch.cuda.is_available() else ('cpu' if value=='auto' else value))


def _interp_features(values,times,song,candidates,previous):
    from .candidates import candidate_features as base_features
    seconds=np.asarray([float(seconds_at_beat(b,song.tempo,song.first)) for b in candidates])
    acoustic=np.column_stack([np.interp(seconds,times,values[:,i],left=0.,right=0.) for i in range(values.shape[1])]).astype(np.float32)
    time_part=base_features(candidates,previous,song.tempo,song.first)
    if song.load_curve_times_sec is None or song.load_curve is None:
        curve=np.zeros((len(seconds),4),dtype=np.float32)
    else:
        curve=np.asarray([resample_load_curve(song.load_curve_times_sec,song.load_curve,
            np.asarray([second-float(song.first)]))[0] for second in seconds],dtype=np.float32)
    return torch.from_numpy(np.concatenate((acoustic,time_part,curve),axis=1))


@torch.inference_mode()
def generate(args):
    device=_device(args.device);model,payload=load(args.checkpoint,device=device);model.eval()
    atoms=payload.get('atom_catalog') or []
    if not atoms: raise ValueError('Composer checkpoint has no atom catalog')
    audio=extract_rational_audio(args.audio,cache_dir=Path(args.checkpoint).parent/'audio_cache',evidence_profile='balanced_evidence_v1')
    tempo=(normalize_tempo_map(json.loads(Path(args.tempo_map).read_text(encoding='utf-8-sig')))
           if args.tempo_map else [EventTempo(Fraction(0),args.bpm or audio.tempo_bpm)])
    song=organize_song(audio,tempo,args.first,max_seconds=999.,quality_profile='musical_evidence_v1')
    values,times,_=planner_audio_queries(song);values=np.asarray(values,dtype=np.float32);times=np.asarray(times,dtype=np.float64)
    bounds=generation_bounds(audio,trim_exact_silence=True);end_sec=bounds['effective_end_sec']
    condition=GenerationCondition(level=args.level,radar=GENERATION_DEFAULTS,style=GENERATION_DEFAULTS)
    if args.condition_file:condition=GenerationCondition.from_dict(json.loads(Path(args.condition_file).read_text(encoding='utf-8-sig')))
    points=[];history=[];previous=Fraction(0);future_state=torch.zeros(1,model.config.future_dim,device=device)
    rng=random.Random(args.seed);max_events=args.max_events or 100000
    while len(points)<max_events:
        candidates=candidate_beats(previous,tempo=song.tempo,first=song.first,audio_end_sec=end_sec,
            candidate_count=model.config.candidate_count)
        if not candidates:break
        feats=_interp_features(values,times,song,candidates,previous)[None].to(device)
        hist=torch.tensor([history[-model.config.history_events:]],dtype=torch.long,device=device)
        action=torch.zeros(1,50,device=device);control=condition.to_vector(device=device)[None]
        output=model(feats,hist,control,action_state=action,future_state=future_state,
                     candidate_mask=torch.zeros(1,len(candidates),dtype=torch.bool,device=device),
                     history_padding_mask=torch.zeros_like(hist,dtype=torch.bool))
        logits=output['candidate_logits'][0].float()/max(args.temperature,1e-6)
        choice=int(torch.multinomial(logits.softmax(-1),1,generator=torch.Generator(device=device).manual_seed(args.seed+len(points)*104729)).item()) if args.temperature>0 else int(logits.argmax())
        beat=candidates[choice]
        action_logits=output['action_logits'][0]
        selected=[]
        for row in action_logits:
            atom=int(row.float().argmax())
            if atom>0 and atom<=len(atoms) and atoms[atom-1] not in selected:selected.append(atoms[atom-1])
        if not selected: selected=[atoms[0]]
        text='/'.join(selected)
        future_state=output['future'].detach()
        points.append(EventPoint(beat,text));history.append(atoms.index(selected[0])+1 if selected[0] in atoms else 0);previous=beat
        if float(seconds_at_beat(beat,song.tempo,song.first))>=end_sec:break
    export_end=Fraction(int(np.ceil(float(max([p.beat for p in points],default=0))))+1)
    text,_=compile_event_points(points,tempo_changes=song.tempo,first=args.first,end_beat=export_end,level=str(condition.level))
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(text,encoding='utf-8')
    report={'schema':'maichart-chart-composer-generation-v1','family':'chart_composer_v1','event_count':len(points),
            'checkpoint':str(Path(args.checkpoint).resolve()),'device':str(device),'seed':args.seed,
            'audio_bounds':bounds,'structured_event_generation':True,'hard_constraints':'export_and_parser_only'}
    output.with_suffix('.composer.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def main(argv=None):
    args=parser().parse_args(argv);print(json.dumps(generate(args),ensure_ascii=False),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
