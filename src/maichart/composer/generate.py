"""Generate a structured event chart with chart_composer_v1."""
from __future__ import annotations
import argparse,json,random
from fractions import Fraction
from pathlib import Path
import numpy as np
import torch

from maichart.event_timing import EventTempo,EventPoint,seconds_at_beat
from maichart.form.data import organize_song
from maichart.form.rational_audio import extract_rational_audio
from maichart.generation.conditions import GenerationCondition,GENERATION_DEFAULTS
from maichart.generation.event_compile import compile_event_points
from maichart.generation.event_cli import _NoteGrammar
from maichart.stream.grounded_inference import IncrementalState
from maichart.stream.grounding import detail_queries
from maichart.generation.timeline import normalize_tempo_map
from maichart.stream.audio_bounds import generation_bounds
from maichart.stream.grounded_model import phrase_features
from .checkpoint import load
from .candidates import candidate_beats,candidate_features,composer_audio_nodes,interp_candidate_audio
from .model import ChartComposer
from maichart.tempo_clock import TempoClock


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--bpm',type=float);p.add_argument('--first',type=float,default=0.);p.add_argument('--level',type=float,default=13.)
    p.add_argument('--temperature',type=float,default=.8);p.add_argument('--seed',type=int,default=0);p.add_argument('--device',default='auto')
    p.add_argument('--max-events',type=int,default=0);p.add_argument('--condition-file');p.add_argument('--tempo-map')
    return p


def _device(value):return torch.device('cuda' if value=='auto' and torch.cuda.is_available() else ('cpu' if value=='auto' else value))


def _interp_features(values,times,song,candidates,previous,*,clock=None):
    clock=clock or TempoClock(song.tempo,song.first)
    seconds=np.asarray([clock.at(b) for b in candidates],dtype=np.float64)
    acoustic=interp_candidate_audio(values,times,seconds)
    time_part=candidate_features(candidates,previous,song.tempo,song.first,clock=clock,candidate_seconds=seconds)
    return torch.from_numpy(np.concatenate((acoustic,time_part),axis=1))


def _history_features(history, atoms, tempo, first, action_slots, *, clock=None,preceding_beat=Fraction(0)):
    rows=[];previous_beat=preceding_beat
    previous_second=float(clock.at(previous_beat) if clock is not None else seconds_at_beat(previous_beat,tempo,first))
    for ids, point in history:
        text='/'.join(atoms[index-1] for index in ids if 0 < index <= len(atoms))
        atom_count=sum(1 for index in ids if index > 0)
        second=float(clock.at(point) if clock is not None else seconds_at_beat(point,tempo,first))
        rows.append((min(1.,max(0.,float(point-previous_beat)/4.)),
            min(1.,max(0.,(second-previous_second)/2.)),min(1.,atom_count/4.),
            float('/' in text),float('h' in text),
            float(any(char in text for char in '-<>^pqszvwV')),
            float(any(char in text for char in 'ABCDE')),float(any(char in text for char in 'bx'))))
        previous_beat=point; previous_second=second
    return rows


def _sample_actions(logits, atoms, grammar, temperature, generator, occupied=(), max_heads=2):
    selected=[]
    used_buttons=set(occupied)
    for slot,row in enumerate(logits):
        allowed=[]
        for atom_id in range(0,len(atoms)+1):
            if slot == 0 and atom_id == 0:
                continue
            if atom_id == 0:
                allowed.append(atom_id); continue
            atom=atoms[atom_id-1]
            if atom in selected:
                continue
            atom_heads={f'A{head}' if str(head).isdigit() else str(head)
                        for head in (grammar.heads.get(atom) or ())}
            if atom_heads & used_buttons or len(used_buttons | atom_heads) > max_heads:
                continue
            allowed.append(atom_id)
        if not allowed:
            break
        values=row.float().clone()
        mask=torch.full_like(values,-1e4)
        mask[torch.tensor(allowed,device=values.device)]=values[torch.tensor(allowed,device=values.device)]
        if temperature > 0:
            choice=int(torch.multinomial((mask/max(temperature,1e-6)).softmax(-1),1,generator=generator).item())
        else:
            choice=int(mask.argmax().item())
        if choice == 0:
            break
        selected_atom=atoms[choice-1]
        selected.append(selected_atom)
        used_buttons.update(f'A{head}' if str(head).isdigit() else str(head)
                            for head in (grammar.heads.get(selected_atom) or ()))
    return selected


def _active_heads(state, beat, tempo, first, *, clock=None):
    """Return concrete source heads still occupied at the proposed beat.

    SustainStateIndex also exposes a hand-space pad coverage feature. That is
    useful as a soft model input but must not be treated as exact key
    ownership for a hard same-head filter.
    """
    second=float(clock.at(beat) if clock is not None else seconds_at_beat(beat,tempo,first)); result=set()
    for marker in getattr(state,'markers',()):
        onset=float(marker.get('second',0.)); end=float(marker.get('end_second',-1.))
        if onset > second or end <= second + 1e-8:
            continue
        kind=marker.get('type')
        if kind == 'hold':
            lane=marker.get('lane')
            if lane is not None: result.add(f'A{int(lane)}')
        elif kind=='touch_hold' and marker.get('touch_pad'):
            result.add(str(marker['touch_pad']))
    return result


@torch.inference_mode()
def generate(args):
    device=_device(args.device);model,payload=load(args.checkpoint,device=device);model.eval()
    atoms=payload.get('atom_catalog') or []
    if not atoms: raise ValueError('Composer checkpoint has no atom catalog')
    audio=extract_rational_audio(args.audio,cache_dir=Path(args.checkpoint).parent/'audio_cache',evidence_profile='balanced_evidence_v1')
    tempo=(normalize_tempo_map(json.loads(Path(args.tempo_map).read_text(encoding='utf-8-sig')))
           if args.tempo_map else [EventTempo(Fraction(0),args.bpm or audio.tempo_bpm)])
    song=organize_song(audio,tempo,args.first,max_seconds=999.,quality_profile='musical_evidence_v1')
    clock=TempoClock(song.tempo,song.first)
    values,times=composer_audio_nodes(song)
    bounds=generation_bounds(audio,trim_exact_silence=True);end_sec=bounds['effective_end_sec']
    condition=GenerationCondition(level=args.level,radar=GENERATION_DEFAULTS,style=GENERATION_DEFAULTS)
    if args.condition_file:condition=GenerationCondition.from_dict(json.loads(Path(args.condition_file).read_text(encoding='utf-8-sig')))
    points=[];history=[];previous=None;max_events=args.max_events or 100000
    action_index=IncrementalState(song.tempo,song.first)
    grammar=_NoteGrammar.from_atoms(atoms)
    rng=torch.Generator(device=device).manual_seed(args.seed)
    while len(points)<max_events:
        candidates=candidate_beats(previous,tempo=song.tempo,first=song.first,audio_end_sec=end_sec,
            candidate_count=model.config.candidate_count,clock=clock)
        if not candidates:break
        occupied_by_candidate=([_active_heads(action_index,cand,song.tempo,song.first,clock=clock) for cand in candidates]
                               if getattr(action_index,'markers',None) else [set() for _ in candidates])
        playable=[len(occ)<2 for occ in occupied_by_candidate]
        if not any(playable):
            previous=candidates[-1]
            continue
        cand_mask=torch.tensor([[not ok for ok in playable]],dtype=torch.bool,device=device)
        feats=_interp_features(values,times,song,candidates,previous,clock=clock)[None].to(device)
        history_tail=history[-model.config.history_events:]
        if history_tail:
            hist_atoms=torch.tensor([row for row,_ in history_tail],dtype=torch.long,device=device)[None]
            preceding_beat=history[-model.config.history_events-1][1] if len(history)>model.config.history_events else Fraction(0)
            hist_features=torch.tensor(_history_features(history_tail,atoms,song.tempo,song.first,model.config.action_slots,
                clock=clock,preceding_beat=preceding_beat),dtype=torch.float32,device=device)[None]
        else:
            hist_atoms=torch.zeros((1,0,model.config.action_slots),dtype=torch.long,device=device)
            hist_features=torch.zeros((1,0,8),dtype=torch.float32,device=device)
        try:
            states,mask,metadata=action_index.at(Fraction(0) if previous is None else previous,include_current=True)
            active=states[~mask]
            action=active.mean(0).to(device) if active.numel() else torch.zeros(50,device=device)
        except Exception:
            action=torch.zeros(50,device=device)
        control=condition.to_vector(device=device)[None]
        output=model(feats,hist_atoms,control,action_state=action[None],history_features=hist_features,
                     candidate_mask=cand_mask,
                     history_padding_mask=torch.zeros((1,hist_atoms.size(1)),dtype=torch.bool,device=device),
                     decode_actions=False)
        logits=output['candidate_logits'][0].float()/max(args.temperature,1e-6)
        if not all(playable):
            logits=logits.masked_fill(cand_mask[0],float('-inf'))
        choice=int(torch.multinomial(logits.softmax(-1),1,generator=rng).item()) if args.temperature>0 else int(logits.argmax())
        beat=candidates[choice]
        decision=model.decode_candidate(output['candidate_context'],
            torch.tensor([choice],dtype=torch.long,device=device))
        occupied=occupied_by_candidate[choice]
        selected=_sample_actions(decision['action_logits'][0],atoms,grammar,args.temperature,rng,occupied=occupied,max_heads=2)
        if not selected:
            for atom in atoms:
                heads={f'A{head}' if str(head).isdigit() else str(head)
                       for head in (grammar.heads.get(atom) or ())}
                if not (heads & occupied) and len(occupied | heads) <= 2:
                    selected=[atom]; break
        if not selected:
            break
        text='/'.join(selected)
        event=EventPoint(beat,text)
        points.append(event); action_index.append(event)
        atom_ids=tuple([atoms.index(atom)+1 for atom in selected]+[0]*(model.config.action_slots-len(selected)))
        history.append((atom_ids,beat));previous=beat
        eos_probability=float(decision['eos_logits'][0].sigmoid())
        beat_sec=float(clock.at(beat))
        if beat_sec>=end_sec or (eos_probability>=.9 and beat_sec>=max(0.,end_sec-2.)):
            break
    export_end=Fraction(int(np.ceil(float(max([p.beat for p in points],default=0))))+1)
    text,_=compile_event_points(points,tempo_changes=song.tempo,first=args.first,end_beat=export_end,level=str(condition.level))
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(text,encoding='utf-8')
    report={'schema':'maichart-chart-composer-generation-v2','family':'chart_composer_v1','event_count':len(points),
            'checkpoint':str(Path(args.checkpoint).resolve()),'device':str(device),'seed':args.seed,
            'audio_bounds':bounds,'structured_event_generation':True,'hard_constraints':'export_and_parser_only'}
    output.with_suffix('.composer.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def main(argv=None):
    args=parser().parse_args(argv);print(json.dumps(generate(args),ensure_ascii=False),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
