"""Continuous chart_stream_v1 generation with no musical segment EOS."""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path

import torch
import numpy as np

from maichart.event_timing import EventPoint, EventTempo
from maichart.form.codec import RationalCodec
from maichart.form.data import organize_song, _form_descriptor
from maichart.form.rational_audio import extract_rational_audio, planner_audio_queries, beat_at_second
from maichart.form.rational_generate import PositionGrammar
from maichart.generation.conditions import GenerationCondition, GENERATION_DEFAULTS
from maichart.generation.event_cli import _NoteGrammar
from maichart.generation.event_compile import compile_event_points
from maichart.generation.sustain_state import SustainStateIndex
from .memory import StreamMemoryBank
from maichart.generation.timeline import normalize_tempo_map
from .checkpoint import load


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--bpm',type=float,default=120.);p.add_argument('--first',type=float,default=0.)
    p.add_argument('--level',type=float,default=13.);p.add_argument('--temperature',type=float,default=.7)
    p.add_argument('--position-temperature',type=float,default=.15);p.add_argument('--max-events',type=int,default=0)
    p.add_argument('--max-denominator',type=int,default=384);p.add_argument('--max-interval-denominator',type=int,default=96)
    p.add_argument('--min-spacing-ms',type=float,default=20.);p.add_argument('--seed',type=int,default=0);p.add_argument('--device',default='cpu')
    p.add_argument('--condition-file'); return p


def _allowed_note(codec,grammar,text):
    chars=[(i,t[2:]) for i,t in enumerate(codec.tokens) if t.startswith('N:')]
    values=[i for i,c in chars if grammar.can_continue(text+c)]
    if grammar.is_complete(text): values.append(codec.end_id)
    return values


def generate(args):
    model,payload=load(args.checkpoint,args.device); model.eval()
    codec=RationalCodec(); audio=extract_rational_audio(args.audio,cache_dir=Path(args.checkpoint).parent/'audio_cache')
    tempo=[EventTempo(Fraction(0),args.bpm)]
    song=organize_song(audio,tempo,args.first,max_seconds=999.,quality_profile='musical_evidence_v1')
    values,times,_=planner_audio_queries(song)
    condition=GenerationCondition(level=args.level,radar=GENERATION_DEFAULTS,style=GENERATION_DEFAULTS)
    condition_file=Path(args.condition_file) if args.condition_file else None
    if condition_file and condition_file.is_file(): condition=GenerationCondition.from_dict(json.loads(condition_file.read_text(encoding='utf-8')))
    batch={'audio':values[None], 'condition':condition.to_vector()[None],
           'audio_padding_mask':torch.zeros((1,len(values)),dtype=torch.bool),
           'condition_null':torch.zeros(1,dtype=torch.bool),
           'action_state':torch.zeros((1,1,50)),
           'action_state_padding_mask':torch.ones((1,1),dtype=torch.bool)}
    batch={k:v.to(args.device) for k,v in batch.items()}
    memory = memory_mask = None
    ids=[codec.bos_id]; history_tokens=None; points=[]; previous=Fraction(-1); state='event'; text=''; position=None
    memory_bank = StreamMemoryBank(max_entries=256, max_retrieval=model.config.max_stream_memory)
    def refresh_memory(summary=None):
        with torch.inference_mode():
            if previous < 0:
                query_index = 0
            else:
                query_second = float(__import__('maichart.event_timing', fromlist=['seconds_at_beat']).seconds_at_beat(
                    previous, song.tempo, song.first))
                query_index = int(np.searchsorted(times, query_second, side='left'))
                query_index = max(0, min(query_index, len(values) - 1))
            query_vector=model.audio_projection(batch['audio'][:, query_index])[0]
        archive=memory_bank.tensor(query_vector, current_beat=previous)
        if archive is not None:
            archive=archive.to(args.device)[None]
            summary=archive if summary is None else torch.cat((summary,archive),1)
            summary=summary[:,-model.config.max_stream_memory:]
        with torch.inference_mode():
            return model.encode(batch['audio'],batch['condition'],
                audio_padding_mask=batch['audio_padding_mask'],condition_null=batch['condition_null'],
                action_state=batch['action_state'],action_state_padding_mask=batch['action_state_padding_mask'],
                stream_memory=summary)
    memory,memory_mask=refresh_memory()
    cache=None; consumed=0
    note_grammar=_NoteGrammar.from_atoms(payload.get('atom_catalog') or ['1','2','3','4','5','6','7','8'])
    rng=torch.Generator(device=args.device).manual_seed(args.seed)
    audio_end=beat_at_second(audio.duration_sec,song.tempo,song.first)
    while len(points) < (args.max_events or 100000):
        decoder=torch.tensor([ids],dtype=torch.long,device=args.device)
        history = (torch.tensor([history_tokens],dtype=torch.long,device=args.device)
                   if history_tokens else None)
        if history is not None and cache is None:
            # History is a static memory branch. Prefix it once before the
            # cached decoder starts; subsequent tokens only update self-K/V.
            history_values=model.token_embedding(history)
            memory_for_decode=torch.cat((history_values,memory),dim=1)
            history_mask=(history==0)
            memory_mask_for_decode=torch.cat((history_mask,memory_mask),dim=1)
        else:
            memory_for_decode, memory_mask_for_decode=memory, memory_mask
        token=torch.tensor([[ids[-1]]],dtype=torch.long,device=args.device)
        with torch.inference_mode():
            output,cache=model.decode_step(token, len(ids)-1, memory_for_decode,
                                           memory_mask_for_decode, cache)
        logits=output['logits'][0,-1].float()
        if state=='event':
            allowed=[codec.at_id]
            if points:
                allowed.append(codec.eos_id)
            # EOS is a whole-song decision, never a compute-chunk decision.
            # A soft remaining-time penalty discourages premature termination
            # while preserving a learned early musical ending when evidence is
            # strong; long rests remain representable by a later Fraction.
            scores=logits[allowed]
            if codec.eos_id in allowed and audio_end > previous:
                remaining=float(audio_end-previous)
                scores=scores.clone()
                scores[allowed.index(codec.eos_id)] -= min(8., remaining/8.)
            token=allowed[int(torch.multinomial(torch.softmax(scores/args.temperature,dim=-1),1,generator=rng))]
            ids.append(token)
            if token==codec.eos_id: break
            state='position'; text=''; continue
        if state=='position':
            lower=max(previous, Fraction(0))
            grammar=PositionGrammar(lower,audio_end,characters=40,
                max_denominator=args.max_denominator,
                min_spacing=Fraction(str(max(0,args.min_spacing_ms)/1000*args.bpm/60)),
                max_delta_denominator=args.max_interval_denominator,delta_reference=previous,
                allowed_ranges=((lower+Fraction(1,10**12),audio_end),))
            allowed=grammar.allowed(text,codec)
            if not allowed: raise ValueError(f'No exact continuation after {previous}')
            scores=logits[allowed]
            if args.position_temperature <= 0:
                token = allowed[int(scores.argmax())]
            else:
                token=allowed[int(torch.multinomial(torch.softmax(scores/args.position_temperature,dim=-1),1,generator=rng))]
            ids.append(token)
            if token==codec.note_id:
                position=Fraction(text); points.append(EventPoint(position,'')); previous=position; state='note'; text=''
            else: text += codec.tokens[token][2:]
            continue
        allowed=_allowed_note(codec,note_grammar,text)
        if not allowed: raise ValueError(f'No valid note continuation for {text!r}')
        scores=logits[allowed]
        token=allowed[int(torch.multinomial(torch.softmax(scores/args.temperature,dim=-1),1,generator=rng))]
        ids.append(token)
        if token==codec.end_id:
            points[-1]=EventPoint(points[-1].beat,text); state='event'; text=''
            with torch.inference_mode():
                event_tokens=torch.tensor([codec.encode([points[-1]])],dtype=torch.long,device=args.device)
                event_vector=model.token_embedding(event_tokens).mean(1)[0]
                acoustic_vector=model.audio_projection(batch['audio'].mean(1))[0]
            memory_bank.append(points[-1].beat, points[-1].note_text, event_vector+acoustic_vector)
            if len(ids) >= model.config.max_tokens:
                # A compute boundary carries a summary and recent authored
                # history; it never emits EOS or moves the musical clock.
                with torch.inference_mode():
                    summary=model.summarize_tokens(torch.tensor([ids],dtype=torch.long,device=args.device),
                        slots=model.config.max_stream_memory)
                    try:
                        state_index=SustainStateIndex(points, song.tempo, song.first)
                        action, action_mask, _=state_index.at(points[-1].beat)
                        batch['action_state']=action[None].to(args.device)
                        batch['action_state_padding_mask']=action_mask[None].to(args.device)
                    except (ValueError, KeyError):
                        pass
                    memory,memory_mask=refresh_memory(summary); cache=None
                history_tokens=ids[-model.config.max_history_tokens:]
                ids=[codec.bos_id]
        else: text += codec.tokens[token][2:]
    points=[p for p in points if p.note_text]
    text_out,_=compile_event_points(points,tempo_changes=song.tempo,bpm=args.bpm,first=args.first,end_beat=audio_end)
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(text_out,encoding='utf-8')
    report={'schema':'maichart-chart-stream-generation-v1','family':'chart_stream_v1','event_count':len(points),
            'audio_duration':audio.duration_sec,'audio_end_beat':str(audio_end),'stream_continuity':True,
            'checkpoint':str(Path(args.checkpoint).resolve()),
            'limitations':['Model quality requires fresh chart_stream training.']}
    output.with_suffix('.stream.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def main(argv=None):
    args=parser().parse_args(argv); print(json.dumps(generate(args),ensure_ascii=False),flush=True); return 0


if __name__=='__main__': raise SystemExit(main())
