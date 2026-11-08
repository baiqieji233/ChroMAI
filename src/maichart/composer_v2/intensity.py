"""Low-frequency chart workload as learned, optional decoding context.

Rates describe authored event onsets and extra simultaneous note heads. They
are not difficulty labels, hard event budgets or a second timing sampler.
"""
from __future__ import annotations
import json
import math
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

POLICY = 'rate_curve_v1'
SCHEMA = 'composer-v2-intensity-v1'
CHOICES = ('auto', 'off', POLICY)
LOOKAHEAD_SEC = (0., 2., 4.)
AUX_WEIGHT = .05
BIN_SECONDS = 2.


@lru_cache(maxsize=65536)
def note_heads(text):
    from maichart.notes import parse_timing_point_notes
    from maichart.timing import RawTimingPoint
    from fractions import Fraction
    point=RawTimingPoint(index=0,raw=text,note_text=text,beat=Fraction(0),tick=0,time_sec=0.,division=4,bpm=120.)
    return len(parse_timing_point_notes(point).notes)


def resolve_policy(requested, parent=None, *, resume=False):
    previous=(parent or {}).get('intensity_policy','off')
    policy=previous if requested=='auto' and resume else POLICY if requested=='auto' else requested
    if requested=='auto' and parent and parent.get('sequence_policy','window_v1')!='continuous_events_v1':policy='off'
    if policy not in ('off',POLICY):raise ValueError('Unknown intensity curve policy')
    if resume and policy!=previous:raise ValueError('Changing intensity policy requires finetune in a new directory')
    return policy, parent is not None and policy!=previous


def curve_grid(song):
    cached=song.descriptor_cache.get(POLICY+'_grid')
    if cached is not None:return cached
    duration=float(song.audio.duration_sec)
    edges=np.r_[np.arange(0,duration,BIN_SECONDS),duration]
    times=(edges[:-1]+edges[1:])/2
    source=np.asarray(song.summary['times_sec'],dtype=np.float64)
    right=np.clip(np.searchsorted(source,times,side='right'),0,len(source)-1)
    left=np.maximum(0,right-1)
    mix=np.clip((times-source[left])/np.maximum(source[right]-source[left],1e-8),0,1).astype(np.float32)
    result=dict(edges=edges,times=times,widths=np.diff(edges).astype(np.float32),
                curve_memory_left=torch.as_tensor(left),curve_memory_right=torch.as_tensor(right),
                curve_memory_mix=torch.as_tensor(mix))
    song.descriptor_cache[POLICY+'_grid']=result
    return result


def authored_targets(song):
    """Fixed-duration coverage, including silent and deliberately empty bins."""
    cached=song.descriptor_cache.get(POLICY)
    if cached is not None:return cached
    grid=curve_grid(song);edges=grid['edges'];duration=float(song.audio.duration_sec)
    seconds=song.clock.at_floats(np.asarray([float(p.beat) for p in song.points]))
    # Zero-length simultaneous representations are grouped into one event.
    events={}
    for point,second in zip(song.points,seconds):
        if 0<=second<=duration:
            events[float(second)]=events.get(float(second),0)+note_heads(point.note_text)
    stamps=np.asarray(list(events),dtype=np.float64)
    count=np.histogram(stamps,bins=edges)[0]
    extra=np.histogram(stamps,bins=edges,weights=np.asarray([max(0,n-1) for n in events.values()]))[0]
    counts=np.stack((count,extra),-1).astype(np.float32)
    result=dict(counts=counts,widths=grid['widths'],times=grid['times'])
    song.descriptor_cache[POLICY]=result
    return result


def query_indices(song, clocks):
    times=curve_grid(song)['times']
    seconds=song.clock.at_floats(np.asarray(clocks,dtype=np.float64))
    queries=seconds[:,None]+np.asarray(LOOKAHEAD_SEC)
    right=np.clip(np.searchsorted(times,queries,side='right'),0,len(times)-1)
    left=np.maximum(0,right-1)
    weight=np.clip((queries-times[left])/np.maximum(times[right]-times[left],1e-8),0.,1.)
    return dict(curve_left=torch.as_tensor(left),curve_right=torch.as_tensor(right),
                curve_mix=torch.as_tensor(weight,dtype=torch.float32))


def validate_profile(profile):
    if not isinstance(profile,dict) or profile.get('schema')!=SCHEMA:
        raise ValueError('请选择 Composer v2 新版强度曲线')
    duration=profile.get('duration_sec')
    if isinstance(duration,bool) or not isinstance(duration,(float,int)) or not math.isfinite(duration) or duration<=0:
        raise ValueError('曲线时长必须是正的有限数值')
    points=profile.get('points')
    if not isinstance(points,list) or not 1<=len(points)<=2048:raise ValueError('曲线需包含1–2048个控制点')
    previous=-1.
    for point in points:
        if not isinstance(point,dict):raise ValueError('曲线控制点格式无效')
        for key in ('time_sec','event_rate','object_rate'):
            value=point.get(key)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ValueError('曲线时间和物量须为非负有限数值')
            if key!='time_sec' and value>1e6:raise ValueError('曲线物量数值超出可表示范围')
        if not previous<point['time_sec']<=duration:raise ValueError('曲线控制点时间必须递增且位于歌曲内')
        if point['object_rate']<point['event_rate']:raise ValueError('每秒物量不能少于每秒出音次数')
        if point['event_rate']==0 and point['object_rate']>0:raise ValueError('出音频率为0时物量也必须为0')
        previous=point['time_sec']
    return profile


def profile_rates(profile,song):
    validate_profile(profile)
    identity=getattr(song.audio,'identity',{})
    if profile.get('audio_sha256')!=identity.get('audio_sha256'):
        raise ValueError('强度曲线对应的音频已变化，请重新预测或导入匹配曲线')
    if abs(profile['duration_sec']-song.audio.duration_sec)>.05:raise ValueError('强度曲线与音频时长不符')
    points=profile['points'];times=[p['time_sec'] for p in points]
    events=np.interp(curve_grid(song)['times'],times,[p['event_rate'] for p in points])
    extra=np.interp(curve_grid(song)['times'],times,[p['object_rate']-p['event_rate'] for p in points])
    return torch.as_tensor(np.stack((events,extra),-1),dtype=torch.float32)


def add_item(item,song,clocks,*,training):
    item.update(query_indices(song,clocks))
    grid=curve_grid(song)
    item.update({k:grid[k] for k in ('curve_memory_left','curve_memory_right','curve_memory_mix')})
    if training:
        targets=authored_targets(song)
        item['curve_counts']=torch.as_tensor(targets['counts'])
        item['curve_widths']=torch.as_tensor(targets['widths'])
    override=getattr(song,'intensity_override',None)
    if override is not None:item['curve_override']=override


def collate(items,out):
    for key in ('curve_left','curve_right','curve_mix','curve_counts','curve_widths','curve_override',
                'curve_memory_left','curve_memory_right','curve_memory_mix'):
        if key in items[0]:
            out[key]=nn.utils.rnn.pad_sequence([item[key] for item in items],batch_first=True)


class IntensityContext(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.predictor=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,max(16,width//4)),nn.SiLU(),
                                     nn.Linear(max(16,width//4),2))
        self.level=nn.Linear(1,width,bias=False)
        self.projection=nn.Linear(2*len(LOOKAHEAD_SEC),width,bias=False)
        nn.init.zeros_(self.projection.weight)

    def plan(self,memory,batch,*,training):
        left=torch.gather(memory,1,batch['curve_memory_left'][...,None].expand(-1,-1,memory.size(-1)))
        right=torch.gather(memory,1,batch['curve_memory_right'][...,None].expand(-1,-1,memory.size(-1)))
        mix=batch['curve_memory_mix'][...,None]
        memory=left*(1-mix)+right*mix
        # Do not teach the automatic predictor to copy chart-derived Radar or
        # Style statistics that are unknown for a new song.
        condition=self.level(batch['condition'][:,:1].float())
        rates=F.softplus(self.predictor(memory+condition[:,None]).float())+1e-6
        context=rates.detach()
        if training and 'curve_counts' in batch:
            # Coarse, slightly noisy hints are present for half the examples.
            # No exact beat/grid labels or chart-derived hints in validation.
            widths=batch['curve_widths'].float()[:,None]
            # Pool counts and exposure separately: a 40ms tail with one note
            # must not contribute an equal-weight 25 notes/s spike.
            smooth=(F.avg_pool1d(batch['curve_counts'].float().transpose(1,2),3,1,1)/
                    F.avg_pool1d(widths,3,1,1).clamp_min(1e-6)).transpose(1,2)
            scale=torch.exp(torch.randn((rates.size(0),1,2),device=rates.device)*.15)
            use=torch.rand((rates.size(0),1,1),device=rates.device)<.5
            context=torch.where(use,smooth*scale,context)
        elif 'curve_override' in batch:
            strength=float(batch.get('intensity_strength',1.))
            context=context*(1-strength)+batch['curve_override']*strength
        return rates,torch.log1p(context)

    def forward(self,context,batch,*,start):
        offset=int(batch.get('curve_query_start',0))
        values=[]
        for name in ('curve_left','curve_right'):
            indices=batch[name][:,start-offset:]
            values.append(torch.gather(context,1,indices.flatten(1)[...,None].expand(-1,-1,2))
                          .reshape(*indices.shape,2))
        mix=batch['curve_mix'][:,start-offset:, :,None]
        return self.projection((values[0]*(1-mix)+values[1]*mix).flatten(-2))


def curve_loss(rates,batch):
    """Poisson deviance per second; zero-count intervals supply actual supervision."""
    if 'curve_counts' not in batch:return rates.sum()*0
    widths=batch['curve_widths'].float()
    target=batch['curve_counts'].float()
    expected=rates*widths[...,None]
    dev=expected-target+target*(torch.log(target.clamp_min(1e-8))-torch.log(expected.clamp_min(1e-8)))
    valid=widths>0
    per_sample=(dev*valid[...,None]).sum((1,2))/widths.sum(1).clamp_min(1e-6)
    quality=batch.get('sample_weight',torch.ones_like(per_sample))
    return (per_sample*quality).sum()/quality.sum().clamp_min(1e-6)


def serialize(rates,song,**metadata):
    values=rates.detach().float().cpu().numpy()
    return dict(schema=SCHEMA,policy=POLICY,duration_sec=float(song.audio.duration_sec),
        audio_sha256=song.audio.identity.get('audio_sha256'),units='onsets_and_note_heads_per_second',
        points=[dict(time_sec=float(t),event_rate=float(v[0]),object_rate=float(v.sum()))
                for t,v in zip(curve_grid(song)['times'],values)],**metadata)


@torch.inference_mode()
def predict(args):
    from .checkpoint import read
    from .codec import ComponentCodec
    from .model import ComposerV2,ComposerV2Config
    from .generate import prepare_inputs
    from .data import ComposerV2Song,_beats_at_seconds
    from .conditions import resolve_generation_condition
    from .stream_data import StreamDataset,collate_stream
    from maichart.tempo_clock import TempoClock
    from fractions import Fraction
    if args.cpu_threads<1:raise ValueError('cpu_threads must be positive')
    if args.device not in ('auto','cuda','cpu'):raise ValueError('Invalid curve prediction device')
    payload=read(args.checkpoint);config=ComposerV2Config.from_dict(payload['model_config'])
    if config.intensity_policy!=POLICY:raise ValueError('此检查点尚未训练强度曲线，请新建训练或在新目录微调')
    torch.set_num_threads(args.cpu_threads)
    device=args.device if args.device!='auto' else 'cuda' if torch.cuda.is_available() else 'cpu'
    model=ComposerV2(config,ComponentCodec.from_dict(payload['codec']));model.load_state_dict(payload['model_state_dict'])
    model=model.to(device).eval()
    tempo,first,timing,audio=prepare_inputs(args,config,device)
    clock=TempoClock(tempo,first)
    condition,known,report=resolve_generation_condition(args.level,args.condition,config.condition_policy)
    song=ComposerV2Song('curve',Path(args.audio),Path(args.audio),5,args.level,[],clock,
        Fraction(str(float(_beats_at_seconds(clock,[audio.duration_sec])[0]))),condition,audio=audio,condition_known=known)
    song.summary=dict(audio.summary(160));song.summary['beats']=_beats_at_seconds(clock,song.summary['times_sec'])
    # Reuse precisely the normal generator's audio encoding and condition path.
    data=StreamDataset.__new__(StreamDataset)
    data.codec=model.codec;data.max_tokens=config.max_tokens;data.history_beats=config.history_beats
    data.window_beats=config.window_beats;data.future_beats=config.future_beats;data.intensity_policy=POLICY
    data.onset_readout_policy='off';data.structure_policy=config.structure_policy
    data.reference_augmentation=False;data.refresh_events=config.refresh_events
    item=data.build_stream(song,[],Fraction(0))
    batch={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in collate_stream([item]).items()}
    # Curve prediction itself does not need the phase-readout branch.
    batch['onset_strength']=0.
    rates=model.encode_audio(batch)['curve_rates'][0]
    profile=serialize(rates,song,checkpoint=str(args.checkpoint),condition=report,timing=timing,
                      checkpoint_epoch=(payload.get('training_state') or {}).get('epoch'))
    validate_profile(profile)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(profile,ensure_ascii=False,indent=2),encoding='utf-8')
    print('composer_v2_curve '+json.dumps(dict(output=str(output)),ensure_ascii=False),flush=True)
    return profile


def main(argv=None):
    from .generate import parser
    return predict(parser().parse_args(argv))


if __name__=='__main__':main()
