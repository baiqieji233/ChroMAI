"""Post-event hand readout. The decoder sees only its own past predictions."""
import torch
from torch import nn
from torch.nn import functional as F
from maichart.generation.muri.model import CENTERS, DISTANCES
from maichart.legacy_hands import HAND_FIELDS as FIELDS, MISSING, NO_RIVAL

from . import hand_timeline as tl
from .spatial_history import DiskSpatialHistory

READOUT = 'contact_memory_readout_v2'
FEEDBACK = 'contact_memory_v2'
POLICIES = (READOUT, FEEDBACK)
FEATURES = 19
FAMILIES = ('rim_tap','with_hold','with_touch','with_hold_and_touch')


def evidence(logits):
    direction, lean, stronger = logits.float().sigmoid().unbind(-1)
    return direction, lean, lean*stronger


def domain(rows):
    return torch.stack((rows[...,tl.FIRST]>0,rows[...,tl.SECOND]>0,rows[...,tl.PREVIOUS]>0,
                        rows[...,tl.SECOND]>0,rows[...,tl.RETURN]>0),dim=-1)


class HandMemory(nn.Module):
    def __init__(self,width,classes):
        super().__init__()
        self.readout = nn.Linear(width,len(FIELDS)*3)
        self.execute = nn.Linear(FEATURES,classes,bias=False)
        nn.init.zeros_(self.execute.weight)
        with torch.no_grad():
            bias=self.readout.bias.view(-1,3)
            bias[:,0].zero_()
            bias[:,1:].fill_(-1.)
        # Actual pad position stays distinct from the model's hand estimate.
        xy = [(0.,0.)]
        for pad in tl.PADS:
            button=pad.isdigit()
            x,y=CENTERS['A'+pad if button else pad]
            radius=DISTANCES['A'] if button else 240.
            xy.append((x/radius,-y/radius))
        self.register_buffer('pad_xy',torch.tensor(xy),persistent=False)

    def forward(self,hidden,rows,start,state,mechanics,feedback):
        if mechanics is not None:
            hidden=torch.where(mechanics[:,None,None],hidden.detach(),hidden)
        logits=self.readout(hidden).unflatten(-1,(len(FIELDS),3))
        if state is not None:
            DiskSpatialHistory._remember(state,'logits',logits.detach(),start)
        if not feedback or rows is None:
            return logits,None
        rows=rows.long()
        p,lean,_=evidence(logits.detach())
        signed=2*p-1
        accepted=signed * lean * (lean>=.65) * (signed.abs()>=.3) * domain(rows)
        # Each pointer refers to one concrete, already closed event. Re-reading
        # it never increments support; a branch rewind overwrites that suffix.
        history=torch.cat((accepted,rows[...,:2].to(accepted.dtype),
                           rows[...,tl.PREVIOUS,None].to(accepted.dtype)),dim=-1)
        if state is not None:
            DiskSpatialHistory._remember(state,'evidence',history,start)
            history=state['evidence'][:,:start+rows.size(1)]
        def fetch(index):
            data=history.gather(1,(index-1).clamp_min(0)[...,None].expand(-1,-1,history.size(-1)))
            return data*(index>0)[...,None]
        recent=fetch(rows[...,tl.LATEST])
        held=fetch(rows[...,tl.HOLD]).gather(-1,rows[...,tl.HOLD_SLOT,None])
        holds=rows[...,tl.N_HOLD]
        facts=torch.stack((holds,rows[...,tl.WAIT],rows[...,tl.TRACE]),-1).clamp(max=2).float()/2
        key_xy=self.pad_xy[recent[...,5:7].long()]
        # Up to four heads can refer to one gesture. Two mirror solutions can
        # agree on a chain even when its first head has no usable side.
        link=recent[...,2].clamp_min(0.)
        predecessor=fetch(recent[...,7].long())
        two_links=torch.minimum(link,predecessor[...,2].clamp_min(0.))
        anchor=fetch(predecessor[...,7].long())
        three_links=torch.minimum(two_links,anchor[...,2].clamp_min(0.))
        # Bound an inherited side by its weakest link, never add evidence.
        inherited=predecessor[...,0].sign()*torch.minimum(predecessor[...,0].abs(),link)
        features=torch.cat((recent[...,:5],(recent[...,:2,None]*key_xy).flatten(-2),
                            held,self.pad_xy[rows[...,tl.HOLD_PAD]],facts,(holds==1)[...,None],
                            torch.stack((two_links,three_links,inherited),-1)),dim=-1)
        if self.training:
            features=features*(torch.rand((len(rows),1,1),device=rows.device)>=.15)
        adjustment=.25*torch.tanh(self.execute(features.to(hidden.dtype)))
        return logits,adjustment


def add_targets(item,song,configuration,begin,end,authored):
    import numpy as np
    labels=getattr(song,'hand_memory_labels',None)
    if not authored or labels is None:
        return
    targets=labels[begin:end].copy()
    close=item['hand_completed'].numpy()
    base=item['token_weights'][item['hand_positions']].numpy()*float(item['sample_weight'])
    weights=np.broadcast_to(base[:,None],targets.shape).copy()
    facts=item['hand_timeline'].numpy()
    weights*=facts[close.clip(min=0),tl.FAMILY:]!=0
    for j,pos in enumerate(close):
        if pos<0 or not base[j]:
            weights[j]=0.
            continue
        for field,back,col in ((2,1,tl.PREVIOUS),(4,2,tl.RETURN)):
            if targets[j,field]==MISSING:
                continue
            # At an abbreviated context boundary only train edges with both
            # authored endpoints present in this exact target slice.
            if j<back or facts[pos,col]!=close[j-back]+1:
                weights[j,field]=0.
            else:
                a,b=configuration.seconds[[begin+j-back,begin+j]]
                weights[j,field]*=min(1.,configuration.quality.covered(a,np.nextafter(b,np.inf),1))
    weights[targets==MISSING]=0.
    item['memory_targets']=torch.from_numpy(targets)
    item['memory_weights']=torch.from_numpy(weights/len(FIELDS))
    item['hand_recording']=getattr(song.audio,'identity',{}).get('audio_sha256')


def collate(items,batch):
    records=[(i,it) for i,it in enumerate(items) if 'memory_targets' in it]
    if not records:
        return
    batch['memory_positions']=torch.cat([torch.stack((torch.full_like(it['hand_completed'],i),
                                                       it['hand_completed']),-1) for i,it in records])
    for key in ('memory_targets','memory_weights'):
        batch[key]=torch.cat([it[key] for _,it in records])
    batch['hand_recordings']=[it.get('hand_recording') for it in items]
    batch['mechanics_mass']+=sum(float(it['memory_weights'].sum()) for _,it in records if it.get('mechanics_only'))


def supervision(logits,batch,*,by_recording=False):
    rows,pos=batch['memory_positions'].unbind(-1)
    raw=logits[rows,pos.clamp_min(0)].float()
    tags=batch['memory_targets'].float()
    weights=batch['memory_weights']
    active=weights.gt(0)
    magnitude=torch.where(tags==NO_RIVAL,0.,tags.abs())
    def bce(column,target):
        return F.binary_cross_entropy_with_logits(raw[...,column],target.float(),reduction='none')
    # A missing rival carries zero loss. Genuine ambiguity teaches abstention.
    # Conditional strong evidence exists only after evidence for a lean exists.
    error=bce(1,magnitude>=1)
    error+=(bce(0,tags>0)+bce(2,magnitude==2))*(magnitude>=1)
    error=error/(1+2*(magnitude>=1).float())
    p,lean,strong=evidence(raw.detach())
    call=(strong>=.5)&(lean>=.65)&((2*p-1).abs()>=.3)&active
    wrong=call & ((magnitude!=2)|((p>=.5)!=(tags>0)))
    stats=None
    if by_recording:
        counts=torch.stack((active,call,wrong),dim=1).long()
        family=batch['hand_timeline'][rows,pos.clamp_min(0),tl.FAMILY:].long()
        classes=F.one_hot(family,num_classes=1+len(FAMILIES))[...,1:]
        stats=counts.new_zeros((len(batch['input_ids']),3,len(FIELDS),len(FAMILIES)))
        stats.index_add_(0,rows,counts[...,None]*classes[:,None])
    return (error*weights).sum(),weights.sum(),stats


def qualified_fields(per_recording):
    # A fixed publishing rule, assessed separately per field. It is not fed
    # back into the optimizer. Limit dominance by any one recording.
    qualified=[]
    for i,name in enumerate(FIELDS):
        for j,family in enumerate(FAMILIES):
            records=[(int(stats[1,i,j]),int(stats[2,i,j])) for stats in per_recording.values() if stats[1,i,j]]
            total=sum(n for n,_ in records)
            if (sum(n>=8 for n,_ in records)>=3 and total>=64 and
                    all(errors/n<=.10 for n,errors in records if n>=8) and
                    sum(errors for _,errors in records)/total<=.05):
                qualified.append(f'{name}/{family}')
    return qualified


def report(logits,row,qualified=()):
    p,lean,strong=evidence(logits)
    values=torch.stack((p,lean,strong),-1).tolist()
    valid=domain(row).tolist()
    results={}
    for i,(name,(prob,support,high),ok) in enumerate(zip(FIELDS,values,valid)):
        if not ok:
            results[name]=None
            continue
        category=f'{name}/{FAMILIES[int(row[tl.FAMILY+i])-1]}'
        display='uncertain' if support<.65 or abs(2*prob-1)<.3 else (
            'supported' if category in qualified and high>=.5 else 'lean')
        results[name]=dict(tendency=prob,at_least_lean=support,supported=high,grade=display)
        if i<2:
            results[name]['pad']=tl.PADS[int(row[i])-1]
    return results
