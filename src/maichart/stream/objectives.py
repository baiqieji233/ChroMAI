"""Event-balanced and authored-audio objectives with independent normalizers."""
import torch
from torch.nn import functional as F


def event_balanced_loss(ce,logits,batch,codec):
    target=batch['target'];valid=~batch['target_padding_mask']
    local=batch.get('target_quality',torch.ones_like(ce))
    valid&=(local>0)&(target!=codec.eos_id)
    group=(target==codec.at_id).long().cumsum(1)
    quality=batch.get('quality_weight',ce.new_ones(ce.size(0)))[:,None]
    groups=target.size(1)+1
    predicted=logits.argmax(-1)
    def grouped(mask):
        counts=ce.new_zeros(ce.size(0),groups).scatter_add(1,group,mask.to(ce))
        losses=ce.new_zeros(ce.size(0),groups).scatter_add(1,group,ce*mask*local*quality)
        active=counts>0
        loss=(losses/counts.clamp_min(1)*active).sum()/active.sum().clamp_min(1)
        wrong=ce.new_zeros(ce.size(0),groups).scatter_add(1,group,((predicted!=target)&mask).to(ce))
        return loss,active.sum().detach(),((wrong==0)&active).sum().detach()
    position=valid&(((target>=7)&(target<18))|(target==codec.at_id)|(target==codec.note_id))
    configuration=valid&((target>=18)|(target==codec.end_id))
    pos,pn,pc=grouped(position);note,nn,nc=grouped(configuration)
    _,en,ec=grouped(valid)
    return .5*(pos+note),{'_whole_position_count':pn,'_whole_position_correct':pc,
        '_whole_configuration_count':nn,'_whole_configuration_correct':nc,
        '_whole_event_count':en,'_whole_event_correct':ec}


def alignment_loss(output,batch):
    logits=output['alignment_event_logits'].float()
    labels=batch['alignment_audio_index'].clamp_min(0)
    mask=batch['alignment_event_mask']&(batch['alignment_event_indices']>=0)
    quality=batch['quality_weight'].float()
    if logits.size(1):
        eligible=mask&~batch['detail_audio_padding_mask'].gather(1,labels)
        safe=logits.masked_fill(~eligible[...,None],0.)
        ce=F.cross_entropy(safe.transpose(1,2),labels,reduction='none')
        event=(ce*eligible*batch['alignment_event_weight']*quality[:,None]).sum()/eligible.sum().clamp_min(1)
        correct=((safe.argmax(-1)==labels)&eligible).sum().detach()
        count=eligible.sum().detach()
    else:
        event=logits.sum()*0;correct=count=logits.new_zeros(())
    values=output['position_alignment_logits'].float()
    target=batch['position_alignment_target'].to(values)
    audio_mask=batch['position_alignment_mask']&~batch['detail_audio_padding_mask']
    weights=batch['detail_time_weight'].to(values)*audio_mask
    positive=(target*weights).sum(1)
    negative=((1-target)*weights).sum(1)
    pos_weight=(negative/positive.clamp_min(1e-6)).clamp(1,12)[:,None]
    ce=F.binary_cross_entropy_with_logits(values,target,reduction='none',pos_weight=pos_weight)
    selection=((ce*weights*batch['alignment_audio_quality']).sum(1)/weights.sum(1).clamp_min(1e-9)
               *quality*batch['curve_example_weight']).mean()
    return event+.25*selection,{'_alignment_count':count,'_alignment_correct':correct,
        'alignment_coverage':mask.sum().float()/(batch['alignment_event_indices']>=0).sum().clamp_min(1)}
