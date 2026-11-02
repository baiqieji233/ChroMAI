"""Runtime-only batching of the existing teacher-forced detail GRUCell.

No parameters, architecture contracts or inference behavior are added here.
"""
import torch
from torch.nn import functional as F
from .stream_duration import select_duration_context


def packed_regular_detail_log_probs(model,context,targets,groups,detail_bank,duration_context=None):
    """Return one packed phase-row log likelihood tensor per sparse shape group.

    Recurrence and the two shared projections are batched across shapes. Each
    shape/slot still owns exactly its old categorical support and duration read.
    """
    if not groups:return []
    maximum=max(len(model.codec.detail_sizes[group[0]]) for group in groups)
    hidden_parts=[];teacher_parts=[];prepared=[]
    offset=0
    for shape,indices,rows,phases in groups:
        absolute=indices.index_select(0,rows)
        ctx=context.index_select(0,absolute)
        target=targets.index_select(0,absolute)
        slots=len(model.codec.detail_sizes[shape])
        weights=[model.shared_details.candidate_weights(shape,slot,detail_bank) for slot in range(slots)]
        hidden_parts.append(ctx+model.shape_embedding.weight[shape]+model.phase_embedding(phases))
        if slots>1:
            teacher=torch.stack([F.embedding(target[:,slot],weights[slot]) for slot in range(slots-1)],1)
            teacher=F.pad(teacher,(0,0,0,maximum-slots))
        else:teacher=ctx.new_zeros(ctx.size(0),maximum-1,32)
        teacher_parts.append(teacher)
        prepared.append((shape,offset,ctx.size(0),target,weights,
            select_duration_context(duration_context,absolute)))
        offset+=ctx.size(0)
    hidden=torch.cat(hidden_parts,0)
    teachers=model.detail_input(torch.cat(teacher_parts,0)) if maximum>1 else None
    previous=model.detail_start.expand(hidden.size(0),-1)
    states=[]
    for slot in range(maximum):
        hidden=model.detail_cell(previous,hidden)
        states.append(hidden)
        if slot+1<maximum:previous=teachers[:,slot]
    projected=model.detail_output(torch.stack(states,1))
    results=[]
    for shape,start,count,target,weights,duration in prepared:
        result=context.new_zeros(count,dtype=torch.float32)
        for slot,weight in enumerate(weights):
            logits=F.linear(projected[start:start+count,slot],weight,model.detail_bias[shape][slot])
            logits=model.duration_detail_logits(logits,shape,slot,duration)
            result=result-F.cross_entropy(logits.float(),target[:,slot],reduction='none')
        results.append(result)
    return results
