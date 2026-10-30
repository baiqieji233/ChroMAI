"""Packed semantic-type blocks for a literal event's exact likelihood.

The model predicts when to finish each required type block separately from
which atom comes next. Multiple Touch notes before a TouchHold are ordinary
targets; there is no missing-type deadline and no fixed Touch count.
"""
import torch
from torch.nn.utils.rnn import pad_sequence

TYPE_BITS=(1,2,4,8,16)


def pack_literal_type_blocks(targets,atom_types):
    """CPU producer packing; only real blocks enter the GRU during training.

    Rows refer to the already packed literal-event rows, not the whole batch.
    Atom order within a type is its original codec order. Earlier type blocks
    are known autoregressive context; later blocks never enter this context.
    """
    if targets.device.type!='cpu':raise ValueError('Literal block packing belongs in the CPU producer')
    mapping=atom_types.tolist() if torch.is_tensor(atom_types) else list(atom_types)
    groups=[([],[],[]) for _ in TYPE_BITS]
    for row_id,row in enumerate(targets.tolist()):
        atoms=[]
        for atom in row:
            if atom==0:break
            if atom<0:continue
            kind=int(mapping[atom])
            if kind not in TYPE_BITS:raise ValueError('Type-block literal atoms must have exactly one parsed note type')
            atoms.append((atom,kind))
        previous=[]
        for type_id,bit in enumerate(TYPE_BITS):
            block=[atom for atom,kind in atoms if kind==bit]
            if not block:continue
            rows,blocks,prefixes=groups[type_id]
            rows.append(row_id)
            blocks.append(torch.tensor([*block,0],dtype=torch.long))
            prefixes.append(torch.tensor(previous or [0],dtype=torch.long))
            previous.extend(block)
    result=[]
    for rows,blocks,prefixes in groups:
        result.append((torch.tensor(rows,dtype=torch.long),
            pad_sequence(blocks,batch_first=True,padding_value=-1) if blocks else torch.empty(0,1,dtype=torch.long),
            pad_sequence(prefixes,batch_first=True,padding_value=0) if prefixes else torch.empty(0,1,dtype=torch.long)))
    return tuple(result)


def tensor_literal_type_blocks(targets,atom_types):
    """Small direct-call fallback without device-to-host shape discovery.

    Production collators use the sparse CPU packing above. This fallback
    retains fixed tensor shapes for standalone model callers and diagnostics.
    """
    positions=torch.arange(targets.size(1),device=targets.device)[None].expand_as(targets)
    values=atom_types[targets.clamp_min(0)]*targets.gt(0)
    rows=torch.arange(targets.size(0),device=targets.device)
    result=[]
    for bit in TYPE_BITS:
        chosen=values.eq(bit)&targets.gt(0)
        count=chosen.sum(-1)
        order=positions.masked_fill(~chosen,targets.size(1)).argsort(dim=-1,stable=True)
        packed=targets.gather(-1,order)
        slots=positions
        packed=torch.where(slots<count[:,None],packed,
            torch.where(slots==count[:,None],torch.zeros_like(packed),torch.full_like(packed,-1)))
        # Events without this type produce a dummy STOP-only block, whose
        # likelihood is masked to zero by the model.
        prefix=targets.masked_fill(~(values.lt(bit)&targets.gt(0)),0)
        result.append((rows,packed,prefix))
    return tuple(result)
