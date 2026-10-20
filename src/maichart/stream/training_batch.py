"""Transfer only fields used by continuation and its supervised objectives."""
from dataclasses import dataclass
import torch

CPU_ONLY=frozenset(('query_committed','final','chart_index','chunk_index',
    'decoder_input_padding_mask','history_clock_padding_mask','plan_labels_padding_mask',
    'target_quality_padding_mask','curve_quality_padding_mask'))

# audio_times_sec feeds phrase-energy pooling over the planner curve.
TIME_FLOAT32=frozenset(('query_seconds','detail_times_sec','archive_beats','archive_seconds','audio_times_sec'))


@dataclass
class PackedBatch:
    """A few pinned slabs instead of one CUDA allocation/copy per field."""
    buffers: tuple
    fields: dict
    extras: dict

    @property
    def nbytes(self):
        return sum(value.numel()*value.element_size() for value in self.buffers)


def pack_batch(batch, *, pin_memory=True):
    """Run in the CPU collator thread, including time casts and page locking.

    Slabs are owned by this batch, never reused while asynchronous DMA may be
    reading them. Exact Fraction/token representations are never converted.
    """
    groups, extras = {}, {}
    for key, value in batch.items():
        if key in CPU_ONLY:
            continue
        if not torch.is_tensor(value):
            extras[key] = value
            continue
        if value.device.type != 'cpu':
            raise ValueError('Packed training inputs must be CPU tensors')
        if key in TIME_FLOAT32 and value.dtype == torch.float64:
            value = value.float()
        groups.setdefault(value.dtype, []).append((key, value))
    buffers, fields = [], {}
    for dtype, values in groups.items():
        alignment = max(1, 256 // values[0][1].element_size())
        sizes = [((v.numel()+alignment-1)//alignment)*alignment for _, v in values]
        slab = torch.empty(sum(sizes), dtype=dtype, pin_memory=pin_memory)
        offset = 0
        for (key, value), padded in zip(values, sizes):
            count = value.numel()
            slab[offset:offset+count].copy_(value.reshape(-1))
            fields[key] = (len(buffers), offset, count, tuple(value.shape))
            offset += padded
        buffers.append(slab)
    return PackedBatch(tuple(buffers), fields, extras)


def transfer_batch(batch,device):
    if isinstance(batch, PackedBatch):
        buffers = [value.to(device, non_blocking=device.type=='cuda') for value in batch.buffers]
        result = dict(batch.extras)
        for key, (group, offset, count, shape) in batch.fields.items():
            result[key] = buffers[group][offset:offset+count].view(shape)
        return result
    result={}
    for key,value in batch.items():
        if key in CPU_ONLY:
            continue
        if torch.is_tensor(value):
            moved=value.to(device,non_blocking=device.type=='cuda')
            result[key]=moved.float() if key in TIME_FLOAT32 and moved.dtype==torch.float64 else moved
        else:
            result[key]=value
    return result
