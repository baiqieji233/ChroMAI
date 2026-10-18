"""Transfer only fields used by continuation and its supervised objectives."""
import torch

CPU_ONLY=frozenset(('audio_times_sec','query_committed','final','chart_index','chunk_index',
    'decoder_input_padding_mask','history_clock_padding_mask','plan_labels_padding_mask',
    'target_quality_padding_mask','curve_quality_padding_mask'))


def transfer_batch(batch,device):
    return {key:value.to(device,non_blocking=device.type=='cuda') if torch.is_tensor(value) else value
            for key,value in batch.items() if key not in CPU_ONLY}
