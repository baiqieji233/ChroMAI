"""Runtime-only activation recomputation; checkpoint weights and targets stay unchanged."""
import sys

POLICIES = ('auto', 'off', 'on')


def configure(model, requested, device):
    if requested not in POLICIES:
        raise ValueError('Unknown activation checkpointing policy')
    import torch
    # Windows can page CUDA allocations into shared system RAM before CUDA
    # reports OOM. Leave room for the desktop on smaller display GPUs.
    small_display_gpu = (device.type == 'cuda' and sys.platform == 'win32'
                         and torch.cuda.get_device_properties(device).total_memory <= 16 * 2**30)
    enabled = requested == 'on' or (requested == 'auto' and small_display_gpu)
    model.activation_checkpointing = enabled
    # Decoder-only recomputation gives display headroom without rerunning the
    # whole audio encoder. Explicit on selects the lowest-memory full mode.
    model.activation_checkpointing_scope = 'decoder' if requested == 'auto' else 'all'
    return dict(requested=requested, enabled=enabled,scope=model.activation_checkpointing_scope,
                reason='windows_display_gpu_headroom' if requested == 'auto' and enabled else requested)


def cuda_stats(device):
    if device.type != 'cuda':
        return {}
    import torch
    free, total = torch.cuda.mem_get_info(device)
    return dict(gpu_allocated_bytes=torch.cuda.memory_allocated(device),
                gpu_reserved_bytes=torch.cuda.memory_reserved(device),
                gpu_free_bytes=free, gpu_total_bytes=total)
