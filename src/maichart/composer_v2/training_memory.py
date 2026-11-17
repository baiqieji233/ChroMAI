"""Training runtime policies with unchanged checkpoint weights and targets."""
import sys

POLICIES = ('auto', 'off', 'on')
PROJECTION_PACKING_CHOICES = ('off', 'on')


def checkpointing_scope(requested, device_type, total_bytes):
    if requested not in POLICIES:
        raise ValueError('Unknown activation checkpointing policy')
    # Windows can page CUDA allocations into shared system RAM before CUDA
    # reports OOM. Leave room for the desktop on smaller display GPUs.
    if requested == 'on':
        return 'all'
    if (requested == 'auto' and device_type == 'cuda' and sys.platform == 'win32'
            and total_bytes is not None and total_bytes <= 16 * 2**30):
        return 'decoder'
    return 'off'


def configure(model, requested, device):
    import torch
    total = (torch.cuda.get_device_properties(device).total_memory
             if requested == 'auto' and device.type == 'cuda' and sys.platform == 'win32' else None)
    scope = checkpointing_scope(requested, device.type, total)
    enabled = scope != 'off'
    model.activation_checkpointing = enabled
    # Decoder-only recomputation gives display headroom without rerunning the
    # whole audio encoder. Explicit on selects the lowest-memory full mode.
    model.activation_checkpointing_scope = scope if enabled else ('decoder' if requested == 'auto' else 'all')
    return dict(requested=requested, enabled=enabled,scope=model.activation_checkpointing_scope,
                reason='windows_display_gpu_headroom' if requested == 'auto' and enabled else requested)


def configure_projection_packing(model, requested, device):
    from .model import Attention

    if requested not in PROJECTION_PACKING_CHOICES:
        raise ValueError('Unknown attention projection packing policy')
    enabled = requested == 'on'
    for module in model.modules():
        if isinstance(module, Attention):
            module.pack_projections = enabled
    return dict(requested=requested, enabled=enabled, device=device.type)


def configure_decoder_flash_attention(model, requested, device):
    import torch
    from .model import DecoderBlock

    if requested not in ('off', 'on'):
        raise ValueError('Unknown decoder FlashAttention policy')
    available = device.type == 'cuda' and torch.backends.cuda.is_flash_attention_available()
    enabled = requested == 'on' and available
    for layer in model.modules():
        if isinstance(layer, DecoderBlock):
            layer.flash_self_attention = enabled
    return dict(requested=requested, enabled=enabled, flash_compiled=available,
                scope='decoder_self_attention')


def cuda_stats(device):
    if device.type != 'cuda':
        return {}
    import torch
    free, total = torch.cuda.mem_get_info(device)
    return dict(gpu_allocated_bytes=torch.cuda.memory_allocated(device),
                gpu_reserved_bytes=torch.cuda.memory_reserved(device),
                gpu_free_bytes=free, gpu_total_bytes=total)
