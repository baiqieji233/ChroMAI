"""Versioned history-memory scale and cross-attention numerical policies.

These policies live in ComposerConfig. Legacy checkpoints retain their original
forward; changing either policy requires an explicit training-stage migration.
"""
from contextlib import contextmanager, nullcontext
import torch


def bounded_history_memory(memory, padding_mask, bound):
    """Limit each history vector's RMS; preserve direction and small vectors.

    Compute the scale in FP32 with a peak-normalized norm to avoid squaring
    large finite activations. Padding and grouped history dropout are not
    usable keys or tail events, so return exact zeros at those positions.
    This bounds the exposed memory, not the encoder's internal residuals.
    """
    values = memory.float()
    peak = values.abs().amax(-1, keepdim=True).clamp_min(1e-6)
    unit_rms = (values / peak).square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    scale = ((float(bound) / peak) / unit_rms).clamp_max(1.)
    result = (values * scale).to(memory.dtype)
    return result.masked_fill(padding_mask[..., None], 0.)


@contextmanager
def _math_attention_context(backend_context):
    getter = getattr(torch.backends.cuda, 'fp16_bf16_reduction_math_sdp_allowed', None)
    setter = getattr(torch.backends.cuda, 'allow_fp16_bf16_reduction_math_sdp', None)
    previous = getter() if getter is not None and setter is not None else None
    try:
        if previous is not None:
            setter(False)
        with backend_context:
            yield
    finally:
        if previous is not None:
            setter(previous)


def history_attention_context(policy, device_type):
    """Select only the local cross-attention core; keep its dropout intact.

    BF16 fused backward can incur cancellation errors for concentrated
    attention with dropout. Math uses FP32 intermediates for half inputs.
    The context restores backend flags even when the attention call fails.
    CPU uses its normal math implementation without changing CUDA flags.
    """
    if policy == 'auto_v1':
        return nullcontext()
    if policy != 'math_v1':
        raise ValueError('Unknown history_attention_policy')
    if device_type != 'cuda':
        return nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:  # PyTorch 2.0/2.1 compatibility
        backend = torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False)
    else:
        backend = sdpa_kernel(SDPBackend.MATH)
    return _math_attention_context(backend)
