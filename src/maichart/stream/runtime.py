"""Device resolution shared by stream loading, curve inference and generation."""
from __future__ import annotations

import torch
import os
from contextlib import contextmanager, nullcontext


@contextmanager
def attention_backend(requested, device):
    """Optional SDPA priority, with safe supported-kernel fallbacks.

    Windows PyTorch wheels may omit Flash Attention entirely. cuDNN can be
    faster there; keep the choice explicit because hardware/builds differ.
    """
    original=requested
    if requested == 'auto':
        # Tested on the Windows Blackwell/cuDNN 9.7 wheel which ships no
        # Flash kernels. Default SDPA ranks the slower CUTLASS path first.
        prefer_cudnn=(device.type=='cuda' and os.name=='nt'
            and torch.cuda.get_device_capability(device)[0]>=10
            and (torch.backends.cudnn.version() or 0)>=90700
            and not getattr(torch.backends.cuda,'is_flash_attention_available',lambda:True)())
        if not prefer_cudnn:
            yield {'requested':'auto','policy':'pytorch_default'}
            return
        requested='cudnn'
    if requested not in {'cudnn','efficient'} or device.type != 'cuda':
        raise ValueError('指定注意力后端需要 CUDA；CPU 请选择自动')
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import inspect
    if 'set_priority' not in inspect.signature(sdpa_kernel).parameters:
        if original=='auto':
            yield {'requested':'auto','policy':'pytorch_default'}
            return
        raise ValueError('此 PyTorch 不支持注意力后端优先级；请选择自动或升级 PyTorch')
    first=SDPBackend.CUDNN_ATTENTION if requested=='cudnn' else SDPBackend.EFFICIENT_ATTENTION
    choices=list(dict.fromkeys((first,SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,SDPBackend.CUDNN_ATTENTION,SDPBackend.MATH)))
    with sdpa_kernel(choices,set_priority=True):
        yield {'requested':original,'preferred':requested,'policy':'priority_with_fallback'}


class CudaBatchTimer:
    """One sampled batch per log interval; no per-batch synchronization."""
    def __init__(self):
        self.events=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
        self.events[0].record()

    def transferred(self):self.events[1].record()

    def computed(self):self.events[2].record()

    def metrics(self):
        self.events[-1].synchronize()
        return {'sample_cuda_transfer_ms':self.events[0].elapsed_time(self.events[1]),
                'sample_cuda_forward_backward_ms':self.events[1].elapsed_time(self.events[2]),
                'cuda_timing_scope':'sampled_batch_stream_elapsed_not_kernel_busy'}


@contextmanager
def training_cpu_budget(requested, device):
    """Bound nested CPU pools while independent songs prepare in threads.

    The scope owns the whole training run, not each worker: changing BLAS
    limits in worker contexts would race because these pools are global.
    """
    previous=torch.get_num_threads()
    count=int(requested) or min(2 if device.type=='cuda' else 4,os.cpu_count() or 1)
    torch.set_num_threads(count)
    try:
        from threadpoolctl import threadpool_limits
        context=threadpool_limits(limits=count,user_api='blas')
        controlled=True
    except ImportError:
        context=nullcontext();controlled=False
    try:
        with context:yield {'torch_threads':count,'blas_threads':count if controlled else None}
    finally:torch.set_num_threads(previous)


def resolve_device(requested='auto'):
    if requested is None or str(requested).strip().lower() == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        device = torch.device(requested)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError('生成设备必须为 auto、cpu 或 cuda[:编号]') from exc
    if device.type not in {'cpu', 'cuda'} or (device.type == 'cpu' and device.index is not None):
        raise ValueError('生成设备必须为 auto、cpu 或 cuda[:编号]')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA 不可用，请选择 CPU 或自动设备')
        index = torch.cuda.current_device() if device.index is None else device.index
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError(f'CUDA 设备编号不可用：{index}')
        device = torch.device('cuda', index)
    return device
