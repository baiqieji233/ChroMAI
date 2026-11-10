"""Optional inference acceleration, kept separate from checkpoint architecture."""
from functools import lru_cache
import logging


AUDIO_DEVICES = ('auto', 'cpu', 'cuda', 'openvino-gpu', 'openvino-npu')


@lru_cache(maxsize=1)
def openvino_devices():
    try:
        import openvino as ov
        return tuple(ov.Core().available_devices)
    except (ImportError, RuntimeError, OSError):
        return ()


def resolve_audio_device(requested='auto', torch_device='cpu'):
    if requested.startswith('cuda:') and requested[5:].isdigit():
        return requested
    if requested not in AUDIO_DEVICES:
        raise ValueError('Audio device must be ' + '/'.join(AUDIO_DEVICES))
    if requested != 'auto':
        return requested
    if str(torch_device).startswith('cuda'):
        return str(torch_device)
    devices = openvino_devices()
    if any(d.split('.')[0] == 'GPU' for d in devices):
        return 'openvino-gpu'
    if 'NPU' in devices:
        return 'openvino-npu'
    return 'cpu'


def audio_runtime(requested, torch_device='cpu'):
    selected = resolve_audio_device(requested, torch_device)
    logging.getLogger(__name__).info('Composer v2 audio device: %s (requested %s)', selected, requested)
    return selected


def training_device(requested='auto'):
    import torch
    if requested not in ('auto','cpu','cuda','xpu'):
        raise ValueError('Training device must be auto/cpu/cuda/xpu')
    xpu=hasattr(torch,'xpu') and torch.xpu.is_available()
    if requested=='auto':
        return 'cuda' if torch.cuda.is_available() else 'xpu' if xpu else 'cpu'
    if requested=='xpu' and not xpu:
        raise RuntimeError('Intel XPU unavailable; install PyTorch from https://download.pytorch.org/whl/xpu and check the Intel GPU driver')
    return requested
