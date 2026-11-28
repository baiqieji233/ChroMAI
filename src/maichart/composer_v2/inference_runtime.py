"""CPU thread budgets shared by generation workers and curve prediction."""
import json
import os


def available_cpus():
    try:
        return max(1,len(os.sched_getaffinity(0)))
    except (AttributeError,OSError):
        count=getattr(os,'process_cpu_count',os.cpu_count)()
        return max(1,count or 1)


def thread_budget(requested,device,parallel=1):
    if isinstance(requested,bool) or not isinstance(requested,int) or requested<0:
        raise ValueError('cpu_threads must be a nonnegative integer (0 = automatic)')
    cpus=available_cpus()
    workers=max(1,int(parallel))
    # Autoregressive decoding uses many small matrix operations. Allocating
    # every logical CPU to every worker generally increases thread contention.
    threads=requested or min(8 if str(device)=='cpu' else 2,max(1,cpus//workers))
    return dict(requested=requested,threads=threads,parallel=workers,available_cpus=cpus)


def configure_threads(requested,device,parallel=1):
    import torch
    report=thread_budget(requested,device,parallel)
    if torch.get_num_threads()!=report['threads']:torch.set_num_threads(report['threads'])
    print('composer_v2_cpu_threads '+json.dumps(report),flush=True)
    return report


def configure_model(model, args, device):
    """Attach runtime-only acceleration after loading/moving checkpoint weights."""
    requested=getattr(args,'inference_backend','auto')
    selection = (str(device), requested)
    if getattr(model, '_inference_backend', None) == selection:
        runtime = getattr(model, '_decoder_runtime', None)
        if getattr(runtime, 'selected', None) is not None:
            import torch
            torch.set_num_threads(runtime.threads)
        return
    runtime = None
    if str(device).startswith('cuda') and requested == 'auto':
        import torch
        from .cuda_decoder import CudaDecoderRuntime
        if torch.version.cuda is not None:
            runtime = CudaDecoderRuntime(model.decoder)
    elif str(device) == 'cpu' and requested != 'torch':
        from .acceleration import openvino_devices
        if openvino_devices():
            from .openvino_decoder import DecoderRuntime
            runtime = DecoderRuntime(model,args)
        elif requested != 'auto':
            raise RuntimeError('OpenVINO unavailable; install the intel-acceleration extra')
    elif requested not in ('auto', 'torch'):
        raise ValueError('OpenVINO decoder requires --device cpu (audio device is independent)')
    model._inference_backend = selection
    model._decoder_runtime = runtime
