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
    if getattr(model,'_inference_backend',None)==requested:
        runtime=getattr(model,'_decoder_runtime',None)
        if runtime is not None and runtime.selected is not None:
            import torch
            torch.set_num_threads(runtime.threads)
        return
    model._inference_backend=requested
    model._decoder_runtime=None
    if str(device)!='cpu' or requested=='torch':
        if str(device)!='cpu' and requested not in ('auto','torch'):
            raise ValueError('OpenVINO decoder requires --device cpu (audio device is independent)')
        return
    from .acceleration import openvino_devices
    if not openvino_devices():
        if requested!='auto':
            raise RuntimeError('OpenVINO unavailable; install the intel-acceleration extra')
        return
    from .openvino_decoder import DecoderRuntime
    model._decoder_runtime=DecoderRuntime(model,args)
