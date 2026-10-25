import pytest
import torch
from maichart.composer.train import _nonfinite_tensors


def reference(mapping):
    return [key for key,value in mapping.items() if torch.is_tensor(value) and not torch.isfinite(value).all()]


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float32,torch.float64])
def test_nonfinite_long_mixed_tensors(device,dtype):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    mapping={}
    for index,(position,value) in enumerate([(0,float('nan')),(8191,float('inf')),(16383,-float('inf'))]):
        data=torch.full((16384,),torch.finfo(dtype).max,device=device,dtype=dtype)
        data[position]=value
        mapping[f'bad{index}']=data
        mapping[f'good{index}']=torch.full((8192,),torch.finfo(dtype).max,device=device,dtype=dtype)
    mapping['empty']=torch.empty(0,device=device,dtype=dtype)
    mapping['integer']=torch.arange(20,device=device)
    mapping['bool']=torch.ones(20,device=device,dtype=torch.bool)
    mapping['text']='ignored'
    assert _nonfinite_tensors(mapping)==reference(mapping)==['bad0','bad1','bad2']


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_nonfinite_strides_complex_order_and_no_grad(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    base=torch.ones((64,128),device=device,requires_grad=True)
    with torch.no_grad():base[31,62]=float('nan')
    mapping={'complex':torch.tensor([1+2j,complex(0,float('inf'))],device=device),
             'transpose':base.T,'slice':base[:,::2],'good':torch.ones(10,device=device),
             'cpu_bad':torch.tensor([float('nan')]),'empty':torch.empty((3,0),device=device)}
    assert _nonfinite_tensors(mapping)==reference(mapping)==['complex','transpose','slice','cpu_bad']
    assert base.grad is None


def test_nonfinite_empty_mapping():
    assert _nonfinite_tensors({'ignored':None})==[]


def test_nonfinite_cuda_foreach_unsupported_falls_back(monkeypatch):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    def unsupported(*args,**kwargs):raise NotImplementedError('unsupported norm')
    monkeypatch.setattr(torch,'_foreach_norm',unsupported)
    mapping={'good':torch.ones(1024,device='cuda'),
             'bad':torch.full((1024,),float('nan'),device='cuda')}
    assert _nonfinite_tensors(mapping)==['bad']
