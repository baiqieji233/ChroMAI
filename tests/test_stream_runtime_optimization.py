"""Execution optimizations preserve probabilities, masks and precise clocks."""
from functools import partial

import pytest
import torch
from torch.utils.data import DataLoader

from maichart.stream.attention import cached_decoder_layer
from maichart.stream.grounded_model import phrase_features,prepare_phrase_bank,query_phrase_bank,PHRASE_WINDOWS
from maichart.stream.prefetch import StreamBatchPrefetch
from maichart.stream.training_batch import pack_batch,transfer_batch
from maichart.generation.event_model import _cached_decoder_layer_with_cross_cache
from test_stream_grounding import cpu_threads,source


def test_packed_transfer_preserves_dtypes_shapes_empty_values_and_noncontiguous_views():
    values={'audio':torch.randn(2,5,120),'query_seconds':torch.tensor([[.123456789]],dtype=torch.float64),
        'decoder_input':torch.arange(12).reshape(2,6).T,'empty':torch.zeros(2,0,4),
        'mask':torch.tensor([[True,False]]),'chart_index':torch.tensor([12]),'extra':'kept'}
    packed=pack_batch(values,pin_memory=False)
    result=transfer_batch(packed,torch.device('cpu'))
    reference=transfer_batch(values,torch.device('cpu'))
    assert len(packed.buffers)==3 and result.keys()==reference.keys()
    for key in result:
        if torch.is_tensor(result[key]):
            torch.testing.assert_close(result[key],reference[key],rtol=0,atol=0)
            assert result[key].dtype==reference[key].dtype
            assert (packed.fields[key][1]*result[key].element_size())%256==0
        else:assert result[key]==reference[key]


class Inputs:
    def __len__(self):return 7
    def __getitem__(self,index):return {'x':torch.tensor([index]),'mask':torch.tensor([index%2==0])}
    def prepare_concurrency_key(self,index):return index


def test_packed_prefetch_preserves_batch_order_and_budget_accounting():
    loader=DataLoader(Inputs(),batch_size=2,num_workers=0)
    with StreamBatchPrefetch(loader,depth=2,workers=2,
            prepare_batch=partial(pack_batch,pin_memory=False)) as batches:
        actual=[transfer_batch(batch,torch.device('cpu'))['x'].flatten().tolist() for batch in batches]
        stats=batches.stats()
    assert actual==[[0,1],[2,3],[4,5],[6]]
    assert stats['buffered_tensor_bytes']==0 and stats['max_buffered_tensor_bytes']>0


@pytest.mark.parametrize('norm_first',[True,False])
def test_cached_projection_slicing_preserves_complete_layer_and_caches(norm_first):
    torch.manual_seed(37)
    layer=torch.nn.TransformerDecoderLayer(16,2,32,dropout=0,batch_first=True,norm_first=norm_first).eval()
    memory=torch.randn(2,9,16)
    mask=torch.tensor([[False]*9,[False]*6+[True]*3])
    additive=torch.zeros_like(mask,dtype=torch.float32).masked_fill(mask,float('-inf'))
    reference=actual=(None,None,None,None)
    with torch.inference_mode():
        for _ in range(3):
            hidden=torch.randn(2,1,16)
            expected,reference=_cached_decoder_layer_with_cross_cache(layer,hidden,memory,*reference,mask)
            output,actual=cached_decoder_layer(layer,hidden,memory,*actual,additive)
            torch.testing.assert_close(output,expected,atol=1e-6,rtol=1e-5)
            for a,b in zip(actual,reference):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)


def reference_phrase(curve,times,padding,seconds):
    """Independent small-loop reference for the existing trapezoid-prefix rule."""
    rows=[]
    for values,clock,mask,queries in zip(curve,times,padding,seconds):
        valid=~mask
        values,clock=values[valid].double(),clock[valid].double()
        if not len(clock):
            rows.append(torch.zeros(len(queries),16,dtype=torch.float64));continue
        if len(clock)==1:
            rows.append(values[0].repeat(len(queries),4));continue
        area=(.5*(values[:-1]+values[1:])*(clock[1:]-clock[:-1])[:,None]).cumsum(0)
        integral=torch.cat((area.new_zeros(1,4),area))
        def at(point):
            if point<=clock[0]:return values.new_zeros(4)
            if point>=clock[-1]:return integral[-1]+(point-clock[-1])*values[-1]
            right=int(torch.searchsorted(clock,point));left=right-1
            return integral[left]+(integral[right]-integral[left])*(point-clock[left])/(clock[right]-clock[left])
        rows.append(torch.stack([torch.cat([(at(q+hi)-at(q+lo))/(hi-lo) for lo,hi in PHRASE_WINDOWS]) for q in queries]))
    return torch.stack(rows)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_vectorized_and_cached_phrase_matches_independent_nonuniform_reference(dtype):
    torch.manual_seed(17)
    curve=torch.rand(3,80,4)
    times=torch.rand(3,80,dtype=dtype).cumsum(1)
    padding=torch.zeros(3,80,dtype=torch.bool);padding[1,57:]=True;padding[2,:]=True
    seconds=torch.tensor([[-20.,0.,1.,3.,10.,30.,90.]]*3,dtype=dtype)
    expected=reference_phrase(curve,times,padding,seconds)
    actual=phrase_features(curve,times,padding,seconds)
    torch.testing.assert_close(actual.double(),expected,atol=5e-6,rtol=5e-6)
    bank=prepare_phrase_bank(curve,times,padding)
    for i in range(seconds.size(1)):
        torch.testing.assert_close(query_phrase_bank(bank,seconds[:,i:i+1]),actual[:,i:i+1],rtol=1e-6,atol=1e-6)


def test_phrase_empty_and_single_node_are_preserved():
    for length in (0,1):
        result=phrase_features(torch.ones(2,length,4),torch.zeros(2,length),
            torch.zeros(2,length,dtype=torch.bool),torch.zeros(2,3))
        assert result.shape==(2,3,16)
        assert torch.all(result==(1 if length else 0))


def test_cpu_clock_tensor_is_reused_only_while_clock_row_is_unchanged():
    import numpy as np
    from maichart.stream.grounded_inference import GroundedInference
    inference=object.__new__(GroundedInference)
    inference.device=torch.device('cpu');inference._timing_source=inference._timing_tensor=None
    values=np.arange(20,dtype=np.float32)
    a=inference._feature_tensor(values)
    assert inference._feature_tensor(values) is a
    changed=values.copy();changed[0]=2
    b=inference._feature_tensor(changed)
    assert b is not a and b[0,0,0]==2 and a[0,0,0]==0


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA backend comparison')
@pytest.mark.parametrize('with_padding',[False,True])
def test_cudnn_priority_matches_efficient_forward_and_gradients(with_padding):
    from maichart.stream.runtime import attention_backend
    torch.manual_seed(43)
    values=[torch.randn(2,4,64,48,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    mask=None
    if with_padding:
        mask=torch.zeros(2,1,1,64,device='cuda',dtype=torch.bfloat16)
        mask[0,...,59:]=float('-inf');mask[1,...,48:]=float('-inf')
    results=[]
    for name in ('efficient','cudnn'):
        inputs=[value.clone().requires_grad_() for value in values]
        with attention_backend(name,torch.device('cuda:0')):
            output=torch.nn.functional.scaled_dot_product_attention(*inputs,attn_mask=mask,is_causal=True)
            output.float().square().mean().backward()
        results.append([output.detach().float().cpu(),*[value.grad.float().cpu() for value in inputs]])
    for a,b in zip(*results):torch.testing.assert_close(a,b,atol=5e-3,rtol=3e-2)


def test_warmup_validates_complete_cache_without_reassembling_samples(source,monkeypatch):
    from test_stream_pipeline import make_dataset
    from maichart.stream.data import StreamDataset
    dataset=make_dataset(source,'mapped')
    for index in range(len(dataset)):dataset[index]
    def fail(*_):pytest.fail('Complete deterministic cache must not reassemble epoch samples')
    monkeypatch.setattr(StreamDataset,'__getitem__',fail)
    result=dataset.warm_input_cache()
    assert result['rebuilt_samples']==0 and result['validated_artifacts']>0


def test_warmup_rebuilds_a_missing_artifact_and_rejects_corruption(source,monkeypatch):
    import json
    from test_stream_pipeline import make_dataset
    from maichart.stream.data import StreamDataset
    dataset=make_dataset(source,'mapped')
    for index in range(len(dataset)):dataset[index]
    identity=dataset._structure_identity(dataset.chunks[0])
    key=dataset.input_cache.key(identity)
    index_file=dataset.input_cache.root/(key+'.json')
    info=json.loads(index_file.read_text())
    info['file']=key+'.missing.pt'
    index_file.write_text(json.dumps(info),encoding='utf-8')
    assert not dataset.input_cache.validate_existing(identity)
    # Corrupt only the metadata; preserve any still-live Windows file mapping.
    index_file.write_text('invalid json',encoding='utf-8')
    dataset._structures.clear();dataset.structure_bytes=0
    assert not dataset.input_cache.validate_existing(identity)
    original=StreamDataset.__getitem__;calls=[]
    def counted(self,index):calls.append(index);return original(self,index)
    monkeypatch.setattr(StreamDataset,'__getitem__',counted)
    result=dataset.warm_input_cache()
    assert result['rebuilt_samples']==1 and calls==[0]
    assert dataset.input_cache.validate_existing(identity)


def test_kinematic_duration_scan_is_once_per_prefix_and_preserves_future_causality():
    import numpy as np
    from maichart.stream.symmetry import kinematic_rows
    scans=[0]
    class Tracked(tuple):
        def __iter__(self):scans[0]+=1;return super().__iter__()
    lanes=[(i%8+1,) for i in range(1000)]
    times=np.arange(1000,dtype=np.float64)
    trajectories=[Tracked() for _ in times]
    counts=np.arange(300,900,3)
    actual=kinematic_rows(lanes,times,counts,times[counts-1],trajectories=trajectories)
    assert scans[0]<1000+4*len(counts)
    changed=list(trajectories)
    changed[950]=((1,1,5,950.,10000.),)
    torch.testing.assert_close(actual,kinematic_rows(lanes,times,counts,times[counts-1],trajectories=changed),rtol=0,atol=0)
    for row in (0,50,100):
        count=counts[row]
        single=kinematic_rows(lanes[:count],times[:count],[count],[times[count-1]],trajectories=trajectories[:count])
        torch.testing.assert_close(actual[row],single[0],rtol=0,atol=0)
