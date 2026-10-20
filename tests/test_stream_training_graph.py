"""Training execution must preserve objectives and accumulated gradients."""
from copy import deepcopy

import pytest
import torch

from maichart.form.codec import RationalCodec
from maichart.stream.model import ChartStreamTransformer,StreamConfig
from maichart.stream.training_graph import install_training_graphs,MetricAccumulator
from maichart.stream.train import _mean_metrics,parser,validate_training_args
from test_stream_grounding import cpu_threads


def test_metric_accumulation_preserves_integer_counts_and_mean_contract():
    accumulator=MetricAccumulator();reference={}
    for i in range(5):
        row={'loss':torch.tensor(.12*i),'_note_count':torch.tensor(20_000_001+i),
             '_note_correct':torch.tensor(10_000_001),'token_accuracy':torch.tensor(.2)}
        accumulator.update(row)
        for name,value in row.items():reference[name]=reference.get(name,0)+value
    for name,value in accumulator.as_dict().items():
        assert value.dtype==reference[name].dtype
        torch.testing.assert_close(value,reference[name],atol=0,rtol=0)
    assert _mean_metrics(accumulator,5)==_mean_metrics(reference,5)


def test_graph_options_reject_invalid_memory_and_explicit_cpu_capture():
    base=['--manifest','fixture','--checkpoint-dir','out']
    for value in ('nan','-1','33'):
        with pytest.raises(ValueError,match='显存预算'):
            validate_training_args(parser().parse_args(base+['--cuda-graph-budget-gb',value]))
    with pytest.raises(ValueError,match='需要 CUDA'):
        validate_training_args(parser().parse_args(base+['--device','cpu','--execution-mode','cuda_graph']))


def small_model(dropout=0):
    return ChartStreamTransformer(StreamConfig(len(RationalCodec()),d_model=32,nhead=2,layers=1,
        dim_feedforward=64,max_tokens=16,max_audio_tokens=16,max_history_tokens=8,
        max_stream_memory=2,dropout=dropout)).train()


def test_cpu_auto_preserves_parameter_names_and_regular_execution():
    torch.manual_seed(7)
    model=small_model();reference=deepcopy(model)
    original=list(model.state_dict())
    runner=install_training_graphs(model,max_tokens=16,max_audio_tokens=16,batch_size=2)
    assert runner.prepare(model,'fp32')['captured']==[]
    assert list(model.state_dict())==original
    values=torch.randn(2,13,32,requires_grad=True);mask=torch.zeros(2,13,dtype=torch.bool)
    a=runner.encoder(model.audio_encoder,values,mask)
    b=reference.audio_encoder(values,src_key_padding_mask=mask)
    torch.testing.assert_close(a,b,rtol=0,atol=0)


cuda=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA graph execution')


@cuda
@pytest.mark.parametrize('kind',['encoder','decoder'])
@pytest.mark.parametrize('precision',['fp32','bf16'])
def test_graph_matches_eager_with_padding_new_masks_accumulation_and_optimizer_updates(kind,precision):
    torch.manual_seed(11)
    model=small_model().cuda();reference=deepcopy(model)
    runner=install_training_graphs(model,mode='cuda_graph',max_tokens=16,max_audio_tokens=16,batch_size=2,budget_gb=.25,report=lambda _:None)
    # Existing gradients are part of the contract: capture must not clear them.
    for a,b in zip(model.parameters(),reference.parameters()):
        a.grad=torch.ones_like(a);b.grad=torch.ones_like(b)
    before=torch.cuda.get_rng_state().clone()
    runner.prepare(model,precision)
    assert torch.equal(torch.cuda.get_rng_state(),before)
    assert sorted(runner.stats()['captured'])==['decoder','encoder'],runner.stats()
    for parameter in model.parameters():assert torch.all(parameter.grad==1)
    left=torch.optim.AdamW(model.parameters(),lr=1e-3)
    right=torch.optim.AdamW(reference.parameters(),lr=1e-3)
    for step in range(4):
        values=torch.randn(2,13,32,device='cuda',requires_grad=True)
        duplicate=values.detach().clone().requires_grad_()
        mask=torch.zeros(2,13,dtype=torch.bool,device='cuda');mask[1,9+step%3:]=True
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
            if kind=='encoder':
                a=runner.encoder(model.audio_encoder,values,mask)
                b=reference.audio_encoder(duplicate,src_key_padding_mask=mask)
            else:
                memory=torch.randn(2,26,32,device='cuda',requires_grad=True)
                other=memory.detach().clone().requires_grad_()
                padding=torch.zeros(2,26,dtype=torch.bool,device='cuda');padding[0,21+step%4:]=True
                a=runner.decoder(model.decoder,values,memory,padding)
                causal=torch.ones(13,13,dtype=torch.bool,device='cuda').triu(1)
                b=reference.decoder(duplicate,other,tgt_mask=causal,memory_key_padding_mask=padding,tgt_is_causal=True)
        tolerance=3e-3 if precision=='bf16' else 3e-5
        torch.testing.assert_close(a,b,atol=tolerance,rtol=tolerance)
        a.square().mean().backward();b.square().mean().backward()
        torch.testing.assert_close(values.grad,duplicate.grad,atol=tolerance,rtol=tolerance)
        for p,q in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(p.grad,q.grad,atol=tolerance,rtol=tolerance)
        if step%2:
            left.step();right.step();left.zero_grad(set_to_none=True);right.zero_grad(set_to_none=True)
            for p,q in zip(model.parameters(),reference.parameters()):
                torch.testing.assert_close(p,q,atol=tolerance,rtol=tolerance)
    assert runner.stats()['replays'][kind]>=4


@cuda
def test_graph_dropout_changes_on_replay_and_short_batch_does_not_add_slots():
    model=small_model(.2).cuda()
    runner=install_training_graphs(model,mode='cuda_graph',max_tokens=16,max_audio_tokens=16,batch_size=2,budget_gb=.25,report=lambda _:None)
    runner.prepare(model,'bf16')
    values=torch.randn(2,16,32,device='cuda',requires_grad=True)
    mask=torch.zeros(2,16,dtype=torch.bool,device='cuda')
    outputs=[]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for _ in range(3):
            output=runner.encoder(model.audio_encoder,values,mask)
            outputs.append(output.detach().clone());output.float().square().mean().backward()
            model.zero_grad(set_to_none=True)
        count=len(runner.slots)
        runner.encoder(model.audio_encoder,values[:1],mask[:1]).sum().backward()
    assert len(runner.slots)==count
    assert 'encoder' in runner.slots,runner.stats()
    assert not torch.equal(outputs[0],outputs[1]) and not torch.equal(outputs[1],outputs[2])
    assert all(torch.isfinite(value).all() for value in outputs)


@cuda
def test_graph_budget_zero_and_capture_failure_fall_back_without_parameter_changes(monkeypatch):
    model=small_model().cuda();original={k:v.clone() for k,v in model.state_dict().items()}
    runner=install_training_graphs(model,max_tokens=16,max_audio_tokens=16,batch_size=2,budget_gb=0,report=lambda _:None)
    assert runner.prepare(model,'fp32')['captured']==[]
    def fail(*_a,**_k):raise RuntimeError('unsupported backend capture')
    monkeypatch.setattr(torch.cuda,'make_graphed_callables',fail)
    runner=install_training_graphs(model,mode='cuda_graph',max_tokens=16,max_audio_tokens=16,batch_size=2,budget_gb=.25,report=lambda _:None)
    stats=runner.prepare(model,'fp32')
    assert not stats['captured'] and set(stats['disabled'])=={'encoder','decoder'}
    values=torch.randn(2,13,32,device='cuda',requires_grad=True)
    result=runner.encoder(model.audio_encoder,values,torch.zeros(2,13,dtype=torch.bool,device='cuda'))
    result.square().mean().backward()
    assert torch.isfinite(values.grad).all()
    for name,value in model.state_dict().items():torch.testing.assert_close(value,original[name],rtol=0,atol=0)
