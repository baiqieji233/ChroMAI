"""Existing detail weights and duration likelihood agree under shape batching."""
import copy
import torch
from maichart.composer.stream_data import collate_stream
from maichart.composer.stream_detail_compute import packed_regular_detail_log_probs
from maichart.composer.stream_duration import select_duration_context
from test_stream_phrase_execution import execution_fixture


def test_packed_recurrence_matches_shape_loop_outputs_and_gradients(monkeypatch):
    torch.set_num_threads(1)
    ds,model,_,_=execution_fixture(monkeypatch)
    other=copy.deepcopy(model)
    groups=[];targets=[];offset=0
    # Include every catalog shape and multiple distinct real phases per shape.
    for shape in range(1,ds.codec.num_shapes):
        slots=len(ds.codec.detail_sizes[shape])
        if not slots:continue
        n=3
        target=torch.zeros(n,max(len(x) for x in ds.codec.detail_sizes),dtype=torch.long)
        for slot,size in enumerate(ds.codec.detail_sizes[shape]):target[:,slot]=torch.arange(n)%size
        targets.append(target)
        indices=torch.arange(offset,offset+n)
        rows=torch.tensor([0,0,1,2]);phases=torch.tensor([0,3,6,1])
        groups.append((shape,indices,rows,phases));offset+=n
    torch.manual_seed(62)
    a=torch.randn(offset,model.config.d_model,requires_grad=True)
    b=a.detach().clone().requires_grad_(True)
    target=torch.cat(targets)
    expected=[]
    bank=model.shared_details.encode_catalog()
    for shape,indices,rows,phases in groups:
        absolute=indices[rows]
        expected.append(model.regular_detail_log_probs(a[absolute],shape,phases,target[absolute],detail_bank=bank))
    actual=packed_regular_detail_log_probs(other,b,target,groups,other.shared_details.encode_catalog())
    for left,right in zip(expected,actual):torch.testing.assert_close(left,right,rtol=1e-5,atol=2e-6)
    sum(x.sum() for x in expected).backward();sum(x.sum() for x in actual).backward()
    torch.testing.assert_close(a.grad,b.grad,rtol=2e-5,atol=2e-6)
    for (name,left),(_,right) in zip(model.named_parameters(),other.named_parameters()):
        if left.grad is None:assert right.grad is None,name
        else:torch.testing.assert_close(left.grad,right.grad,rtol=3e-5,atol=3e-6,msg=name)


def test_real_sparse_phase_loss_duration_and_checkpoint_keys_unchanged(monkeypatch):
    torch.set_num_threads(1)
    ds,left,_,_=execution_fixture(monkeypatch)
    with torch.no_grad():left.duration_audio.score.weight.fill_(.01)
    right=copy.deepcopy(left);right._packed_detail_enabled=True
    assert left.state_dict().keys()==right.state_dict().keys()
    batch=collate_stream([ds[i] for i in range(len(ds))])
    a=left(batch);b=right(batch)
    loss_a,metrics_a=left.loss(a,batch);loss_b,metrics_b=right.loss(b,batch)
    torch.testing.assert_close(loss_a,loss_b,rtol=1e-5,atol=2e-6)
    for key in ('detail_loss','configuration_loss','loss'):
        torch.testing.assert_close(metrics_a[key],metrics_b[key],rtol=1e-5,atol=2e-6)
    loss_a.backward();loss_b.backward()
    for (name,p),(_,q) in zip(left.named_parameters(),right.named_parameters()):
        if p.grad is None:assert q.grad is None,name
        else:torch.testing.assert_close(p.grad,q.grad,rtol=5e-5,atol=5e-6,msg=name)
