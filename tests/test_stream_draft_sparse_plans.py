"""Supervised-only planning preserves the real model's outputs and gradients."""
import copy

import torch

from maichart.composer.stream_data import NOTE, collate_stream, pack_draft_queries
from test_stream_draft_integration import build
from test_stream_ordered_draft import fixture


def _dense_queries(batch,prefix=''):
    starts=batch[prefix+'phrase_start_beat'];ends=batch[prefix+'phrase_end_beat']
    rows=[];positions=[];keys=[];mapping=[]
    for row,(left,right) in enumerate(zip(starts.tolist(),ends.tolist())):
        seen={};indices=[]
        for position,key in enumerate(zip(left,right)):
            if key not in seen:
                seen[key]=len(rows);rows.append(row);positions.append(position);keys.append((row,*key))
            indices.append(seen[key])
        mapping.append(indices)
    batch[prefix+'draft_query_rows']=torch.tensor(rows)
    batch[prefix+'draft_query_positions']=torch.tensor(positions)
    batch[prefix+'draft_query_inverse']=torch.tensor(mapping)
    batch[prefix+'draft_query_keys']=keys


def test_supervised_draft_plans_and_live_graph_reuse_preserve_real_model(monkeypatch):
    torch.manual_seed(17291)
    ds,sparse,_,_=build(monkeypatch)
    with torch.no_grad():
        for layer in (sparse.draft_action_projection,sparse.draft_detail_projection,sparse.draft_rate_projection):
            layer.weight.normal_(0,.04)
        sparse.phrase_context.ordered.alignment_output.weight.normal_(0,.03)
    sparse.train();dense=copy.deepcopy(sparse)
    batch=collate_stream([ds[len(ds)-2],ds[len(ds)-1]])
    full={**batch};_dense_queries(full);_dense_queries(full,'candidate_')
    assert len(batch['draft_query_keys'])<len(full['draft_query_keys'])
    dense_draft=dense.phrase_context.draft
    original=dense_draft.forward
    def no_reuse(*args,**kwargs):
        # Baseline recomputes both departure and landing plans, as before.
        values=list(args)
        if len(values)>5 and values[5] is not None:
            values[5]={key:value for key,value in values[5].items() if key!='draft_forward_plans'}
        return original(*values,**kwargs)
    monkeypatch.setattr(dense_draft,'forward',no_reuse)
    workloads={'dense':0,'sparse':0}
    def count(name):
        def hook(module,args):workloads[name]+=args[0].size(0)
        return hook
    dense_draft.organize.register_forward_pre_hook(count('dense'))
    sparse.phrase_context.draft.organize.register_forward_pre_hook(count('sparse'))
    torch.manual_seed(911);expected=dense(full);expected_loss,_=dense.loss(expected,full)
    expected_loss.backward()
    torch.manual_seed(911);actual=sparse(batch);actual_loss,_=sparse.loss(actual,batch)
    actual_loss.backward()
    torch.testing.assert_close(actual_loss,expected_loss,rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(actual['decoder_hidden'],expected['decoder_hidden'],rtol=0,atol=0)
    active=batch['loss_mask'].bool()&batch['sample_weight'].gt(0)
    marks=active&batch['target_kind'].eq(NOTE)&batch['configuration_mask'].bool()
    for name in ('execution_kind_logits','execution_delta_logits','departure_execution_prior_logits'):
        torch.testing.assert_close(actual[name][active],expected[name][active],rtol=2e-5,atol=2e-5)
    for name in ('draft_action_context','action_execution_prior_logits','execution_route_log_probs'):
        torch.testing.assert_close(actual[name][marks],expected[name][marks],rtol=2e-5,atol=2e-5)
    for name,parameter in sparse.named_parameters():
        reference=dict(dense.named_parameters())[name]
        assert (parameter.grad is None)==(reference.grad is None),name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad,reference.grad,rtol=3e-4,atol=2e-5,msg=name)
    assert workloads['sparse']<workloads['dense']


def test_time_only_landing_uses_one_finite_dummy_and_inference_keeps_queries():
    starts=torch.tensor([[0.,0.,4.,4.],[8.,8.,12.,12.]])
    batch=dict(phrase_start_beat=starts,phrase_end_beat=starts+4,
        candidate_phrase_start_beat=starts,candidate_phrase_end_beat=starts+4,
        loss_mask=torch.tensor([[False,False,True,True],[False,False,True,True]]),
        sample_weight=torch.ones(2,4),target_kind=torch.full((2,4),NOTE),
        configuration_mask=torch.zeros(2,4,dtype=torch.bool))
    pack_draft_queries(batch);pack_draft_queries(batch,'candidate_')
    assert len(batch['draft_query_keys'])==2
    assert len(batch['candidate_draft_query_keys'])==1
    assert batch['candidate_draft_query_inverse'].eq(0).all()
    inference={key:value for key,value in batch.items() if key not in ('target_kind',)}
    pack_draft_queries(inference)
    assert len(inference['draft_query_keys'])==4


def test_live_plan_reuse_requires_equal_cpu_entry_content_for_same_phrase():
    _,draft,cache,batch=fixture()
    batch['draft_query_entry_keys']=[tuple(batch['draft_entry_state'][0,0].tolist())]
    state={};calls=[]
    draft.organize.register_forward_pre_hook(lambda module,args:calls.append(args[0].size(0)))
    def run(value):return draft(None,cache,torch.tensor([0]),value,torch.empty(0,2,128),state)
    a=run(batch);same=run(batch)
    assert len(calls)==1
    torch.testing.assert_close(a['_draft_plan_bank'],same['_draft_plan_bank'],rtol=0,atol=0)
    entry=batch['draft_entry_state'].clone();entry[...,0]=3.
    changed={**batch,'draft_entry_state':entry,'draft_query_entry_keys':[tuple(entry[0,0].tolist())]}
    b=run(changed)
    assert len(calls)==2
    assert not torch.allclose(a['_draft_plan_bank'],b['_draft_plan_bank'])
    # External/fixture callers without the CPU identity always recompute;
    # they must not be treated as proving entry equality from GPU data.
    external={key:value for key,value in batch.items() if key!='draft_query_entry_keys'}
    run(external);run(external)
    assert len(calls)==4
