"""Plan identity and retry scope follow actual prefixes, never a whole-block ban."""
from fractions import Fraction
from types import SimpleNamespace

import torch

from maichart.form.data import EventPoint
from maichart.composer.stream_data import StreamStep
from maichart.composer.stream_phrase_runtime import draft_context_signature,attach_draft_runtime
from maichart.composer.stream_search import _failed_proposal_scope


def step(text):
    return StreamStep(2,Fraction(0),Fraction(1),0.,1.,text)


def test_entry_signature_detects_equal_length_changed_history_and_control_edits():
    context=SimpleNamespace(_committed=[step('1')],condition=torch.tensor([.8,.2]))
    before=draft_context_signature(context,2.,0)
    context._committed[0]=step('2')
    after=draft_context_signature(context,2.,0)
    assert before!=after
    context.condition[1]=.4
    assert after!=draft_context_signature(context,2.,0)


def test_plan_entry_identity_survives_appending_inside_same_phrase():
    context=SimpleNamespace(_committed=[step('1')],condition=torch.tensor([.8,.2]))
    before=draft_context_signature(context,1.,2)
    context._committed.append(StreamStep(2,Fraction(1),Fraction(2),1.,2.,'3'))
    assert before==draft_context_signature(context,1.,2)


def test_local_failure_scope_does_not_exclude_other_configuration_at_same_beat():
    first=EventPoint(Fraction(3),'1h[4:1]')
    other=EventPoint(Fraction(3),'5')
    def failed(point):
        return dict(cursor=Fraction(3),guard=SimpleNamespace(last_proposal=point,
            failure_trials=[(point,[])]))
    assert _failed_proposal_scope(failed(first))==_failed_proposal_scope(failed(first))
    assert _failed_proposal_scope(failed(first))!=_failed_proposal_scope(failed(other))


def test_runtime_packs_only_last_query_without_truncating_decoder_history():
    context=SimpleNamespace(_committed=[],condition=torch.tensor([.8,.2]))
    model=SimpleNamespace(config=SimpleNamespace(draft_policy='beat_ordered_draft_v1'),training=False)
    batch=dict(phrase_start_beat=torch.tensor([[0.,0.,8.,8.]]),
        phrase_end_beat=torch.tensor([[8.,8.,16.,16.]]),
        history_kind=torch.tensor([[1,2,2,3]]),
        draft_query_rows=torch.tensor([0,0]),draft_query_positions=torch.tensor([0,2]),
        draft_query_inverse=torch.tensor([[0,0,1,1]]),
        draft_query_keys=[(0,0.,8.),(0,8.,16.)])
    attached=attach_draft_runtime(context,batch,model)
    assert attached['_inference_last_query'] is True
    assert attached['_timing_query_positions'].tolist()==[[3]]
    assert attached['_timing_original_flat'].tolist()==[3]
    assert attached['_timing_packed_flat'].tolist()==[0]
    assert attached['draft_query_rows'].tolist()==[0]
    assert attached['draft_query_positions'].tolist()==[3]
    assert attached['draft_query_keys']==[(0,8.,16.)]
    assert attached['draft_query_inverse'].tolist()==[[0,0,0,0]]
    assert attached['history_kind'] is batch['history_kind']
    assert '_inference_last_query' not in batch
    assert len(batch['draft_query_keys'])==2


def test_candidate_and_training_batches_keep_their_existing_query_contract():
    context=SimpleNamespace(_committed=[],condition=torch.tensor([.8,.2]))
    model=SimpleNamespace(config=SimpleNamespace(draft_policy='beat_ordered_draft_v1'),training=False)
    batch=dict(candidate_phrase_start_beat=torch.tensor([[8.]]),
        candidate_phrase_end_beat=torch.tensor([[16.]]),
        candidate_draft_query_keys=[(0,8.,16.)])
    candidate=attach_draft_runtime(context,batch,model,candidate=True)
    assert '_inference_last_query' not in candidate
    assert candidate['candidate_draft_query_keys'] is batch['candidate_draft_query_keys']
    model.training=True
    assert attach_draft_runtime(context,batch,model,candidate=True) is batch
