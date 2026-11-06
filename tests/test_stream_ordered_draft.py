"""Contracts for executable drafts, not musical-quality acceptance tests."""
from types import SimpleNamespace

import torch

from maichart.composer.event_set_codec import EventSetCodec
from maichart.composer.stream_draft import OrderedPhraseDraft, read_draft, draft_mode_nll


def fixture():
    codec = EventSetCodec.fit(['1','5','1/5','2/6','C','A1','1-5[8:1]','7h[4:1]'])
    draft = OrderedPhraseDraft(128,16,codec)
    beats = torch.linspace(-2.,2.,17,dtype=torch.float64)[None]
    cache = dict(beats=beats,seconds=(beats+2)*.5,mask=torch.ones_like(beats,dtype=torch.bool),
        ordered_configuration_keys=torch.randn(1,17,64))
    batch = dict(phrase_start_beat=torch.zeros(1,3,dtype=torch.float64),
        phrase_end_beat=torch.full((1,3),2.,dtype=torch.float64),
        step_beat=torch.tensor([[0.,.5,1.]],dtype=torch.float64),
        condition=torch.zeros(1,16),draft_entry_state=torch.zeros(1,3,50),
        phrase_memory_start_beat=torch.zeros(1,3,4,dtype=torch.float64),
        phrase_memory_end_beat=torch.zeros(1,3,4,dtype=torch.float64),
        phrase_memory_mask=torch.zeros(1,3,4,dtype=torch.bool),
        phrase_sequence_inverse=torch.full((1,3,4),-1,dtype=torch.long),
        phrase_sequence_kind=torch.zeros(0,2,dtype=torch.long),
        phrase_sequence_mask=torch.zeros(0,2,dtype=torch.bool),
        phrase_sequence_atoms=torch.zeros(0,2,1,dtype=torch.long),
        phrase_sequence_beats=torch.zeros(0,2,dtype=torch.float64),
        draft_query_rows=torch.tensor([0]),draft_query_positions=torch.tensor([0]),
        draft_query_inverse=torch.zeros(1,3,dtype=torch.long),draft_query_keys=((0,0.,2.),))
    return codec,draft,cache,batch


def predict(draft,cache,batch):
    return draft(None,cache,torch.tensor([0]),batch,torch.empty(0,2,128))


def test_future_target_arrays_never_enter_causal_plan():
    _,draft,cache,batch = fixture()
    original = predict(draft,cache,batch)
    changed = {**batch,'target_kind':torch.full((1,3),999),
        'target_phrase_sequence_atoms':torch.full((5,64,33),999),
        'target_phrase_features':torch.full((1,3,96),999.)}
    other = predict(draft,cache,changed)
    torch.testing.assert_close(original['_draft_plan_bank'],other['_draft_plan_bank'],rtol=0,atol=0)


def test_draft_read_is_absolute_beat_owned_and_does_not_extend_phrase():
    _,draft,cache,batch = fixture()
    output = predict(draft,cache,batch)
    query = torch.tensor([[[.5,2.1],[.5,2.1],[.5,2.1]]],dtype=torch.float64)
    read = read_draft(output,batch,query)
    torch.testing.assert_close(read[:,0,0],read[:,2,0],rtol=0,atol=0)
    assert torch.count_nonzero(read[:,:,1])==0
    projection = torch.nn.Linear(96,3)
    torch.testing.assert_close(read_draft(output,batch,query,projection),projection(read).masked_fill(query[...,None,None]>=2,0.),rtol=1e-5,atol=1e-6)


def test_runtime_cache_reuses_plan_but_rebuilds_growing_query_mapping():
    _,draft,cache,batch = fixture();draft.eval()
    runtime = {};batch={**batch,'_draft_runtime_cache':runtime,'_draft_runtime_key':('same_entry',)}
    with torch.no_grad():
        original = predict(draft,cache,batch)
        longer={key:(torch.cat((value,value[:,-1:]),1) if isinstance(value,torch.Tensor)
            and value.ndim>=2 and value.shape[:2]==(1,3) else value) for key,value in batch.items()}
        other=predict(draft,cache,longer)
    assert not original['draft_runtime_cache_hit'] and other['draft_runtime_cache_hit']
    assert other['draft_context'].shape==(1,4,4,96)
    torch.testing.assert_close(original['_draft_plan_bank'],other['_draft_plan_bank'],rtol=0,atol=0)


def test_wait_splitting_in_reference_cannot_change_draft():
    codec,draft,cache,batch = fixture()
    batch['phrase_memory_mask'][:,:,0]=True
    batch['phrase_memory_start_beat'][:,:,0]=-2.
    batch['phrase_memory_end_beat'][:,:,0]=0.
    batch['phrase_sequence_inverse'][:,:,0]=0
    def with_steps(kinds,beats,atoms):
        return {**batch,'phrase_sequence_kind':torch.tensor([kinds]),
            'phrase_sequence_beats':torch.tensor([beats],dtype=torch.float64),
            'phrase_sequence_atoms':torch.tensor([[ [a] for a in atoms ]]),
            'phrase_sequence_mask':torch.ones(1,len(kinds),dtype=torch.bool)}
    a=with_steps([2,2],[-1.5,-.25],[codec._atoms['1'],codec._atoms['5']])
    b=with_steps([2,3,3,2],[-1.5,-1.,-.5,-.25],[codec._atoms['1'],0,0,codec._atoms['5']])
    torch.testing.assert_close(predict(draft,cache,a)['_draft_plan_bank'],predict(draft,cache,b)['_draft_plan_bank'],rtol=1e-6,atol=1e-6)


def test_all_actual_dense_targets_receive_modewise_geometry_gradients():
    codec,draft,cache,batch = fixture()
    events=[codec.encode(text) for text in ('1/5','2/6','C')]
    width=max(len(x['details']) for x in events)
    details=torch.full((1,3,width),-1,dtype=torch.long)
    for i,event in enumerate(events):details[0,i,:len(event['details'])]=torch.tensor(event['details'])
    batch.update(target_shape=torch.tensor([[x['shape'] for x in events]]),target_details=details,
        target_phase_mask=torch.tensor([[x['phase_mask'] for x in events]]),target_kind=torch.full((1,3),2),
        target_beat=torch.tensor([[.1,.15,.2]],dtype=torch.float64),
        history_kind=torch.tensor([[1,2,2]]),history_atoms=torch.tensor([[[0],[codec._atoms['1']],[codec._atoms['5']]]]))
    model=SimpleNamespace(phrase_context=SimpleNamespace(draft=draft))
    output=predict(draft,cache,batch)
    loss=draft_mode_nll(model,output,batch)
    assert loss.shape==(1,3,4) and torch.isfinite(loss).all()
    loss.sum().backward()
    assert draft.reconstruct[-1].weight.grad.abs().sum()>0
    physical=draft.target_features(model,batch)
    assert not torch.equal(physical[0,0,5:13],physical[0,1,5:13])
    assert physical[0,2,13:46].sum()==1


def test_empty_or_short_phrase_keeps_finite_plan_outputs():
    _,draft,cache,batch=fixture()
    for end in (0.,.01):
        value={**batch,'phrase_end_beat':torch.full((1,3),end,dtype=torch.float64)}
        output=predict(draft,cache,value)
        assert torch.isfinite(output['_draft_plan_bank']).all()


def test_cross_phrase_first_note_uses_landing_draft_supervision():
    codec,draft,cache,batch=fixture()
    event=codec.encode('1')
    batch.update(target_shape=torch.full((1,3),event['shape']),
        target_details=torch.tensor([[event['details']]*3]),
        target_phase_mask=torch.tensor([[event['phase_mask']]*3]),target_kind=torch.full((1,3),2),
        target_beat=torch.tensor([[2.1,2.15,2.2]],dtype=torch.float64),
        candidate_phrase_start_beat=torch.full((1,3),2.,dtype=torch.float64),
        candidate_phrase_end_beat=torch.full((1,3),4.,dtype=torch.float64),
        history_kind=torch.tensor([[1,2,2]]),history_atoms=torch.tensor([[[0],[codec._atoms['1']],[codec._atoms['1']]]]))
    output=predict(draft,cache,batch)
    # This is the actual candidate branch's predicted context, not a label.
    output['draft_action_context']=torch.randn(1,3,4,96,requires_grad=True)
    model=SimpleNamespace(phrase_context=SimpleNamespace(draft=draft))
    loss=draft_mode_nll(model,output,batch)
    assert loss.gt(0).all()
    loss.sum().backward()
    assert output['draft_action_context'].grad.abs().sum()>0
