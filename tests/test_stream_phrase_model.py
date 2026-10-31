"""Phrase causality contracts; intentionally small CPU fixtures."""
import torch

from maichart.composer.stream_phrase import PHRASE_FEATURE_DIM
from maichart.composer.stream_phrase_model import CausalPhraseContext, phrase_auxiliary_loss


class AudioReader:
    @staticmethod
    def _read_audio(cache, queries, inverse, **kwargs):
        # Same interval always has the same music, independent of decoder prefix.
        return queries.float()[..., None].expand(*queries.shape, 32) / 16


def phrase_batch():
    f = PHRASE_FEATURE_DIM
    return dict(phrase_start_beat=torch.zeros(1, 2), phrase_end_beat=torch.full((1, 2), 4.),
        phrase_progress=torch.randn(1, 2, f), phrase_memory_features=torch.zeros(1, 2, 4, f),
        phrase_memory_start_beat=torch.zeros(1, 2, 4), phrase_memory_end_beat=torch.zeros(1, 2, 4),
        phrase_memory_mask=torch.zeros(1, 2, 4, dtype=torch.bool), condition=torch.zeros(1, 3),
        step_mask=torch.ones(1, 2, dtype=torch.bool), target_phrase_features=torch.randn(1, 2, f),
        target_phrase_mask=torch.ones(1, 2, dtype=torch.bool), target_phrase_weight=torch.ones(1, 2))


def test_intent_is_anchor_sticky_and_targets_are_not_forward_inputs():
    module = CausalPhraseContext(8, 3).eval()
    batch = phrase_batch()
    cache = {'delta_keys': torch.zeros(1, 1, 32)}
    residual, output = module(AudioReader(), cache, torch.tensor([0]), batch)
    torch.testing.assert_close(output['phrase_intent'][:, 0], output['phrase_intent'][:, 1])
    assert torch.equal(residual, torch.zeros_like(residual))
    batch['target_phrase_features'].fill_(float('nan'))
    batch['phrase_progress'].normal_()
    _, second = module(AudioReader(), cache, torch.tensor([0]), batch)
    torch.testing.assert_close(output['phrase_intent'], second['phrase_intent'])
    assert torch.equal(output['phrase_reference_weights'][..., 0], torch.ones(1, 2))


def test_masked_recovery_target_nan_is_safe_and_receives_no_gradient():
    batch = phrase_batch()
    batch['target_phrase_mask'].zero_()
    batch['target_phrase_features'].fill_(float('nan'))
    prediction = torch.randn_like(batch['target_phrase_features'], requires_grad=True)
    loss = phrase_auxiliary_loss({'phrase_prediction': prediction}, batch)
    assert loss.item() == 0
    loss.backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


def ordered_fixture():
    from types import SimpleNamespace
    from maichart.composer.stream_model import StreamComposer
    model=SimpleNamespace(kind_embedding=torch.nn.Embedding(5,16),shape_embedding=torch.nn.Embedding(3,16),
        phase_embedding=torch.nn.Embedding(8,16),atom_embedding=torch.nn.Embedding(8,16,padding_idx=0),
        _read_audio=StreamComposer._read_audio)
    batch=phrase_batch()
    batch.update(phrase_start_beat=torch.full((1,2),4.),phrase_end_beat=torch.full((1,2),8.),
        step_beat=torch.tensor([[4.,5.]]),phrase_sequence_inverse=torch.tensor([[[0,-1,-1,-1],[0,-1,-1,-1]]]))
    batch['phrase_memory_mask'][...,0]=True
    batch['phrase_memory_end_beat'][...,0]=4.
    table=dict(kind=torch.zeros(1,32,dtype=torch.long),shape=torch.zeros(1,32,dtype=torch.long),
        phase_mask=torch.zeros(1,32,8,dtype=torch.bool),atoms=torch.zeros(1,32,2,dtype=torch.long),
        beats=torch.zeros(1,32,dtype=torch.float64),seconds=torch.zeros(1,32,dtype=torch.float64),
        gap_beats=torch.zeros(1,32,dtype=torch.float64),gap_seconds=torch.zeros(1,32,dtype=torch.float64),
        mask=torch.zeros(1,32,dtype=torch.bool))
    table['kind'][:,:4]=2;table['shape'][:,:4]=1;table['phase_mask'][:,:4,0]=True
    table['atoms'][0,:4,0]=torch.tensor([1,2,3,4]);table['mask'][:,:4]=True
    table['beats'][0,:4]=torch.tensor([0.,.5,1.,1.5]);table['seconds']=table['beats']/2
    table['gap_beats'][:,:4]=.5;table['gap_seconds'][:,:4]=.25
    for key,value in table.items():
        batch['phrase_sequence_'+key]=value
        batch['target_phrase_sequence_'+key]=value.clone()
    batch.update(target_phrase_sequence_inverse=torch.zeros(1,2,dtype=torch.long),
        target_phrase_sequence_positive_ids=torch.tensor([0]),
        target_phrase_sequence_fingerprint=torch.tensor([[123,456]]))
    beats=torch.arange(41)[None].float()/4
    cache=dict(beats=beats,seconds=beats/2,mask=torch.ones_like(beats,dtype=torch.bool),
        local=torch.randn(1,41,16),delta_keys=torch.randn(1,41,32))
    return model,batch,cache


def test_ordered_migration_preserves_parent_predictions_and_only_adds_ordered_prefix():
    model,batch,cache=ordered_fixture()
    parent=CausalPhraseContext(16,3).eval()
    with torch.no_grad():parent.intent_output.weight.normal_(0,.03)
    upgraded=CausalPhraseContext(16,3,ordered=True).eval()
    missing=upgraded.load_state_dict(parent.state_dict(),strict=False).missing_keys
    assert missing and all(name.startswith('ordered.') for name in missing)
    expected,original=parent(model,cache,torch.tensor([0]),batch)
    actual,output=upgraded(model,cache,torch.tensor([0]),batch)
    torch.testing.assert_close(expected,actual,atol=0,rtol=0)
    for key in ('phrase_intent','phrase_prediction','phrase_reference_weights'):
        torch.testing.assert_close(original[key],output[key],atol=0,rtol=0)
    assert torch.count_nonzero(upgraded.ordered.actor_residual(torch.randn(1,2,16),output,batch))==0


def test_ordered_encoding_preserves_order_and_shared_unique_table_is_encoded_once():
    model,batch,cache=ordered_fixture()
    module=CausalPhraseContext(16,3,ordered=True)
    before=module.ordered.encode_sequences(model,batch)
    changed=dict(batch,phrase_sequence_atoms=batch['phrase_sequence_atoms'].clone())
    changed['phrase_sequence_atoms'][0,:4]=changed['phrase_sequence_atoms'][0,:4].flip(0)
    after=module.ordered.encode_sequences(model,changed)
    assert not torch.allclose(before,after)
    calls=[]
    hook=module.ordered.sequence_gru.register_forward_pre_hook(lambda _,args:calls.append(args[0].shape))
    _,output=module(model,cache,torch.tensor([0]),batch)
    hook.remove()
    assert calls==[torch.Size([1,32,16])]
    torch.testing.assert_close(output['phrase_intent'][:,0],output['phrase_intent'][:,1])
    changed=dict(batch,target_phrase_sequence_atoms=torch.full_like(batch['target_phrase_sequence_atoms'],999999))
    _,future_changed=module(model,cache,torch.tensor([0]),changed)
    torch.testing.assert_close(output['phrase_intent'],future_changed['phrase_intent'],atol=0,rtol=0)


def test_ordered_auxiliary_cosine_teacher_and_multi_positive_preserve_query_budget():
    from maichart.composer.stream_phrase_model import ordered_phrase_auxiliary_loss
    model,batch,cache=ordered_fixture()
    module=CausalPhraseContext(16,3,ordered=True)
    model.phrase_context=module
    _,output=module(model,cache,torch.tensor([0]),batch)
    loss,metrics=ordered_phrase_auxiliary_loss(model,output,batch)
    assert metrics['phrase_ordered_match_loss']==0  # One label: no fake negatives.
    assert torch.isfinite(loss) and loss>0
    loss.backward()
    assert module.ordered.predict[-1].weight.grad.abs().sum()>0
    duplicate=dict(batch,target_phrase_weight=batch['target_phrase_weight']/2)
    half,_=ordered_phrase_auxiliary_loss(model,output,duplicate)
    torch.testing.assert_close(half,loss/2)
    repeated=dict(batch)
    for key,value in batch.items():
        if key.startswith('target_phrase_sequence_') and key not in ('target_phrase_sequence_inverse',
                'target_phrase_sequence_positive_ids','target_phrase_sequence_fingerprint'):
            repeated[key]=value.repeat(2,*([1]*(value.ndim-1)))
    repeated['target_phrase_sequence_positive_ids']=torch.tensor([7,7])
    repeated['target_phrase_sequence_fingerprint']=torch.tensor([[123,456],[123,456]])
    repeated_loss,repeated_metrics=ordered_phrase_auxiliary_loss(model,output,repeated)
    assert repeated_metrics['phrase_ordered_match_loss']==0
    torch.testing.assert_close(repeated_loss,loss)


def test_ordered_contrastive_targets_receive_loss_gradients_and_report_distinct_codes():
    from maichart.composer.stream_phrase_model import ordered_phrase_auxiliary_loss
    model,batch,cache=ordered_fixture()
    module=CausalPhraseContext(16,3,ordered=True)
    model.phrase_context=module
    for key,value in list(batch.items()):
        if key.startswith('target_phrase_sequence_') and key not in ('target_phrase_sequence_inverse',
                'target_phrase_sequence_positive_ids','target_phrase_sequence_fingerprint'):
            batch[key]=value.repeat(2,*([1]*(value.ndim-1)))
    batch['target_phrase_sequence_atoms'][1,:4]=batch['target_phrase_sequence_atoms'][1,:4].flip(0)
    batch['target_phrase_sequence_inverse']=torch.tensor([[0,1]])
    batch['target_phrase_sequence_positive_ids']=torch.tensor([0,1])
    batch['target_phrase_sequence_fingerprint']=torch.tensor([[123,456],[321,654]])
    # Only the supervised loss accesses the target tower; reference inputs stay
    # unchanged and cannot expose these target rows to actor forward.
    _,output=module(model,cache,torch.tensor([0]),batch)
    prediction=torch.randn_like(output['phrase_ordered_prediction'],requires_grad=True)
    output=dict(output,phrase_ordered_prediction=prediction)
    loss,metrics=ordered_phrase_auxiliary_loss(model,output,batch)
    assert metrics['phrase_ordered_code_variance']>0
    assert metrics['phrase_ordered_query_match_loss']>0
    assert metrics['phrase_ordered_target_match_loss']>0
    loss.backward()
    assert module.ordered.sequence_gru.weight_ih_l0.grad.abs().sum()>0
    assert torch.isfinite(module.ordered.sequence_gru.weight_ih_l0.grad).all()
    assert prediction.grad.abs().sum()>0
    # An all-masked recovery label table cannot create NaN or a label gradient.
    module.zero_grad(set_to_none=True)
    masked=dict(batch,target_phrase_mask=torch.zeros_like(batch['target_phrase_mask']))
    clean_prediction=torch.randn_like(prediction,requires_grad=True)
    masked_loss,_=ordered_phrase_auxiliary_loss(model,dict(output,phrase_ordered_prediction=clean_prediction),masked)
    assert masked_loss==0
    masked_loss.backward()
    assert torch.isfinite(clean_prediction.grad).all() and torch.count_nonzero(clean_prediction.grad)==0
    assert module.ordered.sequence_gru.weight_ih_l0.grad is not None
    assert torch.count_nonzero(module.ordered.sequence_gru.weight_ih_l0.grad)==0


def test_ordered_music_keys_are_reused_from_encode_audio_cache():
    from maichart.composer.event_set_codec import EventSetCodec
    from maichart.composer.stream_model import StreamComposer,StreamComposerConfig
    model=StreamComposer(StreamComposerConfig(d_model=16,nhead=4,layers=1,dim_feedforward=32,
        dropout=0.,audio_dim=6,phrase_context_policy='ordered_phrase_v2'),EventSetCodec.fit(['1','2'])).eval()
    calls=[]
    hook=model.phrase_context.ordered.audio_keys.register_forward_hook(lambda *_:calls.append(1))
    seconds=torch.arange(8)[None].float()/2
    cache=model.encode_audio(torch.randn(1,8,6),torch.ones(1,8,dtype=torch.bool),seconds,seconds*2)
    assert calls==[1]
    first=model.phrase_context.ordered.prepare_audio(cache)
    second=model.phrase_context.ordered.prepare_audio(cache)
    assert first is cache and second is cache
    assert calls==[1]
    hook.remove()


def test_unique_music_table_preserves_nonzero_ordered_outputs_and_gradients():
    import copy
    from maichart.composer.stream_phrase import collate_phrase_music
    model,batch,cache=ordered_fixture()
    batch['song_inverse']=torch.tensor([0])
    module=CausalPhraseContext(16,3,ordered=True)
    with torch.no_grad():
        for layer in (module.ordered.anchor_output,module.ordered.reference_output,
                      module.ordered.interaction_output):layer.weight.normal_(0,.03)
        module.intent_output.weight.normal_(0,.03)
    other=copy.deepcopy(module)
    other_model=copy.deepcopy(model)
    original_cache={key:value.clone().requires_grad_(key=='local') for key,value in cache.items()}
    unique_cache={key:value.clone().requires_grad_(key=='local') for key,value in cache.items()}
    from maichart.composer.stream_duration import prepare_interval_clock
    # A cached seconds clock must not be reused for the new beat-domain table.
    unique_cache['_duration_clock']=prepare_interval_clock(unique_cache)
    unique_batch={**batch,**collate_phrase_music(batch)}
    expected,expected_output=module(model,original_cache,torch.tensor([0]),batch)
    actual,actual_output=other(other_model,unique_cache,torch.tensor([0]),unique_batch)
    for a,b in ((expected,actual),*((expected_output[key],actual_output[key]) for key in
            ('phrase_intent','phrase_reference_weights','phrase_prediction','phrase_ordered_prediction'))):
        torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
    # Include the actual execution bridge as well as all nonzero readout paths.
    actor=torch.randn(1,2,16)
    expected_actor=module.ordered.actor_residual(actor,expected_output,batch)
    actual_actor=other.ordered.actor_residual(actor,actual_output,unique_batch)
    torch.testing.assert_close(expected_actor,actual_actor,atol=1e-5,rtol=1e-5)
    def objective(residual,output,actor_residual):
        return residual.square().mean()+output['phrase_ordered_prediction'].square().mean()+actor_residual.square().mean()
    objective(expected,expected_output,expected_actor).backward()
    objective(actual,actual_output,actual_actor).backward()
    torch.testing.assert_close(original_cache['local'].grad,unique_cache['local'].grad,atol=1e-5,rtol=1e-5)
    for (name,a),(other_name,b) in zip(module.named_parameters(),other.named_parameters()):
        assert name==other_name
        assert (a.grad is None)==(b.grad is None),name
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,atol=1e-5,rtol=1e-5,msg=name)
    # Each interval contributes nine reads. Repeated queries share the table.
    original_queries=9*(batch['phrase_start_beat'].numel()+batch['phrase_memory_start_beat'].numel())
    unique_queries=9*unique_batch['phrase_music_start_beat'].numel()
    assert original_queries==90 and unique_queries==27


def test_music_interval_deduplication_keeps_song_identity_and_exact_boundaries():
    from maichart.composer.stream_phrase import collate_phrase_music
    _,batch,_=ordered_fixture()
    for key in ('phrase_start_beat','phrase_end_beat','phrase_memory_start_beat','phrase_memory_end_beat'):
        batch[key]=batch[key].repeat(2,*([1]*(batch[key].ndim-1)))
    batch['song_inverse']=torch.tensor([0,1])
    packed=collate_phrase_music(batch)
    assert packed['phrase_music_start_beat'].numel()==6
    anchors=packed['phrase_music_anchor_inverse']
    assert torch.equal(anchors[:,0],anchors[:,1])
    assert anchors[0,0]!=anchors[1,0]
    for target,start,end in (('anchor','phrase_start_beat','phrase_end_beat'),
                             ('reference','phrase_memory_start_beat','phrase_memory_end_beat')):
        ids=packed['phrase_music_'+target+'_inverse']
        torch.testing.assert_close(packed['phrase_music_start_beat'][ids],batch[start].double(),atol=0,rtol=0)
        torch.testing.assert_close(packed['phrase_music_end_beat'][ids],batch[end].double(),atol=0,rtol=0)
