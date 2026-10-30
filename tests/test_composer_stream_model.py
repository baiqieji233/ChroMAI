import copy
import math

import torch

from maichart.composer.event_set_codec import EventSetCodec
from maichart.composer.stream_model import StreamComposer, StreamComposerConfig, to_device_batch, BOS, NOTE, WAIT, EOS


def fixture(*,timing_output_policy='relative_delta_v1',event_family_policy='legacy_v1'):
    torch.manual_seed(18)
    codec = EventSetCodec.fit(['1/5', '1h[4:1]/5h[8:1]', 'A1/Cf'])
    config = StreamComposerConfig(d_model=16, nhead=4, layers=1, dim_feedforward=32,
        dropout=0., audio_dim=6, condition_dim=3, audio_summary_tokens=4,rollout_steps=4,
        delta_beats=('0', '0', '1/2', '1', '2'), max_event_atoms=4,
        timing_output_policy=timing_output_policy,event_family_policy=event_family_policy)
    model = StreamComposer(config, codec).eval()
    regular, literal = codec.encode('1/5'), codec.encode('A1/Cf')
    details = torch.full((1, 4, 3), -1, dtype=torch.long)
    details[0, 0, :len(regular['details'])] = torch.tensor(regular['details'])
    details[0, 2, :len(literal['details'])] = torch.tensor(literal['details'])
    if literal['details'][-1] != 0:
        details[0, 2, len(literal['details'])] = 0
    atoms = torch.zeros(1, 4, 3, dtype=torch.long)
    atoms[0, 1, :2] = torch.tensor(regular['atoms'])
    atoms[0, 3, :2] = torch.tensor(literal['atoms'])
    phase_mask = torch.ones(1, 4, 8, dtype=torch.bool)
    phase_mask[0, 0] = torch.tensor(regular['phase_mask'])
    history_phase_mask=torch.zeros(1,4,8,dtype=torch.bool)
    history_phase_mask[0,1]=torch.tensor(codec.encode('4/8')['phase_mask'])
    history_phase_mask[0,3]=torch.tensor(literal['phase_mask'])
    batch = dict(song_audio=torch.randn(1, 12, 6), song_mask=torch.ones(1, 12, dtype=torch.bool),
        song_seconds=torch.arange(12)[None].float()*.5, song_beats=torch.arange(12)[None].float(),
        song_inverse=torch.tensor([0]), condition=torch.zeros(1, 3),
        history_kind=torch.tensor([[BOS, NOTE, WAIT, NOTE]]),
        history_shape=torch.tensor([[0, regular['shape'], 0, 0]]),
        history_phase=torch.tensor([[0, 3, 0, 0]]), history_delta=torch.tensor([[0, 2, 4, 2]]),
        history_phase_mask=history_phase_mask,
        history_atoms=atoms, step_second=torch.tensor([[0., .5, 1.5, 2.]]),
        step_beat=torch.tensor([[0., 1., 3., 4.]]), step_mask=torch.ones(1, 4, dtype=torch.bool),
        target_second=torch.tensor([[.5, 1.5, 2., 3.]]),audio_end_sec=torch.tensor([3.]),audio_end_beat=torch.tensor([6.]),
        target_kind=torch.tensor([[NOTE, WAIT, NOTE, EOS]]), target_delta=torch.tensor([[3, 4, 3, 0]]),
        target_shape=torch.tensor([[regular['shape'], 0, 0, 0]]), target_phase_mask=phase_mask,
        target_details=details, loss_mask=torch.ones(1, 4), sample_weight=torch.ones(1, 4),
        action_state=torch.zeros(1, 4, 2, 50), action_state_mask=torch.ones(1, 4, 2, dtype=torch.bool),
        audio_plan_target=torch.zeros(1, 12, 3), audio_plan_weight=torch.ones(1, 12)/12,
        _detail_groups=[(regular['shape'], torch.tensor([0]))], _literal_indices=torch.tensor([2]))
    return model, batch


def test_transition_exit_reuses_full_regular_and_literal_joint_likelihood():
    model,batch=fixture()
    out=model(batch)
    _,base=model.loss(out,batch)
    exits=dict(batch,transition_exit_mask=torch.ones_like(batch['target_kind'],dtype=torch.bool),
               transition_exit_labels=torch.ones_like(batch['target_kind'],dtype=torch.bool))
    total,metrics=model.loss(out,exits)
    expected=.5*(base['time_loss']+base['configuration_loss'])
    torch.testing.assert_close(metrics['transition_exit_loss'],expected)
    torch.testing.assert_close(metrics['base_loss'],base['base_loss'])
    torch.testing.assert_close(total,base['loss']+expected)
    assert metrics['exit_count']==4
    assert metrics['exit_note_count']==2 and metrics['exit_wait_count']==1 and metrics['exit_eos_count']==1
    total.backward()
    assert model.literal_head.weight.grad.abs().sum()>0
    assert model.phase_head.weight.grad.abs().sum()>0
    # A continuation label does not add loss or change the original metrics.
    exits['transition_exit_labels'].zero_()
    _,continuing=model.loss(out,exits)
    assert continuing['transition_exit_loss']==0
    assert continuing['exit_count']==0


def test_action_context_returns_reused_per_lane_target_time_load():
    model,batch=fixture()
    cache=model.encode_audio(batch['song_audio'],batch['song_mask'],batch['song_seconds'],batch['song_beats'])
    out=model(batch,audio_cache=cache)
    assert out['button_load'].shape==(1,4,8,4)
    original=model._prefix_load
    calls=[]
    def counted(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    model._prefix_load=counted
    ctx,load=model.action_context_at(out['hidden'],batch['target_second'],cache,
        batch['song_inverse'],batch,return_button_load=True)
    assert len(calls)==1
    torch.testing.assert_close(load,out['button_load'])
    torch.testing.assert_close(ctx,out['action_context'])
    delta_ctx,delta_load=model.action_context_for_delta(out,batch,batch['target_delta'],cache,
        event_seconds=batch['target_second'],return_button_load=True)
    torch.testing.assert_close(delta_ctx,ctx)
    torch.testing.assert_close(delta_load,load)


def test_targets_cannot_change_time_logits_or_prefix_hidden():
    model, batch = fixture()
    before = model(batch)
    changed = copy.deepcopy(batch)
    changed['target_second'] += 1.
    changed['target_shape'].zero_()
    after = model(changed)
    for name in ('kind_logits', 'delta_logits', 'hidden'):
        torch.testing.assert_close(before[name], after[name], atol=0, rtol=0)
    assert not torch.equal(before['action_context'], after['action_context'])


def test_parallel_teacher_forcing_is_causal_and_matches_prefix():
    model, batch = fixture()
    full = model(batch)
    altered = copy.deepcopy(batch)
    altered['history_atoms'][:, 2:] = 0
    altered['history_kind'][:, 2:] = BOS
    changed = model(altered)
    torch.testing.assert_close(full['hidden'][:, :2], changed['hidden'][:, :2], atol=1e-6, rtol=1e-6)
    prefix = {k: v[:, :2] if (k.startswith(('history_', 'target_', 'step_')) or k in (
        'action_state', 'action_state_mask', 'loss_mask', 'sample_weight')) else v for k, v in batch.items()}
    short = model(prefix)
    torch.testing.assert_close(full['hidden'][:, :2], short['hidden'], atol=1e-6, rtol=1e-6)
    assert full['reference_phase'][0, 2] == 3  # WAIT retains the previous NOTE's phase.
    assert full['reference_phase'][0, 3] == 3  # Literal Touch also retains that phase.


def test_full_loss_trains_regular_fallback_time_and_uniform_plan():
    model, batch = fixture()
    out = model(batch)
    loss, metrics = model.loss(out, batch)
    loss.backward()
    assert torch.isfinite(loss)
    for head in (model.kind_head, model.delta_head, model.shape_head,
                 model.phase_head, model.literal_head, model.plan_head):
        assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
        assert head.weight.grad.abs().sum() > 0
    assert metrics['detail_loss'] > 0


def test_phase_loss_marginalizes_full_event_stabilizer():
    model, batch = fixture()
    batch['loss_mask'][:, 1:] = 0
    out = model(batch)
    out['phase_logits'] = torch.zeros_like(out['phase_logits'], requires_grad=True)
    _, metrics = model.loss(out, batch)
    assert torch.isclose(metrics['phase_loss'], torch.tensor(math.log(4.)/4))


def test_inference_cache_sampled_time_and_checkpoint_are_supported():
    model, batch = fixture()
    cache = model.encode_audio(batch['song_audio'], batch['song_mask'], batch['song_seconds'], batch['song_beats'])
    inference = {k: v for k, v in batch.items() if not k.startswith('target_')}
    out = model(inference, audio_cache=cache)
    sampled = torch.full_like(batch['history_delta'], 3)
    action = model.action_context_for_delta(out, inference, sampled, cache)
    direct = model.action_context_at(out['hidden'], batch['step_second']+.5, cache, batch['song_inverse'],batch)
    torch.testing.assert_close(action, direct)
    event = model.decode_event(action[0, 0], reference_phase=int(out['reference_phase'][0, 0]),
                               reference_phase_mask=out['reference_phase_mask'][0,0],temperature=0)
    assert event['text'] == model.codec.decode(event['shape'], event['phase'], event['details'])
    restored = StreamComposer(StreamComposerConfig.from_dict(model.config.to_dict()), model.codec).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(inference)['kind_logits'], out['kind_logits'])


def test_audio_padding_does_not_change_reads_or_global_pool():
    model, batch = fixture()
    cache = model.encode_audio(batch['song_audio'], batch['song_mask'], batch['song_seconds'], batch['song_beats'])
    # Keep summary-token count fixed and append arbitrary padded values/clocks.
    pad = dict(song_audio=torch.cat((batch['song_audio'], torch.randn(1, 4, 6)), 1),
        song_mask=torch.cat((batch['song_mask'], torch.zeros(1, 4, dtype=torch.bool)), 1),
        song_seconds=torch.cat((batch['song_seconds'], torch.zeros(1, 4)), 1),
        song_beats=torch.cat((batch['song_beats'], torch.zeros(1, 4)), 1))
    other = model.encode_audio(**pad)
    query = torch.tensor([[0., 1.2, 8.]])
    torch.testing.assert_close(model._read_audio(cache, query, batch['song_inverse']),
                               model._read_audio(other, query, batch['song_inverse']))
    torch.testing.assert_close(cache['global'], other['global'], atol=1e-6, rtol=1e-6)


def test_symbolic_pretraining_bypasses_audio_and_plan_but_trains_symbols():
    model, batch = fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError('Symbolic pretraining must not encode the audio bank')
    model.encode_audio = forbidden
    before = model(batch, symbolic_only=True)
    altered = copy.deepcopy(batch)
    altered['song_audio'] = altered['song_audio']*100+123
    altered['target_second'] += 3
    altered['audio_plan_target'] += 10
    altered['symbolic_only'] = True
    after = model(altered)
    for key in ('kind_logits', 'delta_logits'):
        torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
    assert 'plan_logits' not in before
    loss, metrics = model.loss(before, batch)
    loss.backward()
    assert metrics['plan_loss'] == 0
    assert model.kind_embedding.weight.grad.abs().sum() > 0
    assert model.audio_input[0].weight.grad is None


def test_sparse_groups_move_with_nested_batch_without_changing_shape_ids():
    _, batch = fixture()
    moved = to_device_batch(batch, 'cpu')
    assert isinstance(moved['_detail_groups'][0][0], int)
    torch.testing.assert_close(moved['_detail_groups'][0][1], batch['_detail_groups'][0][1])
    torch.testing.assert_close(moved['_literal_indices'], batch['_literal_indices'])


def test_fixed_window_weights_are_not_cancelled_by_batch_normalization():
    model, batch = fixture()
    out = model(batch)
    _, whole = model.loss(out,batch)
    half = dict(batch,sample_weight=batch['sample_weight']*.5,
                audio_plan_weight=batch['audio_plan_weight']*.5)
    _, divided = model.loss(out,half)
    for key in ('time_loss','configuration_loss','plan_loss','loss'):
        torch.testing.assert_close(divided[key],whole[key]*.5)
    recovery = dict(batch,_loss_normalizer=1)
    _, single = model.loss(out,recovery)
    torch.testing.assert_close(single['time_loss'],whole['time_loss']*4)


def test_cursor_uses_physical_delta_values_and_real_song_end():
    model,batch=fixture()
    captured=[]
    hook=model.cursor_clock.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
    batch['audio_end_sec']=torch.tensor([10.])
    batch['audio_end_beat']=torch.tensor([20.])
    model(batch,symbolic_only=True)
    hook.remove()
    clock=captured[0]
    torch.testing.assert_close(clock[...,-3],model.delta_values[batch['history_delta']].float().log1p())
    torch.testing.assert_close(clock[...,-2],(10.-batch['step_second']).log1p())
    torch.testing.assert_close(clock[...,-1],(20.-batch['step_beat']).log1p())


def test_wait_and_eos_do_not_train_note_delta_but_notes_do():
    model,batch=fixture()
    out=model(batch)
    out['delta_logits'].retain_grad()
    loss,_=model.loss(out,batch)
    loss.backward()
    assert out['delta_logits'].grad[0,1].eq(0).all()
    assert out['delta_logits'].grad[0,3].eq(0).all()
    assert out['delta_logits'].grad[0,0].abs().sum()>0
    assert out['delta_logits'].grad[0,2].abs().sum()>0


def test_training_time_support_matches_generation_sampling():
    from fractions import Fraction
    from maichart.composer.stream_sampling import kind_logits_for_audio,note_delta_logits_for_audio
    from maichart.composer.stream_data import StreamInferenceContext
    from maichart.event_timing import EventTempo
    from maichart.tempo_clock import TempoClock
    model,batch=fixture()
    out=model(batch)
    clock=TempoClock([EventTempo(Fraction(0),120.)],0.)
    for i in range(4):
        cursor=Fraction(str(float(batch['step_beat'][0,i])))
        kinds=kind_logits_for_audio(torch.zeros(5),cursor,Fraction(6))
        deltas=note_delta_logits_for_audio(torch.zeros(len(model.config.delta_beats)),model.config.delta_beats,
            cursor,Fraction(6),clock,3.,first_step=i==0)
        assert torch.equal(out['kind_logits'][0,i]>-9999,torch.isfinite(kinds))
        assert torch.equal(out['delta_logits'][0,i]>-9999,torch.isfinite(deltas))
    assert torch.isfinite(out['kind_logits']).all() and torch.isfinite(out['delta_logits']).all()


def test_unknown_phase_targets_and_padding_have_finite_zero_loss():
    model,batch=fixture()
    batch['sample_weight'].zero_()
    batch['target_phase_mask'].zero_()
    batch['audio_plan_weight'].zero_()
    batch['_detail_groups']=[]
    batch['_literal_indices']=torch.empty(0,dtype=torch.long)
    # Every delta can be disallowed at the file end, yet inactive rows remain finite.
    batch['step_beat'][:,-1]=batch['audio_end_beat'][0]
    batch['step_mask'][:,-1]=False
    out=model(batch)
    loss,metrics=model.loss(out,batch)
    assert torch.isfinite(loss) and loss==0
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_opposite_pair_rotation_has_constant_orbit_movement_across_wrap():
    model,_=fixture()
    events=[model.codec.encode(text) for text in ('1/5','2/6','3/7','4/8','1/5')]
    kinds=torch.full((1,len(events)),NOTE)
    shapes=torch.tensor([[event['shape'] for event in events]])
    phase=torch.tensor([[event['phase'] for event in events]])
    masks=torch.tensor([[event['phase_mask'] for event in events]])
    embedded,movement,_=model.history_phase_features(kinds,shapes,phase,masks)
    expected=(model.phase_change.weight[1]+model.phase_change.weight[5])*.5
    torch.testing.assert_close(movement[0,1:],expected.expand(4,-1))
    assert movement[0,0].eq(0).all()
    torch.testing.assert_close(embedded[0,0],(model.phase_embedding.weight[0]+model.phase_embedding.weight[4])*.5)
    # Choosing the other equivalent canonical representative cannot affect inputs.
    changed=model.history_phase_features(kinds,shapes,(phase+4)%8,masks)
    torch.testing.assert_close(changed[0],embedded)
    torch.testing.assert_close(changed[1],movement)


def test_full_hold_details_prevent_false_phase_symmetry_and_wait_keeps_orbit():
    model,_=fixture()
    event=model.codec.encode('1h[4:1]/5h[8:1]')
    assert sum(event['phase_mask'])==1
    kinds=torch.tensor([[NOTE,WAIT,NOTE]])
    shapes=torch.tensor([[event['shape'],0,event['shape']]])
    phases=torch.tensor([[event['phase'],0,event['phase']]])
    masks=torch.tensor([[event['phase_mask'],[False]*8,event['phase_mask']]])
    embedded,movement,_=model.history_phase_features(kinds,shapes,phases,masks)
    torch.testing.assert_close(embedded[0,0],model.phase_embedding.weight[event['phase']])
    assert embedded[0,1].eq(0).all() and movement[0,1].eq(0).all()
    torch.testing.assert_close(movement[0,2],model.phase_change.weight[0])


def test_old_checkpoint_phase_policy_retains_canonical_input_behavior():
    model,_=fixture()
    legacy=model.config.to_dict();legacy.pop('history_phase_policy')
    model.config=StreamComposerConfig.from_dict(legacy)
    assert model.config.history_phase_policy=='canonical_v1'
    kinds=torch.tensor([[NOTE,NOTE]])
    shapes=torch.ones_like(kinds)
    phase=torch.tensor([[3,0]])
    embedded,movement,_=model.history_phase_features(kinds,shapes,phase)
    torch.testing.assert_close(embedded,model.phase_embedding(phase))
    torch.testing.assert_close(movement[0,1],model.phase_change.weight[5])


def test_orbit_output_is_independent_of_equivalent_canonical_reference():
    model,_=fixture()
    context=torch.randn(2,model.config.d_model)
    shape=torch.ones(2,dtype=torch.long)
    mask=torch.tensor([[True,False,False,False,True,False,False,False]]*2)
    left=model.event_phase_logits(context,shape,torch.tensor([0,0]),mask)
    right=model.event_phase_logits(context,shape,torch.tensor([4,4]),mask)
    torch.testing.assert_close(left,right,atol=0,rtol=0)


def test_reference_head_learns_contextual_choice_after_symmetric_pair():
    base,_=fixture()
    codec=EventSetCodec.fit(['1/5','1'])
    model=StreamComposer(base.config,codec).eval()
    with torch.no_grad():
        model.reference_phase_head.weight.zero_();model.reference_phase_head.bias.zero_()
        model.phase_head.weight.zero_();model.phase_head.bias.fill_(-8)
        model.phase_head.bias[1]=8
    context=torch.zeros(2,model.config.d_model);context[0,0]=1;context[1,0]=-1
    shape=torch.full((2,),codec.encode('1')['shape'],dtype=torch.long)
    mask=torch.tensor([[True,False,False,False,True,False,False,False]]*2)
    refs=torch.zeros(2,dtype=torch.long)
    before=model.event_phase_logits(context,shape,refs,mask)
    torch.testing.assert_close(before[0],before[1])
    optimizer=torch.optim.SGD(model.reference_phase_head.parameters(),lr=2.)
    loss=torch.nn.functional.cross_entropy(before,torch.tensor([1,5]))
    loss.backward();optimizer.step()
    after=model.event_phase_logits(context,shape,refs,mask)
    assert after[0,1]>after[0,5] and after[1,5]>after[1,1]
    assert model.reference_phase_head.weight.grad.abs().sum()>0


def test_single_reference_matches_old_probabilities_and_legacy_strict_state():
    model,_=fixture()
    legacy_dict=model.config.to_dict();legacy_dict.pop('phase_output_policy')
    legacy=StreamComposer(StreamComposerConfig.from_dict(legacy_dict),model.codec).eval()
    assert not hasattr(legacy,'reference_phase_head')
    old_state={key:value for key,value in model.state_dict().items() if not key.startswith('reference_phase_head.')}
    legacy.load_state_dict(old_state,strict=True)
    context=torch.randn(2,model.config.d_model)
    shape=torch.ones(2,dtype=torch.long);reference=torch.tensor([1,6])
    mask=torch.nn.functional.one_hot(reference,8).bool()
    actual=model.event_phase_logits(context,shape,reference,mask).log_softmax(-1)
    expected=legacy.event_phase_logits(context,shape,reference).log_softmax(-1)
    torch.testing.assert_close(actual,expected)
