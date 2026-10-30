"""Contracts for learned continuation, distinct motion and physical Touch load."""
import math

import torch

from maichart.composer.event_set_codec import EventSetCodec
from maichart.composer.stream_model import StreamComposer,StreamComposerConfig,NOTE,WAIT,BOS


def tiny(**kwargs):
    kwargs.setdefault('event_family_policy','legacy_v1')
    codec=EventSetCodec.fit(['1','2','A1','A2','C','C1'])
    config=StreamComposerConfig(d_model=8,nhead=2,layers=1,dim_feedforward=16,
        audio_dim=6,condition_dim=3,dropout=0.,audio_summary_tokens=2,
        delta_beats=('0','1/4','1/2','1'),**kwargs)
    return StreamComposer(config,codec).eval()


def test_legacy_missing_metadata_keeps_strict_state_contract():
    old=dict(history_window_policy='rolling_legacy_v1',continuation_policy='off',
             sensor_load_policy='button_only_v1',gap_continuation_policy='additive_v1',
             lane_load_policy='off',
             motion_continuation_policy='motion_continuation_v1')
    model=tiny(**old)
    metadata=model.config.to_dict()
    for name in ('history_window_policy','continuation_policy','sensor_load_policy','gap_continuation_policy','lane_load_policy'):
        metadata.pop(name)
    restored=StreamComposer(StreamComposerConfig.from_dict(metadata),model.codec)
    restored.load_state_dict(model.state_dict(),strict=True)
    assert not hasattr(restored,'touch_load_projection')
    assert not hasattr(restored,'keep_gap_audio')
    assert not hasattr(restored,'lane_load_projection')
    assert restored.config.history_window_policy=='rolling_legacy_v1'


def test_copy_probability_is_gate_mass_not_an_additive_logit_bonus():
    model=tiny()
    with torch.no_grad():model.keep_gap_head.bias.fill_(math.log(3.))
    hidden=torch.zeros(1,3,8)
    batch=dict(history_delta=torch.tensor([[2,2,2]]),history_kind=torch.tensor([[NOTE,WAIT,BOS]]))
    allowed=torch.tensor([[[False,True,True,True]]]).expand(1,3,4)
    logits=torch.tensor([[[0.,-20.,4.,20.]]]).expand(1,3,4).clone().requires_grad_()
    out,repeat=model.repeat_delta_log_probs(hidden,batch,logits,allowed,{},torch.tensor([0]),symbolic_only=True)
    assert repeat.tolist()==[[2,-1,-1]]
    torch.testing.assert_close(out[0,0,2].exp(),torch.tensor(.75))
    torch.testing.assert_close(out.exp().sum(-1),torch.ones(1,3))
    torch.testing.assert_close(out[0,1],logits[0,1].masked_fill(~allowed[0,1],-1e4).log_softmax(-1).masked_fill(~allowed[0,1],-1e4))
    (-out[0,0,1]).backward()
    assert torch.isfinite(logits.grad).all()
    assert model.keep_gap_head.bias.grad.abs().sum()>0


def test_copy_invalid_or_only_legal_class_has_finite_normalized_support():
    model=tiny();hidden=torch.zeros(1,2,8)
    batch=dict(history_delta=torch.tensor([[2,2]]),history_kind=torch.tensor([[NOTE,NOTE]]))
    allowed=torch.tensor([[[False,True,False,True],[False,False,True,False]]])
    out,repeat=model.repeat_delta_log_probs(hidden,batch,torch.zeros(1,2,4),allowed,{},torch.tensor([0]),symbolic_only=True)
    assert repeat.tolist()==[[-1,2]]
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.exp().sum(-1),torch.ones(1,2))
    assert out[0,1,2]==0


def test_touch_load_uses_33_separate_physical_pads_and_c_alias():
    model=tiny();codec=model.codec
    ids={s:codec.encode(s)['atoms'][0] for s in ('1','A1','A2','C','C1')}
    assert model.atom_touch[ids['1']].sum()==0
    assert model.atom_lanes[ids['A1']].sum()==0
    assert model.atom_touch[ids['A1']].sum()==1
    assert not torch.equal(model.atom_touch[ids['A1']],model.atom_touch[ids['A2']])
    assert torch.equal(model.atom_touch[ids['C']],model.atom_touch[ids['C1']])
    atoms=torch.tensor([[[ids['A1']],[ids['A2']]]]);kinds=torch.full((1,2),NOTE)
    sec=torch.tensor([[0.,1.]],dtype=torch.float64)
    cursor=model._pad_load(model.atom_touch,atoms,kinds,sec)
    target=model._pad_load(model.atom_touch,atoms,kinds,sec,sec+1.)
    assert cursor.shape==(1,2,132)
    assert target[0,-1,:99].sum()<cursor[0,-1,:99].sum()
    assert torch.count_nonzero(model.touch_load_projection.weight)==0
    assert 'atom_touch' not in model.state_dict()


def test_distinct_motion_deduplicates_stationary_paths_and_preserves_orbits():
    kinds=torch.full((1,2),NOTE);shapes=torch.ones(1,2,dtype=torch.long)
    masks=torch.nn.functional.one_hot(torch.tensor([[2,2]]),8).bool()
    old=StreamComposer.motion_phase_features(kinds,shapes,masks)
    new=StreamComposer.motion_phase_features(kinds,shapes,masks,distinct=True)
    assert old[0,-1,:,2].tolist()==[1.,1.,1.]
    assert new[0,-1,:,2].tolist()==[0.,0.,1.]
    masks[:]=False;masks[:,:,[1,5]]=True
    base=StreamComposer.motion_phase_features(kinds,shapes,masks,distinct=True)
    assert base[0,-1].sum(-1).tolist()==[.5,.5,1.]
    for rotation in range(8):
        moved=StreamComposer.motion_phase_features(kinds,shapes,masks.roll(rotation,-1),distinct=True)
        torch.testing.assert_close(moved,base.roll(rotation,-1))
    masks=torch.nn.functional.one_hot(torch.tensor([[1,3]]),8).bool()
    moving=StreamComposer.motion_phase_features(kinds,shapes,masks,distinct=True)
    assert moving[0,-1].argmax(-1).tolist()==[5,1,3]


def test_direct_lane_load_averages_actual_decoded_heads_for_each_phase():
    from maichart.composer.events import event_metadata
    codec=EventSetCodec.fit(['1','1/5','1h[4:1]/5','A1'])
    config=StreamComposerConfig(d_model=8,nhead=2,layers=1,dim_feedforward=16,
        audio_dim=6,condition_dim=3,dropout=0.,audio_summary_tokens=2)
    model=StreamComposer(config,codec).eval()
    ctx=torch.zeros(1,8);ref=torch.zeros(1,dtype=torch.long);mask=torch.ones(1,8,dtype=torch.bool)
    load=torch.zeros(1,8,4);load[0,:,0]=torch.arange(8).float()
    for shape in range(codec.num_shapes):
        shapes=torch.tensor([shape])
        plain=model.event_phase_logits(ctx,shapes,ref,mask)
        zero=model.event_phase_logits(ctx,shapes,ref,mask,button_load=load)
        torch.testing.assert_close(zero,plain)
    with torch.no_grad():model.lane_load_projection.weight[0,0]=1.
    for shape in range(codec.num_shapes):
        shapes=torch.tensor([shape])
        plain=model.event_phase_logits(ctx,shapes,ref,mask)
        scored=model.event_phase_logits(ctx,shapes,ref,mask,button_load=load)
        if shape==0:
            torch.testing.assert_close(scored,plain)
            continue
        expected=[]
        for phase in range(8):
            text=codec.decode(shape,phase,[0]*len(codec.shape_catalog[shape]))
            heads={int(head)-1 for head in event_metadata(text).heads}
            expected.append(sum(heads)/len(heads))
        torch.testing.assert_close((scored-plain)[0],torch.tensor(expected))
    assert 'shape_button_lanes' not in model.state_dict()


def test_direct_lane_projection_gets_orientation_gradient_at_zero_init():
    model=tiny();shape=torch.tensor([model.codec.encode('1')['shape']])
    load=torch.zeros(1,8,4);load[0,0,0]=5.
    logits=model.event_phase_logits(torch.zeros(1,8),shape,torch.tensor([0]),
        torch.ones(1,8,dtype=torch.bool),button_load=load)
    (-logits.log_softmax(-1)[0,0]).backward()
    assert torch.isfinite(model.lane_load_projection.weight.grad).all()
    assert model.lane_load_projection.weight.grad.abs().sum()>0


def test_batched_decode_preserves_distinct_per_row_button_load():
    model=tiny();shape=model.codec.encode('1')['shape']
    # Remove other phase preferences so only the shared lane readout selects.
    with torch.no_grad():
        model.phase_head.weight.zero_();model.phase_head.bias.zero_()
        model.lane_load_projection.weight.fill_(1.)
    loads=torch.zeros(2,8,4);loads[0,2,0]=10.;loads[1,6,0]=10.
    events=model.decode_event(torch.zeros(2,8),shape_override=shape,temperature=0.,
        reference_phase_mask=torch.tensor([True]+[False]*7),button_load=loads)
    assert [x['phase'] for x in events]==[2,6]
