from fractions import Fraction
from types import SimpleNamespace

import torch
import pytest

from maichart.composer import stream_proposal
from maichart.composer.stream_sampling import sample_guarded_event
from maichart.composer.stream_train import parser, resolve_learning_policies
from maichart.event_timing import EventPoint


def test_timed_greedy_chooses_exit_mass_before_individual_candidate_and_passes_motion(monkeypatch):
    calls=[]
    motion=torch.tensor(-.7);shape=torch.tensor(3)
    def guarded(model,context,reference,beat,guard,generator,**kwargs):
        calls.append((context,beat,kwargs))
        return EventPoint(beat,'2'),0
    monkeypatch.setattr(stream_proposal,'sample_guarded_event',guarded)
    scores=torch.tensor([-torch.inf,.4,.3,.3]);scores[1:]=scores[1:].log()
    keep=torch.tensor([[[False,False,True,True],[False,True,False,False]]])
    result=stream_proposal.sample_timed_event(None,scores,[Fraction(i) for i in range(4)],Fraction(0),
        lambda index,beat:dict(context=index,motion_keep_logit=motion,motion_reference_shape=shape),
        0,None,None,None,pointer_temperature=0.,timing_keep_mask=keep)
    assert result[:2]==(EventPoint(Fraction(2),'2'),2)
    assert calls[0][2]['motion_keep_logit'] is motion
    assert calls[0][2]['motion_reference_shape'] is shape


def test_timing_retry_reconditions_remaining_keep_exit_mass(monkeypatch):
    chosen=[]
    def guarded(model,context,reference,beat,guard,generator,**kwargs):
        chosen.append(beat)
        return (None,8) if len(chosen)==1 else (EventPoint(beat,'1'),0)
    monkeypatch.setattr(stream_proposal,'sample_guarded_event',guarded)
    scores=torch.tensor([-torch.inf,.4,.3,.3]);scores[1:]=scores[1:].log()
    result=stream_proposal.sample_timed_event(None,scores,[Fraction(i) for i in range(4)],Fraction(0),
        lambda index,beat:index,0,None,None,None,pointer_temperature=0.,
        timing_keep_mask=torch.tensor([False,True,False,False]))
    assert chosen==[Fraction(2),Fraction(1)]
    assert result==(EventPoint(Fraction(1),'1'),1,8,1)


def test_guarded_geometry_retries_retain_candidate_motion_decision():
    seen=[];accepted=[]
    logit=torch.tensor(.3);shape=torch.tensor(7)
    def decode(context,**kwargs):
        seen.append(kwargs)
        return dict(shape=1,phase=0,text='1',details=[])
    def accepts(point):accepted.append(point);return len(accepted)>1
    model=SimpleNamespace(config=SimpleNamespace(event_family_policy='legacy_v1'),decode_event=decode)
    guard=SimpleNamespace(accepts=accepts,commit=lambda point:None)
    point,retries=sample_guarded_event(model,torch.zeros(4),0,Fraction(1),guard,None,
        motion_keep_logit=logit,motion_reference_shape=shape)
    assert point.note_text=='1' and retries==1
    assert len(seen)==2
    for options in seen:
        assert options['motion_keep_logit'] is logit
        assert options['motion_reference_shape'] is shape


@pytest.mark.parametrize('policy',['music_exit_v1','music_exit_v2','music_exit_v3'])
def test_guarded_retries_redraw_joint_template_and_family_intent(policy):
    seen=[];intents=[];accepted=[]
    logit=torch.tensor(-.2);shape=torch.tensor(1)
    def intent(context,**kwargs):intents.append(kwargs);return 2,False
    def decode(context,**kwargs):
        seen.append(kwargs)
        return dict(shape=2,phase=0,text='1-5[4:1]',family=2,details=[])
    def accepts(point):accepted.append(point);return len(accepted)>1
    model=SimpleNamespace(config=SimpleNamespace(event_family_policy='audio_prior_transition_v3',
        transition_policy=policy),sample_event_intent=intent,decode_event=decode)
    guard=SimpleNamespace(accepts=accepts,commit=lambda point:None)
    point,retries=sample_guarded_event(model,torch.zeros(4),0,Fraction(1),guard,None,
        family_logits=torch.zeros(31),template_keep_logit=logit,template_reference_shape=shape)
    assert point is not None and retries==1 and len(intents)==2
    assert intents[0]['template_keep_logit'] is logit
    for options in seen:
        assert options['family_override']==2
        assert options['template_keep_override'] is False
        assert options['template_reference_shape'] is shape


def test_new_training_stage_defaults_use_shared_state_without_exit_override():
    args=parser().parse_args(['--manifest','unused.json','--checkpoint-dir','unused'])
    policies=resolve_learning_policies(args)
    assert policies['transition_policy']=='off'
    assert policies['event_family_policy']=='shared_continuation_v1'
    assert policies['recovery_target_policy']=='causal_correspondence_v1'
    assert policies['gap_continuation_policy']=='off'
    assert policies['timing_grid_audio_policy']=='query_pool_v3'
    assert policies['prefix_augmentation_input_policy']=='musical_loop_recovery_v7'
    assert policies['recovery_policy']=='greedy_mixed_v4'


@pytest.mark.parametrize('prefix_policy',['musical_loop_recovery_v6','musical_loop_recovery_v7','musical_loop_recovery_v8'])
@pytest.mark.parametrize('transition_policy',['music_exit_v1','music_exit_v2','music_exit_v3'])
def test_resume_inherits_saved_long_history_course_version(prefix_policy,transition_policy):
    from maichart.composer.stream_model import StreamComposerConfig
    parent=StreamComposerConfig.from_dict(dict(prefix_augmentation_input_policy=prefix_policy,
        transition_policy=transition_policy,recovery_policy='greedy_mixed_v4',
        event_family_policy='audio_prior_transition_v3'))
    policies=resolve_learning_policies(SimpleNamespace(),parent,resume=True)
    assert policies['prefix_augmentation_input_policy']==prefix_policy
    assert policies['transition_policy']==transition_policy
    assert policies['recovery_policy']=='greedy_mixed_v4'


@pytest.mark.parametrize('source_policy,target_policy,reset',[
    ('music_exit_v1','music_exit_v2',True),
    ('music_exit_v1','music_exit_v3',True),
    ('music_exit_v2','music_exit_v3',True),
    ('music_exit_v1','music_exit_v1',False),
    ('music_exit_v2','music_exit_v2',False),
    ('music_exit_v3','music_exit_v3',False),
])
def test_learning_stage_resets_only_incompatible_exit_outputs(source_policy,target_policy,reset):
    from maichart.composer.event_set_codec import EventSetCodec
    from maichart.composer.stream_model import StreamComposer,StreamComposerConfig
    from maichart.composer.stream_train import upgrade_learning_stage
    parent=StreamComposer(StreamComposerConfig.from_dict(dict(d_model=8,nhead=2,layers=1,
        dim_feedforward=16,transition_policy=source_policy,
        event_family_policy='audio_prior_transition_v3')),EventSetCodec.fit(['1','2']))
    with torch.no_grad():
        parent.music_exit_head[0].weight.fill_(.17)
        parent.music_exit_head[0].bias.fill_(.23)
        parent.music_exit_head[2].weight.fill_(.9)
        parent.music_exit_head[2].bias.fill_(.7)
    upgraded=upgrade_learning_stage(parent,dict(transition_policy=target_policy))
    for key,value in parent.state_dict().items():
        actual=upgraded.state_dict()[key]
        if reset and key in ('music_exit_head.2.weight','music_exit_head.2.bias'):
            assert torch.count_nonzero(actual)==0,key
        else:torch.testing.assert_close(actual,value,rtol=0,atol=0)


def test_vector_query_upgrade_keeps_the_trained_scalar_and_adds_zero_vector_readout():
    from maichart.composer.event_set_codec import EventSetCodec
    from maichart.composer.stream_model import StreamComposer,StreamComposerConfig
    from maichart.composer.stream_train import upgrade_learning_stage
    parent=StreamComposer(StreamComposerConfig.from_dict(dict(d_model=8,nhead=2,layers=1,
        dim_feedforward=16,timing_output_policy='absolute_grid_v1',timing_grid_audio_policy='query_pool_v2')),
        EventSetCodec.fit(['1','2']))
    with torch.no_grad():parent.timing_grid.query_projection.weight.fill_(.37)
    upgraded=upgrade_learning_stage(parent,dict(timing_grid_audio_policy='query_pool_v3'))
    before=parent.state_dict();after=upgraded.state_dict()
    assert set(after)-set(before)=={'timing_grid.query_interaction_projection.weight'}
    assert torch.count_nonzero(upgraded.timing_grid.query_interaction_projection.weight)==0
    for key,value in before.items():torch.testing.assert_close(after[key],value,rtol=0,atol=0)
