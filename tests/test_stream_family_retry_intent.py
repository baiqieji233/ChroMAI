from fractions import Fraction
from types import SimpleNamespace
import torch
from maichart.composer.stream_sampling import sample_guarded_event


def test_failed_configuration_redraws_the_music_family_at_the_same_onset():
    families=[];draws=[];accepted=[]
    def choose(context,**kwargs):draws.append(context);return 1
    def decode(context,**kwargs):
        families.append(kwargs['family_override'])
        return dict(shape=1,phase=0,family=kwargs['family_override'],details=[],text='1')
    def accepts(point):accepted.append(point);return len(accepted)>=3
    model=SimpleNamespace(config=SimpleNamespace(event_family_policy='audio_semantic_v2'),
        sample_event_family=choose,decode_event=decode)
    guard=SimpleNamespace(accepts=accepts,commit=lambda point:None)
    point,retries=sample_guarded_event(model,torch.zeros(4),0,Fraction(1),guard,None,
        family_context=torch.ones(4))
    assert point.note_text=='1' and retries==2
    assert len(draws)==3 and families==[1,1,1]
    assert [trial['accepted'] for trial in guard.last_sampling_trace]==[False,False,True]


def test_unaccepted_joint_proposals_exhaust_exactly_eight_attempts():
    families=[]
    def decode(context,**kwargs):
        families.append(kwargs['family_override'])
        return dict(shape=1,phase=0,family=kwargs['family_override'],details=[],text='1')
    model=SimpleNamespace(config=SimpleNamespace(event_family_policy='audio_semantic_v2'),
        sample_event_family=lambda *a,**kw:1,decode_event=decode)
    guard=SimpleNamespace(accepts=lambda point:False,commit=lambda point:None)
    point,retries=sample_guarded_event(model,torch.zeros(4),0,Fraction(1),guard,None,
        family_context=torch.ones(4))
    assert point is None and retries==8 and families==[1]*8
    assert len(guard.last_sampling_trace)==8


def test_rejected_family_does_not_force_a_timing_change_or_mask_other_details():
    families=iter([2,1]);seen=[];committed=[]
    def decode(context,**kwargs):
        seen.append(kwargs)
        return dict(shape=0,phase=0,family=kwargs['family_override'],details=[0],
            text='1h[4:1]' if kwargs['family_override']==2 else '2')
    model=SimpleNamespace(config=SimpleNamespace(event_family_policy='audio_semantic_v2'),
        sample_event_family=lambda *a,**kw:next(families),decode_event=decode)
    guard=SimpleNamespace(accepts=lambda point:point.note_text=='2',commit=committed.append)
    point,retries=sample_guarded_event(model,None,0,Fraction(3,2),guard,None,
        family_logits=torch.zeros(3),temperature=.65)
    assert point.beat==Fraction(3,2) and point.note_text=='2' and retries==1
    assert len(committed)==1
    assert all('blocked_shape_phases' not in options for options in seen)
    assert [trial['family'] for trial in guard.last_sampling_trace]==[2,1]
    assert all(options['temperature']==.65 for options in seen)


def _real_greedy_duration_model():
    from maichart.composer.event_set_codec import EventSetCodec
    from maichart.composer.stream_model import StreamComposer,StreamComposerConfig
    codec=EventSetCodec.fit(['1-5[64:1]','1-5[4:1]','1'])
    short=codec.encode('1-5[64:1]');long=codec.encode('1-5[4:1]')
    assert short['shape']==long['shape'] and short['phase']==long['phase']
    model=StreamComposer(StreamComposerConfig(d_model=8,nhead=2,layers=1,dim_feedforward=16,
        detail_policy='phase_autoregressive_v2',event_family_policy='audio_semantic_v2',transition_policy='off'),codec).eval()
    with torch.no_grad():
        for parameter in model.parameters():parameter.zero_()
        model.literal_route_head.bias.fill_(-100.)
        model.absolute_phase_bias.fill_(-100.);model.absolute_phase_bias[short['phase']]=0.
        for shape in model.detail_bias:
            for bias in shape:bias.fill_(-100.)
        for slot,bias in enumerate(model.detail_bias[short['shape']]):
            bias[long['details'][slot]]=0.
            bias[short['details'][slot]]=.1
    scores=torch.full((31,),-torch.inf)
    family=model._semantic_shape_families[short['shape']]
    scores[family-1]=0.
    return model,codec,short,long,scores


def _real_reference_mask(codec):
    reference=codec.encode('1')
    return torch.tensor(codec.phase_equivalents(reference['shape'],reference['phase'],reference['details']),dtype=torch.bool)


def test_real_greedy_decoder_retries_same_slide_phase_with_another_duration():
    model,codec,short,long,scores=_real_greedy_duration_model()
    context=torch.zeros(8);trials=[]
    mask=_real_reference_mask(codec)
    first=model.decode_event(context,temperature=0.,family_logits=scores,
        family_override=model._semantic_shape_families[short['shape']],reference_phase_mask=mask)
    def accepts(point):
        trials.append(codec.encode(point.note_text))
        return trials[-1]['details']==long['details']
    guard=SimpleNamespace(accepts=accepts,commit=lambda p:None)
    point,retries=sample_guarded_event(model,context,0,Fraction(7,4),guard,None,
        family_logits=scores,temperature=0.,reference_phase_mask=mask)
    assert point is not None and point.beat==Fraction(7,4) and 0<retries<8
    assert trials[0]['details']==first['details']==short['details']
    assert all(row['shape']==short['shape'] and row['phase']==short['phase'] for row in trials)
    assert trials[-1]['details']==long['details']
    assert guard.last_sampling_trace[-1]['queue_peak']<=64


def test_real_greedy_decoder_can_explore_another_family_at_the_same_onset():
    model,codec,short,long,scores=_real_greedy_duration_model()
    tap=codec.encode('1');tap_family=model._semantic_shape_families[tap['shape']]
    scores[tap_family-1]=-1.
    with torch.no_grad():
        for slot,bias in enumerate(model.detail_bias[short['shape']]):
            bias.fill_(-100.);bias[short['details'][slot]]=0.
    guard=SimpleNamespace(accepts=lambda p:codec.encode(p.note_text)['shape']==tap['shape'],
        commit=lambda p:None)
    point,retries=sample_guarded_event(model,torch.zeros(8),0,Fraction(5,4),guard,None,
        family_logits=scores,temperature=0.,reference_phase_mask=_real_reference_mask(codec))
    assert point is not None and point.beat==Fraction(5,4) and 0<retries<8
    trace=guard.last_sampling_trace
    assert trace[0]['family']!=trace[-1]['family']==tap_family


def test_deterministic_frontier_partitions_do_not_replay_rejected_leaves():
    from maichart.composer.stream_sampling import GreedyDecisionFrontier
    frontier=GreedyDecisionFrontier();leaves=[]
    while frontier.begin():
        leaf=tuple(frontier.choose(torch.tensor([.2,0.])) for _ in range(3))
        assert leaf not in leaves
        leaves.append(leaf);frontier.reject()
    assert leaves[0]==(0,0,0) and len(leaves)==8
