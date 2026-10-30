from fractions import Fraction
from types import SimpleNamespace
import torch
from maichart.composer.stream_sampling import sample_guarded_event


def test_failed_configuration_preserves_the_sampled_family_instead_of_switching_to_touch():
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
    assert len(draws)==1 and families==[1,1,1]


def test_infeasible_family_returns_control_to_time_selection_with_no_type_substitution():
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
