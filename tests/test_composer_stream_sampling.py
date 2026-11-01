from fractions import Fraction
from types import SimpleNamespace
import torch

from maichart.composer.event_set_codec import EventSetCodec
from maichart.composer.stream_sampling import sample_guarded_event


class Guard:
    def __init__(self): self.committed=[]
    def accepts(self,point): return point.note_text!='1'
    def commit(self,point): self.committed.append(point.note_text)


def test_guard_retry_excludes_failed_singleton_phase_and_forwards_orbit_motion():
    codec=EventSetCodec.fit(['1']);shape=codec.encode('1')['shape'];calls=[]
    mask=torch.ones(8,dtype=torch.bool);motion=torch.zeros(3,8)
    def decode(context,**kwargs):
        calls.append(kwargs)
        blocked=kwargs.get('blocked_shape_phases',set())
        phase=next((p for p in range(8) if (shape,p) not in blocked),None)
        if phase is None:return None
        return dict(shape=shape,phase=phase,details=[0],text=codec.decode(shape,phase,[0]))
    guard=Guard()
    point,retries=sample_guarded_event(SimpleNamespace(codec=codec,decode_event=decode),None,0,Fraction(1),guard,None,
        reference_phase_mask=mask,motion_phase_distribution=motion)
    assert point.note_text=='2' and retries==1 and guard.committed==['2']
    assert (shape,0) in calls[1]['blocked_shape_phases']
    assert calls[1]['reference_phase_mask'] is mask and calls[1]['motion_phase_distribution'] is motion


def test_fallback_phase_can_be_revisited_with_different_legal_details():
    calls=[]
    def decode(context,**kwargs):
        calls.append(kwargs)
        if (0,0) in kwargs.get('blocked_shape_phases',set()):return None
        return dict(shape=0,phase=0,details=[1,0],text='1' if len(calls)==1 else 'A1')
    guard=Guard()
    point,retries=sample_guarded_event(SimpleNamespace(decode_event=decode),None,0,Fraction(1),guard,None)
    assert point.note_text=='A1' and retries==1 and len(calls)==2
    assert all('blocked_shape_phases' not in options for options in calls)
