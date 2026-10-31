import torch

from maichart.composer.event_set_codec import EventSetCodec
from maichart.composer.stream_model import StreamComposer,StreamComposerConfig,NOTE


def model():
    codec=EventSetCodec.fit(['1','2','1/5','1b/5','2/6','1h[4:1]/5','1/5h[4:1]'])
    return StreamComposer(StreamComposerConfig(d_model=8,nhead=2,layers=1,
        dim_feedforward=16,event_family_policy='legacy_v1',transition_policy='off'),codec)


def test_modifier_asymmetry_cannot_split_one_physical_chord_between_keep_and_exit():
    subject=model();plain=subject.codec.encode('1/5');modified=subject.codec.encode('1b/5')
    assert plain['shape']==modified['shape']
    assert sum(plain['phase_mask'])==2 and sum(modified['phase_mask'])==1
    shape=torch.tensor([[plain['shape'],modified['shape']]])
    distribution=subject.motion_phase_features(torch.full((1,2),NOTE),shape,
        torch.tensor([[plain['phase_mask'],modified['phase_mask']]]),distinct=True,adjacent=True)[:,-1]
    raw=distribution[...,0,:].gt(0)
    assert raw.sum()==1  # The old direct support split the target orbit.
    keep=subject.motion_keep_mask(shape[:,-1],distribution,shape[:,-1])
    assert keep.sum()==2
    assert keep[0,torch.tensor(plain['phase_mask'])].all()


def test_head_types_are_preserved_when_building_shape_stabilizers():
    subject=model();same=subject.codec.encode('1/5')['shape'];mixed=subject.codec.encode('1h[4:1]/5')['shape']
    assert subject.shape_motion_stabilizer[same].nonzero().flatten().tolist()==[0,4]
    assert subject.shape_motion_stabilizer[mixed].nonzero().flatten().tolist()==[0]
    distribution=torch.zeros(1,3,8);distribution[0,0,2]=1.
    keep=subject.motion_keep_mask(torch.tensor([mixed]),distribution,torch.tensor([mixed]))
    assert keep.nonzero().tolist()==[[0,2]]


def test_motion_orbit_closure_broadcasts_over_batch_and_time_and_requires_reference():
    subject=model();pair=subject.codec.encode('1/5')['shape'];single=subject.codec.encode('1')['shape']
    shapes=torch.tensor([[pair,single],[pair,single]])
    reference=torch.tensor([[pair,single],[-1,0]])
    distribution=torch.zeros(2,2,3,8);distribution[...,0,1]=1.
    keep=subject.motion_keep_mask(shapes,distribution,reference)
    assert keep.shape==(2,2,8) and keep.dtype==torch.bool
    assert keep[0,0].nonzero().flatten().tolist()==[1,5]
    assert keep[0,1].nonzero().flatten().tolist()==[1]
    assert not keep[1].any()
    assert subject.motion_keep_mask(shapes,None,reference) is None
    assert subject.motion_keep_mask(shapes,distribution,None) is None


def test_geometry_buffers_do_not_change_checkpoint_parameter_contract():
    subject=model();state=subject.state_dict()
    assert 'shape_motion_stabilizer' not in state
    assert 'motion_rotation_differences' not in state
