"""Joint survival sampling and generation action-conditioning contracts."""
import torch

from maichart.composer.generate import _sample_survival_tail
from maichart.composer.action_guidance import action_logits_with_touch_guard
from maichart.composer.generate import _filter_rhythm_and_active_span_candidates,_pointer_bin_settings


def test_survival_tail_is_not_an_event():
    generator=torch.Generator().manual_seed(19)
    assert _sample_survival_tail(torch.tensor([-float('inf'),-float('inf')]),
        torch.tensor([-float('inf'),-float('inf')]),torch.tensor(0.),torch.tensor(0.),1.,generator)


def test_survival_joint_blend_includes_tail_endpoints():
    events=torch.tensor([0.,-float('inf')])
    absent=torch.full((2,),-float('inf'))
    generator=torch.Generator().manual_seed(19)
    assert _sample_survival_tail(events,absent,torch.tensor(-float('inf')),torch.tensor(0.),0.,generator)
    assert not _sample_survival_tail(events,absent,torch.tensor(-float('inf')),torch.tensor(0.),1.,generator)


def test_button_load_conditions_full_action_only():
    class Model:
        def __init__(self):self.calls=[]
        def action_slot(self,context,slot,selected_ids,**kwargs):
            self.calls.append(kwargs)
            return torch.tensor([0.,1.,0.])
    model=Model()
    features=torch.ones(8,3)
    action_logits_with_touch_guard(model,torch.zeros(2),0,[],audio_context=torch.zeros(2),
        enabled=True,touch_mask=torch.tensor([False,True,False]),button_load_features=features)
    assert model.calls[0]['button_load_features'] is features
    assert 'button_load_features' not in model.calls[1]


def test_survival_keeps_learned_long_rests_instead_of_legacy_span_pruning():
    from fractions import Fraction
    import numpy as np
    from maichart.composer.candidates import MUSIC_CANDIDATE_POLICY_V6
    opts=dict(policy=MUSIC_CANDIDATE_POLICY_V6,plan_centers=np.array([1.,2.,3.]),
        plan_values=np.array([[1.,1.,0.,0.,0.,0.]]*3))
    candidates=[Fraction(1),Fraction(8)]
    assert _filter_rhythm_and_active_span_candidates(candidates,[True,True],np.array([.5,4.]),Fraction(0),0.,**opts)==[True,False]
    assert _filter_rhythm_and_active_span_candidates(candidates,[True,True],np.array([.5,4.]),Fraction(0),0.,learned_rest=True,**opts)==[True,True]
    assert _pointer_bin_settings(.5,learned_rest=True)['bin_min_p']==0.
