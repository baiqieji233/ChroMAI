from fractions import Fraction
from dataclasses import replace
import numpy as np
import torch

from maichart.composer.history_clock import history_beat_ages
from maichart.composer.grounded_model import _LocalLeadInContext
from maichart.composer.generate import _filter_rhythm_and_active_span_candidates
from maichart.composer.candidates import MUSIC_CANDIDATE_POLICY_V6


def test_true_history_clock_preserves_long_rest_and_small_subdivision():
    beats=[Fraction(10000),Fraction(10032),Fraction(10032)+Fraction(1,8)]
    ages=history_beat_ages(beats)
    np.testing.assert_array_equal(ages,[32.125,.125,0.])
    features=torch.zeros(1,3,8,dtype=torch.bfloat16)
    features[0,:,0]=torch.tensor([0.,1.,.03125])
    actual,steps=_LocalLeadInContext._history_timing(features,torch.tensor([3]),3,
        exact_ages=torch.from_numpy(ages)[None])
    assert actual.dtype==torch.float32
    torch.testing.assert_close(actual,torch.tensor([[32.125,.125,0.]]))
    torch.testing.assert_close(steps,torch.tensor([[0.,32.,.125]]))


def test_dataset_and_generated_prefix_use_same_unclipped_clock(tmp_path,monkeypatch):
    from test_composer_event_views import dataset_fixture,points
    from test_composer_recovery import builder,point
    from maichart.composer.data import collate_composer
    source=points((0,'1'),(8,'2'),(9,'3'),(10,'4'))
    ds=dataset_fixture(tmp_path,monkeypatch,source=source)
    ds.history_events=4
    row=next(i for i,sample in enumerate(ds.samples) if sample.event_index==2)
    sample=ds[row]
    assert sample['history_features'][-1,0]==1.  # old eight-feature input remains clipped
    assert sample['history_beat_ages'].tolist()==[8.,0.]
    rec=builder(source=[point(0),point(8,'2'),point(9,'3'),point(10,'4')])
    recovered,_=rec._inputs([point(0),point(8,'2')],0)
    torch.testing.assert_close(recovered['history_beat_ages'],sample['history_beat_ages'])
    batch=collate_composer([ds[0],sample])
    assert batch['history_beat_ages'].shape==batch['history_padding_mask'].shape
    torch.testing.assert_close(batch['history_beat_ages'][1],sample['history_beat_ages'])


def test_candidate_lookback_does_not_collapse_long_jump_to_sixteen_beats():
    # Inspect the input to the learnable projection: 16-beat counterpart hit is
    # present at +16, absent at +64. This must work before training out_proj.
    module=_LocalLeadInContext(8).eval()
    history=torch.ones(1,1,1,dtype=torch.long)
    heads=torch.zeros(2,8,dtype=torch.bool);heads[1,0]=True
    captured=[]
    hook=module.in_proj.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach()))
    try:
        module(torch.zeros(1,1,8),history,torch.zeros(1,1,8),torch.zeros(1,1,dtype=torch.bool),
            torch.tensor([1]),heads,heads,torch.tensor([[[16.,8.],[64.,32.]]]),
            torch.ones(1,2,dtype=torch.bool),history_beat_ages=torch.zeros(1,1))
    finally:hook.remove()
    # Last 30 features are rhythm: index 19 is the 16-beat hit.
    assert captured[0][0,0,-30+19]==1.
    assert captured[0][0,1,-30+19]==0.


def test_exact_repeat_feature_does_not_count_a_chord_superset_as_same():
    module=_LocalLeadInContext(8).eval()
    heads=torch.zeros(3,8,dtype=torch.bool);heads[1,0]=True;heads[2,[0,4]]=True
    history=torch.tensor([[[2],[1]]])
    captured=[]
    hook=module.in_proj.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach()))
    try:
        module(torch.zeros(1,2,8),history,torch.zeros(1,2,8),torch.zeros(1,2,dtype=torch.bool),
            torch.tensor([2]),heads,heads,torch.ones(1,1,2),torch.ones(1,1,dtype=torch.bool),
            history_beat_ages=torch.tensor([[4.,0.]]))
    finally:hook.remove()
    # flow = 64 geometric + 12 symmetry + 16 repetition + 12 type features.
    repeats=captured[0][0,0,16+64+12+5]
    expected=torch.exp(torch.tensor(-(4./2.5)**2))/(1+torch.exp(torch.tensor(-(4./2.5)**2)))
    torch.testing.assert_close(repeats,expected)


def test_compact_chord_anchor_gets_same_microstep_rule_as_slash_chord():
    candidates=[Fraction(1,8),Fraction(1,4)]
    kwargs=dict(policy=MUSIC_CANDIDATE_POLICY_V6)
    def check(text):
        return _filter_rhythm_and_active_span_candidates(candidates,[True,True],np.array([.05,.1]),
            Fraction(0),0.,committed_note_text=text,**kwargs)
    assert check('17')==check('1/7')==[False,True]
    assert check('1')==[True,True]
