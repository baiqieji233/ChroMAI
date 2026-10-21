"""Semantic checks for overlap training and authored repetition departures."""
from fractions import Fraction as F
from dataclasses import replace
from types import SimpleNamespace
import random
import torch
import pytest

from maichart.form.codec import RationalCodec
from maichart.event_timing import EventPoint,EventTempo
from maichart.stream.continuation import continuation_ranges,cycle_exit_targets,replay_start
from maichart.stream.grounding import masked_note_inputs,clock_inputs,action_summaries
from maichart.stream.model import ChartStreamTransformer,StreamConfig
from maichart.stream.train import _loss,_mean_metrics
from maichart.generation.sustain_state import SustainStateIndex


def test_overlap_covers_every_supervised_token_once_and_keeps_complete_events():
    c=RationalCodec();ids=c.encode([EventPoint(F(i,4),'4/5' if i%3 else '3h[4:2]') for i in range(35)])
    seen=[];overlaps=0
    for warm,start,end in continuation_ranges(ids,72,c.at_id,c.eos_id,24):
        assert end-warm<=72
        assert warm==start or ids[warm]==c.at_id
        assert warm<=start<end
        seen.extend(range(start,end));overlaps+=warm<start
    assert seen==list(range(1,len(ids)))
    assert overlaps>0


def test_long_individual_cell_reduces_replay_without_dropping_target():
    c=RationalCodec();ids=c.encode([EventPoint(F(0),'1'),EventPoint(F(1),'1h[4:123456]'),EventPoint(F(2),'2')])
    length=len(c.cell(EventPoint(F(1),'1h[4:123456]')))
    rows=list(continuation_ranges(ids,length,c.at_id,c.eos_id,12))
    assert sum(end-start for _,start,end in rows)==len(ids)-1
    with pytest.raises(ValueError,match='单个配置'):
        list(continuation_ranges(ids,length-1,c.at_id,c.eos_id,12))


def test_cycle_exit_marks_actual_departure_and_preserves_legitimate_repetition():
    c=RationalCodec()
    cells=['4/5']*8+['4/6']+['1','2']*3+['3']
    ids=c.encode([EventPoint(F(i),text) for i,text in enumerate(cells)])
    mask=cycle_exit_targets(ids,c)
    flagged=[c.tokens[t] for t,m in zip(ids,mask) if m]
    # v14 also preserves authored exits from a confined lane set; these are
    # weights on real departure targets, never a penalty on continuing a run.
    assert flagged==['N:6','N:1','<end>','N:2','N:3']
    assert not mask[-1]
    shorter=c.encode([EventPoint(F(i),s) for i,s in enumerate(['1/2','1/2','1'])])
    assert [c.tokens[t] for t,m in zip(shorter,cycle_exit_targets(shorter,c)) if m]==['<end>']


def test_shape_cycle_exit_targets_structure_not_lane_digit_and_ignores_end_marker():
    c=RationalCodec()
    # Tap cycle exits into a Hold: the departure is the 'h' modifier, never the lane digit.
    cells=['1','2']*3+['3h[4:1]']
    ids=c.encode([EventPoint(F(i),text) for i,text in enumerate(cells)])
    flagged=[c.tokens[t] for t,m in zip(ids,cycle_exit_targets(ids,c)) if m]
    assert 'N:h' in flagged and flagged.count('N:h')==1
    # An all-Tap chart must produce no exits (END's '>' used to look like a Slide on every cell).
    plain=c.encode([EventPoint(F(i),str(1+i%4)) for i in range(16)])
    assert not any(cycle_exit_targets(plain,c))
    # Downbeat-Break trills (1b, 5, 1, 5 -> 8 / 8b) exit a 2-cycle lane pattern even though 'b' differs.
    trill_exit=c.encode([EventPoint(F(i,4),text) for i,text in enumerate(['1b','5','1','5','8'])])
    assert [c.tokens[t] for t,m in zip(trill_exit,cycle_exit_targets(trill_exit,c)) if m]==['N:8']
    trill_break_exit=c.encode([EventPoint(F(i,4),text) for i,text in enumerate(['1b','5','1','5','8b'])])
    assert [c.tokens[t] for t,m in zip(trill_break_exit,cycle_exit_targets(trill_break_exit,c)) if m]==['N:8','N:b']


def test_cycle_exit_uses_rational_timing_not_changes_in_fraction_spelling():
    c=RationalCodec()
    # Canonical positions cross single/multiple digits and whole beats, while
    # every actual onset delta remains 1/7. None is an authored rhythm exit.
    points=[EventPoint(F(i,7),'1') for i in range(17)]
    regular=c.encode(points)
    assert not any(cycle_exit_targets(regular,c))
    # The author then skips one onset: preserve supervision at the first
    # differing position character (18/7 versus the expected 17/7).
    departure=c.encode(points+[EventPoint(F(18,7),'1')])
    flagged=[c.tokens[t] for t,m in zip(departure,cycle_exit_targets(departure,c)) if m]
    assert flagged==['P:8']


def test_span_mask_keeps_clocks_and_drops_only_targets_that_follow_a_mask():
    c=RationalCodec();ids=c.encode([EventPoint(F(i),'1/2') for i in range(12)])
    decoder=ids[:-1];target=ids[1:]
    masked,weights,hidden=masked_note_inputs(decoder,c,random.Random(1),.5,targets=target,
        quality=[1.]*len(target),span_events=4,return_events=True)
    assert [x for x in masked if 7<=x<18]==[x for x in decoder if 7<=x<18]
    assert weights[-1]==1.
    # The only unsupervised rows are those fed a MASK: they would ask the model
    # to continue a cell it cannot see. Everything else, including <at> and the
    # position tokens of masked events, still teaches timing and recovery.
    assert [i for i,w in enumerate(weights) if w==0.]==[i for i,x in enumerate(masked) if x==c.mask_id]
    starts=[i for i,t in enumerate(target) if t==c.at_id]
    assert [weights[i] for i in starts]==[1.]*len(starts)
    assert hidden and set(hidden)<=set(range(12))
    # Every event after a masked span is supervised from its first token.
    assert sum(w==0. for w in weights)<len(target)//4


def test_cycle_exit_loss_has_zero_gradient_for_unmarked_and_excluded_targets():
    # A minimal loss batch avoids any dependence on acoustic preprocessing.
    logits=torch.randn(1,4,55,requires_grad=True)
    batch={'target':torch.tensor([[18,19,20,2]]),'target_padding_mask':torch.zeros(1,4,dtype=torch.bool),
        'decoder_input':torch.tensor([[4,18,19,5]]),'plan_mask':torch.zeros(1,4,dtype=torch.bool),
        'plan_mask_padding_mask':torch.zeros(1,4,dtype=torch.bool),'plan_labels':torch.zeros(1,4,8),
        'curve_target':torch.zeros(1,2,4),'curve_target_padding_mask':torch.zeros(1,2,dtype=torch.bool),
        'audio_padding_mask':torch.zeros(1,2,dtype=torch.bool),'quality_weight':torch.ones(1),
        'target_quality':torch.tensor([[1.,0.,1.,1.]]),'cycle_exit_mask':torch.tensor([[True,True,False,False]])}
    out={'logits':logits,'plan':torch.zeros(1,4,8),'curve':torch.zeros(1,2,4)}
    a,_=_loss(out,batch,cycle_exit_weight=0.)
    b,m=_loss(out,batch,cycle_exit_weight=.1)
    grad=torch.autograd.grad(b-a,logits)[0]
    assert grad[0,0].abs().sum()>0
    torch.testing.assert_close(grad[0,1:],torch.zeros_like(grad[0,1:]),rtol=0,atol=1e-7)
    assert _mean_metrics(m,1)['cycle_exit_targets']==1
