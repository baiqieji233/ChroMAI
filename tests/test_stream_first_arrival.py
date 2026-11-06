import math
import torch
from maichart.composer.stream_arrival import first_arrival


def law(times,rate=2.,start=0.,stop=1.):
    seconds=torch.tensor(times,dtype=torch.float64)
    score=torch.full_like(seconds,math.log(math.expm1(rate)),requires_grad=True)
    return first_arrival(score,seconds,torch.tensor(start),torch.tensor(stop),torch.ones_like(seconds,dtype=torch.bool)),score


def test_unordered_phase_catalog_has_one_normalized_mass_and_tail():
    result,_=law([.7,.1,.4,.9])
    ordered,_=law([.1,.4,.7,.9])
    torch.testing.assert_close(result['joint'].exp().sum()+result['log_tail'].exp(),torch.tensor(1.),rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(result['joint'][torch.tensor([1,2,0,3])],ordered['joint'])
    assert abs(float(result['log_tail'])+2)<1e-6


def test_refining_cells_preserves_physical_interval_survival():
    a,_=law([.25,.75]);b,_=law([.125,.375,.625,.875])
    torch.testing.assert_close(a['log_tail'],b['log_tail'])
    torch.testing.assert_close(a['joint'][0].exp(),b['joint'][:2].exp().sum())


def test_zero_bos_candidate_has_positive_exposure_without_duplicate_pad():
    score=torch.zeros(4,requires_grad=True)
    result=first_arrival(score,torch.tensor([0.,0.,.5,.8]),torch.tensor(0.),torch.tensor(1.),torch.tensor([False,True,True,True]))
    assert result['joint'][0]<-1000 and result['joint'][1]>-1000
    torch.testing.assert_close(result['joint'].exp().sum()+result['log_tail'].exp(),torch.tensor(1.))
    (-result['joint'][1]).backward();assert torch.isfinite(score.grad).all()


def test_rest_and_onset_losses_are_exact_factorization_and_have_gradients():
    result,score=law([.1,.4,.8])
    torch.testing.assert_close(result['log_note']+result['conditional'],result['joint'])
    (-result['log_tail']-result['joint'][1]).backward()
    assert torch.isfinite(score.grad).all() and score.grad.abs().sum()>0


def test_note_exactly_at_wait_horizon_remains_supported():
    result,_=law([.5,1.,2.],start=0.,stop=2.)
    assert result['widths'][-1]>0 and result['joint'][-1]>-1000
    masked=first_arrival(torch.zeros(3),torch.tensor([.5,1.,2.]),torch.tensor(0.),torch.tensor(2.),torch.tensor([True,True,False]))
    assert masked['joint'][-1]<-1000  # actual audio end is excluded by caller
