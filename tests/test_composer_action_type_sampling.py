import pytest
import torch
from maichart.composer.generate import _category_truncate_actions


def distribution():
    p=torch.tensor([.1,.3,.2,.1,.05,.05,.02,.03,.04,.05,.06])
    ids=torch.tensor([-1,0,0,1,2,2,3,4,5,6,7])
    return p,[ids.eq(i) for i in range(8)]


def test_default_preserves_type_and_touch_ring_mass():
    p,masks=distribution();q=_category_truncate_actions(p.log(),.65,masks)
    for mask in masks:torch.testing.assert_close(q[mask].sum(),p[mask].sum())
    torch.testing.assert_close(q[0],p[0])


@pytest.mark.parametrize('type_temperature',[0.,1e-50,.5,.85,1.5])
def test_type_control_preserves_stop_and_conditional_touch_rings(type_temperature):
    p,masks=distribution();q=_category_truncate_actions(p.log(),.65,masks,type_temperature=type_temperature)
    torch.testing.assert_close(q[0],p[0]);torch.testing.assert_close(q.sum(),torch.tensor(1.))
    if type_temperature<=1e-6:
        assert not q[3:].any()
        assert (q[1:3]>0).all()  # Type choice is greedy; within-type keys still sample.
    else:
        torch.testing.assert_close(q[6:]/q[6:].sum(),p[6:]/p[6:].sum())
        assert not torch.allclose(q[6:].sum(),p[6:].sum())


def test_type_floor_removes_rare_kinds_without_pruning_touch_rings():
    p,masks=distribution();q=_category_truncate_actions(p.log(),.65,masks,type_min_p=.3)
    assert not q[3:6].any()
    assert q[6:].gt(0).all()
    torch.testing.assert_close(q[6:]/q[6:].sum(),p[6:]/p[6:].sum())
    masked=p.log();masked[1:3]=float('-inf')
    q=_category_truncate_actions(masked,.65,masks,type_temperature=0.)
    assert not q[1:6].any() and q[6:].sum()>0
