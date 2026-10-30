from fractions import Fraction as Q
from types import SimpleNamespace

import pytest
import torch

from maichart.composer.stream_data import StreamDataset
from maichart.event_timing import EventPoint, EventTempo


@pytest.mark.parametrize('history_policy',['block_aligned_v2','rolling_legacy_v1'])
def test_sqrt_weights_preserve_chart_budget_and_restore_dense_transition_mass(history_policy):
    song=SimpleNamespace(audio=SimpleNamespace(duration_sec=16.),
        tempo=[EventTempo(Q(0),120.)],first=0.)
    points=[EventPoint(Q(i,8),str(i%8+1)) for i in range(128)]
    points += [EventPoint(Q(17+4*i),str(i+1)) for i in range(4)]
    base=SimpleNamespace(songs=[(song,points,12.,1.,None,torch.zeros(16),(),None)])
    common=dict(base=base,lead_in=4,targets=16,window_seconds=8.,history_window_policy=history_policy)
    before=StreamDataset(**common,window_balance_policy='equal_window_v1')
    after=StreamDataset(**common,window_balance_policy='sqrt_window_v2')
    old=torch.tensor(before.target_window_weights[0]);new=torch.tensor(after.target_window_weights[0])
    assert new.sum().item()==pytest.approx(old.sum().item())
    assert new.max()/new.min()<=4.00001
    assert old.max()/old.min()>new.max()/new.min()
    assert new[0]>old[0]
    assert after.streams[0]==before.streams[0]
    assert sum(after.plan_weights)==pytest.approx(sum(before.plan_weights))
    for _,start,end,_ in after.samples:
        assert torch.isfinite(new[start:end]).all()
