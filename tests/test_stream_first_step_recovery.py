"""V6 keeps the model-prefix decision before a teacher-assisted seam."""
from collections import Counter

import pytest
import torch

from maichart.composer.stream_data import NOTE, collate_stream
from maichart.composer.stream_recovery import _configuration_recovery_row, _qualified_continuation
from test_stream_certified_phrase_recovery import make


@pytest.mark.parametrize('bridge_budget',[0,8])
def test_native_first_step_survives_teacher_bridge_with_only_family_configuration(monkeypatch,bridge_budget):
    ds,model,source,prefix,context,guard=make(monkeypatch)
    model.config.recovery_policy='first_step_phrase_v6'
    model.config.recovery_bridge_steps=bridge_budget
    target,trusted,reason=_qualified_continuation(ds,0,prefix,source[2],guard,0,2,'causal_correspondence_v1')
    assert not trusted
    original=list(prefix);counts=Counter()
    row=_configuration_recovery_row(ds,model,0,prefix,target,guard,context,0,2,trusted,reason,counts)
    first=int(row['loss_mask'].nonzero().flatten()[0])
    assert prefix==original
    assert row['step_beat'][first]==float(prefix[-1].end)
    assert row['target_beat'][first]==float(target.end)
    assert row['target_kind'][first]==NOTE
    assert not row['configuration_mask'][first]
    assert row['configuration_family_mask'][first]
    assert int(row['configuration_family_mask'].sum())==1
    assert counts['native_first_family_targets']==1
    assert torch.isclose(row['_training_target_budget'].sum(),torch.tensor(1.))
    packed=collate_stream([row])
    assert packed['_loss_normalizer']==pytest.approx(1.)
    if 'draft_landing_group' in packed:
        assert packed['draft_landing_group'][0,first]>=0
    if bridge_budget:
        assert row['_recovery_configuration_count']>0
        assert row['_recovery_target_count']>row['_recovery_configuration_count']


def test_legacy_rows_collate_with_family_mask_defaulting_to_false(monkeypatch):
    ds,_,_,_,_,_=make(monkeypatch)
    row=ds[0]
    row.pop('configuration_family_mask',None)
    packed=collate_stream([row])
    assert not packed['configuration_family_mask'].any()


def test_unknown_source_never_supplies_a_family_witness(monkeypatch):
    ds,model,source,prefix,context,guard=make(monkeypatch)
    model.config.recovery_policy='first_step_phrase_v6'
    ds.unknown_steps[0][2]=True
    row=_configuration_recovery_row(ds,model,0,prefix,source[2],guard,context,0,2,
        False,'time_only',Counter())
    assert not row['configuration_family_mask'].any()
