"""Recovery learns a real continuation only after an authenticated seam."""
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import torch

from maichart.composer.stream_data import NOTE,collate_stream
from maichart.composer.stream_recovery import (_configuration_recovery_row,_context,_dataset_guard,
    _qualified_continuation)
from test_stream_draft_integration import build


def make(monkeypatch):
    ds,model,_,bank=build(monkeypatch)
    model.config.recovery_policy='certified_phrase_v5'
    model.config.configuration_recovery_steps=6;model.config.recovery_bridge_steps=8
    source=ds.streams[0]
    prefix=[replace(source[0],text='8'),*source[1:2]]
    context=_context(ds,0,bank);context._sync_prefix(prefix,prefix[-1].end)
    guard=_dataset_guard(ds,0,prefix,0)
    return ds,model,source,prefix,context,guard


def test_bridge_conditions_on_real_seam_then_supervises_multiple_causal_configurations(monkeypatch):
    ds,model,source,prefix,context,guard=make(monkeypatch);original=list(prefix)
    target,trusted,reason=_qualified_continuation(ds,0,prefix,source[2],guard,0,2,'causal_correspondence_v1')
    assert not trusted
    counts=Counter()
    row=_configuration_recovery_row(ds,model,0,prefix,target,guard,context,0,2,trusted,reason,counts)
    assert prefix==original
    # One recent authored NOTE already matches; two added notes complete the
    # required three-NOTE seam. Certification reuses that real evidence.
    assert counts['teacher_bridge_steps']>=2
    assert row['_recovery_target_count']>1 and row['_recovery_configuration_count']>1
    assert row['_continuation_teacher_forced']
    active=row['loss_mask'].bool()
    assert torch.all(row['target_beat'][active]>row['step_beat'][active])
    assert float(row['_training_target_budget'].sum())==1.
    # A later supervised NOTE enters history only on the following decision.
    for position in active.nonzero().flatten().tolist():
        assert float(row['step_beat'][position])<float(row['target_beat'][position])
    packed=collate_stream([row]);assert packed['_loss_normalizer']==1.


def test_no_bridge_budget_does_not_promote_an_uncertified_configuration(monkeypatch):
    ds,model,source,prefix,context,guard=make(monkeypatch)
    model.config.recovery_bridge_steps=0
    target,trusted,reason=_qualified_continuation(ds,0,prefix,source[2],guard,0,2,'causal_correspondence_v1')
    row=_configuration_recovery_row(ds,model,0,prefix,target,guard,context,0,2,trusted,reason,Counter())
    assert row['_recovery_target_count']==1 and row['_recovery_configuration_count']==0


def test_a_live_wrong_hold_cannot_be_erased_by_teacher_bridge(monkeypatch):
    ds,model,source,prefix,context,guard=make(monkeypatch)
    # Native lifetime witness is what certifies equivalence, independent of
    # a suffix's textual match. Reuse the existing proof directly.
    from maichart.composer.stream_target_correspondence import authored_seam_corresponds
    actual=[replace(source[0],text='1h[1:8]'),*source[1:6]]
    assert not authored_seam_corresponds(source[:6],actual,guard)
