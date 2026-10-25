from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from maichart.composer.event_views import EventViews
from maichart.composer.events import HEAD_PADS, head_timing_features, event_metadata
from maichart.event_timing import EventPoint, EventTempo


TEMPO = [EventTempo(Fraction(0), 120.)]


def points(*rows):
    return [EventPoint(Fraction(beat), text) for beat, text in rows]


def test_views_preserve_unknown_context_and_never_fabricate_tail_eos():
    source = points((-1, '8'), (0, '1'), (1, '2'), (2, '3'), (3, '4'), (12, '5'))
    views = EventViews.build(source, TEMPO, 0., 6., excluded=[(.4, .8)])
    assert views.source_indices == (1, 2, 3, 4)
    assert views.trusted.tolist() == [True, False, True, True]
    assert views.context_points[1].note_text == '2'
    assert views.eos(0) == (0., 1.)
    assert views.eos(3) == (0., 0.)
    assert views.future_mask(0, 4)[1:7].sum() == 0
    assert views.window_weight(.5, 1.) == 0
    assert views.window_weight(.8, 1.) == 1
    assert not views.seconds.flags.writeable
    with pytest.raises(ValueError): views.seconds[0] = 1.


def test_unknown_tail_zero_weight_region_masks_end_and_not_earlier_continuation():
    source = points((0, '1'), (1, '2'), (2, '3'))
    annotation = {'regions': [{'start_sec': 1.2, 'end_sec': 6., 'weight': 0.}]}
    views = EventViews.build(source, TEMPO, 0., 6., annotation=annotation)
    assert views.eos(0) == (0., 1.)
    assert views.eos(2) == (0., 0.)
    assert views.window_weight(.8, 1.) == 1.
    assert views.window_weight(1., 1.5) == 0.
    assert views.window_weight(5., 6.) == 0.
    complete = EventViews.build(source, TEMPO, 0., 6.)
    assert complete.eos(2) == (1., 1.)
    assert views.source_digest != complete.source_digest


def test_views_do_not_silently_truncate_atoms_or_accept_unsorted_events():
    with pytest.raises(ValueError, match='action_slots'):
        EventViews.build(points((0, '1/2/3')), TEMPO, 0., 6., action_slots=2)
    with pytest.raises(ValueError, match='strictly increasing'):
        EventViews.build(points((1, '1'), (0, '2')), TEMPO, 0., 6.)


def test_carry_in_hold_is_unknown_until_release_not_a_false_empty_start():
    source=points((0, '1h[1:1]'), (1, '2'), (4, '3'))
    views=EventViews.build(source,TEMPO,-.5,5.)
    assert views.source_indices==(1,2)
    assert views.trusted.tolist()==[False,True]
    assert views.window_weight(0.,1.)==0.


def dataset_fixture(tmp_path, monkeypatch, source=None, excluded=()):
    from maichart.composer import data
    from maichart.generation.conditions import GenerationCondition
    from maichart.generation.sustain_state import SustainStateIndex
    source = source or points((-1, '8'), (0, '1'), (1, '2'), (2, '12'), (3, '3'), (12, '4'))
    audio = SimpleNamespace(duration_sec=6., source='fixture', metadata={'audio_sha256': 'fixture'})
    song = SimpleNamespace(tempo=TEMPO, first=0., audio=audio, meters=[], spans=[], regions=[])
    condition = GenerationCondition(level=13.)
    state = (SimpleNamespace() if any(p.beat<0 for p in source)
             else SustainStateIndex(source, TEMPO, 0.))
    record = (song, source, 13., 1., state, condition, excluded, {})
    monkeypatch.setattr(data, 'FormDataset', lambda *a, **kw: SimpleNamespace(songs=[record], audit={}))
    monkeypatch.setattr(data.ComposerDataset, '_audio_end_sec', lambda *a: (6., False))
    times = np.linspace(0., 6., 121)
    features = np.zeros((len(times), 120), dtype=np.float32)
    monkeypatch.setattr(data.ComposerDataset, '_song_features', lambda *a: (features, times))
    manifest = tmp_path / 'manifest.json'
    if not manifest.exists(): manifest.write_text('{}')
    return data.ComposerDataset(manifest, cache_dir=None, candidate_count=96, history_events=2)


def test_dataset_uses_one_context_index_for_history_state_recency_and_labels(tmp_path, monkeypatch):
    dataset = dataset_fixture(tmp_path, monkeypatch, excluded=[(.4, .8)])
    table = dataset._song_table(0)
    assert [p.beat for p in table['points']] == [0, 1, 2, 3]
    assert table['head_counts'].tolist() == [1, 1, 2, 1]
    assert [sample.event_index for sample in dataset.samples] == [0, 2, 3]
    sample = dataset.samples[1]
    assert sample.previous == 1
    assert sample.pointer_weight == 0
    item = dataset[1]
    assert item['history'].size(0) == 2  # includes unknown authored event 2
    assert item['history_features'][-1, 2] == .25
    # Current compact chord heads must not leak into their own recency input.
    sec = table['last_head_seconds'][sample.event_index]
    assert sec[HEAD_PADS.index('A1')] == 0.
    assert sec[HEAD_PADS.index('A2')] == .5
    assert np.isnan(sec[HEAD_PADS.index('A8')])
    assert item['head_timing'].shape == (96, 99)
    assert float(item['eos_weight']) == 1.  # a later trusted context event proves continuation
    assert float(dataset[2]['eos_weight']) == 0.  # unknown/out-of-range source tail does not prove EOS
    assert table['history_features'][2, 2] == .5  # compact chord has two heads
    first = dataset[0]
    assert torch.count_nonzero(first['head_timing']) == 0


def test_dataset_expands_slots_without_losing_ninth_atom(tmp_path, monkeypatch):
    text = '1/2/3/4/5/6/7/8/C'
    dataset = dataset_fixture(tmp_path, monkeypatch, source=points((0, text), (1, '1')))
    assert dataset.action_slots_requested == 8
    assert dataset.action_slots == dataset.max_required_action_slots == 9
    item = dataset[0]
    assert item['action_target'].shape[-1] == 9
    assert torch.count_nonzero(item['action_target']) == 9


def test_workload_and_trailing_unknown_masks_use_independent_support(tmp_path, monkeypatch):
    dataset = dataset_fixture(tmp_path, monkeypatch, excluded=[(1.5, 2.)])
    item = dataset[0]
    times = item['workload_seconds'].numpy()
    before = (times > .5) & (times < 1.5)
    before[:dataset.candidate_count] = False
    assert before.any()
    assert torch.all(item['workload_weight'][before] == 0)
    assert torch.all(item['trailing_weight'][before, 0] > 0)


def test_fingerprint_tracks_real_events_not_just_manifest_or_event_counts(tmp_path, monkeypatch):
    first = dataset_fixture(tmp_path, monkeypatch, source=points((0, '1'), (1, '2'), (2, '3')))
    second = dataset_fixture(tmp_path, monkeypatch, source=points((0, '1'), (1, '3'), (2, '2')))
    assert len(first) == len(second)
    assert first.fingerprint != second.fingerprint


def test_shared_slide_heads_and_touch_hold_have_parser_semantics():
    assert len(event_metadata('12').heads) == 2
    meta = event_metadata('Ch[4:1]')
    assert meta.hold and meta.touch
    shared = event_metadata('1-5[4:1]*-7[4:1]')
    assert len(shared.heads) == 1


def test_pad_timing_is_causal_and_uses_seconds_across_tempo_changes():
    unseen = np.full(len(HEAD_PADS), np.nan)
    seen_seconds, seen_beats = unseen.copy(), unseen.copy()
    seen_seconds[0], seen_beats[0] = .5, 1.
    features = head_timing_features(seen_seconds, seen_beats, [1., 2.], [2., 5.])
    assert features[0, 0] == 1
    assert features[0, len(HEAD_PADS)] == pytest.approx(np.log1p(.5))
    assert features[1, 2*len(HEAD_PADS)] == pytest.approx(np.log1p(4.))
    assert np.all(features[:, 1:len(HEAD_PADS)] == 0)


def test_unknown_supervision_masks_do_not_train_false_gap_future_or_eos():
    from maichart.composer.train import _loss
    def run(future, eos):
        output={'candidate_logits':torch.zeros(2,3,requires_grad=True),
            'action_logits':torch.zeros(2,2,4,requires_grad=True),
            'future':future,'eos_logits':eos}
        batch={'candidate_target':torch.tensor([0,1]),'action_target':torch.tensor([[1,0],[2,0]]),
            'future_target':torch.zeros(2,8),'eos_target':torch.tensor([0.,1.]),
            'future_weight':torch.zeros(2,8),'eos_weight':torch.zeros(2),
            'pointer_weight':torch.zeros(2)}
        loss,metrics=_loss(output,batch,4)
        loss.backward()
        return metrics
    future=torch.randn(2,8,requires_grad=True)
    eos=torch.randn(2,requires_grad=True)
    metrics=run(future,eos)
    assert metrics['future_loss']==metrics['eos_loss']==metrics['pointer_loss']==0
    assert torch.count_nonzero(future.grad)==0
    assert torch.count_nonzero(eos.grad)==0
