"""Contract checks for ordered phrases; deliberately not run during editing."""
from fractions import Fraction
from types import SimpleNamespace

import torch

from maichart.composer.stream_phrase import PhraseIndex, PHRASE_ROW_FIELDS, describe_phrase


def _bank():
    beats = torch.arange(0., 24., .25, dtype=torch.float64)
    return torch.zeros(len(beats), 238), beats/2, beats


def _note(start, end, text):
    return SimpleNamespace(kind=2, start=Fraction(start), end=Fraction(end), text=text)


def test_future_chart_events_cannot_change_query_inputs():
    prefix = [_note(0, 1, '1'), _note(1, 2, '2')]
    first = PhraseIndex(_bank(), 24, prefix+[_note(2, 3, '3')])
    second = PhraseIndex(_bank(), 24, prefix+[_note(2, 3, '8')])
    a, b = first.row(2, 2, supervise=True), second.row(2, 2, supervise=True)
    for key in PHRASE_ROW_FIELDS:
        if not key.startswith('target_'):
            assert torch.equal(a[key], b[key]), key
    assert not torch.equal(a['target_phrase_features'], b['target_phrase_features'])


def test_incremental_and_edited_prefix_match_fresh_archive():
    source = [_note(i, i+1, str(i%8+1)) for i in range(18)]
    original = PhraseIndex(_bank(), 24, source)
    changed = source[:9]+[_note(9, 10, '8')]+source[10:]
    edited = original.edited(changed, 9, len(changed))
    fresh = PhraseIndex(_bank(), 24, changed)
    for query in (0, 9, 10, 18):
        a, b = edited.row(query, query), fresh.row(query, query)
        for key in PHRASE_ROW_FIELDS:
            assert torch.equal(a[key], b[key]), key
    incremental = PhraseIndex(_bank(), 24)
    for step in changed: incremental.append(step)
    assert torch.equal(incremental.row(18,18)['phrase_memory_features'], fresh.row(18,18)['phrase_memory_features'])


def test_ordered_descriptor_and_stable_anchor_memory():
    forward = [_note(0, 1, '1'), _note(5, 6, '5')]
    reverse = [_note(0, 1, '5'), _note(5, 6, '1')]
    assert not torch.equal(describe_phrase(forward, 0, 8), describe_phrase(reverse, 0, 8))
    steps = [_note(i, i+1, str(i%8+1)) for i in range(18)]
    archive = PhraseIndex(_bank(), 24, steps)
    a, b = archive.row(9,9), archive.row(13,13)
    assert a['phrase_start_beat'] == b['phrase_start_beat']
    assert torch.equal(a['phrase_memory_features'], b['phrase_memory_features'])
    assert not torch.equal(a['phrase_progress'], b['phrase_progress'])
