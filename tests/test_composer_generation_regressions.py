import json
from fractions import Fraction

import numpy as np
import pytest
import soundfile as sf
import torch

from maichart.composer import generate as module
from maichart.composer.candidates import MUSIC_CANDIDATE_POLICY_V6, candidate_beats
from maichart.composer.checkpoint import save
from maichart.composer.model import ChartComposer, ComposerConfig
from maichart.event_timing import EventPoint, EventTempo, parse_event_chart
from maichart.generation.conditions import GenerationCondition
from maichart.generation.event_cli import _NoteGrammar


@pytest.mark.parametrize('temperature', [.85, .1, .05, .01, 1e-12])
def test_category_mass_survives_low_positive_temperature(temperature):
    logits = torch.tensor([float('-inf'), 8., *([0.] * 1000)])
    tap = torch.zeros(1002, dtype=torch.bool)
    tap[1] = True
    slide = torch.zeros_like(tap)
    slide[2:] = True
    empty = torch.zeros_like(tap)
    result = module._category_truncate_actions(logits, temperature, (tap, empty, slide, empty))
    torch.testing.assert_close(result[slide].sum(), logits.softmax(-1)[slide].sum())
    torch.testing.assert_close(result.sum(), torch.tensor(1.))
    assert torch.isfinite(result).all() and result[0] == 0


def test_category_sampling_keeps_stop_and_missing_categories():
    empty = torch.zeros(3, dtype=torch.bool)
    logits = torch.tensor([0., float('-inf'), float('-inf')])
    result = module._category_truncate_actions(logits, .05, (empty,) * 4)
    torch.testing.assert_close(result, torch.tensor([1., 0., 0.]))
    # With ordinary logits the distribution is the category mass times its
    # temperature-scaled conditional distribution, independently of other types.
    logits = torch.tensor([-.5, 1., .2, -.7, -1.3])
    tap = torch.tensor([False, True, True, False, False])
    hold = torch.tensor([False, False, False, True, True])
    empty = torch.zeros_like(tap)
    result = module._category_truncate_actions(logits, .85, (tap, hold, empty, empty))
    macro = logits.softmax(-1)
    torch.testing.assert_close(result[0], macro[0])
    for group in (tap, hold):
        torch.testing.assert_close(result[group], macro[group].sum() * (logits[group] / .85).softmax(-1))


@pytest.mark.parametrize('temperature', [.01, .9, 1., 2.])
def test_touch_rings_keep_predicted_mass_and_masked_pads_stay_impossible(temperature):
    atoms=['1','B1','C',*[f'E{i}{suffix}' for suffix in ('','f') for i in range(1,9)]]
    groups=module._head_metadata(atoms,_NoteGrammar.from_atoms(atoms))[4]
    assert len(groups)==8
    torch.testing.assert_close(torch.stack(groups).sum(0),torch.tensor([0]+[1]*len(atoms)))
    logits=torch.tensor([-.5,-.2,-1.,-1.5,*([0.]*16)])
    result=module._category_truncate_actions(logits,temperature,groups)
    for group in groups:
        torch.testing.assert_close(result[group].sum(),logits.softmax(-1)[group].sum())
    # The previous pooled Touch truncation erased both B and C behind common E
    # pads. Preserving their existing probability introduces no ring quota.
    pooled=(*groups[:3],torch.stack(groups[3:]).any(0))
    old=module._category_truncate_actions(logits,.9,pooled)
    assert old[2]==old[3]==0 and result[2]>0 and result[3]>0
    logits[3]=float('-inf')
    assert module._category_truncate_actions(logits,temperature,groups)[3]==0


def test_zero_length_holds_remain_sampleable_and_obey_occupancy():
    atoms = ['1', '1h[1:0]', '2h[4:0]']
    grammar = _NoteGrammar.from_atoms(atoms)
    metadata = module._head_metadata(atoms, grammar, touch_layout_policy='open')
    assert metadata[1].tolist() == [0, 1, 1, 1]
    logits = torch.tensor([[-20., -20., 20., -20.]])
    assert module._sample_actions(logits, atoms, grammar, 0., None,
                                  head_metadata=metadata) == ['1h[1:0]']
    assert module._sample_actions(logits, atoms, grammar, 0., None,
        occupied={'1'}, head_metadata=metadata) == ['2h[4:0]']
    assert module._sample_actions(logits, atoms, grammar, 0., None,
        occupied={'1', '2'}, head_metadata=metadata) == []


@pytest.mark.parametrize('committed', [None, Fraction(0)])
def test_phase_mask_uses_committed_event_instead_of_uncommitted_cursor(committed):
    candidates = candidate_beats(Fraction(1, 16), tempo=[EventTempo(Fraction(0), 200)],
        first=0., audio_end_sec=20., candidate_count=96, policy=MUSIC_CANDIDATE_POLICY_V6)
    actual = module._candidate_playability(candidates, [True] * len(candidates),
                                          committed, MUSIC_CANDIDATE_POLICY_V6)
    assert any(actual)
    assert all(beat.denominator in {1, 2, 3, 4, 6, 8}
               for beat, allowed in zip(candidates, actual) if allowed)
    assert not actual[candidates.index(Fraction(3, 16))]


def test_existing_phase_policy_and_fallback_are_preserved():
    tempo = [EventTempo(Fraction(0), 120)]
    canonical = {1, 2, 3, 4, 6, 8}
    for committed in (None, Fraction(0), Fraction(1, 8), Fraction(1, 3)):
        candidates = candidate_beats(committed, tempo=tempo, first=0., audio_end_sec=20.,
                                    candidate_count=96, policy=MUSIC_CANDIDATE_POLICY_V6)
        base = [index % 3 != 0 for index in range(len(candidates))]
        expected = [ok and beat.denominator in canonical and
            (committed is None or (beat - committed).denominator in canonical)
            for ok, beat in zip(base, candidates)]
        assert module._candidate_playability(candidates, base, committed,
            MUSIC_CANDIDATE_POLICY_V6) == (expected if any(expected) else base)
        assert module._candidate_playability(candidates, base, committed, 'legacy') == base
    # Do not silently change the explicit no-phase-candidate fallback policy.
    assert module._candidate_playability([Fraction(1, 16)], [True], Fraction(0),
                                        MUSIC_CANDIDATE_POLICY_V6) == [True]


@pytest.fixture
def music_checkpoint(tmp_path, monkeypatch):
    sample_rate = 8000
    time = np.arange(sample_rate * 4) / sample_rate
    audio = tmp_path / 'track.wav'
    sf.write(audio, (.1 * np.sin(2 * np.pi * 220 * time)).astype(np.float32), sample_rate)
    model = ChartComposer(ComposerConfig(atom_vocab_size=3, d_model=16, nhead=4,
        layers=1, dim_feedforward=32, candidate_count=96, history_events=8, action_slots=3,
        architecture_profile='music_context_v2', head_timing_dim=99))
    with torch.no_grad():
        model.eos_head.weight.zero_()
        model.eos_head.bias.fill_(-50.)
        model.action_head.weight.zero_()
        model.action_head.bias.copy_(torch.tensor([39., 40., -40.]))
    checkpoint = tmp_path / 'composer.pt'
    save(checkpoint, model, None, epoch=1, args={}, manifest=tmp_path / 'manifest.json',
         atom_catalog=['1', '2'])
    original = ChartComposer.forward

    def nearest(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        score = -torch.arange(result['candidate_logits'].size(-1)).float()[None]
        score = score.expand_as(result['candidate_logits']).masked_fill(kwargs['candidate_mask'], float('-inf'))
        result['candidate_logits'] = result['audio_candidate_logits'] = score
        return result

    monkeypatch.setattr(ChartComposer, 'forward', nearest)
    return audio, checkpoint


@pytest.mark.parametrize('commit_first', [False, True])
def test_generation_retry_does_not_turn_search_cursor_into_history(
        tmp_path, monkeypatch, music_checkpoint, commit_first):
    audio, checkpoint = music_checkpoint
    calls = 0
    masks = []
    real_mask = module._candidate_playability

    def reject_then_accept(*args, **kwargs):
        nonlocal calls
        calls += 1
        if commit_first and calls == 1:
            return ['1']
        failures_end = 49 if commit_first else 96
        if calls <= failures_end:
            return []
        return ['2'] if commit_first else ['1']

    def record_mask(candidates, base, committed, policy):
        result = real_mask(candidates, base, committed, policy)
        masks.append((list(candidates), committed, result))
        return result

    monkeypatch.setattr(module, '_sample_actions', reject_then_accept)
    monkeypatch.setattr(module, '_candidate_playability', record_mask)
    args = module.parser().parse_args(['--audio', str(audio), '--checkpoint', str(checkpoint),
        '--output', str(tmp_path / 'generated.txt'), '--bpm', '120', '--temperature', '0',
        '--device', 'cpu', '--max-events', '2' if commit_first else '1'])
    report = module.generate(args)
    assert report['event_count'] == (2 if commit_first else 1)
    assert report['guard']['skipped_candidates'] == (1 if commit_first else 2)
    candidates, committed, mask = masks[-1]
    assert candidates[0] == Fraction(1, 8)
    assert committed == (Fraction(0) if commit_first else None)
    assert all(beat.denominator in {1, 2, 3, 4, 6, 8}
               for beat, allowed in zip(candidates, mask) if allowed)


@pytest.mark.parametrize('difficulty', [5, 6])
def test_early_generation_exports_future_tempo_and_requested_slot(
        tmp_path, music_checkpoint, difficulty):
    audio, checkpoint = music_checkpoint
    tempo = tmp_path / 'tempo.json'
    tempo.write_text(json.dumps([{'beat': '0', 'bpm': 120}, {'beat': '4', 'bpm': 150}]))
    condition = tmp_path / 'condition.json'
    condition.write_text(json.dumps({'difficulty': difficulty}))
    output = tmp_path / 'generated.txt'
    args = module.parser().parse_args(['--audio', str(audio), '--checkpoint', str(checkpoint),
        '--output', str(output), '--tempo-map', str(tempo), '--condition-file', str(condition),
        '--temperature', '0', '--device', 'cpu', '--max-events', '1'])
    report = module.generate(args)
    chart = parse_event_chart(output, difficulty, strict_notes=True)
    assert report['event_count'] == len(chart.points) == 1
    assert chart.points[0].beat == 0 and chart.end_beat == 5
    assert [(change.beat, change.bpm) for change in chart.tempo_changes] == [(0, 120), (4, 150)]
    assert report['condition']['difficulty'] == difficulty
    assert output.with_suffix('.composer.json').is_file()


@pytest.mark.parametrize('difficulty', [0, 7])
def test_invalid_export_slot_fails_before_loading_weights(tmp_path, monkeypatch, difficulty):
    condition = tmp_path / 'condition.json'
    condition.write_text(json.dumps({'difficulty': difficulty}))
    monkeypatch.setattr(module, 'load', lambda *args, **kwargs: pytest.fail('Loaded weights before slot validation'))
    args = module.parser().parse_args(['--audio', 'unused.wav', '--checkpoint', 'unused.pt',
        '--output', str(tmp_path / 'unused.txt'), '--condition-file', str(condition)])
    with pytest.raises(ValueError, match='difficulty'):
        module.generate(args)


def test_constant_tempo_export_keeps_previous_tail_and_default_slot():
    points = [EventPoint(Fraction(9, 4), '1')]
    text, _ = module._compile_composer_chart(points, [EventTempo(Fraction(0), 120)],
                                            0., GenerationCondition(level=13.))
    from maichart.event_timing import parse_event_chart_text
    chart = parse_event_chart_text(text, 5)
    assert chart.end_beat == 4 and chart.points == points
