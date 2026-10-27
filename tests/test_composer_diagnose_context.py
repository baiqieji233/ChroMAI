from fractions import Fraction

import numpy as np
import torch

from maichart.composer.diagnose import (_decode_context, _diagnostic_dataset, _inputs, _map_catalog_ids,
    _omit_last_prefix, interventions)
from maichart.composer.model import ComposerConfig, GROUNDED_PROFILE, build_model


ATOMS = ['1', '2', '3h[4:1]', '4-8[4:1]']


def _config(**kwargs):
    return ComposerConfig(atom_vocab_size=5, d_model=16, nhead=4, layers=1,
        dim_feedforward=32, dropout=0., history_events=8, action_slots=3,
        architecture_profile=GROUNDED_PROFILE, head_timing_dim=99, lens_width=8,
        lens_chunk_size=3, **kwargs)


def _batch():
    return {
        'candidate_features': torch.randn(2, 6, 141),
        'history': torch.tensor([[[1, 0, 0], [2, 0, 0]], [[2, 0, 0], [1, 0, 0]]]),
        'condition': torch.randn(2, 16), 'action_state': torch.randn(2, 50),
        'history_features': torch.randn(2, 2, 8),
        'candidate_padding_mask': torch.zeros(2, 6, dtype=torch.bool),
        'history_padding_mask': torch.zeros(2, 2, dtype=torch.bool),
        'head_timing': torch.rand(2, 6, 99), 'workload_audio': torch.randn(2, 8, 104),
        'ledger': torch.rand(2, 6, 6), 'interval_audio': torch.randn(2, 6, 16),
        'trailing_audio': torch.randn(2, 8, 52), 'audio_lens': torch.randn(2, 8, 31, 26),
        'audio_lens_mask': torch.ones(2, 8, 31, dtype=torch.bool),
        'planner_audio': torch.randn(2, 4, 104), 'planner_mask': torch.zeros(2, 4, dtype=torch.bool),
        'planner_indices': torch.tensor([[[0, 1], [0, 1], [1, 2], [1, 2], [2, 3], [2, 3]]]*2),
        'planner_mix': torch.full((2, 6), .5),
    }


def test_inputs_forwards_selected_state_and_motif_tensors():
    batch = _batch()
    batch.update(selected_action_state=torch.randn(2, 50), motif_ids=torch.ones(2, 3, 2, dtype=torch.long),
        motif_relative_beats=torch.zeros(2, 3), motif_mask=torch.ones(2, 3, dtype=torch.bool))
    result = _inputs(batch, 'cpu')
    for key in ('selected_action_state', 'motif_ids', 'motif_relative_beats', 'motif_mask'):
        torch.testing.assert_close(result[key], batch[key])


def test_context_policies_cannot_silently_use_context_from_a_different_target():
    import pytest

    for settings in ({'action_state_policy': 'selected_prefix_v3'},
                     {'motif_memory_policy': 'audio_recurrence_archive_v1'}):
        model = build_model(_config(**settings), ATOMS).eval()
        with pytest.raises(ValueError, match='causal decode_context'):
            interventions(model, _inputs(_batch(), 'cpu'))


def test_interventions_decode_selected_context_at_same_base_candidate():
    torch.set_num_threads(1)
    model = build_model(_config(action_state_policy='selected_prefix_v3',
        motif_memory_policy='audio_recurrence_archive_v1'), ATOMS).eval()
    inputs = _inputs(_batch(), 'cpu')
    seen = []

    def decode_context(choice, _inputs):
        seen.append(choice.clone())
        batch = choice.shape[0]
        return {
            'selected_action_state': torch.zeros(batch, 50),
            'motif_ids': torch.zeros(batch, 2, 1, dtype=torch.long),
            'motif_relative_beats': torch.zeros(batch, 2),
            'motif_mask': torch.zeros(batch, 2, dtype=torch.bool),
        }

    report = interventions(model, inputs, decode_context=decode_context)
    assert report
    assert len(seen) == 1 + len(report)
    assert all(torch.equal(seen[0], choice) for choice in seen[1:])


def test_decode_context_uses_committed_prefix_and_checkpoint_atom_ids(monkeypatch):
    from types import SimpleNamespace
    from maichart.composer import context_features

    dataset = SimpleNamespace(
        action_state_policy='selected_prefix_v3', motif_memory_policy='audio_recurrence_archive_v1',
        history_events=2, motif_max_events=4, base=SimpleNamespace(songs=[(SimpleNamespace(),)]),
        _song_table=lambda _song: {'state_index': object(), 'motif_index': object()})
    sample = SimpleNamespace(song_index=0, event_index=3)
    called = []

    def selected(index, committed_count, beat):
        called.append(('selected', committed_count, beat))
        return torch.full((50,), float(committed_count))

    def motifs(index, beat, committed_count, history_events, max_events):
        called.append(('motif', committed_count, beat))
        return {'motif_ids': torch.tensor([[1], [2]]),
            'motif_relative_beats': torch.tensor([-.5, -1.]),
            'motif_mask': torch.tensor([True, True])}

    monkeypatch.setattr(context_features, 'selected_action_state', selected)
    monkeypatch.setattr(context_features, 'motif_tensors', motifs)
    callback = _decode_context(dataset, sample, [Fraction(1), Fraction(2)],
        np.asarray([0, 4, 7]), torch.device('cpu'), committed_count=2)
    result = callback(torch.tensor([1]), {})
    assert called == [('selected', 2, Fraction(2)), ('motif', 2, Fraction(2))]
    assert result['selected_action_state'].shape == (1, 50)
    assert result['motif_ids'].flatten().tolist() == [4, 7]


def test_intervention_symbolic_unread_masks_shuffled_nonzero_motif_residual():
    torch.set_num_threads(1)
    torch.manual_seed(81)
    model = build_model(_config(motif_memory_policy='audio_recurrence_archive_v1'), ATOMS).eval()
    with torch.no_grad():
        model.motif_memory.output.weight.normal_(0., .05)
    inputs = _inputs(_batch(), 'cpu')
    observed = []
    original_decode = model.decode_candidate

    active = []

    def spy_decode(*args, **kwargs):
        output = original_decode(*args, **kwargs)
        keep = kwargs.get('motif_keep')
        if keep is not None and not bool(keep.any()):
            observed.append(output['action_logits'][..., 0, :].detach().clone())
        elif keep is None:
            active.append(output['action_logits'][..., 0, :].detach().clone())
        return output

    model.decode_candidate = spy_decode

    def run(archive):
        def decode_context(_choice, current_inputs):
            motif_ids = torch.tensor(archive, dtype=torch.long).view(1, 2, 1).expand(2, -1, -1)
            return {'motif_ids': motif_ids,
                'motif_relative_beats': torch.zeros(2, 2),
                'motif_mask': torch.ones(2, 2, dtype=torch.bool)}
        return interventions(model, inputs, decode_context=decode_context)

    first = run([1, 2])
    second = run([3, 4])
    assert len(observed) == 2
    torch.testing.assert_close(observed[0], observed[1], atol=0., rtol=0.)
    assert len(active) >= 2 and not torch.allclose(active[0], active[-1])
    assert 'symbolic_unread' in first and 'symbolic_unread' in second


def test_unknown_motif_atom_mapping_is_rejected_instead_of_becoming_stop():
    assert _map_catalog_ids(torch.tensor([[0, 1]]), np.asarray([0, 5]), 'motif').tolist() == [[0, 5]]
    try:
        _map_catalog_ids(torch.tensor([[1, 2]]), np.asarray([0, 5]), 'retrieved motif')
    except ValueError as exc:
        assert 'absent from the checkpoint catalog' in str(exc)
    else:
        raise AssertionError('Unknown motif atom IDs must not silently map to STOP')


def test_diagnostic_dataset_inherits_checkpoint_context_contract(monkeypatch):
    from types import SimpleNamespace
    import maichart.composer.diagnose as diagnose

    seen = {}
    monkeypatch.setattr(diagnose, 'ComposerDataset', lambda *args, **kwargs: seen.update(kwargs) or object())
    args = SimpleNamespace(manifest='manifest.json', cache_dir='cache', limit_charts=2)
    config = SimpleNamespace(candidate_count=96, history_events=128, action_slots=8,
        architecture_profile=GROUNDED_PROFILE, action_state_policy='selected_prefix_v3',
        motif_memory_policy='audio_recurrence_archive_v1', motif_max_events=12,
        head_timing_policy='separate_pads_v2')
    _diagnostic_dataset(args, SimpleNamespace(config=config))
    assert seen['action_state_policy'] == 'selected_prefix_v3'
    assert seen['motif_memory_policy'] == 'audio_recurrence_archive_v1'
    assert seen['motif_max_events'] == 12
    assert seen['head_timing_policy'] == 'separate_pads_v2'


def test_omitted_prefix_recomputes_candidate_state_from_strict_prefix(tmp_path, monkeypatch):
    from fractions import Fraction
    from test_composer_event_views import dataset_fixture, points
    from maichart.composer.context_features import composer_action_state

    dataset = dataset_fixture(tmp_path, monkeypatch,
        source=points((0, '1'), (1, '2'), (2, '3'), (3, '4'), (4, '5')))
    dataset.action_state_policy = 'candidate_prefix_v2'
    sample = dataset.samples[4]
    item = dataset[4]
    id_map = np.arange(len(dataset.codec_atoms) + 1, dtype=np.int64)
    changed, _seconds, candidates = _omit_last_prefix(dataset, sample, item, id_map)
    expected = composer_action_state(dataset._song_table(0)['state_index'],
        sample.event_index - 1, candidates, 'candidate_prefix_v2')
    torch.testing.assert_close(changed['action_state'], expected)
    assert changed['action_state'].shape == (dataset.candidate_count, 50)
    assert 'selected_action_state' not in changed
    assert 'motif_ids' not in changed
