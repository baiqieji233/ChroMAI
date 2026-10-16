"""CPU checks for causal musical-evidence supervision and masked numerics."""
import copy
from dataclasses import asdict, replace
from fractions import Fraction as F

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from maichart.event_timing import EventPoint, EventTempo
from maichart.form.alignment import (
    build_alignment_targets, build_event_alignment_targets, position_alignment_loss,
    temporal_neighbors,
)
from maichart.form.codec import RationalCodec
from maichart.form.model import FormConfig, FormTransformer
from maichart.form.quality_profile import MUSICAL_EVIDENCE
from maichart.form.rational_planner import RationalPlanner, RationalPlannerConfig, rational_planner_loss


TEMPO = [EventTempo(F(0), 120.)]


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _model_and_batch():
    codec = RationalCodec()
    torch.manual_seed(7541)
    config = FormConfig(vocab_size=len(codec), d_model=16, nhead=2, layers=1,
        dim_feedforward=32, dropout=0, audio_dim=6, descriptor_dim=5, condition_dim=3,
        max_tokens=64, max_audio_tokens=8, max_memory_tokens=16,
        quality_profile=MUSICAL_EVIDENCE)
    model = FormTransformer(config).eval()
    points = [EventPoint(F(0), '1'), EventPoint(F(1, 3), '5')]
    ids = torch.tensor(codec.encode(points)[:-1])
    times = torch.tensor([0., .16666667, .23, .65])
    targets = build_event_alignment_targets(points, TEMPO, 0., times, ids, codec)
    batch = {'audio': torch.randn(1, 4, 6), 'form': torch.randn(1, 2, 8),
        'audio_padding_mask': torch.zeros(1, 4, dtype=torch.bool),
        'history': torch.zeros(1, 1, dtype=torch.long),
        'condition': torch.tensor([[.9, .3, .6]]), 'task': torch.zeros(1, dtype=torch.long),
        'span_bounds': torch.tensor([[.1, .2]]), 'decoder_input': ids[None],
        'audio_times_sec': times[None],
        **{key: value[None] for key, value in targets.items()}}
    labels = build_alignment_targets(points, TEMPO, 0., times)
    batch.update(position_alignment_target=labels['labels'][None],
                 position_alignment_mask=labels['mask'][None])
    return model, batch, codec


def test_targets_use_existing_queries_without_inserting_exact_event_positions():
    codec = RationalCodec()
    points = [EventPoint(F(0), '1'), EventPoint(F(1, 3), '2'), EventPoint(F(5, 7), '3')]
    queries = np.array([-.001, .005, .167, .5])
    original = queries.copy()
    prefix = codec.encode(points)[:-1]
    labels = build_event_alignment_targets(points, TEMPO, 0., queries, prefix, codec)
    np.testing.assert_array_equal(queries, original)
    assert labels['alignment_audio_index'].tolist() == [1, 2, 3]
    assert labels['alignment_event_mask'].tolist() == [True, True, False]
    for index in labels['alignment_event_indices'].tolist():
        assert prefix[index] in (codec.bos_id, codec.end_id)
        assert prefix[index+1] == codec.at_id


def test_negative_audio_events_do_not_label_virtual_silence_or_following_audio():
    points = [EventPoint(F(0), '1')]
    result = build_alignment_targets(points, TEMPO, -.005, [-.005, 0., .005])
    assert result['mask'].tolist() == [False, True, True]
    assert not result['labels'].any()


@pytest.mark.parametrize('queries', [[0., float('nan')], [.2, .1], [[0., .1]]])
def test_target_builders_reject_invalid_query_clocks(queries):
    with pytest.raises(ValueError, match='query times'):
        build_alignment_targets([], TEMPO, 0, queries)


def test_alignment_targets_do_not_change_encoded_memory_or_main_logits():
    model, batch, _ = _model_and_batch()
    altered = copy.deepcopy(batch)
    altered['position_alignment_target'] = torch.rand_like(batch['position_alignment_target'])
    altered['alignment_audio_index'] = torch.zeros_like(batch['alignment_audio_index'])
    altered['alignment_event_indices'] = altered['alignment_event_indices'].flip(-1)
    with torch.no_grad():
        memory_a, mask_a = model.encode(batch)
        memory_b, mask_b = model.encode(altered)
        regular, other = model(batch), model(altered)
    torch.testing.assert_close(memory_a, memory_b, rtol=0, atol=0)
    assert torch.equal(mask_a, mask_b)
    torch.testing.assert_close(regular['logits'], other['logits'], rtol=0, atol=0)


def test_event_query_hidden_cannot_read_current_position_digits():
    model, batch, codec = _model_and_batch()
    changed = copy.deepcopy(batch)
    boundary = int(batch['alignment_event_indices'][0, -1])
    digit = boundary + 2
    changed['decoder_input'][0, digit] = codec.ids['P:7']
    with torch.no_grad():
        before, after = model(batch), model(changed)
    torch.testing.assert_close(before['hidden_states'][:, :boundary+1],
                               after['hidden_states'][:, :boundary+1], rtol=0, atol=1e-7)
    torch.testing.assert_close(before['alignment_event_logits'][:, -1],
                               after['alignment_event_logits'][:, -1], rtol=0, atol=1e-7)


def test_audio_only_head_cannot_copy_authored_history_or_revision_draft():
    model, batch, codec = _model_and_batch()
    changed = copy.deepcopy(batch)
    changed['task'] = torch.ones(1, dtype=torch.long)
    changed['history'] = torch.tensor([[codec.bos_id, codec.eos_id]])
    changed['draft'] = torch.tensor([[codec.bos_id, codec.eos_id]])
    changed['right'] = torch.tensor([[codec.bos_id, codec.eos_id]])
    with torch.no_grad():
        before, after = model(batch), model(changed)
    torch.testing.assert_close(before['position_alignment_logits'], after['position_alignment_logits'],
                               rtol=0, atol=0)


def test_new_heads_backpropagate_to_decoder_and_audio_projection():
    model, batch, _ = _model_and_batch()
    output = model(batch)
    assert output['alignment_event_logits'].shape == (1, 2, 4)
    assert output['position_alignment_logits'].shape == (1, 4)
    losses, metrics = position_alignment_loss(output, batch)
    losses.sum().backward()
    for parameter in (model.alignment_query.weight, model.alignment_key.weight,
                      model.audio_projection.weight, model.decoder.layers[0].self_attn.in_proj_weight,
                      model.position_alignment_head[-1].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert all(torch.isfinite(value).all() for value in metrics.values())


def _loss_batch(rows=2, events=2, nodes=3):
    logits = torch.randn(rows, events, nodes, requires_grad=True)
    scores = torch.randn(rows, nodes, requires_grad=True)
    batch = {'alignment_event_indices': torch.zeros(rows, events, dtype=torch.long),
        'alignment_audio_index': torch.zeros(rows, events, dtype=torch.long),
        'alignment_event_mask': torch.ones(rows, events, dtype=torch.bool),
        'position_alignment_target': torch.zeros(rows, nodes),
        'position_alignment_mask': torch.ones(rows, nodes, dtype=torch.bool),
        'audio_padding_mask': torch.zeros(rows, nodes, dtype=torch.bool),
        'task': torch.zeros(rows, dtype=torch.long)}
    return {'alignment_event_logits': logits, 'position_alignment_logits': scores}, batch


def test_revision_auxiliary_has_zero_loss_metrics_and_gradients():
    prediction, batch = _loss_batch()
    batch['task'][1] = 1
    losses, metrics = position_alignment_loss(prediction, batch)
    assert losses[1] == 0
    assert all(value[1] == 0 for value in metrics.values())
    losses.sum().backward()
    for value in prediction.values():
        assert not value.grad[1].any()
        assert torch.isfinite(value.grad).all()


def test_masked_invalid_labels_and_nan_padding_are_zero_safe():
    prediction, batch = _loss_batch()
    batch['alignment_event_mask'][1] = False
    batch['alignment_audio_index'][1] = -17
    batch['alignment_event_indices'][1] = -1
    batch['audio_padding_mask'][1] = True
    with torch.no_grad():
        prediction['alignment_event_logits'][1] = float('nan')
        prediction['position_alignment_logits'][1] = float('nan')
    batch['position_alignment_target'][1] = float('nan')
    losses, metrics = position_alignment_loss(prediction, batch)
    assert torch.isfinite(losses).all() and losses[1] == 0
    losses.sum().backward()
    assert all(torch.isfinite(value.grad).all() for value in prediction.values())


@pytest.mark.parametrize('events,nodes', [(0, 0), (0, 3), (2, 0)])
def test_empty_auxiliary_dimensions_are_finite_and_differentiable(events, nodes):
    prediction, batch = _loss_batch(events=events, nodes=nodes)
    losses, metrics = position_alignment_loss(prediction, batch)
    assert losses.shape == (2,)
    assert torch.isfinite(losses).all()
    losses.sum().backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in prediction.values())


def test_event_loss_uses_its_own_valid_event_denominator():
    prediction, batch = _loss_batch(rows=1, events=3, nodes=2)
    batch['alignment_event_mask'][0] = torch.tensor([True, False, False])
    batch['position_alignment_mask'][:] = False
    loss, metrics = position_alignment_loss(prediction, batch)
    expected = torch.nn.functional.cross_entropy(prediction['alignment_event_logits'][:, 0],
                                                 batch['alignment_audio_index'][:, 0], reduction='none')
    torch.testing.assert_close(loss, expected)
    assert metrics['position_alignment_coverage'][0] == pytest.approx(1/3)


def _neighbor_reference(hidden, times, padding, seconds):
    output = torch.zeros((*hidden.shape[:2], hidden.size(2)*2), dtype=hidden.dtype)
    for row in range(hidden.size(0)):
        for index in range(hidden.size(1)):
            if padding[row, index]:
                continue
            for start, selection in ((0, range(index)), (hidden.size(2), range(index+1, hidden.size(1)))):
                keys = [other for other in selection if not padding[row, other] and
                        abs(times[row, other]-times[row, index]) <= seconds]
                if keys:
                    output[row, index, start:start+hidden.size(2)] = hidden[row, keys].mean(0)
    return output


def test_nonuniform_neighbors_match_reference_with_padding_holes_and_nan():
    times = torch.tensor([[0., float('nan'), .2, .3, float('nan'), 1.4],
                          [float('nan')]*6])
    padding = torch.tensor([[False, True, False, False, True, False], [True]*6])
    hidden = torch.randn(2, 6, 3)
    hidden[padding] = float('nan')
    hidden.requires_grad_()
    actual = temporal_neighbors(hidden, times, padding, seconds=.25)
    expected = _neighbor_reference(hidden.detach(), times, padding, .25)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(hidden.grad).all()
    assert not hidden.grad[padding].any()


def test_zero_context_still_hides_invalid_padded_nodes():
    hidden = torch.tensor([[[1.], [float('nan')]]], requires_grad=True)
    actual = temporal_neighbors(hidden, torch.tensor([[0., float('nan')]]),
                                torch.tensor([[False, True]]), seconds=0)
    torch.testing.assert_close(actual, torch.tensor([[[1., 1.], [0., 0.]]]))
    actual.sum().backward()
    assert torch.isfinite(hidden.grad).all()


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_low_precision_long_prefix_keeps_narrow_neighbor_means_and_gradients(dtype):
    hidden = torch.full((1, 1024, 2), .3, dtype=dtype, requires_grad=True)
    times = torch.arange(1024, dtype=torch.float64)[None] * .01
    reference_hidden = hidden.detach().float().requires_grad_()
    actual = temporal_neighbors(hidden, times, seconds=.015)
    reference = temporal_neighbors(reference_hidden, times, seconds=.015)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, reference.to(dtype), rtol=0, atol=0)
    assert actual[0, 1001, 0] == hidden[0, 1001, 0]
    actual.float().sum().backward()
    reference.sum().backward()
    assert torch.isfinite(hidden.grad).all()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad.to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize('times', [[[0., .2, .1]], [[0., float('nan'), .2]]])
def test_unmasked_invalid_neighbor_clock_is_rejected(times):
    with pytest.raises(ValueError, match='finite and times ordered'):
        temporal_neighbors(torch.ones(1, 3, 2), torch.tensor(times), seconds=1)


def _planner(profile=MUSICAL_EVIDENCE):
    torch.manual_seed(8241)
    return RationalPlanner(RationalPlannerConfig(width=16, form_heads=2, form_layers=1,
        descriptor_dim=5, dropout=0, quality_profile=profile, context_seconds=1.)).eval()


def test_planner_temporal_path_uses_real_seconds_not_adjacent_array_indices():
    planner = _planner()
    with torch.no_grad():
        planner.temporal_projection.weight.normal_(std=.1)
    audio = torch.randn(1, 4, 120)
    form, condition = torch.randn(1, 2, 8), torch.zeros(1, 16)
    near = torch.tensor([[0., .2, 5., 7.]])
    changed_near, changed_far = audio.clone(), audio.clone()
    changed_near[:, 1] += torch.randn(120)
    changed_far[:, 2] += torch.randn(120)
    with torch.no_grad():
        original = planner(audio, form, condition, audio_times_sec=near)['load_curve']
        nearby = planner(changed_near, form, condition, audio_times_sec=near)['load_curve']
        distant = planner(changed_far, form, condition, audio_times_sec=near)['load_curve']
    assert not torch.allclose(original[:, 0], nearby[:, 0])
    torch.testing.assert_close(original[:, 0], distant[:, 0], rtol=0, atol=0)


def test_planner_all_padding_and_empty_form_are_finite():
    planner = _planner()
    audio = torch.randn(2, 4, 120)
    audio[1] = float('nan')
    audio.requires_grad_()
    mask = torch.tensor([[False]*4, [True]*4])
    form = torch.full((2, 2, 8), float('nan'))
    times = torch.tensor([[0., .2, .8, 2.], [float('nan')]*4])
    result = planner(audio, form, torch.zeros(2, 16), audio_padding_mask=mask,
        form_padding_mask=torch.ones(2, 2, dtype=torch.bool), audio_times_sec=times)['load_curve']
    assert torch.isfinite(result).all()
    assert not result[1].any()
    result.sum().backward()
    assert torch.isfinite(audio.grad).all()
    empty = planner(torch.empty(1, 0, 120), torch.empty(1, 0, 8), torch.zeros(1, 16),
                    audio_times_sec=torch.empty(1, 0))
    loss = rational_planner_loss(empty, {'load_curve_target': torch.empty(1, 0, 4),
        'load_curve_mask': torch.empty(1, 0, 4, dtype=torch.bool), 'frame_weight': torch.empty(1, 0)})['loss']
    assert torch.isfinite(loss)
    loss.backward()


def test_legacy_model_has_no_alignment_output_or_parameters():
    config = FormConfig(vocab_size=10, d_model=16, nhead=2, layers=1, dim_feedforward=32,
                         audio_dim=6, descriptor_dim=5, condition_dim=3, dropout=0)
    torch.manual_seed(41)
    default = FormTransformer(config).eval()
    torch.manual_seed(41)
    explicit = FormTransformer(FormConfig(**{**asdict(config), 'quality_profile': 'legacy'})).eval()
    assert not any('alignment' in key or 'span_projection' in key for key in default.state_dict())
    batch = {'audio': torch.randn(1, 3, 6), 'form': torch.randn(1, 2, 8),
        'condition': torch.zeros(1, 3), 'task': torch.zeros(1, dtype=torch.long),
        'decoder_input': torch.tensor([[1, 2, 3]]), 'alignment_event_indices': torch.tensor([[0]])}
    result = default(batch)
    assert set(result) == {'logits', 'relation_logits', 'hidden_states'}
    torch.testing.assert_close(result['logits'], explicit(batch)['logits'], rtol=0, atol=0)


def test_planner_rejects_zero_heads_with_clear_validation():
    with pytest.raises(ValueError, match='form_heads'):
        RationalPlannerConfig(form_heads=0)


def test_planner_loss_mask_hides_nan_before_sigmoid_and_huber():
    logits = torch.full((2, 3, 4), float('nan'), requires_grad=True)
    targets = {'load_curve_target': torch.full((2, 3, 4), float('nan')),
               'load_curve_mask': torch.zeros(2, 3, 4, dtype=torch.bool),
               'frame_weight': torch.ones(2, 3)}
    loss = rational_planner_loss({'load_curve': logits}, targets)['loss']
    assert torch.isfinite(loss) and loss == 0
    loss.backward()
    assert torch.isfinite(logits.grad).all() and not logits.grad.any()


def test_legacy_planner_valid_input_keeps_original_forward_result():
    planner = _planner('legacy')
    audio, form, condition = torch.randn(2, 4, 120), torch.randn(2, 3, 8), torch.randn(2, 16)
    with torch.no_grad():
        positions = torch.arange(form.size(1))
        regions = planner.form_projection(form)+planner.form_position(positions)[None]
        regions = planner.form_encoder(regions, src_key_padding_mask=None)
        hidden = planner.audio_projection(audio)+planner.condition_projection(condition)[:, None]
        context, _ = planner.region_attention(hidden, regions, regions, key_padding_mask=None,
                                               need_weights=False)
        expected = planner.output(planner.norm(hidden+context))
        actual = planner(audio, form, condition)['load_curve']
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_enhanced_audio_measurements_survive_disabling_geometry():
    from maichart.form.data import organize_song
    from test_form_pipeline import fixture_audio
    original = fixture_audio(4)
    audio = SimpleNamespace(**asdict(original), feature_policy='legacy_sparse_v1')
    song = organize_song(audio, TEMPO, 0, quality_profile=MUSICAL_EVIDENCE,
                         rhythm_geometry_profile='none')
    assert song.rhythm_geometry_profile == 'none'
    assert audio.feature_policy == 'native_measured_v2'


def test_revision_span_crossing_region_clips_motif_request_to_region():
    from maichart.form.data import organize_song, input_example
    from maichart.form.timing import build_spans, MeterChange
    from test_form_pipeline import fixture_audio
    regions = build_spans(F(8), [MeterChange(F(0), 4, 4)], bars_per_span=1)
    song = organize_song(fixture_audio(5), TEMPO, 0, spans=regions, quality_profile=MUSICAL_EVIDENCE)
    revised = replace(song, spans=[replace(regions[0], end=F(6))], span_region_indices=[0])
    example = input_example(revised, 0, [], task=1)
    assert example['meta']['span']['end'] == '6'
    assert example['meta']['region']['end'] == '4'
