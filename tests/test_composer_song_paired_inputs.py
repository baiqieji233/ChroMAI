"""Exact event-time song pairing across authored and generated prefixes."""
from fractions import Fraction as F
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from maichart.composer.acoustics import AcousticContext
from maichart.composer.planning import PlanGrid
from maichart.composer.quality_inputs import quality_context_inputs, exact_history_audio
from maichart.event_timing import EventPoint, EventTempo
from maichart.tempo_clock import TempoClock


def fixture():
    tempo = [EventTempo(F(0), 120.), EventTempo(F(4), 180.)]
    clock = TempoClock(tempo, -.2)
    times = np.linspace(0., 9., 361)
    values = np.stack([np.sin(times*(i+1)*.13) for i in range(120)], axis=-1).astype(np.float32)
    context = AcousticContext(values, times, 9.)
    plan = PlanGrid(clock, 9., context)
    return tempo, clock, values, times, context, plan


@pytest.mark.parametrize('beats', [[], [F(1, 3)], [F(1, 3), F(7, 2), F(19, 4), F(13, 2)]])
def test_authored_slice_and_actual_prefix_pair_exact_same_audio(beats):
    _, clock, values, times, context, plan = fixture()
    points = [EventPoint(beat, '1') for beat in beats]
    ids = np.ones((len(points), 2), dtype=np.int64)
    kwargs = dict(atom_ids=ids, candidate_beats=[7., 8.], candidate_seconds=clock.at_floats([7., 8.]),
        state_index=None, audio_features=values, audio_times=times, acoustic_context=context,
        clock=clock, audio_duration=9., policy='song_paired_v3', plan_grid=plan)
    generated = quality_context_inputs(points=points, **kwargs)
    seconds = clock.at_floats(np.asarray(beats, dtype=np.float64))
    authored = quality_context_inputs(points=(), point_beats=beats, point_seconds=seconds,
        history_audio=context.at(seconds), **kwargs)
    for key in generated:
        torch.testing.assert_close(generated[key], authored[key], rtol=0, atol=0)
    assert not any(key.startswith('quality_anchor_') for key in generated)
    np.testing.assert_array_equal(generated['quality_history_audio'], context.at(seconds))
    np.testing.assert_allclose(generated['quality_history_clock'][:, 1], seconds)
    clock_rows = generated['quality_plan_clock'].numpy()
    np.testing.assert_allclose(clock.at_floats(clock_rows[:, 0]), clock_rows[:, 1], atol=2e-6)
    expected_indices, expected_mix = plan.locate(seconds)
    np.testing.assert_array_equal(generated['quality_history_plan_indices'], expected_indices)
    np.testing.assert_array_equal(generated['quality_history_plan_mix'], expected_mix)


def test_exact_prefix_audio_cache_is_bounded_and_uses_synthetic_seconds():
    _, _, _, _, context, _ = fixture()
    seconds = np.linspace(.001, 8.999, 8300)
    rows = exact_history_audio(context, seconds)
    np.testing.assert_array_equal(rows, context.at(seconds))
    assert len(context._quality_history_cache[0]) == 8192
    again = exact_history_audio(context, seconds[-16:])
    np.testing.assert_array_equal(again, rows[-16:])


def test_recovery_tail_aligns_song_pairing_with_symbolic_history_and_padding():
    from maichart.composer.candidates import MUSIC_CANDIDATE_POLICY_V6
    from maichart.composer.data import collate_composer
    from maichart.composer.event_views import EventViews
    from maichart.composer.recovery import RecoveryBuilder
    tempo, clock, values, times, context, plan = fixture()
    source = [EventPoint(F(i), '1') for i in range(12)]
    song = SimpleNamespace(tempo=tempo, first=-.2, audio=SimpleNamespace(duration_sec=9.))
    table = dict(clock=clock, condition=torch.zeros(16), audio_duration=9., planner_data=plan)
    def grounded(*args, **kwargs):
        indices, mix = plan.locate(args[2])
        return dict(planner_audio=torch.from_numpy(plan.audio), planner_mask=torch.zeros(len(plan.centers), dtype=torch.bool),
            planner_target=torch.from_numpy(plan.targets), planner_weight=torch.from_numpy(plan.weights),
            planner_indices=torch.from_numpy(indices), planner_mix=torch.from_numpy(mix))
    dataset = SimpleNamespace(samples=[SimpleNamespace(song_index=0, event_index=1)],
        base=SimpleNamespace(songs=[(song,)]), event_views={0: EventViews.build(source, tempo, -.2, 9.)},
        candidate_count=96, candidate_policy=MUSIC_CANDIDATE_POLICY_V6, architecture_profile='audio_grounded_v3',
        history_events=2, action_slots=3, codec_atoms=['1'], atom_to_id={'1':1},
        quality_inputs_enabled=True, quality_inputs_policy='song_paired_v3',
        _song_table=lambda index:table, _song_features=lambda index:(values,times), _grounded_features=grounded)
    prefix = [EventPoint(F(0), '1'), EventPoint(F(13, 3), '1'), EventPoint(F(19, 3), '1')]
    full, _ = RecoveryBuilder(dataset)._inputs(prefix, 0)
    empty, _ = RecoveryBuilder(dataset)._inputs([], 0)
    assert len(full['quality_history_audio']) == len(full['history']) == 2
    np.testing.assert_allclose(full['quality_history_clock'][:,0], [13/3,19/3])
    assert not full['quality_phrase_weight'].any()
    assert full['quality_phrase_target'].size(-1) == 9
    batch = collate_composer([full, empty])
    assert batch['history_padding_mask'][1].all()
    assert not batch['quality_history_audio'][1].any()
    assert not batch['quality_history_plan_mix'][1].any()
    assert batch['quality_history_plan_indices'].dtype == torch.long
    assert batch['quality_history_audio'].shape[:2] == batch['history_padding_mask'].shape


def test_song_phrase_fine_rhythm_uses_shared_event_ctx_definition():
    from maichart.composer.event_views import EventViews
    from maichart.composer.quality_inputs import quality_targets, phrase_audio_bank
    from maichart.training.music_plan_targets import fine_note_ratio
    tempo, clock, _, _, context, _ = fixture()
    points = [EventPoint(F(1), '1'), EventPoint(F(5,4), '1'), EventPoint(F(2), '1')]
    views = EventViews.build(points, tempo, -.2, 9.)
    bank = phrase_audio_bank(context, clock, 9.)
    legacy = quality_targets(points, clock, bank, views)
    paired = quality_targets(points, clock, bank, views, policy='song_paired_v3')
    assert legacy['phrase_target'].shape[-1] == 8
    assert paired['phrase_target'].shape[-1] == 9
    assert paired['phrase_target'][0,8] == fine_note_ratio([p.beat for p in points])
    torch.testing.assert_close(paired['phrase_target'][...,:8],legacy['phrase_target'])


def test_same_source_different_chart_labels_share_one_raw_song_bank(monkeypatch):
    from maichart.composer.data import collate_composer
    from maichart.composer.event_views import EventViews
    from maichart.composer.quality_inputs import plan_clock_features
    tempo, clock, values, times, context, _ = fixture()
    calls = []
    original_at = context.at
    def at(seconds):
        calls.append(np.asarray(seconds).copy())
        return original_at(seconds)
    monkeypatch.setattr(context, 'at', at)
    chart_a = [EventPoint(F(1), '1')]
    chart_b = [EventPoint(F(1), '1/3'), EventPoint(F(2), '2'), EventPoint(F(3), '4')]
    grids = [PlanGrid(clock, 9., context, EventViews.build(points, tempo, -.2, 9.), share_audio=True)
             for points in (chart_a, chart_b)]
    assert len(calls) == 1
    assert grids[0].audio is grids[1].audio
    assert not np.array_equal(grids[0].targets, grids[1].targets)
    assert grids[0].targets is not grids[1].targets
    rows = []
    for index, grid in enumerate(grids):
        bank = grid._shared_raw_plan_bank
        quality = quality_context_inputs(points=[], atom_ids=np.zeros((0,1),dtype=np.int64),
            candidate_beats=[1.], candidate_seconds=clock.at_floats([1.]), state_index=None,
            audio_features=values, audio_times=times, acoustic_context=context, clock=clock,
            audio_duration=9., policy='song_paired_v3', plan_grid=grid)
        rows.append(dict(candidate_features=torch.zeros(1,8), candidate_target=torch.tensor(0),
            action_target=torch.zeros(1,dtype=torch.long), action_state=torch.zeros(1,8),
            future_target=torch.zeros(8), condition=torch.full((16,),float(index)),
            history=torch.zeros(0,1,dtype=torch.long), history_features=torch.zeros(0,8),
            candidate_mask=torch.zeros(1,dtype=torch.bool), song_index=torch.tensor(index),
            event_index=torch.tensor(0), eos_target=torch.tensor(0.),
            planner_audio=bank['audio_tensor'], planner_mask=bank['mask'],
            planner_target=torch.from_numpy(grid.targets), planner_weight=torch.from_numpy(grid.weights),
            quality_phrase_target=torch.zeros(len(quality['quality_phrase_audio']),9),
            quality_phrase_weight=torch.zeros(len(quality['quality_phrase_audio'])), **quality))
    batch = collate_composer(rows)
    assert batch['song_audio'].shape[0] == 1
    assert batch['song_inverse'].tolist() == [0,0]
    assert 'planner_audio' not in batch and 'planner_mask' not in batch
    assert 'quality_plan_clock' not in batch
    torch.testing.assert_close(batch['song_audio'][batch['song_inverse']],
                               torch.stack([row['planner_audio'] for row in rows]))
    torch.testing.assert_close(batch['song_clock'][batch['song_inverse']],
                               torch.stack([row['quality_plan_clock'] for row in rows]))
    assert not torch.equal(batch['planner_target'][0], batch['planner_target'][1])
    assert plan_clock_features(grids[0],clock).data_ptr() == plan_clock_features(grids[1],clock).data_ptr()


def test_compact_song_batch_forward_and_gradients_match_repeated_inputs():
    import copy
    from maichart.composer.train import _forward_batch
    from test_composer_echo_stage2_integration import _echo_fixture
    model, data = _echo_fixture('song_paired_v3')
    model.train()
    other = copy.deepcopy(model)
    # Both rows refer to the same immutable song but retain distinct controls.
    data['planner_audio'][1] = data['planner_audio'][0]
    data['planner_mask'][1] = data['planner_mask'][0]
    batch = {k: v for k, v in data.items() if k not in ('quality_inputs', 'candidate_indices', 'action_targets', 'candidate_mask')}
    batch.update(data['quality_inputs'])
    batch.update(candidate_target=data['candidate_indices'], action_target=data['action_targets'],
                 candidate_padding_mask=data['candidate_mask'], song_audio=data['planner_audio'][:1],
                 song_mask=data['planner_mask'][:1], song_clock=data['quality_inputs']['quality_plan_clock'][:1],
                 song_inverse=torch.tensor([0, 0]), song_unique_indices=torch.tensor([0]))
    compact = {k: v for k, v in batch.items() if k not in ('planner_audio', 'planner_mask', 'quality_plan_clock')}
    outputs, grads, rng = [], [], []
    for net, sample in ((model, batch), (other, compact)):
        torch.manual_seed(1729)
        result = _forward_batch(net, sample, packed_actions=True)
        loss = result['plan'].square().sum() + result['candidate_logits'].square().sum() + result['action_logits'].square().sum()
        loss.backward()
        outputs.append(result)
        grads.append({k: p.grad for k, p in net.named_parameters() if p.grad is not None})
        rng.append(torch.get_rng_state())
    assert torch.equal(rng[0], rng[1])
    for key in ('plan', 'candidate_logits', 'action_logits'):
        torch.testing.assert_close(outputs[0][key], outputs[1][key], rtol=0, atol=0)
    assert grads[0].keys() == grads[1].keys()
    for key in grads[0]:
        torch.testing.assert_close(grads[0][key], grads[1][key], rtol=0, atol=0)


def test_raw_song_grid_cache_keeps_tempo_offset_and_audio_identity_distinct():
    from maichart.composer.quality_inputs import shared_raw_plan_bank
    tempo, clock, _, _, context, plan = fixture()
    centers = plan.centers
    baseline = shared_raw_plan_bank(context, clock, centers)
    same = shared_raw_plan_bank(context, TempoClock(tempo,-.2), centers.copy())
    assert baseline is same
    changed_first = shared_raw_plan_bank(context, TempoClock(tempo,.1), centers)
    changed_bpm = shared_raw_plan_bank(context, TempoClock([EventTempo(F(0),130.)],-.2), centers)
    assert baseline is not changed_first and baseline is not changed_bpm
    assert baseline['audio_tensor'].data_ptr() != changed_first['audio_tensor'].data_ptr()
    _, _, _, _, other_context, _ = fixture()
    other_source = shared_raw_plan_bank(other_context, clock, centers)
    assert baseline is not other_source
    for shift in range(8):
        shared_raw_plan_bank(context, TempoClock(tempo,float(shift)), centers)
    assert len(context._quality_raw_plan_cache[0]) == 4
