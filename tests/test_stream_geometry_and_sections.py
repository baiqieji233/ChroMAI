"""Mirror symmetry, causal hand kinematics, relative-time grounding and phrase energy."""
from fractions import Fraction as F
import math

import numpy as np
import pytest
import torch

from maichart.form.type_guidance import _counts
from maichart.generation.muri.model import CENTERS, PADS
from maichart.stream.model import StreamConfig
from maichart.stream.symmetry import (KINEMATIC_DIM, TRANSFORMS, _BOUNDARY, _BUTTON, event_lanes,
                                      kinematic_rows, state_permutation, transform_state, transform_text)
from maichart.stream.grounded_model import (GroundedContext, UPGRADE_PARAMETERS, phrase_features,
                                            relative_features)
from maichart.stream.section_profile import SectionTypeProfile

SIGN = {'identity': (1, 1), 'mirror_lr': (-1, 1), 'mirror_ud': (1, -1), 'rotate_180': (-1, -1)}
CELLS = ['1', '1/8', '1-5[8:1]', '3h[4:1]', '1>5[8:1]', '7<3[4:1]', '1p5[4:1]', '2qq7[4:1]', '1s5[4:1]',
         '1V35[4:1]', '1-4[8:1]*-6[8:1]', 'A1/D2', 'E8f', 'Ch[2:1]', 'C1', 'B3/B6', '12', '4bx', '1h[#0.5]']


@pytest.mark.parametrize('transform', TRANSFORMS)
def test_button_maps_follow_cabinet_geometry(transform):
    for k in range(1, 9):
        for letter, fn in (('A', _BUTTON[transform]), ('B', _BUTTON[transform]),
                           ('D', _BOUNDARY[transform]), ('E', _BOUNDARY[transform])):
            x, y = CENTERS[f'{letter}{k}']
            mx, my = CENTERS[f'{letter}{fn(k)}']
            assert math.dist((SIGN[transform][0]*x, SIGN[transform][1]*y), (mx, my)) < 1e-6


@pytest.mark.parametrize('transform', TRANSFORMS)
@pytest.mark.parametrize('text', CELLS)
def test_mirroring_is_an_involution_and_preserves_note_types(transform, text):
    mirrored = transform_text(text, transform)
    assert transform_text(mirrored, transform) == text
    assert (_counts(text) == _counts(mirrored)).all()


def test_mirroring_swaps_left_right_slide_senses_and_keeps_durations():
    assert transform_text('1>5[8:1]', 'mirror_lr') == '8<4[8:1]'
    assert transform_text('1p5[4:1]', 'mirror_ud') == '4q8[4:1]'
    assert transform_text('1h[#0.5]', 'rotate_180') == '5h[#0.5]'
    assert transform_text('C1', 'mirror_lr') == 'C1'     # center sensor has no lane


@pytest.mark.parametrize('transform', TRANSFORMS)
def test_state_permutation_is_a_permutation_matching_pad_geometry(transform):
    permutation = state_permutation(transform)
    assert sorted(permutation.tolist()) == list(range(50))
    state = torch.zeros(1, 51)
    state[0, 4+PADS.index('A1')] = 1
    state[0, -1] = .25
    moved = transform_state(state, transform)
    target = f'A{_BUTTON[transform](1)}'
    assert moved[0, 4+PADS.index(target)] == 1 and moved[0, -1] == .25
    assert moved[:, :50].sum() == 1


def test_event_lanes_ignore_touch_and_cover_every_struck_button():
    assert event_lanes('1/8') == (1, 8) and event_lanes('A1/D2') == () and event_lanes('12') == (1, 2)
    assert event_lanes('3h[4:1]') == (3,) and event_lanes('1-5[8:1]') == (1,)


def test_kinematics_are_causal_and_fill_pure_tap_passages():
    lanes = [(1,), (2,), (3,), (4,), (5,)]
    seconds = [0., .2, .4, .6, .8]
    rows = kinematic_rows(lanes, seconds, [0, 3, 3, 5], [0., .5, .5, 1.])
    assert not rows[0].any()                       # nothing committed: empty
    assert rows[1].abs().sum() > 0 and torch.equal(rows[1], rows[2])
    # Decay vector: lane 3 (latest) is warmer than lane 1 (oldest) and lane 4 was never struck.
    assert rows[1][2] > rows[1][0] > 0 and rows[1][3] == 0
    assert rows[1][8+2] == 1                        # most recent event struck lane 3
    # Future events must never leak: changing event 3/4 cannot change a count=3 query.
    altered = kinematic_rows([(1,), (2,), (3,), (8,), (8,)], seconds, [3], [.5])
    assert torch.equal(altered[0], rows[1])
    assert rows.shape[1] == KINEMATIC_DIM


def test_relative_features_distinguish_past_from_future_beats():
    delta = torch.tensor([-1., 0., 1.])
    values = relative_features(delta, torch.full((3,), 120.), 2.)
    assert values[0, 0] < 0 < values[2, 0] and values[1].abs().sum() == pytest.approx(values[1, 4]+values[1, 6]+values[1, 8])
    # Whole-beat offsets repeat the beat-phase columns; only signed offset separates them.
    one_beat = relative_features(torch.tensor([-.5, .5]), torch.full((2,), 120.), 2.)
    torch.testing.assert_close(one_beat[0, 4], one_beat[1, 4])
    assert one_beat[0, 0] != one_beat[1, 0]


def test_local_attention_can_prefer_the_coming_onset_after_training_step():
    config = StreamConfig(vocab_size=55, d_model=32, nhead=4, layers=1, dim_feedforward=64, local_audio_neighbors=8,
                          grounding_profile='causal_audio_v1')
    context = GroundedContext(config)
    for name, parameter in context.named_parameters():
        assert (name in {n.split('grounding.')[1] for n in UPGRADE_PARAMETERS}) == bool(
            name in {n.split('grounding.')[1] for n in UPGRADE_PARAMETERS})
    times = torch.arange(0, 4, .25, dtype=torch.float64)[None]
    audio = torch.randn(1, len(times[0]), 120)
    bank = context.prepare_bank(audio, times, torch.zeros(1, len(times[0]), dtype=torch.bool))
    clock = torch.zeros(1, 1, 20); clock[..., 14] = .25; clock[..., 19] = .4
    seconds = torch.tensor([[1.]], dtype=torch.float64)
    center = torch.randn(1, 1, 120)
    baseline = context.local(clock, seconds, center, bank)
    with torch.no_grad():
        context.local_distance_value.weight[:, 0].fill_(1.)   # signed seconds now visible
    shifted = context.local(clock, seconds, center, bank)
    assert not torch.allclose(baseline, shifted)              # offsets reach the output


def test_upgrade_parameters_are_zero_initialised_and_registered():
    config = StreamConfig(vocab_size=55, d_model=32, nhead=4, layers=1, dim_feedforward=64,
                          grounding_profile='causal_audio_v1')
    from maichart.stream.model import ChartStreamTransformer
    model = ChartStreamTransformer(config)
    names = dict(model.named_parameters())
    for key in UPGRADE_PARAMETERS:
        assert key in names and float(names[key].abs().sum()) == 0.


def test_phrase_features_pool_only_audio_curve_inside_each_window():
    times = torch.arange(0, 40, 1., dtype=torch.float64)[None]
    curve = torch.zeros(1, 40, 4)
    curve[:, 20:, 0] = 5.                                    # logit 5: loud second half
    padding = torch.zeros(1, 40, dtype=torch.bool)
    early = phrase_features(curve.sigmoid(), times, padding, torch.tensor([[5.]], dtype=torch.float64))
    late = phrase_features(curve.sigmoid(), times, padding, torch.tensor([[30.]], dtype=torch.float64))
    assert early.shape[-1] == 16
    # Window 2 (next 12 s) from t=5 is quiet; from t=30 it is loud. Previous-section window flips too.
    assert late[0, 0, 8] > early[0, 0, 8] + .1
    assert early[0, 0, 12] <= .55 and late[0, 0, 12] > .6


def test_phrase_features_integrate_nonuniform_audio_time_cells():
    # Four nodes occupy unequal portions of the [0, 4 s] phrase window.  A
    # node-count mean would be .5, while the Voronoi time integral is .75.
    times = torch.tensor([[0., .1, 1.9, 2., 3.]], dtype=torch.float64)
    curve = torch.zeros(1, 5, 4)
    curve[0, :, 0] = torch.tensor([0., 0., 1., 1., 100.])
    padding = torch.tensor([[False, False, False, False, True]])
    result = phrase_features(curve, times, padding, torch.tensor([[0.]], dtype=torch.float64))
    assert result[0, 0, 4].item() == pytest.approx(.75)


def test_section_profile_shifts_type_mix_without_changing_song_mean():
    times = np.arange(0, 60, 1.)
    curve = np.zeros((60, 4)); curve[:, 0] = .5; curve[:, 1] = 1.; curve[30:, 1] = .2
    curve[30:, 2] = .8; curve[30:, 3] = .5
    curve[:30, 0] = .2; curve[30:, 0] = .8
    profile = SectionTypeProfile(curve, times)
    early, late = profile(10.), profile(50.)
    assert early[0] > late[0] and late[2] > early[2]          # tap-heavy first half, hold-heavy second
    assert np.all(early > 0) and np.all(late > 0)
    assert profile.density_factor(10.) < 1. < profile.density_factor(50.)


def test_valid_symmetries_supports_mirror_lr_only_mode():
    from maichart.event_timing import EventPoint
    from maichart.stream.data import _valid_symmetries
    from maichart.stream.train import parser, validate_training_args
    points = [EventPoint(F(0), '1-4[8:1]/8-5[8:1]'), EventPoint(F(1), '1/8')]
    assert _valid_symmetries(points, 'none') == ('identity',)
    assert _valid_symmetries(points, 'mirror_lr') == ('identity', 'mirror_lr')
    assert _valid_symmetries(points, 'mirror4') == ('identity', 'mirror_lr', 'mirror_ud', 'rotate_180')
    default_args = parser().parse_args(['--manifest', 'm.json', '--checkpoint-dir', 'out'])
    assert default_args.symmetry_augmentation == 'mirror_lr'
    for mode in ('none', 'mirror_lr', 'mirror4'):
        default_args.symmetry_augmentation = mode
        validate_training_args(default_args)
    default_args.symmetry_augmentation = 'invalid'
    with pytest.raises(ValueError, match='symmetry_augmentation'):
        validate_training_args(default_args)


def test_kinematics_distinguish_sweeps_from_alternating_trills():
    seconds = [0., .125, .25, .375]
    cw_sweep = kinematic_rows([(1,), (2,), (3,), (4,)], seconds, [4], [.5])[0]
    ccw_sweep = kinematic_rows([(8,), (7,), (6,), (5,)], seconds, [4], [.5])[0]
    trill = kinematic_rows([(1,), (5,), (1,), (5,)], seconds, [4], [.5])[0]
    # t-2 hand mask (channels 24..31): in cw_sweep t-2 is lane 3; in trill t-2 is lane 1.
    assert cw_sweep[24 + 2] == 1. and cw_sweep[24:32].sum() == 1.
    assert trill[24 + 0] == 1. and trill[24:32].sum() == 1.
    # 2-step angular velocity (channels 32..33) matches 1-step velocity in a sweep.
    torch.testing.assert_close(cw_sweep[32:34], cw_sweep[18:20])
    # Signed sweep_run (channel 34) is +3/8 for clockwise 4-note sweep, -3/8 for CCW, 0 for trill.
    assert cw_sweep[34] == pytest.approx(3 / 8)
    assert ccw_sweep[34] == pytest.approx(-3 / 8)
    assert trill[34] == pytest.approx(0.)
    # Alternating trill_run (channel 35) is 2/8 after 4 alternating notes, 0 for sweeps.
    assert trill[35] == pytest.approx(2 / 8)
    assert cw_sweep[35] == pytest.approx(0.)


def test_kinematics_keep_a_long_slide_visible_beyond_strike_horizon():
    # The Slide starts at event 0, while the query is ten committed events
    # later.  A short strike horizon must not erase the still-active trajectory.
    count = 10
    lanes = [(1,)] + [(2,)] * (count - 1)
    seconds = [float(index) for index in range(count)]
    trajectories = [((1, 1, 5, 0.5, 20.0),)] + [()] * (count - 1)
    row = kinematic_rows(lanes, seconds, [count], [10.0], horizon=1,
                         trajectories=trajectories)[0]
    assert row[44 + 4] >= .5      # destination lane 5 remains in flight
    assert row[52 + 0] >= .5      # origin lane 1 remains occupied
    assert row[60] == pytest.approx(.25)


def test_kinematics_ignore_uncommitted_trajectory_entries():
    lanes = [(1,), (2,)]
    seconds = [0.0, 1.0]
    trajectories = [(), ((1, 2, 8, 1.1, 4.0),)]
    row = kinematic_rows(lanes, seconds, [1], [1.0], trajectories=trajectories)[0]
    assert not row[44:60].any()


def test_local_audio_lookup_spans_full_horizon_on_dense_8192_grid():
    from maichart.stream.model import StreamConfig
    from maichart.stream.grounded_model import GroundedContext
    cfg = StreamConfig(vocab_size=55, d_model=32, nhead=4, layers=1, dim_feedforward=64,
                       grounding_profile='causal_audio_v1', detail_audio_nodes=8192,
                       local_audio_neighbors=32, local_audio_seconds=2.)
    ctx = GroundedContext(cfg)
    times = torch.linspace(0., 128., 8192, dtype=torch.float64)[None]
    values = torch.zeros(1, 8192, cfg.audio_dim)
    padding = torch.zeros(1, 8192, dtype=torch.bool)
    bank = ctx.prepare_bank(values, times, padding)
    seconds = torch.tensor([[30.]], dtype=torch.float64)
    out = ctx.local(torch.zeros(1, 1, 20), seconds, torch.zeros(1, 1, cfg.audio_dim), bank)
    assert out.shape == (1, 1, 32)
    searchable = times.contiguous()
    right = torch.searchsorted(searchable, seconds.contiguous())
    width = cfg.local_audio_neighbors
    radius = float(cfg.local_audio_seconds)
    back_count = max(1, width // 4)
    fwd_count = width - back_count
    low = torch.searchsorted(searchable, (seconds - .25 * radius).contiguous())
    high = (torch.searchsorted(searchable, (seconds + radius).contiguous(), right=True) - 1).clamp_min(0)
    back_excess = ((right - low) - back_count).clamp_min(0)[..., None]
    fwd_excess = ((high - right) - max(1, fwd_count - 1)).clamp_min(0)[..., None]
    b = torch.arange(back_count, 0, -1)
    f = torch.arange(fwd_count)
    back_offsets = -b - torch.round((b / max(1, back_count)) ** 1.6 * back_excess).long()
    fwd_offsets = f + torch.round((f / max(1, fwd_count - 1)) ** 1.6 * fwd_excess).long()
    raw = (right[..., None] + torch.cat((back_offsets, fwd_offsets), -1)).clamp(0, 8191)
    sampled = times[0, raw[0, 0]]
    assert torch.all(sampled[1:] > sampled[:-1])
    assert float(sampled[-1] - 30.) > 1.95
    assert float(30. - sampled[0]) > .45
    assert float(sampled[back_count + 1] - sampled[back_count]) < .05


def test_masked_clock_inputs_zeros_interval_echo_only_on_hidden_or_dropped_events():
    import random
    from types import SimpleNamespace
    from maichart.form.codec import RationalCodec
    from maichart.event_timing import EventPoint, EventTempo
    from maichart.stream.grounding import clock_inputs, masked_clock_inputs
    codec = RationalCodec()
    points = [EventPoint(F(0), '1'), EventPoint(F(1, 2), '2'), EventPoint(F(1), '3/4')]
    ids = codec.encode(points)
    decoder = ids[:-1]
    targets = ids[1:]
    song = SimpleNamespace(tempo=[EventTempo(F(0), 120.)], first=0.,
                           audio=SimpleNamespace(duration_sec=10.), meters=None)
    rows = clock_inputs(decoder, codec, song)
    untouched_d, untouched_q = masked_clock_inputs(
        rows['decoder_clock'], decoder, codec, random.Random(0), (),
        dropout=0., query_clock=rows['query_clock'],
        query_index=rows['decoder_query_index'], targets=targets)
    assert torch.equal(untouched_d, rows['decoder_clock'])
    assert torch.equal(untouched_q, rows['query_clock'])
    masked_d, masked_q = masked_clock_inputs(
        rows['decoder_clock'], decoder, codec, random.Random(0), [1],
        dropout=0., query_clock=rows['query_clock'],
        query_index=rows['decoder_query_index'], targets=targets)
    # Event 1 (<note> of beat 1/2 until <note> of beat 1) has delta/delta_sec zeroed,
    # while beat/phase/tempo columns remain identical.
    assert torch.equal(masked_d[:, :15], rows['decoder_clock'][:, :15])
    assert torch.equal(masked_d[:, 17:], rows['decoder_clock'][:, 17:])
    assert float(rows['decoder_clock'][:, 15:17].abs().sum()) > float(masked_d[:, 15:17].abs().sum())
    assert float(masked_q[:, 15:17].abs().sum()) < float(rows['query_clock'][:, 15:17].abs().sum())


def test_cycle_exit_targets_marks_rhythm_pattern_exit_and_long_periods():
    from maichart.form.codec import RationalCodec
    from maichart.event_timing import EventPoint
    from maichart.stream.continuation import cycle_exit_targets
    codec = RationalCodec()
    # Two 4-interval composite rhythm cycles [1/2, 1/4, 1/4, 1] followed by a departure to delta=1/2
    deltas = [F(1, 2), F(1, 4), F(1, 4), F(1)] * 2 + [F(1, 2), F(1, 2)]
    beats = [F(0)]
    for d in deltas:
        beats.append(beats[-1] + d)
    # Distinct lane rotations so cell/lane cycle doesn't trigger before rhythm exit
    points = [EventPoint(b, str(1 + (i * 3) % 8)) for i, b in enumerate(beats)]
    ids = codec.encode(points)
    mask = cycle_exit_targets(ids, codec)
    flagged = [codec.tokens[t] for t, m in zip(ids, mask) if m]
    assert any(tok.startswith('P:') for tok in flagged)


def test_rhythm_cycle_cost_penalizes_composite_loops_and_spares_steady_streams():
    from maichart.event_timing import EventPoint
    from maichart.stream.timing import rhythm_cycle_cost
    # Steady 8th-note stream (20 notes) must have zero rhythm cycle cost
    steady = [EventPoint(F(i, 2), '1') for i in range(20)]
    assert rhythm_cycle_cost(F(20, 2), steady, F(19, 2), .6) == 0.
    # Composite 1-bar rhythm pattern from bcf5caf4f5ad:
    # [1/2, 1/4, 1/4, 1/2, 1/4, 1/4, 1/2, 1/2, 1/2, 1/2] (10 intervals summing to 4 beats)
    bar_deltas = [F(1, 2), F(1, 4), F(1, 4), F(1, 2), F(1, 4), F(1, 4), F(1, 2), F(1, 2), F(1, 2), F(1, 2)]
    beats = [F(0)]
    for d in bar_deltas * 3:
        beats.append(beats[-1] + d)
    points = [EventPoint(b, '1') for b in beats]
    prev = points[-1].beat
    # Continuing the 4th bar with F(1, 2) triggers both multi-period and bar-phase penalties
    loop_cost = rhythm_cycle_cost(prev + F(1, 2), points, prev, .6)
    break_cost = rhythm_cycle_cost(prev + F(1), points, prev, .6)
    assert loop_cost > .8
    assert break_cost == 0.


def test_type_guidance_damps_multi_bar_phase_motif_cloning():
    from maichart.form.type_guidance import TypeGuidance
    from maichart.generation.conditions import GenerationCondition, GENERATION_DEFAULTS
    atoms = ['1', '2', '1b', '1-4[8:1]', '2h[4:1]']
    cond = GenerationCondition(level=13., radar=GENERATION_DEFAULTS, style=GENERATION_DEFAULTS)
    g = TypeGuidance(atoms, cond, [], planner_strength=0., style_strength=1.)
    # Commit Slide + Chord + Break at beat phase 1 in bar 0 (beat 1) and bar 1 (beat 5)
    g.commit('1b/2-5[8:1]', F(1))
    g.commit('1b/2-5[8:1]', F(5))
    bias_same_phase = g.event(F(9))
    bias_other_phase = g.event(F(10))
    assert bias_same_phase.phase_chord_run == 2
    assert bias_same_phase.phase_break_run == 2
    assert bias_other_phase.phase_chord_run == 0
    assert bias_same_phase.style_log[1] < bias_other_phase.style_log[1]
    tb_same = bias_same_phase.token_bias('1', ['/', 'b'])
    tb_other = bias_other_phase.token_bias('1', ['/', 'b'])
    assert tb_same[0] < tb_other[0] and tb_same[1] < tb_other[1]


def test_webui_exposes_rhythm_diversity_and_updated_cycle_exit_default():
    from pathlib import Path
    from maichart.stream.generate import parser as gen_parser
    from maichart.stream.train import parser as train_parser
    gen_args = gen_parser().parse_args(['--audio', 'a.wav', '--checkpoint', 'c.pt', '--output', 'o.txt'])
    train_args = train_parser().parse_args(['--manifest', 'm.json', '--checkpoint-dir', 'out'])
    assert gen_args.rhythm_diversity_strength == pytest.approx(.6)
    assert train_args.cycle_exit_loss_weight == pytest.approx(.25)
    html = Path('src/maichart/webui/static/index.html').read_text(encoding='utf-8')
    js = Path('src/maichart/webui/static/app.js').read_text(encoding='utf-8')
    assert 'id="streamRhythmDiversityStrength"' in html
    assert 'streamRhythmDiversityStrength' in js
    assert 'rhythm_diversity_strength' in js
