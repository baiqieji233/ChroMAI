from __future__ import annotations

from fractions import Fraction
import numpy as np
import torch

from maichart.composer.audio_lens import PlacementTargets
from maichart.composer.candidates import (
    MUSIC_CANDIDATE_POLICY_V6,
    is_playable_candidate_transition,
)
from maichart.composer.configuration_observation import configuration_observation
from maichart.composer.conserved_actions import build_audio_macro_groups
from maichart.composer.curve import apply_curve_to_composer
from maichart.composer.generate import _has_future_planned_activity, evaluate_eos_proposal
from maichart.composer.guard import EventGuard
from maichart.composer.model import ComposerConfig, build_model
from maichart.composer.objectives import (
    JOINT_ACTION_OBJECTIVE,
    RECOVERY_ACTION_OBJECTIVE,
    action_objective,
    recovery_objective,
)
from maichart.event_timing import EventPoint, EventTempo


def test_cross_grid_transition_allows_ternary_to_binary_and_blocks_micro_step():
    prev = Fraction(1, 3)
    # 1/3 -> 3/4 (delta = 5/12 >= 1/24) and 1/3 -> 7/8 (delta = 13/24 >= 1/24) are allowed
    assert is_playable_candidate_transition(Fraction(3, 4), prev, MUSIC_CANDIDATE_POLICY_V6)
    assert is_playable_candidate_transition(Fraction(7, 8), prev, MUSIC_CANDIDATE_POLICY_V6)
    # 1/3 -> 3/8 (delta = 1/24, cross-grid denominator 24 < 1/16) is blocked as a micro-bridge
    assert not is_playable_candidate_transition(Fraction(3, 8), prev, MUSIC_CANDIDATE_POLICY_V6)
    # Same-grid 1/3 -> 2/3 is allowed
    assert is_playable_candidate_transition(Fraction(2, 3), prev, MUSIC_CANDIDATE_POLICY_V6)


def test_guard_accepts_zero_duration_hold_and_reports_rhythm_observation():
    guard = EventGuard([EventTempo(Fraction(0), 180.0)], 0.0, 10.0)
    assert guard.accepts(EventPoint(Fraction(0, 1), '1h[16:0]'))
    assert guard.last_reason is None

    obs = configuration_observation([
        EventPoint(Fraction(0, 1), '1'),
        EventPoint(Fraction(1, 4), '2'),
        EventPoint(Fraction(1, 2), '3'),
        EventPoint(Fraction(5, 6), '4'),
        EventPoint(Fraction(1, 1), '5'),
    ])
    assert 'rhythm' in obs
    assert obs['rhythm']['rhythm_family_transitions'] >= 1
    assert 'binary' in obs['rhythm']['family_counts']
    assert 'ternary' in obs['rhythm']['family_counts']


def test_eos_proposal_vetoed_when_future_planned_activity_remains():
    plan_centers = np.asarray([1.0, 3.0, 5.0, 7.0], dtype=np.float64)
    plan_values = np.zeros((4, 6), dtype=np.float32)
    # Block at 5.0s has strong planned head density: log1p(4.0) and low rest probability
    plan_values[2, 1] = float(np.log1p(4.0))
    plan_values[2, 5] = 0.05
    has_activity, reason = _has_future_planned_activity(
        2.0,
        10.0,
        plan_centers=plan_centers,
        plan_values=plan_values,
    )
    assert has_activity
    assert reason == 'planner_activity_remaining'


def test_eos_proposal_five_scenarios_decouple_sustain_tail_from_head_demand():
    # (1) Long Hold tail reaches 160.327s near end_sec=160.737s, while last note head is at 153s
    # and Planner still predicts activity at 156s with EOS=0.999 -> MUST veto EOS.
    plan_centers = np.asarray([150.0, 152.0, 156.0, 159.5], dtype=np.float64)
    plan_values = np.zeros((4, 6), dtype=np.float32)
    plan_values[:, 5] = 0.95
    plan_values[2, 1] = float(np.log1p(3.2))
    plan_values[2, 5] = 0.10
    decision_hold = evaluate_eos_proposal(
        153.0,
        160.737,
        eos_probability=0.999,
        eos_threshold=0.98,
        last_action_end_sec=160.327,
        plan_centers=plan_centers,
        plan_values=plan_values,
    )
    assert decision_hold['proposed'] is True
    assert decision_hold['accepted'] is False
    assert decision_hold['should_stop'] is False
    assert decision_hold['veto_reason'] == 'planner_activity_remaining'

    # (2) Genuine quiet outro (even >10s before end_sec) with no future curve/planner/workload
    # activity -> MUST accept EOS without forcing filler notes to end_sec.
    quiet_plan = np.zeros((4, 6), dtype=np.float32)
    quiet_plan[:, 5] = 0.95
    curve_times = np.arange(145.0, 160.5, 0.5, dtype=np.float64)
    quiet_workload = np.zeros((len(curve_times), 5), dtype=np.float32)
    decision_quiet = evaluate_eos_proposal(
        150.0,
        160.737,
        eos_probability=0.99,
        eos_threshold=0.98,
        last_action_end_sec=150.4,
        curve_times=curve_times,
        song_workload=quiet_workload,
        plan_centers=plan_centers,
        plan_values=quiet_plan,
    )
    assert decision_quiet['proposed'] is True
    assert decision_quiet['accepted'] is True
    assert decision_quiet['should_stop'] is True
    assert decision_quiet['stop_reason'] == 'learned_eos'
    assert decision_quiet['veto_reason'] is None

    # (3) Mid-song silence (60s..75s) followed by resumed activity at 80s -> MUST veto EOS at 60s.
    mid_centers = np.asarray([62.0, 68.0, 72.0, 80.0, 100.0], dtype=np.float64)
    mid_plan = np.zeros((5, 6), dtype=np.float32)
    mid_plan[:, 5] = 0.98
    mid_plan[3, 1] = float(np.log1p(2.5))
    mid_plan[3, 5] = 0.05
    decision_mid = evaluate_eos_proposal(
        60.0,
        160.0,
        eos_probability=0.995,
        eos_threshold=0.98,
        plan_centers=mid_centers,
        plan_values=mid_plan,
    )
    assert decision_mid['should_stop'] is False
    assert decision_mid['veto_reason'] == 'planner_activity_remaining'

    # (4) Manual intensity curve has activity in the second half -> MUST veto EOS even if sustain tail is near end.
    profile = {
        'schema': 'maichart-intensity-curve-v1',
        'duration_sec': 160.0,
        'audio_sha256': 'b' * 64,
        'points': [
            {'time_sec': 0.0, 'intensity': 0.2},
            {'time_sec': 120.0, 'intensity': 0.0},
            {'time_sec': 145.0, 'intensity': 0.7},
            {'time_sec': 160.0, 'intensity': 0.0},
        ],
        'rest_regions': [],
    }
    ctrl = apply_curve_to_composer(
        profile,
        mode='follow',
        span_mode='full',
        strength=1.0,
        duration_sec=160.0,
    )
    decision_curve = evaluate_eos_proposal(
        125.0,
        160.0,
        eos_probability=0.999,
        eos_threshold=0.98,
        last_action_end_sec=159.8,
        curve_controller=ctrl,
        plan_centers=plan_centers,
        plan_values=quiet_plan,
    )
    assert decision_curve['should_stop'] is False
    assert decision_curve['veto_reason'] == 'curve_activity_remaining'

    # (5) Natural end when no legal future candidates remain or audio range is exhausted.
    decision_exhausted = evaluate_eos_proposal(
        140.0,
        160.0,
        eos_probability=0.05,
        has_future_candidates=False,
    )
    assert decision_exhausted['should_stop'] is True
    assert decision_exhausted['stop_reason'] == 'audio_range_exhausted'
    assert decision_exhausted['proposed'] is False


def test_curve_controller_modulates_real_rates_and_respects_rest_regions():
    profile = {
        'schema': 'maichart-intensity-curve-v1',
        'duration_sec': 16.0,
        'audio_sha256': 'a' * 64,
        'points': [
            {'time_sec': 0.0, 'intensity': 0.2},
            {'time_sec': 8.0, 'intensity': 0.8},
            {'time_sec': 16.0, 'intensity': 0.5},
        ],
        'rest_regions': [{'start_sec': 4.0, 'end_sec': 6.0}],
    }
    ctrl = apply_curve_to_composer(
        profile,
        mode='follow',
        span_mode='full',
        strength=1.0,
        duration_sec=16.0,
    )
    assert ctrl.is_in_rest_region(5.0)
    assert not ctrl.is_in_rest_region(7.0)

    workload = torch.full((1, 3, 5), 1.0)
    seconds = np.asarray([1.0, 5.0, 8.0], dtype=np.float64)
    overrides = ctrl.build_window_overrides(
        candidate_seconds=seconds,
        model_workload=workload,
    )
    out = overrides['workload_override']
    assert out.shape == (1, 3, 5)
    # Rest region at t=5.0 zeroes rate channels
    assert float(out[0, 1, 0]) == 0.0
    # High intensity at t=8.0 exceeds low intensity at t=1.0
    assert float(out[0, 2, 0]) > float(out[0, 0, 0])


def test_candidate_competition_placement_grad_and_audio_macro_groups():
    atoms = ['1', '2', '1b', '1h[4:1]', '1-5[4:1]', 'A1', 'C']
    cfg = ComposerConfig(
        atom_vocab_size=len(atoms) + 1,
        d_model=32,
        nhead=4,
        layers=1,
        dim_feedforward=64,
        history_events=8,
        candidate_count=8,
        action_slots=4,
        architecture_profile='audio_grounded_v3',
        head_timing_dim=99,
        lens_width=16,
        lens_chunk_size=8,
    )
    model = build_model(cfg, atoms)
    assert model.candidate_competition is not None
    # Zero-initialized output projection means competition starts as exact identity
    x = torch.randn(2, 8, 32)
    valid = torch.tensor([
        [True, True, False, True, False, False, True, False],
        [False, True, True, True, False, True, False, False],
    ])
    assert torch.allclose(model.candidate_competition(x, valid), x)

    # Non-zero out_proj weights: mutating any invalid (~valid) candidate features,
    # workload_audio, placement_probs, or candidate_time must NOT alter any valid candidate output.
    comp = model.candidate_competition
    torch.manual_seed(42)
    with torch.no_grad():
        comp.out_proj.weight.normal_(0.0, 0.25)
        comp.out_proj.bias.normal_(0.0, 0.1)
    wl_a = torch.randn(2, 8, 104)
    place_a = torch.rand(2, 8, 5)
    time_a = torch.zeros(2, 8, 5)
    time_a[..., 0] = torch.linspace(0.1, 2.0, 8)
    time_a[..., 1] = time_a[..., 0] * 0.4
    out_a = comp(x, valid, workload_audio=wl_a, placement_probs=place_a, candidate_time=time_a)

    x_b = x.clone()
    wl_b = wl_a.clone()
    place_b = place_a.clone()
    time_b = time_a.clone()
    x_b[~valid] = torch.randn_like(x_b[~valid]) * 50.0 + 100.0
    wl_b[~valid] = torch.randn_like(wl_b[~valid]) * 50.0 - 75.0
    place_b[~valid] = 1.0 - place_b[~valid]
    time_b[~valid] = 999.0
    out_b = comp(x_b, valid, workload_audio=wl_b, placement_probs=place_b, candidate_time=time_b)
    torch.testing.assert_close(out_a[valid], out_b[valid], atol=0.0, rtol=0.0)

    # Isolated valid candidates (0 valid neighbors, e.g. batch 0 index 3 and 6) must have
    # zero prominence and zero neighbor diff: changing their placement_probs cannot fabricate a peak.
    place_c = place_a.clone()
    place_c[0, 3] = torch.tensor([0.99, 0.99, 0.99, 0.99, 0.99])
    place_c[0, 6] = torch.tensor([0.01, 0.01, 0.01, 0.01, 0.01])
    out_c = comp(x, valid, workload_audio=wl_a, placement_probs=place_c, candidate_time=time_a)
    torch.testing.assert_close(out_a[0, 3], out_c[0, 3], atol=0.0, rtol=0.0)
    torch.testing.assert_close(out_a[0, 6], out_c[0, 6], atol=0.0, rtol=0.0)

    # Audio macro groups collapse rotational pad symmetries (e.g. '1' and '2')
    audio_gids, num_audio_groups = build_audio_macro_groups(model.action_factors)
    assert int(audio_gids[1]) == int(audio_gids[2])
    assert int(audio_gids[1]) != int(audio_gids[3])
    assert num_audio_groups < len(atoms) + 1

    # Macro + detail decomposition sums to nonstop_ce
    logits = torch.randn(2, 4, len(atoms) + 1, requires_grad=True)
    targets = torch.tensor([[1, 2, 0, 0], [4, 0, 0, 0]], dtype=torch.long)
    weights = torch.ones(2)
    loss, diag = action_objective(
        logits, targets, weights, policy=JOINT_ACTION_OBJECTIVE, macro_group_ids=audio_gids
    )
    assert 'macro_ce' in diag and 'detail_ce' in diag
    assert torch.allclose(diag['macro_ce'] + diag['detail_ce'], diag['nonstop_ce'], atol=1e-5)

    # PlacementTargets.proximity peaks at exact note time, decays smoothly, and unions per-channel peaks within +/-25ms
    pt = PlacementTargets(
        [EventPoint(Fraction(1, 1), '1'), EventPoint(Fraction(1015, 1000), '2h[4:1]')],
        [EventTempo(Fraction(0), 60.0)],
        0.0,
        5.0,
    )
    occ, _ = pt.query([1.005])
    prox = pt.proximity([1.0, 1.005, 1.050], sigma_sec=0.010)
    assert float(prox[0, 0]) == 1.0
    # At t=1.005s, both Tap (at 1.000s, dt=5ms) and Hold (at 1.015s, dt=10ms) are within +/-25ms:
    # occupancy has both presence=1 and hold=1, and proximity has positive Gaussian targets for BOTH channels!
    assert float(occ[0, 0]) == 1.0 and float(occ[0, 1]) == 1.0
    assert float(prox[1, 0]) == np.exp(-0.5 * (0.005 / 0.010) ** 2).astype(np.float32)
    assert float(prox[1, 1]) == np.exp(-0.5 * (0.010 / 0.010) ** 2).astype(np.float32)
    assert float(prox[2, 0]) < 1e-4

    # Recovery objective includes EOS loss when eos_weight > 0
    rec_out = {
        'candidate_logits': torch.randn(2, 4, requires_grad=True),
        'action_logits': torch.randn(2, 4, len(atoms) + 1, requires_grad=True),
        'eos_logits': torch.randn(2, requires_grad=True),
    }
    rec_batch = {
        'sample_weight': torch.ones(2),
        'pointer_weight': torch.ones(2),
        'action_weight': torch.ones(2),
        'eos_weight': torch.ones(2),
        'candidate_target': torch.tensor([0, 1]),
        'action_target': targets,
        'eos_target': torch.zeros(2),
    }
    rec_loss, rec_metrics = recovery_objective(
        rec_out, rec_batch, policy=RECOVERY_ACTION_OBJECTIVE, weight=0.1
    )
    assert 'eos_loss' in rec_metrics
    rec_loss.backward()
    assert rec_out['eos_logits'].grad is not None


def test_recovery_builder_accumulates_reachability_counters_across_builds():
    from types import SimpleNamespace
    from maichart.composer.event_views import EventViews
    from maichart.composer.recovery import RecoveryBuilder
    from maichart.tempo_clock import TempoClock

    tempo = [EventTempo(Fraction(0), 120.0)]
    # Event at beat 2+7/12 is off canonical {1,2,3,4,6,8} denominators; event at beat 4 is canonical.
    source = [
        EventPoint(Fraction(0), '1'),
        EventPoint(Fraction(1), '2'),
        EventPoint(Fraction(31, 12), '3'),
        EventPoint(Fraction(4), '4'),
    ]
    views = EventViews.build(source, tempo, 0.0, 8.0)
    song = SimpleNamespace(tempo=tempo, first=0.0, audio=SimpleNamespace(duration_sec=8.0))
    values = np.zeros((161, 120), np.float32)
    table = {'clock': TempoClock(tempo, 0.0), 'condition': torch.zeros(16)}
    dataset = SimpleNamespace(
        samples=[SimpleNamespace(song_index=0, event_index=1, weight=1.0)],
        base=SimpleNamespace(songs=[(song,)]),
        event_views={0: views},
        candidate_count=96,
        candidate_policy=MUSIC_CANDIDATE_POLICY_V6,
        architecture_profile='audio_grounded_v3',
        history_events=8,
        action_slots=3,
        codec_atoms=['1', '2', '3', '4'],
        atom_to_id={'1': 1, '2': 2, '3': 3, '4': 4},
        _song_table=lambda index: table,
        _song_features=lambda index: (values, np.linspace(0.0, 8.0, 161)),
    )
    rec = RecoveryBuilder(dataset)
    # First build targets beat 31/12 (denominator 12 -> not playable under canonical {1,2,3,4,6,8}).
    prefix_1 = [EventPoint(Fraction(0), '1'), EventPoint(Fraction(1), '2'), EventPoint(Fraction(2), '3')]
    item_1, _ = rec.build(prefix_1, 0, generated_count=2)
    assert rec.playability_unreachable_references == 1

    # Second build targets beat 4 (canonical reachable) -> playability_unreachable_references must NOT reset to 0!
    prefix_2 = [EventPoint(Fraction(0), '1'), EventPoint(Fraction(2), '2'), EventPoint(Fraction(3), '3')]
    item_2, reason_2 = rec.build(prefix_2, 0, generated_count=2)
    assert reason_2 == 'accepted' and item_2 is not None
    assert rec.playability_unreachable_references == 1
    assert rec.reference_reachability_counts['reachable'] == 1
    assert (
        rec.reference_reachability_counts['playability_filtered']
        + rec.reference_reachability_counts['generator_unreachable']
    ) == 1


def test_rhythm_32_48_guard_small_bucket_truncation_active_span_and_local_lead_in():
    from maichart.composer.generate import (
        _anti_repetition_logit_bias,
        _canonical_duration_mask,
        _conserved_action_probabilities,
        _filter_rhythm_and_active_span_candidates,
        _grounded_pointer_probabilities,
        _truncate_probabilities,
    )
    from test_composer_audio_grounded_v3 import inputs

    # 1. {48} micro-step (2/3 -> 3/4, delta = 1/12) and {32} 3/8 hiccup (1/8 -> 1/2, delta = 3/8) are blocked!
    assert not is_playable_candidate_transition(Fraction(3, 4), Fraction(2, 3), MUSIC_CANDIDATE_POLICY_V6)
    assert not is_playable_candidate_transition(Fraction(1, 3), Fraction(1, 4), MUSIC_CANDIDATE_POLICY_V6)
    assert not is_playable_candidate_transition(Fraction(3, 8), Fraction(0, 1), MUSIC_CANDIDATE_POLICY_V6)
    assert not is_playable_candidate_transition(Fraction(1, 2), Fraction(1, 8), MUSIC_CANDIDATE_POLICY_V6)
    # Valid 1/8-beat {32} stream step (0 -> 1/8 -> 1/4) remains allowed:
    assert is_playable_candidate_transition(Fraction(1, 8), Fraction(0, 1), MUSIC_CANDIDATE_POLICY_V6)
    assert is_playable_candidate_transition(Fraction(1, 4), Fraction(1, 8), MUSIC_CANDIDATE_POLICY_V6)

    # 2. Small buckets (<= 16 items) in _truncate_probabilities and _conserved_action_probabilities
    #    prune low-probability tail options while _canonical_duration_mask preserves valid [32:3] and 12b Holds:
    dur_mask = _canonical_duration_mask(('1', '6h[8:1]', '6h[32:3]', '6h[1:3]', 'A8h[48:1]', '2<4[32:13]', '4<7[4:15]'))
    assert dur_mask.tolist() == [True, True, True, True, True, False, False, False]
    bits = np.asarray([0, 1 << 0, 1 << 1, 1 << 2], dtype=np.uint64)
    aba_hist = [((1, 0, 0), Fraction(1, 1)), ((2, 0, 0), Fraction(3, 2)), ((1, 0, 0), Fraction(2, 1))]
    bias = _anti_repetition_logit_bias(aba_hist, Fraction(5, 2), bits, np.uint64(0xFF), device=torch.device('cpu'),
                                       policy='legacy_v1')
    assert bias is not None and float(bias[2]) < -1.0 and float(bias[3]) == 0.0
    # official_tail_v2 (default) leaves a single ABA -> B inside the official p95 ping-pong length.
    assert _anti_repetition_logit_bias(aba_hist, Fraction(5, 2), bits, np.uint64(0xFF),
                                       device=torch.device('cpu')) is None
    small_probs = torch.tensor([0.90, 0.08, 0.01, 0.01])
    truncated = _truncate_probabilities(small_probs, top_p=0.85, min_p=0.08, max_k=12)
    assert float(truncated[2]) == 0.0 and float(truncated[3]) == 0.0
    assert float(truncated[0]) > 0.0

    # Hold macro bucket with 4 durations where [32:3] and [48:1] are < 1% tail:
    hold_probs = torch.tensor([0.0, 0.88, 0.10, 0.01, 0.01])
    hold_logits = hold_probs.log()
    macro_ids = torch.tensor([0, 1, 1, 1, 1])
    conserved = _conserved_action_probabilities(hold_logits, 1.0, macro_ids)
    assert float(conserved[3]) == 0.0 and float(conserved[4]) == 0.0
    torch.testing.assert_close(conserved.sum(), torch.tensor(1.0))

    # Coarse bin pruning removes 1% long-rest Bin 5 noise when bin_min_p > 0:
    ptr_probs = torch.tensor([0.90, 0.09, 0.01])
    ptr_bins = torch.tensor([0, 1, 5])
    pruned_ptr = _grounded_pointer_probabilities(
        ptr_probs.log(), ptr_bins, 0.5, bin_temperature=0.7, bin_min_p=0.04
    )
    assert float(pruned_ptr[2]) == 0.0

    # 3. Active-span long-jump guard blocks >= 4-beat jumps over active music, and outro ramp accepts EOS:
    cands = [Fraction(1, 4), Fraction(1, 2), Fraction(8, 1), Fraction(16, 1)]
    secs = np.asarray([10.08, 10.16, 12.53, 15.05], dtype=np.float64)
    plan_centers = np.asarray([10.5, 11.5, 12.5, 13.5, 14.5], dtype=np.float64)
    plan_values = np.zeros((5, 6), dtype=np.float32)
    plan_values[:, 1] = float(np.log1p(4.0))
    plan_values[:, 5] = 0.05
    filtered_playable = _filter_rhythm_and_active_span_candidates(
        cands,
        [True, True, True, True],
        secs,
        Fraction(0, 1),
        10.0,
        policy=MUSIC_CANDIDATE_POLICY_V6,
        plan_centers=plan_centers,
        plan_values=plan_values,
    )
    assert filtered_playable == [True, True, False, False]

    # Outro curve tapering to 0.0 at 157s..160.74s does not veto EOS at 156.24s:
    outro_profile = {
        'schema': 'maichart-intensity-curve-v1',
        'duration_sec': 160.74,
        'audio_sha256': 'c' * 64,
        'points': [
            {'time_sec': 0.0, 'intensity': 0.5},
            {'time_sec': 155.0, 'intensity': 0.3913},
            {'time_sec': 157.0, 'intensity': 0.0},
            {'time_sec': 160.74, 'intensity': 0.0},
        ],
        'rest_regions': [],
    }
    outro_ctrl = apply_curve_to_composer(
        outro_profile, mode='follow', span_mode='full', strength=1.0, duration_sec=160.74
    )
    assert not outro_ctrl.has_future_activity(156.24, 160.74)
    assert outro_ctrl.is_in_rest_region(160.73)

    # 4. Slide terminal landing lanes (atom_end_lanes) and _LocalLeadInContext receive gradients:
    atoms = ['1', '2', '1-5[8:1]', '6h[8:1]', 'B3']
    cfg = ComposerConfig(
        atom_vocab_size=len(atoms) + 1,
        d_model=16,
        nhead=4,
        layers=1,
        dim_feedforward=32,
        dropout=0.0,
        history_events=8,
        action_slots=3,
        architecture_profile='audio_grounded_v3',
        head_timing_dim=99,
        lens_width=8,
        lens_chunk_size=3,
        cross_event_relation_policy='relative_heads_v1',
    )
    model = build_model(cfg, atoms)
    # Atom 3 is '1-5[8:1]': head lane is 1 (index 0), end lane is 5 (index 4)!
    assert bool(model.action_factors.atom_head_lanes[3, 0])
    assert not bool(model.action_factors.atom_head_lanes[3, 4])
    assert bool(model.action_factors.atom_end_lanes[3, 4])

    batch = inputs()
    batch['history'] = torch.tensor([[[3, 0, 0], [1, 0, 0]], [[2, 0, 0], [3, 0, 0]]])
    out = model(**batch, candidate_indices=torch.tensor([1, 2]), action_targets=torch.tensor([[2, 0, 0], [1, 0, 0]]))
    (out['candidate_logits'].square().mean() + out['action_logits'].square().mean()).backward()
    assert model.local_lead_in.out_proj.weight.grad is not None
    assert float(model.local_lead_in.out_proj.weight.grad.abs().sum()) > 0.0
