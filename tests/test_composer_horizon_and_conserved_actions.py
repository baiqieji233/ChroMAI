from fractions import Fraction
import pytest
import torch
from torch.nn import functional as F

from maichart.composer.checkpoint import inspect_payload, load, migrate_audio_grounded, save
from maichart.composer.conserved_actions import group_logsumexp
from maichart.composer.model import ComposerConfig, GROUNDED_PROFILE, build_model, resolve_composer_policy
from test_composer_audio_grounded_v3 import inputs


ATOMS = [
    '1', '1b', '1x', '2', '2b', '3', '4', '5', '8',
    '1h[4:1]', '1h[2:1]', '2h[4:1]',
    '1-5[8:1]', '1-5[4:1]', '1<5[8:1]', '1>5[8:1]', '8-4[8:1]',
    'A1', 'B2', 'C', 'Cf',
]


def make_config(**overrides):
    values = dict(
        atom_vocab_size=len(ATOMS) + 1, d_model=16, nhead=4, layers=1,
        dim_feedforward=32, dropout=0., history_events=8, action_slots=3,
        architecture_profile=GROUNDED_PROFILE, head_timing_dim=99,
        lens_width=8, lens_chunk_size=3,
    )
    values.update(overrides)
    return ComposerConfig(**values)


def test_policy_resolution_defaults_and_resume_inheritance():
    assert resolve_composer_policy('future_horizon_policy') == 'candidate_horizon_v1'
    assert resolve_composer_policy('action_head_policy') == 'compact_grouped_v3'
    assert resolve_composer_policy('future_horizon_policy', architecture_profile='music_context_v2') == 'off'
    assert resolve_composer_policy('action_head_policy', architecture_profile='music_context_v2') == 'flat_v1'
    old_parent = {'architecture_profile': GROUNDED_PROFILE}
    assert resolve_composer_policy('future_horizon_policy', 'auto', old_parent, resume=True) == 'off'
    assert resolve_composer_policy('action_head_policy', 'auto', old_parent, resume=True) == 'flat_v1'
    assert resolve_composer_policy('future_horizon_policy', 'auto', old_parent, resume=False) == 'candidate_horizon_v1'
    assert resolve_composer_policy('action_head_policy', 'auto', old_parent, resume=False) == 'compact_grouped_v3'


def test_horizon_only_migration_preserves_exact_outputs_at_step_zero():
    torch.manual_seed(21)
    base_model = build_model(make_config(), ATOMS).eval()
    migrated = migrate_audio_grounded(
        base_model, ATOMS,
        future_horizon_policy='candidate_horizon_v1',
    ).eval()
    batch = inputs()
    idx = torch.tensor([1, 2])
    with torch.no_grad():
        before = base_model(**batch, candidate_indices=idx)
        after = migrated(**batch, candidate_indices=idx)
    for key in ('candidate_logits', 'action_logits', 'audio_action_logits', 'future', 'eos_logits'):
        torch.testing.assert_close(before[key], after[key], rtol=1e-5, atol=1e-6)


def test_future_horizon_attends_only_valid_future_window_within_eight_beats():
    torch.manual_seed(22)
    model = build_model(make_config(future_horizon_policy='candidate_horizon_v1'), ATOMS).eval()
    with torch.no_grad():
        model.future_horizon.output.weight.normal_(0, 0.2)
    batch = inputs()
    # Candidate beats: [0.5, 1.0, 2.0, 4.0, 11.0, 12.0]
    beats = torch.tensor([[0.5, 1.0, 2.0, 4.0, 11.0, 12.0],
                          [0.5, 1.0, 2.0, 4.0, 11.0, 12.0]])
    batch['candidate_features'][..., 120] = beats
    batch['candidate_features'][..., 121] = beats * 0.4
    # Query at candidate index 1 (beat 1.0):
    # - index 0 (beat 0.5) is in the PAST relative to index 1 -> must not affect horizon
    # - index 2, 3 (beats 2.0, 4.0) are in [1.0, 9.0] -> MUST affect horizon
    # - index 4, 5 (beats 11.0, 12.0) are > 1.0 + 8.0 -> must NOT affect horizon
    q_idx = torch.tensor([1, 1])

    # Isolate horizon effect on candidate 1 by varying audio_lens (which is local per candidate)
    base_out = model(**batch, candidate_indices=q_idx)

    # 1) Perturb future candidate 2 (beat 2.0, within +8 beats): must change candidate 1's action_context
    fut_batch = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
    fut_batch['audio_lens'][:, 2] += 5.0
    fut_out = model(**fut_batch, candidate_indices=q_idx)
    assert not torch.allclose(base_out['action_context'], fut_out['action_context'], atol=1e-5)

    # 2) Perturb past candidate 0 (beat 0.5 < 1.0) in horizon module directly: must NOT change candidate 1
    ac = torch.randn(2, 6, 16)
    valid = torch.ones(2, 6, dtype=torch.bool)
    h_base = model.future_horizon(ac, batch['candidate_features'], valid, audio_dim=120)
    ac_past = ac.clone(); ac_past[:, 0] += 10.0
    h_past = model.future_horizon(ac_past, batch['candidate_features'], valid, audio_dim=120)
    torch.testing.assert_close(h_base[:, 1], h_past[:, 1], rtol=1e-5, atol=1e-6)

    # 3) Perturb far-future candidate 4 (beat 11.0 > 1.0 + 8.0): must NOT change candidate 1
    ac_far = ac.clone(); ac_far[:, 4] += 10.0
    h_far = model.future_horizon(ac_far, batch['candidate_features'], valid, audio_dim=120)
    torch.testing.assert_close(h_base[:, 1], h_far[:, 1], rtol=1e-5, atol=1e-6)


def test_conserved_grouped_head_strictly_preserves_macro_masses_and_prevents_slide_tap_contention():
    torch.manual_seed(23)
    model = build_model(make_config(action_head_policy='conserved_grouped_v2'), ATOMS).eval()
    head = model.conserved_action_head
    batch = inputs()
    idx = torch.tensor([1, 2])
    with torch.no_grad():
        baseline = model(**batch, candidate_indices=idx)['action_logits']
        base_group_mass = group_logsumexp(baseline, head.atom_group_id, head.num_groups)

        # Perturb all internal detail autoregressive tables (Slide shapes, endpoints, durations, modifiers)
        for param in (head.group_traj_gate.weight, head.shape_detail.weight, head.endpoint_detail.weight,
                      head.break_detail.weight, head.group_dur_gate.weight, head.shape_dur_gate.weight,
                      head.end_dur_gate.weight, head.hold_dur_detail.weight, head.slide_dur_detail.weight):
            param.normal_(0, 4.0)
        head._clear_table_cache()
        perturbed = model(**batch, candidate_indices=idx)['action_logits']
        perturbed_group_mass = group_logsumexp(perturbed, head.atom_group_id, head.num_groups)

    # Within-group Slide/Tap/Hold atom logits change substantially...
    assert float((perturbed - baseline).abs().max()) > 1.0
    # ...yet every (Kind, StartPad) macro group's total probability mass is 100% conserved!
    torch.testing.assert_close(base_group_mass, perturbed_group_mass, rtol=1e-5, atol=1e-5)


def test_decode_candidate_and_action_slot_parity_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(24)
    model = build_model(make_config(
        future_horizon_policy='candidate_horizon_v1',
        action_head_policy='conserved_grouped_v2',
        cross_event_relation_policy='relative_heads_v1',
    ), ATOMS)
    batch = inputs()
    # Include Slides so independent macro classification also has a genuine
    # within-Slide shape target, rather than expecting Tap loss to train it.
    targets = torch.tensor([[13, 4, 0], [15, 0, 0]])
    idx = torch.tensor([1, 2])
    out = model(**batch, candidate_indices=idx, action_targets=targets)
    loss = F.cross_entropy(out['action_logits'].reshape(-1, len(ATOMS) + 1), targets.reshape(-1))
    loss.backward()
    assert float(model.future_horizon.output.weight.grad.abs().sum()) > 0
    assert float(model.conserved_action_head.macro_head.weight.grad.abs().sum()) > 0
    assert float(model.conserved_action_head.shape_detail.weight.grad.abs().sum()) > 0

    # Take one optimizer step so all new weights are non-zero, then verify parity & save/load
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    opt.step()
    model.eval()
    with torch.no_grad():
        full_out = model(**batch, candidate_indices=idx, action_targets=targets)
        ptr_only = model(**batch, decode_actions=False)
        dec = model.decode_candidate(
            ptr_only['candidate_context'], idx, project_actions=False,
            audio_context=ptr_only['audio_action_context'],
            previous_heads=ptr_only['previous_heads'],
        )
        slot0 = torch.stack([
            model.action_slot(dec['action_context'][i], 0, [], previous_heads=dec['previous_heads'][i])
            for i in range(2)
        ])
        slot1 = torch.stack([
            model.action_slot(dec['action_context'][i], 1, [int(targets[i, 0])],
                              previous_heads=dec['previous_heads'][i])
            for i in range(2)
        ])
        torch.testing.assert_close(full_out['action_logits'][:, 0], slot0, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(full_out['action_logits'][:, 1], slot1, rtol=1e-5, atol=1e-5)

    ckpt_path = tmp_path / 'model.pt'
    payload = save(ckpt_path, model, opt, epoch=1, args={}, manifest=tmp_path / 'm.json', atom_catalog=ATOMS)
    assert inspect_payload(payload)['can_resume']
    loaded, _ = load(ckpt_path)
    loaded.eval()
    with torch.no_grad():
        reloaded_out = loaded(**batch, candidate_indices=idx, action_targets=targets)
    torch.testing.assert_close(full_out['action_logits'], reloaded_out['action_logits'], rtol=0, atol=0)
