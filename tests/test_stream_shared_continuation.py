import copy
from dataclasses import replace

import torch

from maichart.composer.stream_data import StreamInferenceContext, collate_stream
from maichart.composer.stream_model import StreamComposer
from maichart.composer.stream_model import StreamComposerConfig
from maichart.composer.stream_proposal import prepare_selected_candidate
from maichart.composer.stream_timing import grouped_logsumexp
from test_composer_stream_model import fixture
from test_stream_landing_integration import fixture as landing_fixture


def _shared_fixture():
    model, batch = fixture(event_family_policy='shared_continuation_v1')
    with torch.no_grad():
        torch.manual_seed(41)
        model.event_family_head.weight.normal_(std=.12)
        model.event_family_head.bias.zero_()
    return model, batch


def test_shared_family_reads_older_prefix_at_fixed_music_and_last_note():
    model, batch = _shared_fixture()
    changed = copy.deepcopy(batch)
    # Preserve the latest NOTE, its time, candidate audio and physical state.
    changed['history_atoms'][0, 1].zero_()
    changed['history_atoms'][0, 1, :2] = torch.tensor(model.codec.encode('A1/Cf')['atoms'][:2])
    changed['history_shape'][0, 1] = 0
    with torch.no_grad():
        before = model(batch)
        after = model(changed)
    torch.testing.assert_close(before['event_family_audio_logits'], after['event_family_audio_logits'])
    assert not torch.allclose(before['event_family_logits'][0, -1], after['event_family_logits'][0, -1])


def test_shared_family_mass_is_independent_of_route_catalog_size():
    model, batch = _shared_fixture()
    with torch.no_grad():
        output = model(batch)
    mass = grouped_logsumexp(output['event_route_log_probs'], model.semantic_route_families - 1, groups=31)
    torch.testing.assert_close(mass.softmax(-1), output['event_family_logits'].softmax(-1), atol=1e-6, rtol=1e-6)


def test_selected_candidate_helper_matches_teacher_action_and_decoding_logits():
    model, batch = _shared_fixture()
    with torch.no_grad():
        cache = model.encode_audio(batch['song_audio'], batch['song_mask'], batch['song_seconds'], batch['song_beats'])
        output = model(batch, audio_cache=cache)
        action, load, prepared = model.candidate_action_state(output, batch, cache, batch['target_second'],
            return_button_load=True)
        torch.testing.assert_close(action, output['action_context'])
        torch.testing.assert_close(load, output['button_load'])
        torch.testing.assert_close(model.family_logits_for_action(action), output['event_family_logits'])
        assert prepared['hidden'].shape == output['hidden'].shape


def test_old_checkpoint_dict_defaults_to_authored_recovery_labels():
    assert StreamComposerConfig.from_dict({}).recovery_target_policy == 'authored_next_v1'
    assert StreamComposerConfig(event_family_policy='shared_continuation_v1').recovery_target_policy == 'causal_correspondence_v1'


def test_aligned_native_selected_landing_matches_training_and_is_idempotent(monkeypatch):
    ds, baseline, song, bank = landing_fixture(monkeypatch)
    config = replace(baseline.config, event_family_policy='shared_continuation_v1',
        transition_policy='off', gap_continuation_policy='off')
    model = StreamComposer(config, ds.codec).eval()
    with torch.no_grad():
        model.event_family_head.weight.normal_(0, .1)
        model.candidate_state_projection.weight.normal_(0, .1)
        model.phrase_context.ordered.interaction_output.weight.normal_(0, .1)
        model.phrase_context.ordered.alignment_output.weight.normal_(0, .1)
    batch = collate_stream([ds[3]])
    _, _, end, _ = ds.samples[3]
    context = StreamInferenceContext(song, ds.codec, ds.delta_vocab, ds.conditions[0], lead_in=8,
        audio_bank=bank, audio_grid_seconds=.05, history_window_policy='block_aligned_v2', rollout_steps=4,
        continuation_policy='pattern_runs_v2', timing_output_policy=ds.timing_output_policy,
        phrase_context_policy='aligned_phrase_v3', candidate_state_policy='selected_native_v1')
    for step in ds.streams[0][:end - 1]:
        context.commit(step)
    raw = context.current_batch()
    target = ds.streams[0][end - 1]
    times = torch.full_like(raw['step_second'], target.end_second)
    with torch.no_grad():
        teacher = model(batch)
        cache = model.encode_audio(raw['song_audio'], raw['song_mask'], raw['song_seconds'], raw['song_beats'])
        departure = model(raw, audio_cache=cache)
        selected, prepared = prepare_selected_candidate(model, context, raw, departure, target.end, cache)
        actual, _, ready = model.candidate_action_state(prepared, selected, cache, times,
            return_button_load=True)
        repeated, _, ready_again = model.candidate_action_state(ready, selected, cache, times,
            return_button_load=True)
        torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
        torch.testing.assert_close(ready['hidden'], ready_again['hidden'], rtol=0, atol=0)
        torch.testing.assert_close(teacher['action_context'][:, -1], actual[:, -1], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(teacher['event_family_logits'][:, -1],
            model.family_logits_for_action(actual[:, -1]), rtol=1e-5, atol=1e-5)
        modified = {**selected, 'candidate_action_state': torch.ones_like(selected['candidate_action_state']),
            'candidate_action_state_mask': torch.zeros_like(selected['candidate_action_state_mask'])}
        changed, _ = model.candidate_action_state(prepared, modified, cache, times)
        assert not torch.equal(changed[:, -1], actual[:, -1])
