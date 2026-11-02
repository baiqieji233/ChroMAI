"""Real chart rows exercise the new opt-in song modules end to end."""
import copy
from dataclasses import replace
import torch
from maichart.composer.stream_model import StreamComposer
from maichart.composer.stream_data import collate_stream,StreamInferenceContext
from maichart.composer.stream_proposal import prepare_selected_candidate
from test_stream_phrase_execution import execution_fixture


def style_fixture(monkeypatch):
    ds,baseline,song,bank=execution_fixture(monkeypatch)
    config=replace(baseline.config,song_profile_policy='masked_joint_v1',song_response_policy='music_response_v1')
    return ds,StreamComposer(config,ds.codec).eval(),song,bank


def test_real_forward_backward_song_modules_are_finite(monkeypatch):
    ds,model,_,_=style_fixture(monkeypatch)
    batch=collate_stream([ds[i] for i in range(len(ds))])
    output=model(batch);loss,metrics=model.loss(output,batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics['song_profile_loss'])
    for module in (model.song_profile,model.song_response):
        gradients=[p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients)>0


def test_future_supervision_cannot_change_response_prior(monkeypatch):
    ds,model,_,_=style_fixture(monkeypatch)
    batch=collate_stream([ds[1]]);changed=copy.deepcopy(batch)
    for key in ('target_second','target_beat','target_relation','target_phrase_features'):
        if key in changed:changed[key]=torch.randn_like(changed[key].float())
    if 'target_phrase_sequence_atoms' in changed:changed['target_phrase_sequence_atoms'].zero_()
    with torch.no_grad():a=model(batch);b=model(changed)
    for key in ('song_response_logits','song_response_phrase_logits','song_profile_audio'):
        torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)


def test_selected_candidate_retains_departure_response_and_locked_choice(monkeypatch):
    ds,model,song,bank=style_fixture(monkeypatch)
    context=StreamInferenceContext(song,ds.codec,ds.delta_vocab,ds.conditions[0],lead_in=8,
        audio_bank=bank,audio_grid_seconds=.05,history_window_policy='block_aligned_v2',rollout_steps=4,
        continuation_policy='pattern_runs_v2',timing_output_policy=ds.timing_output_policy,
        phrase_context_policy='aligned_phrase_v3',candidate_state_policy='selected_native_v1',
        training_window_policy='phrase_complete_v1',max_phrase_events=64)
    end=ds.samples[1][2]
    for step in ds.streams[0][:end-1]:context.commit(step)
    batch=context.current_batch()
    with torch.no_grad():
        cache=model.encode_audio(batch['song_audio'],batch['song_mask'],batch['song_seconds'],batch['song_beats'])
        output=model(batch,audio_cache=cache)
        selected,prepared=prepare_selected_candidate(model,context,batch,output,ds.streams[0][end-1].end,cache)
    for key in ('song_response_logits','song_response_phrase_logits'):
        torch.testing.assert_close(prepared[key],output[key],rtol=0,atol=0)
    for choice in range(model.config.song_response_modes):
        prior=model.execution_prior(prepared,selected,song_choice=choice)
        torch.testing.assert_close(prior,prepared['song_response_phrase_logits'].log_softmax(-1)[:,:,choice,:])


def test_policy_off_retains_strict_checkpoint_contract(monkeypatch):
    ds,old,_,_=execution_fixture(monkeypatch)
    config=replace(old.config,song_response_policy='off',song_profile_policy='off')
    restored=StreamComposer(config,ds.codec)
    restored.load_state_dict(old.state_dict(),strict=True)
    assert set(restored.state_dict())==set(old.state_dict())
    assert not any(key.startswith(('song_response.','song_profile.')) for key in restored.state_dict())
