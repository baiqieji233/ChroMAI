from fractions import Fraction as F
from types import SimpleNamespace
import torch

from maichart.event_timing import EventPoint,EventTempo
from maichart.composer.event_set_codec import EventSetCodec
from maichart.composer.stream_data import StreamDataset,StreamInferenceContext,collate_stream
from maichart.composer.stream_model import StreamComposer,StreamComposerConfig
from maichart.composer.stream_proposal import prepare_selected_candidate


def fixture(monkeypatch):
    beats=torch.arange(0.,24.,.1,dtype=torch.float64)
    bank=(torch.zeros(len(beats),238),beats/2,beats)
    monkeypatch.setattr(StreamDataset,'_bank',lambda self,song:bank)
    song=SimpleNamespace(audio=SimpleNamespace(duration_sec=12.),tempo=[EventTempo(F(0),120.)],first=0.)
    points=[EventPoint(F(i),str(i%8+1)) for i in range(20)]
    codec=EventSetCodec.fit([p.note_text for p in points])
    ds=StreamDataset(base=SimpleNamespace(songs=[(song,points,13.,1.,None,torch.zeros(16),(),None)]),
        codec=codec,lead_in=8,targets=4,audio_grid_seconds=.05,
        history_window_policy='block_aligned_v2',continuation_policy='pattern_runs_v2',
        phrase_context_policy='aligned_phrase_v3',candidate_state_policy='selected_native_v1')
    model=StreamComposer(StreamComposerConfig(d_model=16,nhead=4,layers=1,dim_feedforward=32,
        dropout=0.,history_events=8,rollout_steps=4,audio_grid_seconds=.05,
        delta_beats=('0',*map(str,ds.delta_vocab.values)),phrase_context_policy='aligned_phrase_v3',
        event_family_policy='audio_state_phrase_v4',template_policy='family_preserving_v1',
        candidate_state_policy='selected_native_v1',transition_policy='music_exit_v3'),codec).eval()
    return ds,model,song,bank


def test_landing_inputs_cannot_change_time_logits_and_backward_is_finite(monkeypatch):
    ds,model,_,_=fixture(monkeypatch)
    batch=collate_stream([ds[3]])
    with torch.no_grad():
        model.phrase_context.ordered.alignment_output.weight.normal_(0,.1)
        model.candidate_state_projection.weight.normal_(0,.1)
    original=model(batch)
    changed={**batch,'candidate_action_state':torch.ones_like(batch['candidate_action_state']),
        'candidate_action_state_mask':torch.zeros_like(batch['candidate_action_state_mask'])}
    altered=model(changed)
    torch.testing.assert_close(original['kind_logits'],altered['kind_logits'],rtol=0,atol=0)
    torch.testing.assert_close(original['delta_logits'],altered['delta_logits'],rtol=0,atol=0)
    assert not torch.equal(original['action_context'],altered['action_context'])
    loss,_=model.loss(original,batch);loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_selected_landing_matches_teacher_action_context(monkeypatch):
    ds,model,song,bank=fixture(monkeypatch)
    item=ds[3];batch=collate_stream([item]);_,start,end,_=ds.samples[3]
    with torch.no_grad():
        model.phrase_context.ordered.alignment_output.weight.normal_(0,.1)
        model.candidate_state_projection.weight.normal_(0,.1)
    expected=model(batch)
    context=StreamInferenceContext(song,ds.codec,ds.delta_vocab,ds.conditions[0],lead_in=8,
        audio_bank=bank,audio_grid_seconds=.05,history_window_policy='block_aligned_v2',rollout_steps=4,
        continuation_policy='pattern_runs_v2',timing_output_policy=ds.timing_output_policy,
        phrase_context_policy='aligned_phrase_v3',candidate_state_policy='selected_native_v1')
    for step in ds.streams[0][:end-1]:context.commit(step)
    raw=context.current_batch()
    cache=model.encode_audio(raw['song_audio'],raw['song_mask'],raw['song_seconds'],raw['song_beats'])
    output=model(raw,audio_cache=cache)
    target=ds.streams[0][end-1]
    selected,prepared=prepare_selected_candidate(model,context,raw,output,target.end,cache)
    times=torch.full_like(raw['step_second'],target.end_second)
    actual=model.action_context_at(prepared['hidden'],times,cache,selected['song_inverse'],selected)
    torch.testing.assert_close(expected['action_context'][:,-1],actual[:,-1],rtol=1e-5,atol=1e-5)
