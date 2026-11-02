from dataclasses import replace
import torch
from maichart.composer.stream_model import StreamComposer
from maichart.composer.stream_data import collate_stream
from test_stream_phrase_execution import execution_fixture


def test_new_profile_score_uses_chart_mass_once_and_keeps_plan_floor_diagnostic(monkeypatch):
    torch.set_num_threads(1)
    ds,old,_,_=execution_fixture(monkeypatch)
    model=StreamComposer(replace(old.config,song_profile_policy='masked_ordered_joint_v2'),old.codec).eval()
    batch=collate_stream([ds[i] for i in range(len(ds))])
    output=model(batch)
    rows=batch['profile_chart_rows']
    parts=model.song_profile.loss_parts(output['song_profile_audio'].index_select(0,rows),
        batch['condition'].index_select(0,rows),torch.zeros_like(batch['condition'].index_select(0,rows),dtype=torch.bool))
    weights=batch['profile_chart_weights']
    loss,metrics=model.loss(output,batch)
    off,_=model.loss(output,{**batch,'profile_chart_weights':torch.zeros_like(weights)})
    expected=((parts['joint_nll']+parts['crps'])*weights).sum()
    torch.testing.assert_close(loss-off,model.config.song_profile_weight*expected)
    torch.testing.assert_close(metrics['song_profile_loss'],expected)
    torch.testing.assert_close(metrics['main_prediction_loss'],metrics['time_loss']+metrics['configuration_loss'])
    assert float(metrics['plan_rate_floor'])>=0
    assert float(metrics['plan_rate_excess'])>=-1e-5
    loss.backward()
    assert sum(float(p.grad.abs().sum()) for p in model.song_profile.parameters() if p.grad is not None)>0
