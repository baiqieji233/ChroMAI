"""Read-only inspector and log-consumer regressions; no model training involved."""
import json
from types import SimpleNamespace

import pytest

from maichart.webui import server
from test_webui import webui_api  # noqa: F401 - temporary HTTP fixture


@pytest.mark.parametrize('family', ['chart_composer_v2','chart_composer_v1',
    'song_form_v1','song_form_v2','chart_stream_v1','event_ctx_v1'])
def test_event_manifest_counts_do_not_use_frame_rejection_flags(tmp_path, family):
    manifest=tmp_path/'training_manifest.json'
    manifest.write_text(json.dumps(dict(schema='maichart-training-manifest-v1',
        preprocessing=dict(target_schema='event_v1'),songs=[dict(song_id='song',difficulties=[
            dict(usable_for_training=False,event_usable_for_training=True),
            dict(usable_for_training=False,event_usable_for_training=False)])])))
    replies=[]
    server.WebUIHandler._inspect_manifest(SimpleNamespace(_json_response=replies.append),
        dict(model_family=family,manifest=str(manifest)))
    assert replies[0]['usable_difficulty_count']==1
    assert replies[0]['filtered_difficulty_count']==1
    assert replies[0]['usable_count_basis']=='event_manifest_qc'


def task():
    return server.TaskState(task_id='audit',kind='composer_v2_train',result={'epoch_target':50})


def send(state, name, **data):
    server._update_composer_progress(state,'composer_v2_'+name+' '+json.dumps(data))


def test_epoch_progress_and_checkpoint_cursor_are_separate():
    state=task()
    send(state,'progress',stage='training',epoch=1,batch=50,total=100,loss=4.,optimizer_steps=5)
    assert state.progress==pytest.approx(.01)
    send(state,'checkpoint',epoch=0,batch_cursor=50,optimizer_steps=5)
    assert state.result['epoch']==1
    assert state.progress==pytest.approx(.01)
    assert state.result['composer_v2_training_metrics']['loss']==4.
    send(state,'progress',stage='epoch_complete',epoch=1,total=50,loss=3.,validation_loss=3.)
    assert state.progress==pytest.approx(.02)
    assert state.result['composer_v2_training_metrics']['loss']==4.
    assert state.result['composer_v2_epoch_history'][0]['validation_loss']==3.
    assert state.result['composer_v2_epoch_metrics']['loss_scope']=='epoch_validation'
    send(state,'progress',stage='training',epoch=2,batch=10,total=100,loss=3.5,optimizer_steps=6)
    assert state.progress==pytest.approx(.022)


def test_preparation_and_generation_show_counters_without_fake_percentages():
    state=task()
    send(state,'prepare',stage='audio',completed=12,total=20)
    assert state.result['progress_indeterminate']
    assert '12/20' in state.message
    assert state.progress==0
    send(state,'event',events=128,beat='43/2')
    assert '128' in state.message and '43/2' in state.message
    assert state.result['progress_indeterminate']


def test_epoch_history_replaces_duplicate_summaries_and_is_bounded():
    state=task()
    for epoch in range(1,205):
        send(state,'progress',stage='epoch_complete',epoch=epoch,total=250,
            loss=.5,validation_loss=.5)
    send(state,'progress',stage='epoch_complete',epoch=204,total=250,
        loss=.4,validation_loss=.4)
    history=state.result['composer_v2_epoch_history']
    assert len(history)==200
    assert len({row['epoch'] for row in history})==200
    assert history[-1]['validation_loss']==.4


def test_updated_http_inspector_reports_event_usable_counts(webui_api, tmp_path):
    manifest=tmp_path/'training_manifest.json'
    manifest.write_text(json.dumps(dict(schema='maichart-training-manifest-v1',
        preprocessing=dict(target_schema='event_v1'),songs=[dict(song_id='song',difficulties=[
            dict(usable_for_training=False,event_usable_for_training=True)])])))
    code,data=webui_api('/api/inspect-manifest',dict(model_family='chart_composer_v2',manifest=str(manifest)))
    assert code==200
    assert data['usable_difficulty_count']==1 and data['filtered_difficulty_count']==0
    code,status=webui_api('/api/status')
    assert code==200 and status['training_capabilities']['composer_v2_api_contract']=='composer_v2_continuous_events_v3'
    assert not status['tasks']
