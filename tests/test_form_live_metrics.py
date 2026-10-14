import json

from maichart.webui.form_api import update_form_progress
from maichart.webui.server import TaskState


def test_latest_phase_metrics_survive_checkpoint_and_phase_changes():
    task = TaskState('metrics', 'form_train')
    planner = {'epoch': 3, 'phase': 'blueprint', 'losses': {'loss_onset': .7, 'loss_type': .3}}
    layout = {'epoch': 3, 'phase': 'layout_revision',
              'losses': {'draft_note_configuration_loss': .6},
              'metric_counts': {'draft_note_configuration_loss': 320}}
    update_form_progress(task, 'form_progress '+json.dumps(planner))
    update_form_progress(task, 'form_progress '+json.dumps(layout))
    update_form_progress(task, 'form_checkpoint '+json.dumps({'epoch': 3, 'batch': 500}))
    assert task.result['training_metrics']['blueprint'] == planner
    assert task.result['training_metrics']['layout_revision'] == layout


def test_epoch_metric_history_is_bounded_and_deduplicated():
    task = TaskState('history', 'form_train')
    for epoch in range(1, 25):
        update_form_progress(task, 'form_epoch '+json.dumps({'epoch': epoch, 'draft_revision_loss': .5}))
    update_form_progress(task, 'form_epoch '+json.dumps({'epoch': 24, 'draft_revision_loss': .4}))
    assert len(task.result['metric_history']) == 20
    assert task.result['metric_history'][0]['epoch'] == 5
    assert task.result['metric_history'][-1]['draft_revision_loss'] == .4


def test_onset_diagnostics_are_exposed_without_changing_arrangement_progress():
    task = TaskState('onsets', 'form_generate')
    task.result['phase'] = 'arrangement'
    summary = {'raw_local_peaks': 120, 'gap_diagnostics': [{'duration_sec': 8.8}]}
    update_form_progress(task, 'form_onsets '+json.dumps(summary))
    assert task.result['onset_summary'] == summary
    assert task.result['phase'] == 'arrangement'


def test_rational_decode_diagnostics_reach_task_result_without_changing_progress():
    task=TaskState('rational','form_generate')
    task.result['phase']='arrangement'
    value={'schema':'song_form_rational_diagnostics_v1','event_count':118,'empty_eos_spans':16,'incomplete_spans':[4]}
    update_form_progress(task,'form_rational_diagnostics '+json.dumps(value))
    assert task.result['rational_diagnostics']==value
    assert task.result['phase']=='arrangement'
