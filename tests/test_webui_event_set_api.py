from types import SimpleNamespace

import pytest

from maichart.webui import composer_api, event_set_api


class Handler:
    def _json_response(self, value): self.response = value


class Runtime:
    def __init__(self): self.task = SimpleNamespace(task_id='set-test', result={})
    def _reserve_composer_training_task(self, directory): return self.task
    def _create_task(self, kind): return self.task
    def _run_subprocess_task(self, task, args, **kwargs):
        self.args = args; self.dispatch = kwargs
        if kwargs.get('on_finish'): kwargs['on_finish']()


def test_stream_train_defaults_and_no_legacy_flags(tmp_path):
    manifest = tmp_path / 'manifest.json'; manifest.write_text('{}')
    handler, runtime = Handler(), Runtime()
    composer_api.start_train(handler, dict(model_family='chart_composer_v1', architecture_profile=event_set_api.PROFILE,
        manifest=str(manifest),holdout_fraction=0, checkpoint_dir=str(tmp_path / 'out'), cache_dir=str(tmp_path / 'cache'),
        candidate_count=999, action_slots=33, echo_prior_policy='phase_echo_v1'), runtime=runtime)
    assert runtime.args[:2] == ['-m', 'maichart.composer.stream_train']
    for flag, expected in [('--batch-size','16'),('--grad-accumulation','4'),('--d-model','256'),('--rollout-steps','16'),('--window-seconds','8.0'),('--precision','bf16')]:
        assert runtime.args[runtime.args.index(flag)+1] == expected
    assert '--candidate-count' not in runtime.args
    assert '--echo-prior-policy' not in runtime.args
    assert runtime.task.result['epoch_target'] == 50
    assert runtime.dispatch['expected_epoch'] == 50
    assert runtime.args[runtime.args.index('--audio-grid-seconds')+1]=='0.01'
    assert runtime.args[runtime.args.index('--plan-weight')+1]=='0.1'
    for flag,value in [('--validation-max-batches','32'),('--validation-batch-size','4'),
                       ('--generation-eval-interval-epochs','1'),('--warmup-steps','0'),
                       ('--prefetch-batches','2'),('--sample-workers','1'),('--audio-cache-gb','1.0'),
                       ('--min-lr-ratio','0.1'),('--lr-schedule','constant'),('--lr','0.0002'),('--plan-blank-window-seconds','0.5'),
                       ('--training-window-policy','phrase_complete_v1'),('--max-phrase-events','64'),('--phrase-seam-events','2'),
                       ('--phrase-execution-policy','component_modes_v2'),('--phrase-modes','4'),('--phrase-relation-weight','0.1'),
                       ('--detail-representation-policy','typed_components_v2'),('--duration-feature-policy','log_time_v2')]:
        assert runtime.args[runtime.args.index(flag)+1]==value


def test_stream_generate_routes_checkpoint_and_filters_old_knobs(tmp_path, monkeypatch):
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'fixture')
    audio=tmp_path/'audio.wav';audio.write_bytes(b'fixture')
    monkeypatch.setattr(event_set_api,'checkpoint_profile',lambda path:event_set_api.PROFILE)
    runtime, handler=Runtime(),Handler()
    composer_api.start_generate(handler,dict(checkpoint=str(checkpoint),audio=str(audio),output=str(tmp_path/'maidata.txt'),
        seed=0,level=13.,pointer_grid_policy='obsolete',density_feedback_policy='obsolete'),runtime=runtime)
    assert runtime.args[:2] == ['-m','maichart.composer.stream_generate']
    assert runtime.args[runtime.args.index('--phrase-mode-temperature') + 1] == '1.0'
    assert '--condition-file' in runtime.args
    assert '--pointer-grid-policy' not in runtime.args
    assert '--density-feedback-policy' not in runtime.args
    assert runtime.dispatch['expected_epoch'] is None


def test_phrase_generation_settings_reach_cli_and_effective_settings(tmp_path):
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'fixture')
    audio=tmp_path/'audio.wav';audio.write_bytes(b'fixture')
    runtime=Runtime()
    event_set_api.start_generate(Handler(),dict(checkpoint=str(checkpoint),audio=str(audio),
        output=str(tmp_path/'out.txt'),seed=0,phrase_candidates=4,audio_tracking_strength=0.,
        softcomfort_strength=1.5,phrase_rollback_budget=3),runtime=runtime)
    for flag,value in [('--phrase-candidates','4'),('--audio-tracking-strength','0.0'),
                       ('--softcomfort-strength','1.5'),('--phrase-rollback-budget','3')]:
        assert runtime.args[runtime.args.index(flag)+1]==value
    effective=runtime.task.result['effective_settings']
    assert effective['phrase_candidates']==4
    assert effective['audio_tracking_strength']==0.
    assert effective['search_policy']=='phrase_v1'


@pytest.mark.parametrize('settings',[{'phrase_candidates':3},{'audio_tracking_strength':1.1},
    {'softcomfort_strength':2.1},{'phrase_commit_beats':9.,'phrase_write_beats':8.}])
def test_phrase_generation_rejects_invalid_settings(tmp_path,settings):
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'fixture')
    audio=tmp_path/'audio.wav';audio.write_bytes(b'fixture')
    with pytest.raises(ValueError):
        event_set_api.start_generate(Handler(),dict(checkpoint=str(checkpoint),audio=str(audio),
            output=str(tmp_path/'out.txt'),seed=0,**settings),runtime=Runtime())


def test_stream_rejects_old_checkpoint_initialization(tmp_path,monkeypatch):
    from maichart.composer import stream_checkpoint
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    checkpoint=tmp_path/'old.pt';checkpoint.write_bytes(b'fixture')
    monkeypatch.setattr(event_set_api,'checkpoint_profile',lambda path:'audio_grounded_v3')
    monkeypatch.setattr(stream_checkpoint,'read',lambda path:{'schema':'old'})
    with pytest.raises(ValueError,match='incompatible'):
        event_set_api.start_train(Handler(),dict(mode='finetune',manifest=str(manifest),holdout_fraction=0,checkpoint_dir=str(tmp_path/'out'),init_checkpoint=str(checkpoint)),runtime=Runtime())


@pytest.mark.parametrize('mode', ['resume','finetune'])
def test_grid_inherits_parent_instead_of_ui_default(tmp_path,monkeypatch,mode):
    from maichart.composer import stream_checkpoint
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    checkpoint=tmp_path/'parent.pt';checkpoint.write_bytes(b'fixture')
    monkeypatch.setattr(event_set_api,'checkpoint_profile',lambda path:event_set_api.PROFILE)
    monkeypatch.setattr(stream_checkpoint,'read',lambda path:{'schema':event_set_api.SCHEMA,'model_config':{}})
    runtime=Runtime()
    event_set_api.start_train(Handler(),dict(mode=mode,manifest=str(manifest),holdout_fraction=0,checkpoint_dir=str(tmp_path/'out'),init_checkpoint=str(checkpoint),audio_grid_seconds=.01),runtime=runtime)
    assert runtime.args[runtime.args.index('--audio-grid-seconds')+1]=='0.05'


@pytest.mark.parametrize('extra,pointer,action',[
    ({'temperature':0},'0.0','0.0'),
    ({'temperature':.3,'pointer_temperature':0},'0.0','0.3'),
    ({'temperature':0,'action_temperature':.7},'0.0','0.7'),
    ({'temperature':.4,'pointer_temperature':None},'0.4','0.4'),
])
def test_api_stage_temperatures_inherit_total(tmp_path,extra,pointer,action):
    checkpoint=tmp_path/'cp.pt';checkpoint.write_bytes(b'fixture')
    audio=tmp_path/'audio.wav';audio.write_bytes(b'fixture')
    runtime=Runtime()
    event_set_api.start_generate(Handler(),dict(checkpoint=str(checkpoint),audio=str(audio),output=str(tmp_path/'out.txt'),seed=0,**extra),runtime=runtime)
    assert runtime.args[runtime.args.index('--pointer-temperature')+1]==pointer
    assert runtime.args[runtime.args.index('--action-temperature')+1]==action


@pytest.mark.parametrize('mode', ['fresh','finetune','resume'])
def test_learning_policies_are_real_subprocess_flags_and_resume_inherits(tmp_path,monkeypatch,mode):
    from maichart.composer import stream_checkpoint
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    checkpoint=tmp_path/'parent.pt';checkpoint.write_bytes(b'fixture')
    monkeypatch.setattr(stream_checkpoint,'read',lambda path:{'schema':event_set_api.SCHEMA,'model_config':{'history_events':96,'motion_continuation_policy':'motion_continuation_v1'}})
    runtime=Runtime()
    event_set_api.start_train(Handler(),dict(mode=mode,manifest=str(manifest),holdout_fraction=0,checkpoint_dir=str(tmp_path/'out'),init_checkpoint=str(checkpoint)),runtime=runtime)
    policies=runtime.task.result['effective_settings']
    expected='rolling_legacy_v1' if mode=='resume' else 'block_aligned_v2'
    assert policies['history_window_policy']==expected
    assert policies['continuation_policy']==('off' if mode=='resume' else 'pattern_runs_v2')
    assert policies['motion_continuation_policy']==('motion_continuation_v1' if mode=='resume' else 'context_motion_v3')
    assert policies['recovery_policy']==('persistent_v1' if mode=='resume' else 'greedy_mixed_v4')
    assert policies['prefix_augmentation_input_policy']==('causal_window_config_time_first_decision_v3' if mode=='resume' else 'musical_arrangement_recovery_v10')
    assert policies['plan_kind_policy']==('off' if mode=='resume' else 'phrase_arranged_hazard_v4')
    assert runtime.args[runtime.args.index('--plan-kind-policy')+1]==policies['plan_kind_policy']
    assert policies['timing_grid_audio_policy']==('off' if mode=='resume' else 'query_pool_v3')
    assert policies['transition_policy']=='off'
    assert policies['gap_continuation_policy']==('additive_v1' if mode=='resume' else 'off')
    assert policies['event_family_policy']==('legacy_v1' if mode=='resume' else 'shared_continuation_v1')
    assert policies['recovery_target_policy']==('authored_next_v1' if mode=='resume' else 'causal_correspondence_v1')
    assert runtime.args[runtime.args.index('--gap-continuation-policy')+1]==policies['gap_continuation_policy']
    assert policies['prefix_augmentation_rate']==(0 if mode=='resume' else .15)
    assert policies['generation_eval_temperature']==.65
    assert policies['lane_load_policy']==('off' if mode=='resume' else 'direct_buttons_v1')
    assert runtime.args[runtime.args.index('--lane-load-policy')+1]==policies['lane_load_policy']
    assert policies['transition_exit_weight']==0
    assert runtime.args[runtime.args.index('--history-window-policy')+1]==expected
    assert runtime.args[runtime.args.index('--transition-exit-weight')+1]==str(policies['transition_exit_weight'])
    if mode!='fresh':assert policies['history_events']==96


def test_api_rejects_resume_policy_upgrade_and_supports_explicit_old_baseline(tmp_path,monkeypatch):
    from maichart.composer import stream_checkpoint
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    checkpoint=tmp_path/'parent.pt';checkpoint.write_bytes(b'fixture')
    monkeypatch.setattr(stream_checkpoint,'read',lambda path:{'schema':event_set_api.SCHEMA,'model_config':{}})
    payload=dict(mode='resume',manifest=str(manifest),holdout_fraction=0,checkpoint_dir=str(tmp_path/'out'),init_checkpoint=str(checkpoint),continuation_policy='typed_runs_v1')
    with pytest.raises(ValueError,match='Resume inherits continuation_policy'):
        event_set_api.start_train(Handler(),payload,runtime=Runtime())
    runtime=Runtime()
    event_set_api.start_train(Handler(),{**payload,'mode':'fresh','learning_policy':'legacy','continuation_policy':'off'},runtime=runtime)
    assert runtime.task.result['effective_settings']['transition_exit_weight']==0
    assert runtime.task.result['effective_settings']['history_window_policy']=='rolling_legacy_v1'


@pytest.mark.parametrize('parent_values', [{}, {'history_window_policy':'block_aligned_v2',
    'continuation_policy':'typed_runs_v1','sensor_load_policy':'separate_touch_v1',
    'gap_continuation_policy':'probability_v2','motion_continuation_policy':'motion_distinct_v2',
    'lane_load_policy':'direct_buttons_v1','transition_exit_weight':.3}])
def test_explicit_checkpoint_learning_policy_finetune_inherits_parent(tmp_path,monkeypatch,parent_values):
    from maichart.composer import stream_checkpoint
    from maichart.composer.stream_model import StreamComposerConfig
    from maichart.composer.stream_train import LEARNING_POLICIES
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    checkpoint=tmp_path/'parent.pt';checkpoint.write_bytes(b'fixture')
    monkeypatch.setattr(stream_checkpoint,'read',lambda path:{'schema':event_set_api.SCHEMA,'model_config':parent_values})
    runtime=Runtime()
    event_set_api.start_train(Handler(),dict(mode='finetune',learning_policy='checkpoint',
        manifest=str(manifest),holdout_fraction=0,checkpoint_dir=str(tmp_path/'out'),init_checkpoint=str(checkpoint)),runtime=runtime)
    parent=StreamComposerConfig.from_dict(parent_values)
    for key in (*LEARNING_POLICIES,'transition_exit_weight','prefix_augmentation_rate'):
        assert runtime.task.result['effective_settings'][key]==getattr(parent,key)
        flag='--'+key.replace('_','-')
        assert runtime.args[runtime.args.index(flag)+1]==str(getattr(parent,key))


def test_explicit_checkpoint_learning_policy_without_parent_is_rejected(tmp_path):
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    runtime=Runtime()
    with pytest.raises(ValueError,match='requires a parent checkpoint'):
        event_set_api.start_train(Handler(),dict(mode='fresh',learning_policy='checkpoint',
            manifest=str(manifest),holdout_fraction=0,checkpoint_dir=str(tmp_path/'out')),runtime=runtime)
    assert not hasattr(runtime,'args')


def test_augmentation_and_eval_temperature_are_real_flags_with_resume_checks(tmp_path,monkeypatch):
    from maichart.composer import stream_checkpoint
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    checkpoint=tmp_path/'parent.pt';checkpoint.write_bytes(b'fixture')
    monkeypatch.setattr(stream_checkpoint,'read',lambda path:{'schema':event_set_api.SCHEMA,'model_config':{},'args':{'generation_eval_temperature':.4}})
    payload=dict(manifest=str(manifest),holdout_fraction=0,checkpoint_dir=str(tmp_path/'out'),init_checkpoint=str(checkpoint))
    runtime=Runtime()
    event_set_api.start_train(Handler(),dict(payload,mode='finetune',prefix_augmentation_rate=.2,generation_eval_temperature=.7),runtime=runtime)
    assert runtime.args[runtime.args.index('--prefix-augmentation-rate')+1]=='0.2'
    assert runtime.args[runtime.args.index('--generation-eval-temperature')+1]=='0.7'
    event_set_api.start_train(Handler(),dict(payload,mode='resume'),runtime=runtime)
    assert runtime.task.result['effective_settings']['prefix_augmentation_rate']==0
    assert runtime.task.result['effective_settings']['generation_eval_temperature']==.4
    with pytest.raises(ValueError,match='Resume inherits generation_eval_temperature'):
        event_set_api.start_train(Handler(),dict(payload,mode='resume',generation_eval_temperature=.7),runtime=Runtime())


def test_automatic_split_dispatches_effective_manifests_and_records_evidence(tmp_path):
    import json
    songs=[]
    for i in range(4):
        audio=tmp_path/f'audio_{i}.wav'; audio.write_bytes(bytes([i]))
        songs.append(dict(song_id=str(i),audio_path=str(audio),difficulties=[dict(difficulty_index=4,level_raw='13')]))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(songs=songs)))
    runtime=Runtime()
    event_set_api.start_train(Handler(),dict(manifest=str(manifest),checkpoint_dir=str(tmp_path/'out')),
                              runtime=runtime)
    report=runtime.task.result['data_split']
    assert runtime.args[runtime.args.index('--manifest')+1]==report['splits']['train']['path']
    assert runtime.args[runtime.args.index('--validation-manifest')+1]==report['splits']['dev']['path']
    assert set(report['splits']['train']['audio_sha256']).isdisjoint(report['splits']['dev']['audio_sha256'])
    assert (tmp_path/'out'/'data_split'/'split_report.json').is_file()


def test_rejected_existing_output_does_not_rewrite_split(tmp_path):
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    output=tmp_path/'out'/'data_split';output.mkdir(parents=True)
    sentinel=output/'train_manifest.json';sentinel.write_text('old split')
    with pytest.raises(ValueError,match='new empty checkpoint'):
        event_set_api.start_train(Handler(),dict(manifest=str(manifest),checkpoint_dir=str(tmp_path/'out')),
                                  runtime=Runtime())
    assert sentinel.read_text()=='old split'


def test_same_process_busy_request_does_not_write_split(tmp_path):
    from maichart.composer.training_lock import training_request_lock, TrainingOutputBusyError
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    output=tmp_path/'out'
    with training_request_lock(output):
        with pytest.raises(TrainingOutputBusyError):
            event_set_api.start_train(Handler(),dict(manifest=str(manifest),checkpoint_dir=str(output)),
                                      runtime=Runtime())
    assert not (output/'data_split').exists()


def test_advanced_policies_and_generation_panel_controls_reach_cli(tmp_path):
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    runtime=Runtime()
    event_set_api.start_train(Handler(),dict(manifest=str(manifest),holdout_fraction=0,
        checkpoint_dir=str(tmp_path/'out'),learning_policy='legacy',
        prefix_augmentation_input_policy='musical_loop_recovery_v8',recovery_policy='greedy_mixed_v4',
        generation_eval_phrase_candidates=4,generation_eval_phrase_step_budget=321),runtime=runtime)
    for flag,value in [('--prefix-augmentation-input-policy','musical_loop_recovery_v8'),
                       ('--recovery-policy','greedy_mixed_v4'),('--generation-eval-phrase-candidates','4'),
                       ('--generation-eval-phrase-step-budget','321')]:
        assert runtime.args[runtime.args.index(flag)+1]==value


def test_finetune_backbone_and_resume_objective_values_inherit_parent(tmp_path,monkeypatch):
    from maichart.composer import stream_checkpoint
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    checkpoint=tmp_path/'parent.pt';checkpoint.write_bytes(b'fixture')
    monkeypatch.setattr(stream_checkpoint,'read',lambda path:dict(schema=event_set_api.SCHEMA,
        model_config=dict(d_model=128,nhead=4,history_events=96,dropout=.2),
        args=dict(seed=7,recovery_interval=13,recovery_weight=.3,max_steps=8)))
    for mode in ('finetune','resume'):
        runtime=Runtime()
        event_set_api.start_train(Handler(),dict(mode=mode,manifest=str(manifest),holdout_fraction=0,
            checkpoint_dir=str(tmp_path/'out'),init_checkpoint=str(checkpoint),d_model=512,
            history_events=64,dropout=.1),runtime=runtime)
        settings=runtime.task.result['effective_settings']
        assert settings['d_model']==128 and settings['history_events']==96 and settings['dropout']==.2
        if mode=='resume':
            assert settings['seed']==7 and settings['recovery_interval']==13
            assert settings['recovery_weight']==.3 and settings['max_steps']==8


def test_resume_refuses_new_selected_manifest_before_dispatch(tmp_path,monkeypatch):
    import json
    from maichart.composer import stream_checkpoint
    source=tmp_path/'source.json';source.write_text('{}')
    effective=tmp_path/'train.json';effective.write_text(json.dumps(dict(split_metadata=dict(source_manifest=str(source)))))
    other=tmp_path/'new.json';other.write_text('{}')
    checkpoint=tmp_path/'parent.pt';checkpoint.write_bytes(b'fixture')
    monkeypatch.setattr(stream_checkpoint,'read',lambda path:dict(schema=event_set_api.SCHEMA,model_config={},args=dict(manifest=str(effective))))
    runtime=Runtime()
    with pytest.raises(ValueError,match='use finetune for new data'):
        event_set_api.start_train(Handler(),dict(mode='resume',manifest=str(other),holdout_fraction=0,
            checkpoint_dir=str(tmp_path/'out'),init_checkpoint=str(checkpoint)),runtime=runtime)
    assert not hasattr(runtime,'args')
    event_set_api.start_train(Handler(),dict(mode='resume',manifest=str(source),holdout_fraction=0,
        checkpoint_dir=str(tmp_path/'out'),init_checkpoint=str(checkpoint)),runtime=runtime)
    assert runtime.args[runtime.args.index('--manifest')+1]==str(effective)


def test_validation_sampling_inherits_total_temperature_until_overridden(tmp_path):
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    runtime=Runtime()
    payload=dict(manifest=str(manifest),holdout_fraction=0,checkpoint_dir=str(tmp_path/'out'),
        generation_eval_pointer_temperature=None,generation_eval_action_temperature=0,
        generation_eval_min_same_head_ms=40,generation_eval_touch_layout_policy='open',generation_eval_max_events=70)
    event_set_api.start_train(Handler(),payload,runtime=runtime)
    assert '--generation-eval-pointer-temperature' not in runtime.args
    for flag,value in [('--generation-eval-action-temperature','0'),('--generation-eval-min-same-head-ms','40'),
                       ('--generation-eval-touch-layout-policy','open'),('--generation-eval-max-events','70')]:
        assert runtime.args[runtime.args.index(flag)+1]==value

def test_resume_learning_rate_override_and_timing_compute(tmp_path, monkeypatch):
    from maichart.composer import stream_checkpoint
    manifest=tmp_path/'manifest.json'; manifest.write_text('{}')
    checkpoint=tmp_path/'parent.pt'; checkpoint.write_bytes(b'fixture')
    monkeypatch.setattr(stream_checkpoint, 'read', lambda path: {
        'schema': event_set_api.SCHEMA, 'model_config': {},
        'args': {'lr': .0002, 'lr_schedule': 'cosine', 'warmup_steps': 100, 'min_lr_ratio': .1}})
    payload=dict(mode='resume', manifest=str(manifest), holdout_fraction=0,
        checkpoint_dir=str(tmp_path/'out'), init_checkpoint=str(checkpoint),
        lr=.0005, lr_schedule='constant', warmup_steps=0, min_lr_ratio=1, timing_compute='dense')
    normal=Runtime(); event_set_api.start_train(Handler(), payload, runtime=normal)
    settings=normal.task.result['effective_settings']
    assert (settings['lr'],settings['lr_schedule'],settings['warmup_steps'],settings['min_lr_ratio']) == (.0002,'cosine',100,.1)
    assert '--reset-learning-rate' not in normal.args
    override=Runtime(); event_set_api.start_train(Handler(), {**payload,'reset_learning_rate':True}, runtime=override)
    settings=override.task.result['effective_settings']
    assert (settings['lr'],settings['lr_schedule'],settings['warmup_steps'],settings['min_lr_ratio']) == (.0005,'constant',0,1)
    assert '--reset-learning-rate' in override.args
    assert override.args[override.args.index('--timing-compute')+1]=='dense'
    with pytest.raises(ValueError,match='only available for resume'):
        event_set_api.start_train(Handler(), {**payload,'mode':'fresh','reset_learning_rate':True}, runtime=Runtime())
    with pytest.raises(ValueError,match='Unknown timing compute'):
        event_set_api.start_train(Handler(), {**payload,'timing_compute':'invalid'}, runtime=Runtime())


def test_song_style_default_snapshot_and_explicit_file_are_distinct(tmp_path):
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'fixture')
    audio=tmp_path/'audio.wav';audio.write_bytes(b'fixture')
    condition=tmp_path/'condition.json';condition.write_text('{}')
    base=dict(checkpoint=str(checkpoint),audio=str(audio),output=str(tmp_path/'out.txt'),seed=0,
              song_style_source='auto',song_style_locks={'tap_bias':.2},song_response_mode=2)
    default=Runtime();event_set_api.start_generate(Handler(),base,runtime=default)
    assert '--condition-file-defaults' in default.args
    assert default.args[default.args.index('--song-style-source')+1]=='auto'
    assert 'tap_bias' in default.args[default.args.index('--song-style-locks')+1]
    assert default.args[default.args.index('--song-response-mode')+1]=='2'
    explicit=Runtime();event_set_api.start_generate(Handler(),{**base,'condition_file':str(condition)},runtime=explicit)
    assert '--condition-file-defaults' not in explicit.args


def test_data_worker_runtime_setting_reaches_trainer(tmp_path):
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    base=dict(manifest=str(manifest),holdout_fraction=0,checkpoint_dir=str(tmp_path/'out'))
    runtime=Runtime();event_set_api.start_train(Handler(),{**base,'data_workers':1},runtime=runtime)
    assert runtime.args[runtime.args.index('--data-workers')+1]=='1'
    assert runtime.task.result['effective_settings']['data_workers']==1
    with pytest.raises(ValueError,match='data_workers'):
        event_set_api.start_train(Handler(),{**base,'data_workers':2},runtime=Runtime())


def test_resume_explicit_rebatch_preserves_effective_batch(tmp_path, monkeypatch):
    from maichart.composer import stream_checkpoint
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    checkpoint=tmp_path/'parent.pt';checkpoint.write_bytes(b'fixture')
    monkeypatch.setattr(stream_checkpoint,'read',lambda path:{'schema':event_set_api.SCHEMA,'model_config':{},'args':{'batch_size':32,'grad_accumulation':2}})
    payload=dict(mode='resume',manifest=str(manifest),holdout_fraction=0,checkpoint_dir=str(tmp_path/'out'),init_checkpoint=str(checkpoint),batch_size=16,grad_accumulation=4)
    normal=Runtime();event_set_api.start_train(Handler(),payload,runtime=normal)
    assert (normal.task.result['effective_settings']['batch_size'],normal.task.result['effective_settings']['grad_accumulation'])==(32,2)
    assert '--rebatch-resume' not in normal.args
    edited=Runtime();event_set_api.start_train(Handler(),{**payload,'rebatch_resume':True},runtime=edited)
    assert (edited.task.result['effective_settings']['batch_size'],edited.task.result['effective_settings']['grad_accumulation'])==(16,4)
    assert '--rebatch-resume' in edited.args
    with pytest.raises(ValueError,match='same effective batch'):
        event_set_api.start_train(Handler(),{**payload,'rebatch_resume':True,'grad_accumulation':3},runtime=Runtime())


def test_detail_compute_runtime_setting_reaches_trainer(tmp_path):
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    base=dict(manifest=str(manifest),holdout_fraction=0,checkpoint_dir=str(tmp_path/'out'))
    default=Runtime();event_set_api.start_train(Handler(),base,runtime=default)
    assert default.task.result['effective_settings']['detail_compute']=='serial'
    for mode in ('packed','serial'):
        runtime=Runtime();event_set_api.start_train(Handler(),{**base,'detail_compute':mode},runtime=runtime)
        assert runtime.args[runtime.args.index('--detail-compute')+1]==mode
        assert runtime.task.result['effective_settings']['detail_compute']==mode
    with pytest.raises(ValueError,match='detail compute'):
        event_set_api.start_train(Handler(),{**base,'detail_compute':'invalid'},runtime=Runtime())


def test_unified_song_choice_uses_profile_component_count(tmp_path):
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    runtime=Runtime();event_set_api.start_train(Handler(),dict(manifest=str(manifest),holdout_fraction=0,checkpoint_dir=str(tmp_path/'out'),song_profile_components=6,song_response_modes=3),runtime=runtime)
    assert runtime.task.result['effective_settings']['song_response_modes']==6
    assert runtime.args[runtime.args.index('--song-response-modes')+1]=='6'
