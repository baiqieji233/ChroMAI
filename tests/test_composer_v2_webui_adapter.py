"""Adapter boundary checks: invalid requests must never reserve a task."""
from types import SimpleNamespace
import pytest
from maichart.webui import composer_v2_api as api

class Runtime:
    def __init__(self): self.created=[];self.launched=[]
    def _prepare_training_sources(self,payload): return payload,None
    def _create_task(self,kind,checkpoint_dir=None):
        import threading
        task=SimpleNamespace(task_id="v2",kind=kind,result={},lock=threading.Lock())
        self.created.append(task)
        return task
    def _run_subprocess_task(self,task,args,**kwargs): self.launched.append((task,args,kwargs))

class Handler:
    def _json_response(self,result): self.result=result;return result


def test_onset_readout_training_and_generation_cli_wiring(tmp_path,monkeypatch):
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    parent=tmp_path/'parent.pt';parent.touch()
    audio=tmp_path/'audio.wav';audio.touch()
    info=dict(valid=True,can_init=True,can_resume=True,sequence_policy='continuous_events_v1',
              model_config={'sequence_policy':'continuous_events_v1'})
    monkeypatch.setattr(api,'inspection',lambda _:info)
    runtime=Runtime()
    api.start_train(Handler(),dict(manifest=str(manifest),checkpoint_dir=str(tmp_path/'trial'),
        init_checkpoint=str(parent),mode='finetune',validation_source='off',
        onset_readout='phase_audio_v1'),runtime=runtime)
    from maichart.composer_v2.train import parser as training_parser
    options=training_parser().parse_args(runtime.launched[-1][1][2:])
    assert options.onset_readout=='phase_audio_v1'
    runtime=Runtime()
    with pytest.raises(ValueError,match='微调'):
        api.start_generate(Handler(),dict(checkpoint=str(parent),audio=str(audio),onset_strength=1),runtime=runtime)
    assert not runtime.created
    info['model_config']['onset_readout_policy']='phase_audio_v1'
    api.start_generate(Handler(),dict(checkpoint=str(parent),audio=str(audio),onset_strength=0),runtime=runtime)
    from maichart.composer_v2.generate import parser as generation_parser
    options=generation_parser().parse_args(runtime.launched[-1][1][2:])
    assert options.onset_strength==0

def test_condition_and_timing_contract_before_task(tmp_path, monkeypatch):
    checkpoint=tmp_path/'model.pt';checkpoint.touch()
    audio=tmp_path/'song.wav';audio.touch()
    runtime=Runtime()
    monkeypatch.setattr(api,'inspection',lambda _:dict(valid=True,condition_contract={'policy':'radar_style_v1'}))
    base=dict(checkpoint=str(checkpoint),audio=str(audio))
    for override in ({'unknown_condition':.5},{'note_density':float('nan')},{'peak':1.1}):
        with pytest.raises(ValueError):
            api.start_generate(Handler(),dict(base,bpm=180,condition=override),runtime=runtime)
    assert not runtime.created
    api.start_generate(Handler(),base,runtime=runtime)
    automatic_args=runtime.launched[-1][1]
    assert '--bpm' not in automatic_args and '--first' not in automatic_args
    api.start_generate(Handler(),dict(base,first=0),runtime=runtime)
    manual_args=runtime.launched[-1][1]
    assert manual_args[manual_args.index('--first')+1]=='0.0'
    api.start_generate(Handler(),dict(base,bpm=180,condition={'peak':0.,'melody_following':.6}),runtime=runtime)
    import json
    args=runtime.launched[-1][1]
    assert json.loads(args[args.index('--condition')+1])=={'peak':0.,'melody_following':.6}
    monkeypatch.setattr(api,'inspection',lambda _:dict(valid=True))
    with pytest.raises(ValueError,match='等级'):
        api.start_generate(Handler(),dict(base,bpm=180,condition={'peak':.5}),runtime=runtime)

def test_train_api_defaults_match_executable_parser():
    from maichart.composer_v2.train import parser
    defaults=vars(parser().parse_args(['--manifest','source.json','--checkpoint-dir','new']))
    assert {key:defaults[key] for key in api.TRAIN_DEFAULTS}==api.TRAIN_DEFAULTS


def test_training_resource_settings_reach_cli(tmp_path):
    from maichart.composer_v2.train import parser

    manifest = tmp_path / 'training.json'
    manifest.write_text('{}', encoding='utf-8')
    runtime = Runtime()
    payload = dict(manifest=str(manifest), checkpoint_dir=str(tmp_path / 'output'),
                   validation_source='off', batch_size=2, grad_accumulation=32,
                   cuda_memory_budget_gb=7.5, audio_cache_gb=0,timing_recovery='causal_rhythm_v1')
    api.start_train(Handler(), payload, runtime=runtime)
    task, command, _ = runtime.launched[0]
    options = parser().parse_args(command[2:])
    assert options.sequence_policy == 'continuous_events_v1'
    assert options.timing_recovery == task.result['effective_settings']['timing_recovery'] == 'causal_rhythm_v1'
    for key in ('batch_size', 'grad_accumulation', 'cuda_memory_budget_gb', 'audio_cache_gb'):
        assert getattr(options, key) == task.result['effective_settings'][key] == payload[key]

def test_v2_manual_tempo_map_and_automatic_offset_reach_generator(tmp_path,monkeypatch):
    import json
    from maichart.composer_v2.generate import parser
    checkpoint=tmp_path/'model.pt';checkpoint.touch()
    audio=tmp_path/'song.wav';audio.touch()
    monkeypatch.setattr(api,'inspection',lambda _:dict(valid=True))
    runtime=Runtime()
    tempo_map=[{'beat':'0','bpm':112.},{'beat':'36','bpm':180.}]
    api.start_generate(Handler(),dict(checkpoint=str(checkpoint),audio=str(audio),
        tempo_map=tempo_map,first=None),runtime=runtime)
    command=runtime.launched[-1][1]
    assert '--first' not in command and '--bpm' not in command
    assert json.loads(command[command.index('--tempo-map')+1])==tempo_map
    args=parser().parse_args(command[2:])
    assert args.first is None and args.bpm is None and args.tempo_map==tempo_map

def test_invalid_generate_candidate_has_no_task(tmp_path,monkeypatch):
    ckpt=tmp_path/"model.pt";ckpt.touch()
    audio=tmp_path/"song.wav";audio.touch()
    monkeypatch.setattr(api,"inspection",lambda _:dict(valid=True))
    runtime=Runtime()
    with pytest.raises(ValueError,match="候选"):
        api.start_generate(Handler(),dict(checkpoint=str(ckpt),audio=str(audio),bpm=180,candidates=3),runtime=runtime)
    assert not runtime.created

def test_split_generation_settings_reach_cli_including_zero(tmp_path,monkeypatch):
    from maichart.composer_v2.generate import parser

    checkpoint=tmp_path/'model.pt';checkpoint.touch()
    audio=tmp_path/'song.wav';audio.touch()
    monkeypatch.setattr(api,'inspection',lambda _:dict(valid=True,sequence_policy='continuous_events_v1'))
    runtime=Runtime()
    request=dict(checkpoint=str(checkpoint),audio=str(audio),temperature=.65,top_p=.93,
        action_temperature=0.,action_top_p=.98)
    api.start_generate(Handler(),request,runtime=runtime)
    task,command,_=runtime.launched[0]
    args=parser().parse_args(command[2:])
    for key in ('temperature','top_p','action_temperature','action_top_p'):
        assert getattr(args,key)==task.result['effective_settings'][key]==request[key]


def test_split_sampling_rejects_window_checkpoints_before_launch(tmp_path,monkeypatch):
    checkpoint=tmp_path/'model.pt';checkpoint.touch()
    audio=tmp_path/'song.wav';audio.touch()
    monkeypatch.setattr(api,'inspection',lambda _:dict(valid=True,sequence_policy='window_v1'))
    runtime=Runtime()
    with pytest.raises(ValueError,match='连续事件'):
        api.start_generate(Handler(),dict(checkpoint=str(checkpoint),audio=str(audio),
            action_temperature=1.1),runtime=runtime)
    assert not runtime.created


def test_fresh_mert_contract_is_separate(tmp_path):
    manifest=tmp_path/"manifest.json";manifest.write_text("{}")
    runtime=Runtime();handler=Handler()
    api.start_train(handler,dict(manifest=str(manifest),checkpoint_dir=str(tmp_path/"v2"),holdout_fraction=0),runtime=runtime)
    task,args,kwargs=runtime.launched[0]
    assert task.kind=="composer_v2_train"
    assert args[:2]==["-m","maichart.composer_v2.train"]
    assert args[args.index("--audio-backend")+1]=="mert"
    assert args[args.index("--grad-accumulation")+1]=="8"
    assert all("composer_v2_" in path.name for path in kwargs["result_paths"])

def test_foreign_family_is_rejected_before_task(tmp_path):
    runtime=Runtime()
    with pytest.raises(ValueError,match="chart_composer_v2"):
        api.start_train(Handler(),dict(model_family="chart_composer_v1"),runtime=runtime)
    assert not runtime.created


def test_rollback_bounds_and_cli(tmp_path,monkeypatch):
    ckpt=tmp_path/"model.pt";ckpt.touch()
    audio=tmp_path/"song.wav";audio.touch()
    monkeypatch.setattr(api,"inspection",lambda _:dict(valid=True))
    runtime=Runtime()
    with pytest.raises(ValueError,match="rollback_windows"):
        api.start_generate(Handler(),dict(checkpoint=str(ckpt),audio=str(audio),bpm=180,rollback_windows=9),runtime=runtime)
    assert not runtime.created
    api.start_generate(Handler(),dict(checkpoint=str(ckpt),audio=str(audio),bpm=180),runtime=runtime)
    task,args,_=runtime.launched[0]
    assert args[args.index("--rollback-windows")+1]=="2"
    assert task.result["effective_settings"]["rollback_windows"]==2


def test_exhaustion_policy_reaches_cli_and_invalid_value_never_creates_task(tmp_path,monkeypatch):
    ckpt=tmp_path/'model.pt';ckpt.touch()
    audio=tmp_path/'song.wav';audio.touch()
    monkeypatch.setattr(api,'inspection',lambda _:dict(valid=True))
    runtime=Runtime();base=dict(checkpoint=str(ckpt),audio=str(audio),bpm=190)
    with pytest.raises(ValueError):api.start_generate(Handler(),dict(base,exhaustion_policy='ignore_everything'),runtime=runtime)
    assert not runtime.created
    for policy in ('auto','report','stop'):
        api.start_generate(Handler(),dict(base,exhaustion_policy=policy),runtime=runtime)
        task,args,_=runtime.launched[-1]
        assert args[args.index('--exhaustion-policy')+1]==policy
        assert task.result['effective_settings']['exhaustion_policy']==policy


def test_auto_split_is_default_and_never_reused(tmp_path,monkeypatch):
    import json
    from maichart.composer import evaluation
    source=tmp_path/"source.json";source.write_text(json.dumps({"songs":[]}))
    calls=[]
    def split(manifest,out,**kwargs):
        calls.append((manifest,out,kwargs));out.mkdir(parents=True)
        paths={name:{"path":str(out/(name+"_manifest.json"))} for name in ("train","dev","final")}
        for item in paths.values():Path(item["path"]).write_text("{}")
        return {"splits":paths}
    from pathlib import Path
    monkeypatch.setattr(evaluation,"split_manifest",split)
    monkeypatch.setattr(evaluation,"assert_unseen",lambda *args,**kwargs:None)
    runtime=Runtime();payload=dict(manifest=str(source),checkpoint_dir=str(tmp_path/"out"))
    api.start_train(Handler(),payload,runtime=runtime)
    task,args,_=runtime.launched[0]
    assert calls[0][2]["fraction"]==.05
    assert args[args.index("--manifest")+1].endswith("train_manifest.json")
    assert args[args.index("--validation-manifest")+1].endswith("dev_manifest.json")
    assert task.result["effective_settings"]["split_report"]["splits"]["final"]
    with pytest.raises(ValueError,match="data_split"):
        api.start_train(Handler(),payload,runtime=runtime)
    assert len(runtime.created)==1


def test_resume_inherits_split_without_resplitting(tmp_path,monkeypatch):
    import json,hashlib
    from maichart.composer import evaluation
    source=tmp_path/"source.json";source.write_text("{}")
    train=tmp_path/"train.json";train.write_text(json.dumps({"split_metadata":{
        "source_manifest":str(source),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest()}}))
    dev=tmp_path/"dev.json";dev.write_text("{}")
    ckpt=tmp_path/"last.pt";ckpt.touch()
    monkeypatch.setattr(api,"inspection",lambda _:dict(valid=True,can_resume=True,args={"manifest":str(train),"validation_manifest":str(dev)}))
    monkeypatch.setattr(evaluation,"split_manifest",lambda *args,**kwargs:pytest.fail("resume resplit"))
    monkeypatch.setattr(evaluation,"assert_unseen",lambda *args,**kwargs:None)
    runtime=Runtime()
    api.start_train(Handler(),dict(manifest=str(source),mode="resume",init_checkpoint=str(ckpt),checkpoint_dir=str(tmp_path/"out")),runtime=runtime)
    _,args,_=runtime.launched[0]
    assert args[args.index("--manifest")+1]==str(train)
    assert args[args.index("--validation-manifest")+1]==str(dev)


def test_fractional_window_fails_before_split_or_task(tmp_path):
    source=tmp_path/"source.json";source.write_text("{}")
    runtime=Runtime()
    with pytest.raises(ValueError,match="window_beats"):
        api.start_train(Handler(),dict(manifest=str(source),window_beats=12.5),runtime=runtime)
    assert not runtime.created


def test_real_song_split_is_disjoint_and_preserves_final_holdout(tmp_path):
    import json
    songs=[]
    for i in range(4):
        audio=tmp_path/f"audio{i}.bin";audio.write_bytes(bytes([i+1])*32)
        chart=tmp_path/f"chart{i}.txt";chart.write_text("chart"+str(i))
        songs.append(dict(song_id=str(i),audio_path=str(audio),maidata_path=str(chart),difficulties=[]))
    source=tmp_path/"source.json";source.write_text(json.dumps(dict(songs=songs)))
    runtime=Runtime()
    api.start_train(Handler(),dict(manifest=str(source),checkpoint_dir=str(tmp_path/"out")),runtime=runtime)
    report=runtime.created[0].result["effective_settings"]["split_report"]
    from pathlib import Path
    sets=[{song["song_id"] for song in json.loads(Path(report["splits"][part]["path"]).read_text())["songs"]} for part in ("train","dev","final")]
    assert all(sets)
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]


def test_manual_overlap_and_missing_mert_directory_fail_before_task(tmp_path):
    import json
    audio=tmp_path/"audio.bin";audio.write_bytes(b"same")
    source=tmp_path/"source.json";source.write_text(json.dumps({"songs":[{"song_id":"same","audio_path":str(audio)}]}))
    runtime=Runtime()
    with pytest.raises(ValueError,match="overlap"):
        api.start_train(Handler(),dict(manifest=str(source),validation_source="manual",validation_manifest=str(source)),runtime=runtime)
    assert not runtime.created
    with pytest.raises(ValueError,match="MERT"):
        api.start_train(Handler(),dict(manifest=str(source),mert_model="not-a-local-model",holdout_fraction=0),runtime=runtime)
    assert not runtime.created


def test_resume_architecture_is_checkpoint_authoritative(tmp_path,monkeypatch):
    manifest=tmp_path/"train.json";manifest.write_text("{}")
    parent=tmp_path/"last.pt";parent.touch()
    monkeypatch.setattr(api,"inspection",lambda _:dict(valid=True,can_resume=True,
        args={"manifest":str(manifest),"seed":7},model_config={"d_model":64,"layers":2,"nhead":4,
        "dim_feedforward":128,"window_beats":24,"history_beats":8,"future_beats":2,
        "audio_backend":"native","mert_model":None}))
    runtime=Runtime()
    api.start_train(Handler(),dict(manifest=str(manifest),init_checkpoint=str(parent),mode="resume",
        d_model=384,audio_backend="mert",mert_model="stale-path",checkpoint_dir=str(tmp_path/"out")),runtime=runtime)
    task,args,_=runtime.launched[0]
    assert args[args.index("--d-model")+1]=="64"
    assert args[args.index("--window-beats")+1]=="24"
    assert args[args.index("--audio-backend")+1]=="native"
    assert args[args.index("--seed")+1]=="7"
    assert "--mert-model" not in args


def test_generation_cpu_threads_positive_integer(tmp_path,monkeypatch):
    parent=tmp_path/"last.pt";parent.touch()
    audio=tmp_path/"song.wav";audio.touch()
    monkeypatch.setattr(api,"inspection",lambda _:dict(valid=True))
    for value in (0,-1,1.5):
        runtime=Runtime()
        with pytest.raises(ValueError,match="cpu_threads"):
            api.start_generate(Handler(),dict(checkpoint=str(parent),audio=str(audio),bpm=180,cpu_threads=value),runtime=runtime)
        assert not runtime.created
    runtime=Runtime()
    api.start_generate(Handler(),dict(checkpoint=str(parent),audio=str(audio),bpm=180),runtime=runtime)
    _,args,_=runtime.launched[0]
    assert args[args.index("--cpu-threads")+1]=="2"


def test_manual_validation_ignores_disabled_zero_fraction(tmp_path, monkeypatch):
    from maichart.composer import evaluation
    source=tmp_path/'train.json';source.write_text('{}')
    dev=tmp_path/'dev.json';dev.write_text('{}')
    checked=[]
    monkeypatch.setattr(evaluation,'assert_unseen',lambda *args,**kwargs:checked.append(args))
    monkeypatch.setattr(evaluation,'split_manifest',lambda *args,**kwargs:pytest.fail('manual validation resplit'))
    runtime=Runtime()
    api.start_train(Handler(),dict(manifest=str(source),checkpoint_dir=str(tmp_path/'out'),
        validation_source='manual',validation_manifest=str(dev),holdout_fraction=0),runtime=runtime)
    task,args,_=runtime.launched[0]
    assert task.result['effective_settings']['validation_source']=='manual'
    assert args[args.index('--validation-manifest')+1]==str(dev)
    assert checked==[(source,dev)]


def test_finetune_inherits_refresh_and_clears_stale_mert_model(tmp_path, monkeypatch):
    source=tmp_path/'train.json';source.write_text('{}')
    parent=tmp_path/'last.pt';parent.touch()
    monkeypatch.setattr(api,'inspection',lambda _:dict(valid=True,can_init=True,
        model_config=dict(d_model=64,nhead=4,layers=2,dim_feedforward=128,
            audio_backend='native',mert_model=None,refresh_events=32)))
    runtime=Runtime()
    api.start_train(Handler(),dict(manifest=str(source),checkpoint_dir=str(tmp_path/'out'),
        mode='finetune',init_checkpoint=str(parent),holdout_fraction=0,
        refresh_events=16,mert_model='stale-path',d_model=384),runtime=runtime)
    _,args,_=runtime.launched[0]
    assert args[args.index('--refresh-events')+1]=='32'
    assert args[args.index('--d-model')+1]=='64'
    assert '--mert-model' not in args


@pytest.mark.parametrize('level',[0,16,float('inf')])
def test_bad_level_never_starts_generation(tmp_path,monkeypatch,level):
    parent=tmp_path/'last.pt';parent.touch()
    audio=tmp_path/'song.wav';audio.touch()
    monkeypatch.setattr(api,'inspection',lambda _:dict(valid=True))
    runtime=Runtime()
    with pytest.raises(ValueError):
        api.start_generate(Handler(),dict(checkpoint=str(parent),audio=str(audio),level=level),runtime=runtime)
    assert not runtime.created
