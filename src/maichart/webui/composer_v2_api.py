"""Main-workbench adapter for the independent Composer v2 contract."""
from __future__ import annotations
import json
import hashlib
import math
from pathlib import Path
from .composer_api import _path, _int, _float, _arg

FAMILY = "chart_composer_v2"
SCHEMA = "maichart-composer-v2-checkpoint-v1"
API_CONTRACT = "composer_v2_continuous_events_v3"
from maichart.composer_v2.defaults import TRAIN_DEFAULTS, GENERATION_DEFAULTS, MAX_BATCH_COUNT

def inspection(path):
    from maichart.composer_v2.checkpoint import inspect as inspect_checkpoint
    result = dict(inspect_checkpoint(path))
    if result.get("schema") not in {None, SCHEMA} or result.get("family") not in {None, FAMILY}:
        result.update(valid=False, can_resume=False, can_init=False,
                      error="Composer v2 schema/family mismatch", errors=["Composer v2 schema/family mismatch"])
    result.update(path=str(path), model_family=FAMILY, generation_entry="/")
    result.setdefault("family", FAMILY)
    result.setdefault("errors", [result["error"]] if result.get("error") else [])
    return result

def inspect(handler,payload):
    path = _path(payload.get("checkpoint") or payload.get("checkpoint_path"), "checkpoint", exists=True)
    try:
        result = inspection(path)
    except Exception as exc:
        result = dict(valid=False,model_family=FAMILY,path=str(path),error=str(exc),errors=[str(exc)])
    handler._json_response(result)

def _family(payload):
    if payload.get("model_family") not in {None,FAMILY}:
        raise ValueError(f"此接口只接受 {FAMILY}")

def _integer(payload,key,default,minimum=0):
    value=payload.get(key,default)
    if isinstance(value,bool):raise ValueError(f"{key} 必须是整数")
    converted=_float(payload,key,default)
    if not converted.is_integer():raise ValueError(f"{key} 必须是整数")
    result=int(converted)
    if result<minimum:raise ValueError(f"{key} 不能小于 {minimum}")
    return result

def _choice(payload,key,default,choices):
    value=str(payload.get(key,default))
    if value not in choices: raise ValueError(f"{key} 必须是 {', '.join(map(str,choices))}")
    return value

def _start(handler,runtime,kind,args,settings,paths,checkpoint_dir=None):
    task=runtime._create_task(kind,checkpoint_dir=checkpoint_dir)
    task.result.update(model_family=FAMILY,family=FAMILY,schema=SCHEMA,effective_settings=settings)
    if kind in ('composer_v2_curve','composer_v2_media'):
        task.result.update(audio=settings['audio'],audio_preview_url=f'/api/intensity/audio?id={task.task_id}')
    if kind.endswith("_train"): task.result["epoch_target"]=settings["epochs"]
    try:
        runtime._run_subprocess_task(task,args,expected_epoch=settings.get("epochs"),result_paths=paths)
    except Exception as exc:
        with task.lock:
            task.status="failed"
            task.error=str(exc)
        raise
    return handler._json_response(dict(task_id=task.task_id,model_family=FAMILY))

def start_train(handler,payload,*,runtime):
    _family(payload)
    if payload.get("auto_preprocess"): raise ValueError("Composer v2 请先准备训练 manifest")
    payload,selection=runtime._prepare_training_sources({**payload,"model_family":FAMILY})
    manifest=_path(payload.get("manifest"),"manifest",exists=True)
    directory=_path(payload.get("checkpoint_dir") or "checkpoints/chart_composer_v2","checkpoint_dir")
    cache=_path(payload.get("cache_dir") or "cache/chart_composer_v2","cache_dir")
    mode=_choice(payload,"mode",payload.get("training_mode","fresh"),{"fresh","resume","finetune"})
    parent=None
    if mode!="fresh":
        parent=_path(payload.get("resume") or payload.get("init_checkpoint") or payload.get("training_checkpoint") or payload.get("checkpoint"),"checkpoint",exists=True)
        checked=inspection(parent)
        if not checked.get("valid") or not checked.get("can_resume" if mode=="resume" else "can_init",False):
            raise ValueError("Composer v2 checkpoint 不支持当前训练方式："+str(checked.get("error") or checked.get("errors")))
        if mode=="finetune" and parent.parent==directory:
            raise ValueError("finetune 必须使用新的 checkpoint 目录")
        if mode=="resume":
            saved_args=checked.get("args") or {}
            payload={**{key:value for key,value in saved_args.items() if key in TRAIN_DEFAULTS},**payload}
        # Both modes retain the parent's model and audio configuration.
        # The trainer may append observed vocabulary entries during finetune.
        fixed=checked.get("model_config") or checked.get("config") or {}
        for key in ("audio_layers","song_layers","d_model","layers","nhead","dim_feedforward","dropout","window_beats","history_beats","future_beats","ticks_per_beat","max_tokens","audio_backend","mert_model"):
            if key in fixed and not (key=="audio_layers" and mode=="finetune" and payload.get(key) is not None):payload[key]=fixed[key]
        payload['refresh_events']=fixed.get('refresh_events',16)
    sequence_policy='continuous_events_v1' if mode=='fresh' else (checked.get('model_config') or {}).get('sequence_policy','window_v1')
    args=["-m","maichart.composer_v2.train","--manifest",str(manifest),"--checkpoint-dir",str(directory),"--cache-dir",str(cache)]
    settings={}
    quality=payload.get('quality_enabled',((checked.get('args') or {}).get('quality_enabled',True) if mode=='resume' else True))
    if not isinstance(quality,bool): raise ValueError('质量加权必须是布尔值')
    args.append('--quality-enabled' if quality else '--no-quality-enabled')
    settings['quality_enabled']=quality
    from maichart.composer_v2.intensity import CHOICES as CURVE_CHOICES,resolve_policy as resolve_curve_policy
    curve_requested=_choice(payload,'intensity_curve','auto',CURVE_CHOICES)
    curve_policy,_=resolve_curve_policy(curve_requested,(checked.get('model_config') or {}) if parent else None,resume=mode=='resume')
    _arg(args,'--intensity-curve',curve_requested)
    settings.update(intensity_curve=curve_requested,intensity_policy=curve_policy)
    from maichart.composer_v2.timing_recovery import CHOICES
    recovery_policy=_choice(payload,'timing_recovery','auto',CHOICES)
    _arg(args,'--timing-recovery',recovery_policy)
    settings['timing_recovery']=recovery_policy
    from maichart.composer_v2.onset_policy import CHOICES as ONSET_CHOICES,resolve_policy as resolve_onset_policy
    onset_requested=_choice(payload,'onset_readout','auto',ONSET_CHOICES)
    onset_policy,_=resolve_onset_policy(onset_requested,
        (checked.get('model_config') or {}) if parent else None,resume=mode=='resume')
    if onset_policy!='off' and sequence_policy!='continuous_events_v1':
        raise ValueError('独立声学读出仅适用于连续事件检查点')
    _arg(args,'--onset-readout',onset_requested)
    settings.update(onset_readout=onset_requested,onset_readout_policy=onset_policy)
    from maichart.composer_v2.rhythm_structure import CHOICES as STRUCTURE_CHOICES,resolve_policy as resolve_structure_policy
    structure_requested=_choice(payload,'rhythm_structure','auto',STRUCTURE_CHOICES)
    structure_parent=(checked.get('model_config') or {}) if parent else None
    if structure_requested not in ('auto','off') and sequence_policy!='continuous_events_v1':
        raise ValueError('节拍结构与对齐参考仅适用于连续事件检查点')
    structure_policy,_=resolve_structure_policy(structure_requested,structure_parent,resume=mode=='resume')
    _arg(args,'--rhythm-structure',structure_requested)
    settings.update(rhythm_structure=structure_requested,structure_policy=structure_policy)
    music_context=_choice(payload,'music_context','auto',{'auto','off','song_detail_v1'})
    _arg(args,'--music-context',music_context)
    settings['music_context']=music_context
    for capacity_key,capacity_default in (('audio_layers',4),('song_layers',2)):
        if capacity_key not in TRAIN_DEFAULTS:
            capacity_value=_integer(payload,capacity_key,capacity_default,minimum=1)
            _arg(args,'--'+capacity_key.replace('_','-'),capacity_value)
            settings[capacity_key]=capacity_value
    refresh_events=_integer(payload,'refresh_events',16,minimum=1)
    _arg(args,'--refresh-events',refresh_events);_arg(args,'--sequence-policy',sequence_policy)
    settings.update(refresh_events=refresh_events,sequence_policy=sequence_policy)
    choices=dict(lr_schedule={"cosine","constant"},audio_backend={"mert","native"},prepare_device={"auto","cuda","cpu"},device={"auto","cuda","cpu"},precision={"bf16","fp16","fp32","auto"})
    zeros={"warmup_steps","future_beats","history_beats","validation_max_batches","seed","max_steps","limit_charts","save_every","early_stopping_patience"}
    for key,default in TRAIN_DEFAULTS.items():
        if key in ('cuda_memory_budget_gb', 'audio_cache_gb') and isinstance(payload.get(key), bool):
            raise ValueError(f'{key} 必须是数字')
        if key in choices: value=_choice(payload,key,default,choices[key])
        elif isinstance(default,int): value=_integer(payload,key,default,minimum=0 if key in zeros else 1)
        else: value=_float(payload,key,default,minimum=0)
        if key in ('cuda_memory_budget_gb', 'audio_cache_gb') and value > 1024:
            raise ValueError(f'{key} 不能超过 1024 GiB')
        settings[key]=value
        _arg(args,"--"+key.replace("_","-"),value)
    if settings["d_model"]%settings["nhead"]: raise ValueError("d_model 必须能被 nhead 整除")
    if settings["lr"]<=0: raise ValueError("lr 必须大于0")
    if settings["dropout"]>=1 or settings["min_lr_ratio"]>1: raise ValueError("dropout/min_lr_ratio 超出范围")
    source=_choice(payload,"validation_source","manual" if payload.get("validation_manifest") else "auto",{"auto","manual","off"})
    fraction=_float(payload,"holdout_fraction",.05,minimum=0) if source=="auto" else 0.
    if fraction>=1:raise ValueError("holdout_fraction 必须小于1")
    if source=="auto" and fraction==0:source="off"
    validation=(_path(payload["validation_manifest"],"validation_manifest",exists=True)
        if mode!="resume" and source=="manual" and payload.get("validation_manifest") else None)
    if mode!="resume" and source=="manual" and validation is None:raise ValueError("手动验证需要提供 validation_manifest")
    mert_model=_path(payload["mert_model"],"本地 MERT 目录",exists=True,directory=True) if payload.get("mert_model") else None
    if mert_model:_arg(args,"--mert-model",mert_model)
    artifacts=[directory/name for name in ("composer_v2_last.pt","composer_v2_best.pt","composer_v2_inference.pt")]
    if mode=="fresh" and any(path.exists() for path in artifacts):raise ValueError("fresh 输出目录已有 Composer v2 权重，请选择新目录")
    split_report=None
    source_manifest=manifest
    if mode=="resume":
        saved=checked.get("args") or {}
        manifest=_path(saved.get("manifest"),"父 checkpoint 训练清单",exists=True)
        split_metadata=json.loads(manifest.read_text(encoding="utf-8-sig")).get("split_metadata",{})
        accepted={manifest}
        if split_metadata.get("source_manifest"):
            accepted.add(_path(split_metadata["source_manifest"],"原始训练来源"))
        if source_manifest not in accepted:raise ValueError("resume 必须使用父 checkpoint 的训练清单或其原始划分来源")
        expected=(split_metadata.get("source_sha256") if source_manifest!=manifest else
            ((checked.get("data_contract") or {}).get("train") or {}).get("manifest_sha256"))
        if expected and hashlib.sha256(source_manifest.read_bytes()).hexdigest()!=expected:
            raise ValueError("resume 训练清单内容已变化；请保留原 checkpoint 并使用新目录 finetune")
        validation=_path(saved["validation_manifest"],"父 checkpoint 验证清单",exists=True) if saved.get("validation_manifest") else None
        source="inherited"
    elif source=="auto":
        split_directory=directory/"data_split"
        if split_directory.exists():raise ValueError("输出目录已有 data_split，请使用新的输出目录以保留原划分")
        from maichart.composer.evaluation import split_manifest
        split_report=split_manifest(manifest,split_directory,fraction=fraction,seed=settings["seed"],cache_dir=cache)
        manifest=Path(split_report["splits"]["train"]["path"])
        validation=Path(split_report["splits"]["dev"]["path"])
    if validation is not None:
        from maichart.composer.evaluation import assert_unseen
        assert_unseen(manifest,validation,cache_dir=cache)
        _arg(args,"--validation-manifest",validation)
    args[args.index("--manifest")+1]=str(manifest)
    settings.update(validation_source=source,holdout_fraction=fraction,
        validation_manifest=str(validation) if validation else None,source_manifest=str(source_manifest),
        split_report=split_report,mert_model=str(mert_model) if mert_model else None)
    if parent: _arg(args,"--resume" if mode=="resume" else "--init-checkpoint",parent)
    # The child reads the full manifest from disk. Polling only needs the selection summary.
    selection_summary = {key: value for key, value in selection.items() if key != 'songs'} if selection else None
    settings.update(mode=mode,manifest=str(manifest),cache_dir=str(cache),dataset_selection=selection_summary)
    condition_policy = (checked.get("condition_contract", {}).get("policy", "level_only_v1")
                        if mode == "resume" else payload.get("condition_policy", "radar_style_v1"))
    if condition_policy not in {"level_only_v1", "radar_style_v1"}:
        raise ValueError("未知 Composer v2 condition_policy")
    _arg(args, "--condition-policy", condition_policy)
    settings["condition_policy"] = condition_policy
    return _start(handler,runtime,"composer_v2_train",args,settings,[directory/name for name in ("composer_v2_last.pt","composer_v2_best.pt","composer_v2_inference.pt")],directory)

def start_generate(handler,payload,*,runtime):
    _family(payload)
    checkpoint=_path(payload.get("checkpoint"),"checkpoint",exists=True)
    checked=inspection(checkpoint)
    if not checked.get("valid"): raise ValueError("Composer v2 checkpoint 不可用："+str(checked.get("error") or checked.get("errors")))
    condition = payload.get("condition")
    if condition is None: condition = {}
    if not isinstance(condition, dict): raise ValueError("condition 必须是 JSON 对象")
    from maichart.composer_v2.conditions import SUPPORTED_KEYS
    for key, value in condition.items():
        if key == "level" or key not in SUPPORTED_KEYS:
            raise ValueError("不支持的 Composer v2 条件：" + str(key))
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"条件 {key} 必须是 0 到 1 的有限数值")
    policy = checked.get("condition_contract", {}).get("policy", "level_only_v1")
    if condition and policy != "radar_style_v1":
        raise ValueError("此 checkpoint 仅训练了等级条件，不支持 Radar / Style override；请使用新目录 finetune 后的 radar_style_v1 checkpoint")
    audio=_path(payload.get("audio"),"audio",exists=True)
    output=_path(payload.get("output") or "outputs/composer_v2/maidata.txt","output")
    cache=_path(payload.get("cache_dir") or "cache/chart_composer_v2","cache_dir")
    settings=dict(checkpoint=str(checkpoint),audio=str(audio),output=str(output),cache_dir=str(cache))
    args=["-m","maichart.composer_v2.generate"]
    if condition: _arg(args, "--condition", json.dumps(condition, ensure_ascii=False))
    settings.update(condition=condition, condition_policy=policy)
    phrase_search=_choice(payload,'phrase_search','layout',{'event','layout','joint'})
    phrase_beats=_float(payload,'phrase_beats',4.,minimum=1.)
    phrase_commit_beats=_float(payload,'phrase_commit_beats',2.,minimum=0.)
    if isinstance(payload.get('phrase_beats'),bool) or isinstance(payload.get('phrase_commit_beats'),bool):
        raise ValueError('提前试写长度和每次保留长度必须是数字')
    if phrase_beats>16 or not 0<phrase_commit_beats<=phrase_beats:
        raise ValueError('提前试写长度须为1–16拍；每次保留长度须大于0，且不能超过提前试写长度')
    if checked.get('sequence_policy')=='window_v1' and phrase_search!='event':
        raise ValueError('这个旧模型不支持比较整段写法，请选择“逐步生成（速度优先）”')
    settings.update(phrase_search=phrase_search,phrase_beats=phrase_beats,phrase_commit_beats=phrase_commit_beats)
    for key in ('phrase_search','phrase_beats','phrase_commit_beats'):_arg(args,'--'+key.replace('_','-'),settings[key])
    settings['exhaustion_policy']=_choice(payload,'exhaustion_policy','auto',{'auto','report','stop'})
    settings['tempo_detection']=_choice(payload,'tempo_detection','auto',{'auto','constant','legacy'})
    _arg(args,'--tempo-detection',settings['tempo_detection'])
    intercept=payload.get('intercept_tap_on_slide',GENERATION_DEFAULTS['intercept_tap_on_slide'])
    if not isinstance(intercept,bool):raise ValueError('拦截Slide撞尾必须是布尔值')
    threshold=_float(payload,'tap_on_slide_threshold_ms',200.,minimum=0)
    if isinstance(payload.get('tap_on_slide_threshold_ms'),bool) or threshold>250:
        raise ValueError('撞尾检测阈值必须为0–250 ms')
    settings.update(intercept_tap_on_slide=intercept,tap_on_slide_threshold_ms=threshold)
    args.append('--intercept-tap-on-slide' if intercept else '--no-intercept-tap-on-slide')
    _arg(args,'--tap-on-slide-threshold-ms',threshold)
    _arg(args,'--exhaustion-policy',settings['exhaustion_policy'])
    for key in ("checkpoint", "audio", "output", "cache_dir"): _arg(args,"--"+key.replace("_","-"),settings[key])
    for key,default in dict(level=13.,temperature=.9,top_p=.95).items():
        settings[key]=_float(payload,key,default)
        _arg(args,"--"+key.replace("_","-"),settings[key])
    if settings["temperature"]<0 or not 0<settings["top_p"]<=1: raise ValueError("temperature/top_p 超出范围")
    from maichart.composer_v2.onset_policy import effective_strength
    raw_onset=payload.get('onset_strength')
    if isinstance(raw_onset,bool):raise ValueError('起音参考强度必须是数字')
    onset_strength=effective_strength(None if raw_onset in (None,'') else raw_onset,
        (checked.get('model_config') or {}).get('onset_readout_policy','off'))
    if raw_onset not in (None,''):
        settings['onset_strength']=onset_strength
        _arg(args,'--onset-strength',onset_strength)
    reference_selection=_choice(payload,'reference_selection','auto',{'auto','aligned'})
    if reference_selection=='aligned':
        if checked.get('sequence_policy')!='continuous_events_v1':
            raise ValueError('对齐参考检索仅适用于 Composer v2 连续事件检查点')
        _arg(args,'--reference-selection',reference_selection)
    settings['reference_selection']=reference_selection
    for key in ("action_temperature", "action_top_p"):
        raw = payload.get(key, GENERATION_DEFAULTS[key] if checked.get('sequence_policy') == 'continuous_events_v1' else None)
        if raw is None or raw == "":
            continue
        if checked.get("sequence_policy") != "continuous_events_v1":
            raise ValueError("独立配置采样适用于 Composer v2 连续事件检查点")
        if isinstance(raw, bool):
            raise ValueError(f"{key} 必须是数值")
        value = _float({key:raw}, key, 0.)
        if key == "action_temperature" and value < 0:
            raise ValueError("配置温度必须不小于 0")
        if key == "action_top_p" and not 0 < value <= 1:
            raise ValueError("配置 Top-p 必须大于 0 且不大于 1")
        settings[key] = value
        _arg(args, "--" + key.replace("_", "-"), value)
    if not 1<=settings["level"]<=15:raise ValueError("level 必须在1到15之间")
    for key,default in dict(seed=0,candidates=GENERATION_DEFAULTS["candidates"],max_events=0,rollback_windows=2,window_retries=16,cpu_threads=GENERATION_DEFAULTS['cpu_threads']).items():
        settings[key]=_integer(payload,key,default,minimum=0)
        _arg(args,"--"+key.replace("_","-"),settings[key])
    if not 1<=settings["window_retries"]<=128:raise ValueError("每事件提议重试次数必须在1到128之间")
    if settings["rollback_windows"]>8:raise ValueError("rollback_windows 必须在0到8之间")
    if settings["candidates"] not in {1,2,4}: raise ValueError("候选数只能是1、2、4")
    settings["device"]=_choice(payload,"device","auto",{"auto","cuda","cpu"})
    _arg(args,"--device",settings["device"])
    settings.update(bpm=None,first=None)
    if payload.get("first") not in {None, ""}:
        settings["first"]=_float(payload,"first",0.)
        _arg(args,"--first",settings["first"])
    if payload.get("bpm") not in {None, ""}:
        bpm=_float(payload,"bpm",120,minimum=.001)
        _arg(args,"--bpm",bpm)
        settings["bpm"]=bpm
    for key in ("tempo_map","meter_map"):
        if payload.get(key) is not None:
            value=payload[key]
            if isinstance(value,str): value=json.loads(value)
            if not isinstance(value,list): raise ValueError(f"{key} 必须是 JSON 数组")
            _arg(args,"--"+key.replace("_","-"),json.dumps(value,ensure_ascii=False))
            settings[key]=value
    count=_integer(payload,'count',GENERATION_DEFAULTS['count'],minimum=1)
    if count>MAX_BATCH_COUNT:raise ValueError(f'一次最多生成 {MAX_BATCH_COUNT} 张谱面')
    settings['count']=count
    parallel=_integer(payload,'parallel',GENERATION_DEFAULTS['parallel'],minimum=1)
    if parallel>MAX_BATCH_COUNT:raise ValueError(f'同时生成数不能超过 {MAX_BATCH_COUNT}')
    settings.update(parallel=parallel,effective_parallel=min(count,parallel))
    _arg(args,'--parallel',parallel)
    paths=[output,output.with_suffix('.composer_v2.json')]
    profile=payload.get('intensity_profile')
    if profile is not None:
        from maichart.composer_v2.intensity import validate_profile
        validate_profile(profile)
        if (checked.get('model_config') or {}).get('intensity_policy','off')=='off':
            raise ValueError('此检查点未训练强度曲线，不能使用编辑曲线')
        strength=_float(payload,'intensity_strength',1.,minimum=0)
        if strength>1:raise ValueError('编辑曲线强度必须在0–1之间')
        import uuid
        profile_path=runtime.INTENSITY_ROOT/(uuid.uuid4().hex+'.composer_v2_curve.json')
        profile_path.parent.mkdir(parents=True,exist_ok=True)
        profile_path.write_text(json.dumps(profile,ensure_ascii=False),encoding='utf-8')
        _arg(args,'--intensity-curve',profile_path);_arg(args,'--intensity-strength',strength)
        settings.update(intensity_curve=str(profile_path),intensity_strength=strength)
    if count>1:
        # Separate task directories also protect previews from later overwrites.
        import uuid
        root=output.parent/(output.stem+'_batch_'+uuid.uuid4().hex[:12])
        output=root/output.name
        manifest=root/'batch.composer_v2_batch.json'
        args[args.index('--output')+1]=str(output)
        _arg(args,'--count',count);_arg(args,'--batch-manifest',manifest)
        settings.update(output=str(output),batch_manifest=str(manifest))
        paths=[manifest]  # Successful chart artifacts are collected from it.
    return _start(handler,runtime,"composer_v2_generate",args,settings,paths)


def start_curve(handler,payload,*,runtime):
    """Run prediction in a cancellable subprocess rather than the HTTP thread."""
    _family(payload)
    checkpoint=_path(payload.get('checkpoint'),'checkpoint',exists=True)
    checked=inspection(checkpoint)
    if not checked.get('valid') or (checked.get('model_config') or {}).get('intensity_policy','off')!='rate_curve_v1':
        raise ValueError('此检查点尚未训练新版强度曲线；请从头训练或在新目录微调开启')
    audio=_path(payload.get('audio'),'audio',exists=True)
    import uuid
    output=runtime.INTENSITY_ROOT/(uuid.uuid4().hex+'.composer_v2_curve.json')
    args=['-m','maichart.composer_v2.intensity','--checkpoint',str(checkpoint),'--audio',str(audio),'--output',str(output)]
    settings={'checkpoint':str(checkpoint),'audio':str(audio),'output':str(output)}
    settings.update(cache_dir=str(_path(payload.get('cache_dir') or 'cache/chart_composer_v2','cache_dir')),
        device=_choice(payload,'device','auto',{'auto','cuda','cpu'}),
        tempo_detection=_choice(payload,'tempo_detection','auto',{'auto','constant','legacy'}),
        cpu_threads=_integer(payload,'cpu_threads',GENERATION_DEFAULTS['cpu_threads'],minimum=0),level=_float(payload,'level',13.,minimum=1))
    if settings['level']>15:raise ValueError('level 必须在1到15之间')
    for key in ('cache_dir','device','tempo_detection','cpu_threads','level'):
        _arg(args,'--'+key.replace('_','-'),settings[key])
    for key in ('bpm','first','tempo_map','condition'):
        value=payload.get(key)
        if value is not None and value!='':_arg(args,'--'+key.replace('_','-'),json.dumps(value,ensure_ascii=False) if isinstance(value,(dict,list)) else value)
    return _start(handler,runtime,'composer_v2_curve',args,settings,[output])


def start_media(handler,payload,*,runtime):
    _family(payload)
    audio=_path(payload.get('audio'),'audio',exists=True)
    cache=_path(payload.get('cache_dir') or 'cache/chart_composer_v2','cache_dir')
    import uuid
    output=runtime.INTENSITY_ROOT/(uuid.uuid4().hex+'.editor_media.json')
    args=['-m','maichart.webui.audio_overview','--audio',str(audio),'--cache-dir',str(cache),'--output',str(output)]
    return _start(handler,runtime,'composer_v2_media',args,dict(audio=str(audio),cache_dir=str(cache)),[output])
