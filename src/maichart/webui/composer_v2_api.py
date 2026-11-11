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
from maichart.composer_v2.defaults import TRAIN_DEFAULTS, GENERATION_DEFAULTS, EVALUATION_DEFAULTS, MAX_BATCH_COUNT

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
    result = dict(task_id=task.task_id,model_family=FAMILY)
    if kind == 'composer_v2_auto_quality_audit':
        result.update(manifest=settings['manifest'], auto_scope='preview' if settings['limit_charts'] else 'formal')
    return handler._json_response(result)

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
    try:
        manifest_data=json.loads(manifest.read_text(encoding='utf-8-sig'))
    except (OSError,ValueError,TypeError):
        manifest_data={}
    has_zero_unmarked=bool(
        manifest_data.get('candidate_pool')
        or manifest_data.get('reviewed_training')
        or any(isinstance(s,dict) and (s.get('reviewed_only') or s.get('candidate_pool')) for s in manifest_data.get('songs',()))
    )
    if has_zero_unmarked and not quality:
        raise ValueError('人工确认片段必须启用质量加权；未确认部分仅作上下文，不能作为整谱训练')
    if has_zero_unmarked and sequence_policy!='continuous_events_v1':
        raise ValueError('人工确认片段当前仅支持 Composer v2 连续事件训练')
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
    reviewed_ref=_choice(payload,'reviewed_reference_policy','auto',{'auto','approved_only','off'})
    if reviewed_ref=='approved_only' and sequence_policy!='continuous_events_v1':
        raise ValueError('示范库已认证段落呼应参考仅适用于连续事件检查点')
    _arg(args,'--reviewed-reference-policy',reviewed_ref)
    settings['reviewed_reference_policy']=reviewed_ref
    auto_quality=_choice(payload,'auto_quality_policy','auto',{'auto','on','off'})
    from maichart.composer_v2.train import resolve_auto_quality_policy
    if mode!='resume' and auto_quality=='on':
        resolve_auto_quality_policy(auto_quality,manifest,quality_enabled=quality,
                                    sequence_policy=sequence_policy)
    _arg(args,'--auto-quality-policy',auto_quality)
    settings['auto_quality_policy']=auto_quality
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
    from maichart.composer_v2.music_policy import CHOICES as MUSIC_CHOICES,resolve_policy as resolve_music_policy
    music_context=_choice(payload,'music_context','auto',MUSIC_CHOICES)
    music_policy,_=resolve_music_policy(music_context,(checked.get('model_config') or {}) if parent else None,resume=mode=='resume')
    _arg(args,'--music-context',music_context)
    settings['music_context']=music_context
    settings['music_context_policy']=music_policy
    for capacity_key,capacity_default in (('audio_layers',4),('song_layers',2)):
        if capacity_key not in TRAIN_DEFAULTS:
            capacity_value=_integer(payload,capacity_key,capacity_default,minimum=1)
            _arg(args,'--'+capacity_key.replace('_','-'),capacity_value)
            settings[capacity_key]=capacity_value
    refresh_events=_integer(payload,'refresh_events',16,minimum=1)
    _arg(args,'--refresh-events',refresh_events);_arg(args,'--sequence-policy',sequence_policy)
    settings.update(refresh_events=refresh_events,sequence_policy=sequence_policy)
    from maichart.composer_v2.acceleration import AUDIO_DEVICES
    choices=dict(lr_schedule={"cosine","constant"},audio_backend={"mert","native"},prepare_device=set(AUDIO_DEVICES),device={"auto","cuda","cpu","xpu"},precision={"bf16","fp16","fp32","auto"},activation_checkpointing={'auto','off','on'})
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
        resolve_auto_quality_policy(auto_quality,manifest,checked.get('training_contract'),
                                    resume=True,quality_enabled=quality,sequence_policy=sequence_policy)
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
    record_probabilities=payload.get('record_candidate_probabilities',GENERATION_DEFAULTS['record_candidate_probabilities'])
    if not isinstance(record_probabilities,bool):
        raise ValueError('record_candidate_probabilities 必须是布尔值')
    if record_probabilities and checked.get('sequence_policy')=='window_v1':
        raise ValueError('记录候选评估数据仅支持连续事件模型；这个旧窗口模型不支持')
    settings['record_candidate_probabilities']=record_probabilities
    if record_probabilities:args.append('--record-candidate-probabilities')
    phrase_search=_choice(payload,'phrase_search','layout',{'event','layout','joint','hybrid'})
    phrase_rank_policy=_choice(payload,'phrase_rank_policy',GENERATION_DEFAULTS['phrase_rank_policy'],{'comfort_v1','comfort_v2','likelihood_veto_v1'})
    phrase_beats=_float(payload,'phrase_beats',4.,minimum=1.)
    phrase_commit_beats=_float(payload,'phrase_commit_beats',2.,minimum=0.)
    if isinstance(payload.get('phrase_beats'),bool) or isinstance(payload.get('phrase_commit_beats'),bool):
        raise ValueError('提前试写长度和每次保留长度必须是数字')
    if phrase_beats>16 or not 0<phrase_commit_beats<=phrase_beats:
        raise ValueError('提前试写长度须为1–16拍；每次保留长度须大于0，且不能超过提前试写长度')
    if checked.get('sequence_policy')=='window_v1' and phrase_search!='event':
        raise ValueError('这个旧模型不支持比较整段写法，请选择“逐步生成（速度优先）”')
    if checked.get('sequence_policy')=='window_v1' and phrase_rank_policy!='comfort_v1':
        raise ValueError('模型似然排序仅支持连续事件模型')
    settings.update(phrase_search=phrase_search,phrase_rank_policy=phrase_rank_policy,phrase_beats=phrase_beats,phrase_commit_beats=phrase_commit_beats)
    for key in ('phrase_search','phrase_rank_policy','phrase_beats','phrase_commit_beats'):_arg(args,'--'+key.replace('_','-'),settings[key])
    settings['exhaustion_policy']=_choice(payload,'exhaustion_policy','auto',{'auto','report','stop','downgrade_or_skip'})
    if checked.get('sequence_policy')=='window_v1' and settings['exhaustion_policy'] in {'report','downgrade_or_skip'}:
        raise ValueError('旧窗口模型仅支持 auto 或 stop 耗尽策略')
    settings['tempo_detection']=_choice(payload,'tempo_detection','auto',{'auto','constant','legacy'})
    settings['tempo_octave_policy']=_choice(payload,'tempo_octave_policy',GENERATION_DEFAULTS['tempo_octave_policy'],{'off','maimai_pulse_v1'})
    settings['downbeat_alignment']=_choice(payload,'downbeat_alignment',GENERATION_DEFAULTS['downbeat_alignment'],{'off','bar_downbeat_v1'})
    for key in ('tempo_detection','tempo_octave_policy','downbeat_alignment'):
        _arg(args,'--'+key.replace('_','-'),settings[key])
    intercept=payload.get('intercept_tap_on_slide',GENERATION_DEFAULTS['intercept_tap_on_slide'])
    if not isinstance(intercept,bool):raise ValueError('拦截Slide撞尾必须是布尔值')
    threshold=_float(payload,'tap_on_slide_threshold_ms',200.,minimum=0)
    if isinstance(payload.get('tap_on_slide_threshold_ms'),bool) or threshold>250:
        raise ValueError('撞尾检测阈值必须为0–250 ms')
    intercept_sht=payload.get('intercept_slide_head_tap',GENERATION_DEFAULTS['intercept_slide_head_tap'])
    if not isinstance(intercept_sht,bool):raise ValueError('拦截内屏星头提前触发必须是布尔值')
    from maichart.generation.muri.touch_reachability import TOUCH_REACHABILITY_POLICIES
    touch_policy=_choice(payload,'touch_reachability_policy',GENERATION_DEFAULTS['touch_reachability_policy'],set(TOUCH_REACHABILITY_POLICIES))
    touch_radius=_float(payload,'touch_cohold_radius',GENERATION_DEFAULTS['touch_cohold_radius'],minimum=60.)
    if isinstance(payload.get('touch_cohold_radius'),bool) or touch_radius>260:
        raise ValueError('Touch同手兼顾半径必须为60–260')
    from maichart.composer_v2.forward_feasibility import FORWARD_POLICIES
    forward_policy=_choice(payload,'forward_feasibility',GENERATION_DEFAULTS['forward_feasibility'],set(FORWARD_POLICIES))
    forward_tos_ms=_float(payload,'forward_tap_on_slide_ms',GENERATION_DEFAULTS['forward_tap_on_slide_ms'],minimum=0.)
    if isinstance(payload.get('forward_tap_on_slide_ms'),bool) or forward_tos_ms>250:
        raise ValueError('前向禁止撞尾阈值必须为0–250 ms')
    from maichart.composer_v2.self_refine import SELF_REFINE_POLICIES
    self_refine=_choice(payload,'self_refine',GENERATION_DEFAULTS['self_refine'],set(SELF_REFINE_POLICIES))
    if checked.get('sequence_policy')=='window_v1' and self_refine!='off':
        raise ValueError('自我精修仅支持连续事件模型')
    settings.update(intercept_tap_on_slide=intercept,tap_on_slide_threshold_ms=threshold,
        intercept_slide_head_tap=intercept_sht,touch_reachability_policy=touch_policy,
        touch_cohold_radius=touch_radius,forward_feasibility=forward_policy,
        forward_tap_on_slide_ms=forward_tos_ms,self_refine=self_refine)
    args.append('--intercept-tap-on-slide' if intercept else '--no-intercept-tap-on-slide')
    args.append('--intercept-slide-head-tap' if intercept_sht else '--no-intercept-slide-head-tap')
    _arg(args,'--tap-on-slide-threshold-ms',threshold)
    _arg(args,'--touch-reachability-policy',touch_policy)
    _arg(args,'--touch-cohold-radius',touch_radius)
    _arg(args,'--forward-feasibility',forward_policy)
    _arg(args,'--forward-tap-on-slide-ms',forward_tos_ms)
    _arg(args,'--self-refine',self_refine)
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
    from maichart.composer_v2.acceleration import AUDIO_DEVICES
    for key, choices in (('audio_device',set(AUDIO_DEVICES)),
                         ('inference_backend',{'auto','torch','openvino-cpu','openvino-gpu'})):
        settings[key]=_choice(payload,key,'auto',choices)
        _arg(args,'--'+key.replace('_','-'),settings[key])
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
    if record_probabilities:paths.append(output.with_suffix('.candidate_audit.jsonl'))
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
        tempo_octave_policy=_choice(payload,'tempo_octave_policy',GENERATION_DEFAULTS['tempo_octave_policy'],{'off','maimai_pulse_v1'}),
        downbeat_alignment=_choice(payload,'downbeat_alignment',GENERATION_DEFAULTS['downbeat_alignment'],{'off','bar_downbeat_v1'}),
        cpu_threads=_integer(payload,'cpu_threads',GENERATION_DEFAULTS['cpu_threads'],minimum=0),level=_float(payload,'level',13.,minimum=1))
    if settings['level']>15:raise ValueError('level 必须在1到15之间')
    from maichart.composer_v2.acceleration import AUDIO_DEVICES
    settings['audio_device']=_choice(payload,'audio_device','auto',set(AUDIO_DEVICES))
    for key in ('cache_dir','device','tempo_detection','tempo_octave_policy','downbeat_alignment','cpu_threads','level','audio_device'):
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


def _evaluation_integer(payload, key, default, minimum=0):
    value = payload.get(key, default)
    if type(value) is not int or value < minimum:
        raise ValueError(f'{key} 必须是大于等于 {minimum} 的整数')
    return value


def start_evaluate(handler, payload, *, runtime):
    """Launch bounded CPU-only diagnostics without touching any running task."""
    _family(payload)
    checkpoint = _path(payload.get('checkpoint'), 'checkpoint', exists=True)
    if not checkpoint.is_file(): raise ValueError('checkpoint 必须是文件')
    checked = inspection(checkpoint)
    if not checked.get('valid') or (checked.get('model_config') or {}).get('sequence_policy') != 'continuous_events_v1':
        raise ValueError('离线评估只支持有效的 Composer v2 continuous_events_v1 检查点')
    manifest_value = payload.get('manifest')
    manifest = (_path(manifest_value, 'manifest', exists=True) if manifest_value
                else checkpoint.parent / 'data_split' / 'dev_manifest.json')
    if not manifest.is_file():
        raise ValueError(f'未找到验证清单，请指定 manifest；默认仅查找 {manifest}，不会使用训练清单代替')
    overlap = payload.get('check_training_overlap', EVALUATION_DEFAULTS['check_training_overlap'])
    if type(overlap) is not bool: raise ValueError('check_training_overlap 必须是布尔值')
    settings = dict(checkpoint=str(checkpoint), manifest=str(manifest.resolve()),
        cache_dir=str(_path(payload.get('cache_dir') or 'cache/chart_composer_v2', 'cache_dir')),
        device='cpu', check_training_overlap=overlap,
        window_sampling=_choice(payload, 'window_sampling', EVALUATION_DEFAULTS['window_sampling'], {'spread', 'prefix'}))
    for key, minimum in (('limit_charts', 0), ('max_batches', 0), ('timing_rollouts', 0), ('cpu_threads', 1)):
        settings[key] = _evaluation_integer(payload, key, EVALUATION_DEFAULTS[key], minimum)
    training = checkpoint.parent / 'data_split' / 'train_manifest.json'
    if overlap and not training.is_file():
        raise ValueError(f'无法审计训练重叠：缺少同检查点目录清单 {training}')
    import uuid
    output = runtime.INTENSITY_ROOT / uuid.uuid4().hex / 'quality_eval.json'
    settings['output'] = str(output)
    args = ['-m', 'maichart.composer_v2.evaluate', 'teacher-forced']
    for key in ('checkpoint', 'manifest', 'cache_dir', 'limit_charts', 'max_batches',
                'timing_rollouts', 'cpu_threads', 'window_sampling', 'output'):
        _arg(args, '--' + key.replace('_', '-'), settings[key])
    if overlap:
        _arg(args, '--training-manifest', training)
        settings['training_manifest'] = str(training)
    return _start(handler, runtime, 'composer_v2_evaluate', args, settings, [output])


def start_candidate_audit(handler, payload, *, runtime):
    """Audit existing candidate logs; never generates candidates or uses CUDA."""
    _family(payload)
    source = _path(payload.get('input'), 'input', exists=True)
    root = Path(getattr(runtime, 'PROJECT_ROOT', Path(__file__).resolve().parents[3])).resolve()
    if not source.is_file() or source.suffix.lower() not in {'.json', '.jsonl'}:
        raise ValueError('input 必须是已存在的 JSON 或 JSONL 文件')
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise ValueError('input 必须位于当前项目允许路径内') from exc
    import uuid
    directory = runtime.INTENSITY_ROOT / uuid.uuid4().hex
    output = directory / 'candidate_audit.json'; markdown = directory / 'candidate_audit.md'
    args = ['-m', 'maichart.composer_v2.candidate_audit', '--input', str(source),
            '--output', str(output), '--markdown', str(markdown)]
    settings = dict(input=str(source), output=str(output), markdown=str(markdown), device='cpu')
    return _start(handler, runtime, 'composer_v2_candidate_audit', args, settings, [output, markdown])


def auto_quality_status(handler, payload, *, runtime):
    """Read existing results without re-composing the training selection."""
    from .quality_api import auto_quality_view

    _family(payload)
    scope = _choice(payload, 'auto_scope', 'formal', {'formal', 'preview'})
    policy = _choice(payload, 'auto_quality_policy', 'auto', {'auto', 'on', 'off'})
    quality_enabled = payload.get('quality_enabled', True)
    if not isinstance(quality_enabled, bool):
        raise ValueError('quality_enabled 必须是布尔值')
    resume = payload.get('mode') == 'resume'
    sequence = payload.get('sequence_policy', 'continuous_events_v1')
    empty = dict(available=False, manifest=None, auto_scope=scope, has_annotations=False,
                 has_preview_annotations=False, annotations_path=None, record_count=0, report=None,
                 preview_report=None, training_eligible=False, training_effective=None if resume else False)

    def status(value):
        manifest = _path(value, 'manifest')
        if not manifest.is_file():
            return {**empty, 'manifest': str(manifest)}
        data = json.loads(manifest.read_text(encoding='utf-8-sig'))
        meta, _, _ = auto_quality_view(manifest, data, scope)
        eligible = meta['has_annotations'] and not (data.get('candidate_pool') or data.get('reviewed_training'))
        effective = bool(eligible and quality_enabled and policy != 'off' and sequence == 'continuous_events_v1')
        return dict(meta, training_eligible=eligible, training_effective=None if resume else effective)

    if payload.get('manifest'):
        return handler._json_response(status(payload['manifest']))
    manifests = payload.get('manifests') or []
    if not isinstance(manifests, list) or any(not isinstance(value, str) or not value.strip() for value in manifests):
        raise ValueError('训练数据集列表必须由有效路径组成')
    manifests = list(dict.fromkeys(manifests))
    if len(manifests) == 1:
        return handler._json_response(status(manifests[0]))
    return handler._json_response(dict(empty, sources=[status(value) for value in manifests]))


def start_auto_quality_audit(handler, payload, *, runtime):
    """Inspect the selected corpus in a background CPU process."""
    _family(payload)
    payload, _ = runtime._prepare_training_sources({**payload, 'model_family': FAMILY})
    manifest = _path(payload.get('manifest'), 'manifest', exists=True)
    limit = _integer(payload, 'limit_charts', 0, minimum=0)
    tail_mode = _choice(payload, 'tap_on_slide_mode', 'mask_hard', {'mask_hard', 'mask_all', 'allow'})
    inner_mode = _choice(payload, 'inner_muri_mode', 'allow', {'allow', 'mask_hard'})
    threshold = _float(payload, 'tap_on_slide_ms', 200., minimum=0)
    if isinstance(payload.get('tap_on_slide_ms'), bool) or threshold > 250:
        raise ValueError('撞尾检查范围须为 0–250 ms')
    from maichart.generation.muri.touch_reachability import TOUCH_REACHABILITY_POLICIES
    touch_policy = _choice(payload, 'touch_reachability_policy', 'palm_span_v1', set(TOUCH_REACHABILITY_POLICIES))
    touch_radius = _float(payload, 'touch_cohold_radius', 180., minimum=60.)
    if isinstance(payload.get('touch_cohold_radius'), bool) or touch_radius > 260:
        raise ValueError('Touch同手兼顾半径须为 60–260')
    domain_pathology_mode = _choice(payload, 'domain_pathology_mode', 'maimai_dx_v1', {'maimai_dx_v1', 'off'})
    acoustic_audit_mode = _choice(payload, 'acoustic_audit_mode', 'auto', {'auto', 'on', 'off'})
    highlight_good_mode = _choice(payload, 'highlight_good_mode', 'auto', {'auto', 'acoustic_and_flow_v1', 'chart_only_v1', 'off'})
    good_quantile = _float(payload, 'good_quantile', 0.82, minimum=0.50)
    if isinstance(payload.get('good_quantile'), bool) or good_quantile > 0.98:
        raise ValueError('优选好段分位线须为 0.50–0.98')
    max_problem_fraction = _float(payload, 'max_problem_fraction', 0.45, minimum=0.05)
    if isinstance(payload.get('max_problem_fraction'), bool) or max_problem_fraction > 1.0:
        raise ValueError('整谱排除上限比例须为 0.05–1.0')
    experimental_hand_flow_quantile = _float(payload, 'experimental_hand_flow_quantile', 0.0, minimum=0.0)
    if isinstance(payload.get('experimental_hand_flow_quantile'), bool) or experimental_hand_flow_quantile > 0.25:
        raise ValueError('手顺降权分位比例须为 0–0.25')
    workers = _integer(payload, 'workers', 0, minimum=0)
    if workers > 32:
        raise ValueError('并行体检进程数须为 0–32')
    args = ['-m', 'maichart.composer_v2.auto_quality_audit', '--manifest', str(manifest),
            '--limit-charts', str(limit), '--tap-on-slide-mode', tail_mode,
            '--inner-muri-mode', inner_mode, '--tap-on-slide-ms', str(threshold)]
    if touch_policy != 'palm_span_v1':
        _arg(args, '--touch-reachability-policy', touch_policy)
    if touch_radius != 180.:
        _arg(args, '--touch-cohold-radius', touch_radius)
    if domain_pathology_mode != 'maimai_dx_v1':
        _arg(args, '--domain-pathology-mode', domain_pathology_mode)
    if acoustic_audit_mode != 'auto':
        _arg(args, '--acoustic-audit-mode', acoustic_audit_mode)
    if highlight_good_mode != 'auto':
        _arg(args, '--highlight-good-mode', highlight_good_mode)
    if good_quantile != 0.82:
        _arg(args, '--good-quantile', good_quantile)
    if max_problem_fraction != 0.45:
        _arg(args, '--max-problem-fraction', max_problem_fraction)
    if experimental_hand_flow_quantile != 0.0:
        _arg(args, '--experimental-hand-flow-quantile', experimental_hand_flow_quantile)
    if workers != 0:
        _arg(args, '--workers', workers)
    report = manifest.with_name('auto_quality_audit_preview_report.json' if limit else 'auto_quality_audit_report.json')
    annotations = manifest.with_name('auto_quality_annotations_preview.json' if limit else 'auto_quality_annotations.json')
    settings = dict(manifest=str(manifest), limit_charts=limit, tap_on_slide_mode=tail_mode,
                    inner_muri_mode=inner_mode, tap_on_slide_ms=threshold,
                    touch_reachability_policy=touch_policy,
                    touch_cohold_radius=touch_radius,
                    domain_pathology_mode=domain_pathology_mode,
                    acoustic_audit_mode=acoustic_audit_mode,
                    highlight_good_mode=highlight_good_mode,
                    good_quantile=good_quantile,
                    max_problem_fraction=max_problem_fraction,
                    experimental_hand_flow_quantile=experimental_hand_flow_quantile,
                    workers=workers,
                    device='cpu')
    return _start(handler, runtime, 'composer_v2_auto_quality_audit', args, settings, [report, annotations])
