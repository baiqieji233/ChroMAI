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
TRAIN_DEFAULTS = dict(epochs=50,batch_size=8,grad_accumulation=8,lr=.0002,
    lr_schedule="cosine",warmup_steps=200,min_lr_ratio=.1,d_model=384,layers=8,
    nhead=8,dim_feedforward=1536,dropout=.1,window_beats=16,history_beats=16,
    future_beats=4,ticks_per_beat=48,max_tokens=2048,audio_backend="mert",
    prepare_device="auto",device="auto",precision="bf16",validation_max_batches=32,
    checkpoint_interval_steps=500,log_interval=20,seed=0,max_steps=0,limit_charts=0,cpu_threads=2,
    cuda_memory_budget_gb=0.0,audio_cache_gb=0.25)

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
        # Both weight initialization and exact resume retain the parent's model,
        # codec and audio contract. Only the new-stage optimizer settings differ.
        fixed=checked.get("model_config") or checked.get("config") or {}
        for key in ("d_model","layers","nhead","dim_feedforward","dropout","window_beats","history_beats","future_beats","ticks_per_beat","max_tokens","audio_backend","mert_model"):
            if key in fixed:payload[key]=fixed[key]
        payload['refresh_events']=fixed.get('refresh_events',16)
    sequence_policy='continuous_events_v1' if mode=='fresh' else (checked.get('model_config') or {}).get('sequence_policy','window_v1')
    args=["-m","maichart.composer_v2.train","--manifest",str(manifest),"--checkpoint-dir",str(directory),"--cache-dir",str(cache)]
    settings={}
    from maichart.composer_v2.timing_recovery import CHOICES
    recovery_policy=_choice(payload,'timing_recovery','auto',CHOICES)
    _arg(args,'--timing-recovery',recovery_policy)
    settings['timing_recovery']=recovery_policy
    refresh_events=_integer(payload,'refresh_events',16,minimum=1)
    _arg(args,'--refresh-events',refresh_events);_arg(args,'--sequence-policy',sequence_policy)
    settings.update(refresh_events=refresh_events,sequence_policy=sequence_policy)
    choices=dict(lr_schedule={"cosine","constant"},audio_backend={"mert","native"},prepare_device={"auto","cuda","cpu"},device={"auto","cuda","cpu"},precision={"bf16","fp16","fp32","auto"})
    zeros={"warmup_steps","future_beats","history_beats","validation_max_batches","seed","max_steps","limit_charts"}
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
    settings['exhaustion_policy']=_choice(payload,'exhaustion_policy','auto',{'auto','report','stop'})
    _arg(args,'--exhaustion-policy',settings['exhaustion_policy'])
    for key in ("checkpoint", "audio", "output", "cache_dir"): _arg(args,"--"+key.replace("_","-"),settings[key])
    for key,default in dict(level=13.,temperature=.9,top_p=.95).items():
        settings[key]=_float(payload,key,default)
        _arg(args,"--"+key.replace("_","-"),settings[key])
    if settings["temperature"]<0 or not 0<settings["top_p"]<=1: raise ValueError("temperature/top_p 超出范围")
    if not 1<=settings["level"]<=15:raise ValueError("level 必须在1到15之间")
    for key,default in dict(seed=0,candidates=1,max_events=0,rollback_windows=2,cpu_threads=2).items():
        settings[key]=_integer(payload,key,default,minimum=0)
        _arg(args,"--"+key.replace("_","-"),settings[key])
    if settings["cpu_threads"]<1:raise ValueError("cpu_threads 必须是正整数")
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
    return _start(handler,runtime,"composer_v2_generate",args,settings,[output,output.with_suffix(".composer_v2.json")])
