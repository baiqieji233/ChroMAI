"""WebUI adapter for the independent chart_composer_v1 workflow."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


FAMILY = "chart_composer_v1"
SCHEMA = "maichart-chart-composer-checkpoint-v2"
GENERATION_CONTRACT = 'composer_generation_controls_v11_history_type_sampling'


def _path(value: Any, label: str, *, exists: bool = False, directory: bool = False) -> Path:
    if not value or not str(value).strip():
        raise ValueError(f"缺少 {label} 路径")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        # Match server._resolve_path so a service started outside the repo still
        # resolves relative UI paths against the project root.
        path = Path(__file__).resolve().parents[3] / path
    path = path.resolve()
    if exists and not path.exists():
        raise ValueError(f"{label} 不存在: {path}")
    if directory and exists and not path.is_dir():
        raise ValueError(f"{label} 必须是目录: {path}")
    return path


def _int(payload: dict[str, Any], key: str, default: int, *, minimum: int | None = None) -> int:
    try:
        value = int(payload.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} 必须是整数") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{key} 不能小于 {minimum}")
    return value


def _float(payload: dict[str, Any], key: str, default: float, *, minimum: float | None = None) -> float:
    try:
        value = float(payload.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} 必须是数字") from exc
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{key} 数值无效")
    return value


def _bool(payload,key,default=False):
    value=payload.get(key,default)
    if not isinstance(value,bool):raise ValueError(f'{key} 必须是布尔值')
    return value


def _manifest(payload: dict[str, Any]) -> Path:
    value = payload.get("manifest")
    if isinstance(value, list):
        value = next((item for item in value if str(item).strip()), None)
    return _path(value, "manifest", exists=True)


def _device(value: Any) -> str:
    result = str(value or "auto").strip().lower()
    if result not in {"auto", "cpu", "cuda"}:
        raise ValueError("device 只能是 auto、cpu 或 cuda")
    return result


def _precision(value: Any) -> str:
    result = str(value or "auto").strip().lower()
    if result not in {"auto", "fp32", "fp16", "bf16"}:
        raise ValueError("precision 只能是 auto、fp32、fp16 或 bf16")
    return result


def _arg(args: list[str], name: str, value: Any) -> None:
    args.extend([name, str(value)])


def start_train(handler, payload: dict[str, Any], *, runtime: Any) -> None:
    requested_family = payload.get("model_family")
    if requested_family not in {None, FAMILY}:
        raise ValueError(f"Composer endpoint 只接受 model_family={FAMILY}")
    manifest = _manifest(payload)
    checkpoint_dir = _path(payload.get("checkpoint_dir"), "checkpoint-dir")
    cache_dir = _path(payload.get("cache_dir"), "cache-dir")
    mode = str(payload.get("mode", "fresh")).strip().lower()
    if mode not in {"fresh", "resume", "finetune"}:
        raise ValueError("训练模式只能是 fresh、resume 或 finetune")
    rebatch=payload.get('rebatch_resume',False)
    if not isinstance(rebatch,bool) or (rebatch and mode!='resume'):
        raise ValueError('rebatch_resume requires resume mode and a boolean value')
    if rebatch and _int(payload,'max_steps',0,minimum=0):
        raise ValueError('rebatch_resume requires max_steps=0')
    d_model = _int(payload, "d_model", 192, minimum=1)
    dim_feedforward = _int(payload, "dim_feedforward", max(1, d_model * 3), minimum=1)
    nhead = _int(payload, "nhead", 6, minimum=1)
    if mode != "fresh" and not (payload.get("resume") or payload.get("init_checkpoint") or payload.get("checkpoint")):
        raise ValueError("resume / finetune 模式必须提供 checkpoint")
    parent = payload.get("resume") or payload.get("init_checkpoint") or payload.get("checkpoint")
    parent_path = _path(parent, "checkpoint", exists=True) if mode != "fresh" else None
    architecture_profile = str(payload.get('architecture_profile') or 'audio_grounded_v3').strip()
    if architecture_profile not in {'music_context_v2', 'audio_grounded_v3'}:
        raise ValueError('architecture_profile must be music_context_v2 or audio_grounded_v3')
    parent_config = {}
    parent_args = {}
    if mode in {"resume", "finetune"}:
        # Resume is only valid for the exact current training contract. Older
        # workload/v5 checkpoints remain usable for generation and explicit
        # finetune, but must never be silently treated as resumable.
        from maichart.composer.checkpoint import inspect_payload, read_payload
        inspection = inspect_payload(read_payload(parent_path))
        parent_config = inspection.get('model_config') or {}
        parent_args = inspection.get('args') or {}
        if mode == 'resume':
            architecture_profile = str(parent_config.get('architecture_profile') or 'music_context_v2')
        if mode == 'resume' and not inspection.get("can_resume"):
            detail = inspection.get("error") or inspection.get("training_migration") or "checkpoint contract is not resumable"
            raise ValueError(f"Composer checkpoint cannot resume: {detail}; use finetune with a new output directory")
        if mode == 'finetune' and not inspection.get('can_init'):
            raise ValueError(f"Composer checkpoint cannot initialize finetune: {inspection.get('error') or inspection.get('training_migration')}")
        d_model = _int(payload, 'd_model', parent_config.get('d_model', d_model), minimum=1)
        dim_feedforward = _int(payload, 'dim_feedforward', parent_config.get('dim_feedforward', dim_feedforward), minimum=1)
        nhead = _int(payload, 'nhead', parent_config.get('nhead', nhead), minimum=1)
    if d_model % nhead:
        raise ValueError("d_model 必须能被 nhead 整除")
    dropout=_float(payload,'dropout',float(parent_config.get('dropout',.1)),minimum=0.)
    if dropout>1.:raise ValueError('dropout 不能大于 1')
    if mode!='fresh' and dropout!=float(parent_config.get('dropout',.1)):
        raise ValueError('普通 dropout 继承已有模型；改变它需要 fresh 训练')
    from maichart.composer.model import resolve_history_anchor_policy, resolve_composer_policy, COMPOSER_POLICY_FIELDS, validate_supported_context_policies
    anchor_policy=resolve_history_anchor_policy(payload.get('history_anchor_policy','auto'),parent_config,resume=mode=='resume')

    args = ["-m", "maichart.composer.train"]
    _arg(args, "--manifest", manifest)
    _arg(args, "--checkpoint-dir", checkpoint_dir)
    _arg(args, "--cache-dir", cache_dir)
    _arg(args, '--architecture-profile', architecture_profile)
    _arg(args,'--dropout',dropout)
    _arg(args,'--history-anchor-policy',anchor_policy)
    grounded_settings = {'architecture_profile': architecture_profile,'dropout':dropout,'history_anchor_policy':anchor_policy}
    from maichart.composer.objectives import resolve_action_objective
    objective=resolve_action_objective(payload.get('action_objective_policy','auto'),
        parent_args if mode!='fresh' else None,resume=mode=='resume',architecture_profile=architecture_profile)
    grounded_settings['action_objective_policy']=objective
    _arg(args,'--action-objective-policy',objective)
    cpu_default = (parent_args.get('cpu_threads') or 0) if mode == 'resume' else 0
    cpu_raw = payload.get('cpu_threads', cpu_default)
    if isinstance(cpu_raw, bool) or not isinstance(cpu_raw, (int, str)):
        raise ValueError('cpu_threads 必须是整数')
    cpu_threads = _int(payload, 'cpu_threads', cpu_default, minimum=0)
    if cpu_threads > 256:
        raise ValueError('cpu_threads cannot exceed 256')
    _arg(args, '--cpu-threads', cpu_threads)
    grounded_settings['cpu_threads'] = cpu_threads
    for key in COMPOSER_POLICY_FIELDS:
        value = resolve_composer_policy(key, payload.get(key, 'auto'), parent_config, resume=mode=='resume',architecture_profile=architecture_profile)
        grounded_settings[key] = value
        _arg(args, '--' + key.replace('_', '-'), value)
    validate_supported_context_policies(grounded_settings)
    motif_max = _int(payload, 'motif_max_events', int(parent_config.get('motif_max_events',16)), minimum=1)
    if motif_max > 32:
        raise ValueError('motif_max_events cannot exceed 32')
    if mode == 'resume' and motif_max != int(parent_config.get('motif_max_events',16)):
        raise ValueError('resume inherits motif_max_events; use finetune to change it')
    grounded_settings['motif_max_events'] = motif_max
    _arg(args, '--motif-max-events', motif_max)
    touch_policy = str(payload.get('touch_layout_policy', 'auto'))
    if touch_policy not in {'auto','open','legacy_two_heads'}:
        raise ValueError('unknown touch_layout_policy')
    inherited_touch = str(parent_args.get('touch_layout_policy', 'legacy_two_heads')) if mode!='fresh' else 'open'
    if touch_policy == 'auto': touch_policy = inherited_touch
    if mode=='resume' and touch_policy != inherited_touch:
        raise ValueError('resume inherits touch_layout_policy; use finetune to change it')
    grounded_settings['touch_layout_policy'] = touch_policy
    _arg(args, '--touch-layout-policy', touch_policy)

    for key, default, minimum, maximum in (
        ('lens_width', 64, 8, 256), ('lens_chunk_size', 32, 1, 256),
        ('recovery_interval', 100, 0, None), ('recovery_samples', 2, 1, 8),
        ('recovery_steps', 2, 2, 4),
    ):
        value = _int(payload, key, default, minimum=minimum)
        if maximum is not None and value > maximum:
            raise ValueError(f'{key} cannot exceed {maximum}')
        inherited = parent_config.get(key, parent_args.get(key)) if mode == 'resume' else None
        if inherited is not None:
            if key in payload and value != inherited:
                raise ValueError(f'resume inherits {key}={inherited}; use finetune to change it')
            value = inherited
        grounded_settings[key] = value
        _arg(args, '--' + key.replace('_', '-'), value)
    for key, default, minimum, maximum in (
        ('history_residual_bound', 1., .1, 4.), ('history_dropout', .3, 0., .9),
        ('ledger_dropout', .5, 0., .9), ('placement_weight', .15, 0., 1.),
        ('planner_weight', .2, 0., 1.), ('recovery_weight', .1, 0., 1.),
    ):
        value = _float(payload, key, default, minimum=minimum)
        if value > maximum:
            raise ValueError(f'{key} cannot exceed {maximum}')
        inherited = parent_config.get(key, parent_args.get(key)) if mode == 'resume' else None
        if inherited is not None:
            if key in payload and value != inherited:
                raise ValueError(f'resume inherits {key}={inherited}; use finetune to change it')
            value = inherited
        grounded_settings[key] = value
        _arg(args, '--' + key.replace('_', '-'), value)
    inherited_history_rms = (
        parent_config.get('history_memory_rms_bound', parent_args.get('history_memory_rms_bound', 8.))
        if mode != 'fresh' else None
    )
    history_memory_rms_bound = _float(
        payload, 'history_memory_rms_bound',
        float(inherited_history_rms if inherited_history_rms is not None else 8.), minimum=1.)
    if history_memory_rms_bound > 64.:
        raise ValueError('history_memory_rms_bound cannot exceed 64')
    if mode == 'resume':
        if 'history_memory_rms_bound' in payload and history_memory_rms_bound != float(inherited_history_rms):
            raise ValueError(f'resume inherits history_memory_rms_bound={inherited_history_rms}; use finetune to change it')
        history_memory_rms_bound = float(inherited_history_rms)
    grounded_settings['history_memory_rms_bound'] = history_memory_rms_bound
    _arg(args, '--history-memory-rms-bound', history_memory_rms_bound)
    for key, flag, default, minimum in (
        ("epochs", "--epochs", 10, 1), ("batch_size", "--batch-size", 4, 1),
        ("grad_accumulation", "--grad-accumulation", 1, 1),
        ("prefetch_batches", "--prefetch-batches", 2, 0),
        ("log_interval", "--log-interval", 25, 1), ("max_steps", "--max-steps", 0, 0),
        ("d_model", "--d-model", d_model, 1), ("dim_feedforward", "--dim-feedforward", dim_feedforward, 1),
        ("layers", "--layers", 3, 1),
        ("nhead", "--nhead", nhead, 1), ("history_events", "--history-events", 128, 1),
        ("candidate_count", "--candidate-count", 96, 16), ("action_slots", "--action-slots", 8, 1),
        ("checkpoint_interval_steps", "--checkpoint-interval-steps", 500, 0),
        ("limit_charts", "--limit-charts", 0, 0), ("prepare_workers", "--prepare-workers", 4, 0),
        ("sample_workers", "--sample-workers", 4, 1),
        ("sample_locality", "--sample-locality", 4, 1),
        ("profile_steps", "--profile-steps", 0, 0),
    ):
        value = _int(payload, key, default, minimum=minimum)
        if key == 'sample_locality' and value > 8:
            raise ValueError('sample_locality cannot exceed 8')
        if key == 'sample_locality' and mode == 'resume' and parent_args.get(key) is not None:
            inherited = int(parent_args[key])
            if key in payload and value != inherited:
                raise ValueError(f'resume inherits sample_locality={inherited}; use finetune to change it')
            value = inherited
        if key == 'action_slots' and value > 64:
            raise ValueError('action_slots cannot exceed 64')
        if key in {'d_model','dim_feedforward','layers','nhead','history_events','candidate_count','action_slots'}:
            inherited = parent_config.get(key) if mode in {'resume','finetune'} else None
            if inherited is not None:
                if mode == 'resume' and key in payload and value != inherited:
                    raise ValueError(f'resume inherits {key}={inherited}; use finetune to change it')
                if key not in payload or mode == 'resume':
                    value = int(inherited)
        _arg(args, flag, value)
    if mode != 'resume' or 'lr' in payload:
        _arg(args, "--lr", _float(payload, "lr", 0.0002, minimum=0.0))
    _arg(args, "--audio-cache-gb", _float(payload, "audio_cache_gb", 4.0, minimum=0.0))
    for key,maximum,default in (('symbolic_dropout',.5,.15),('audio_pointer_weight',1.,.15)):
        value=_float(payload,key,default,minimum=0.)
        if value>maximum: raise ValueError(f'{key} cannot exceed {maximum}')
        inherited = parent_args.get(key) if mode == 'resume' else None
        if inherited is not None:
            if key in payload and value != inherited:
                raise ValueError(f'resume inherits {key}={inherited}; use finetune to change it')
            value = float(inherited)
        _arg(args,'--'+key.replace('_','-'),value)
    _arg(args, "--device", _device(payload.get("device")))
    _arg(args, "--precision", _precision(payload.get("precision")))
    _arg(args, "--seed", _int(payload, "seed", 0, minimum=0))
    implementation=str(payload.get('optimizer_implementation','auto'))
    if implementation not in {'auto','standard','foreach','fused'}:
        raise ValueError('optimizer_implementation must be auto, standard, foreach or fused')
    _arg(args,'--optimizer-implementation',implementation)
    if _int(payload,'prefetch_batches',2,minimum=0)>16:
        raise ValueError('prefetch_batches cannot exceed 16')
    if _int(payload,'sample_workers',4,minimum=1)>8:
        raise ValueError('sample_workers cannot exceed 8')
    if _int(payload,'sample_locality',4,minimum=1)>8:
        raise ValueError('sample_locality cannot exceed 8')
    if payload.get('export_inference') is False:
        args.append('--no-export-inference')
    if mode == "resume":
        _arg(args, "--resume", parent_path)
        if rebatch:args.append('--rebatch-resume')
    elif mode == "finetune":
        _arg(args, "--init-checkpoint", parent_path)

    reserve = getattr(runtime, '_reserve_composer_training_task', None)
    task = reserve(checkpoint_dir) if reserve else runtime._create_task("composer_train")
    task.result.update({"family": FAMILY, "schema": SCHEMA, "mode": mode,
                        "manifest": str(manifest), "checkpoint_dir": str(checkpoint_dir),
                        "epoch_target": _int(payload, "epochs", 10, minimum=1),
                        "rebatch_resume_requested": rebatch,
                        "effective_settings": grounded_settings,
                        "settings_policy": "resume_inherited" if mode == 'resume' else "requested"})
    artifacts=[checkpoint_dir/'composer_last.pt']
    if payload.get('export_inference') is not False: artifacts.append(checkpoint_dir/'composer_inference.pt')
    try:
        runtime._run_subprocess_task(task, args, expected_epoch=task.result["epoch_target"],result_paths=artifacts)
    except Exception as exc:
        # Thread creation can fail before the worker owns task finalization.
        # Release the directory reservation by publishing a terminal state.
        lock = getattr(task, 'lock', None)
        if lock is not None:
            with lock:
                task.status = 'failed'
                task.error = str(exc)
                task.message = '任务启动失败'
        else:
            task.status = 'failed'
        raise
    handler._json_response({"task_id": task.task_id, "family": FAMILY})


def start_generate(handler, payload: dict[str, Any], *, runtime: Any) -> None:
    requested_family = payload.get("model_family")
    if requested_family not in {None, FAMILY}:
        raise ValueError(f"Composer endpoint 只接受 model_family={FAMILY}")
    audio = _path(payload.get("audio"), "audio", exists=True)
    checkpoint = _path(payload.get("checkpoint"), "checkpoint", exists=True)
    requested_output = _path(payload.get("output") or 'webui_workspace/generated/maidata.txt', "output")
    from maichart.composer.generation_settings import resolve_seed,resolve_condition
    from maichart.generation.timeline import serialize_tempo_map
    seed,seed_randomized=resolve_seed(payload.get('seed'))
    level=_float(payload,'level',13.,minimum=1.)
    condition_file=(_path(payload['condition_file'],'condition-file',exists=True)
                    if payload.get('condition_file') else None)
    condition,condition_source=resolve_condition(level,payload.get('condition'),condition_file)
    tempo=serialize_tempo_map(payload['tempo_map']) if payload.get('tempo_map') is not None else None
    first=_float(payload,'first',0.)
    temperature=_float(payload,'temperature',.8,minimum=0.)
    pointer_temperature=_float(payload,'pointer_temperature',temperature,minimum=0.)
    action_temperature=_float(payload,'action_temperature',temperature,minimum=0.)
    touch_layout_policy=str(payload.get('touch_layout_policy','open'))
    if touch_layout_policy not in {'open','legacy_two_heads'}:
        raise ValueError('unknown touch_layout_policy')
    motif_strength=_float(payload,'motif_strength',1.,minimum=0.)
    if motif_strength>2.:raise ValueError('motif_strength cannot exceed 2')
    curve_strength=_float(payload,'curve_strength',1.,minimum=0.)
    if curve_strength>2.:
        raise ValueError('curve_strength 不能大于 2')
    pointer_history_strength = _float(payload, 'pointer_history_strength', 1., minimum=0.)
    if pointer_history_strength > 1.:
        raise ValueError('pointer_history_strength 不能大于 1')
    for key in ('action_history_strength','action_type_temperature','action_type_min_p'):
        if isinstance(payload.get(key), bool):
            raise ValueError(f'{key} 必须是数字')
    action_history_strength=_float(payload,'action_history_strength',1.,minimum=0.)
    if action_history_strength>1.:raise ValueError('action_history_strength 不能大于 1')
    action_type_temperature=_float(payload,'action_type_temperature',1.,minimum=0.)
    if action_type_temperature>2.:raise ValueError('action_type_temperature 不能大于 2')
    action_type_min_p=_float(payload,'action_type_min_p',0.,minimum=0.)
    if action_type_min_p>1.:raise ValueError('action_type_min_p 不能大于 1')
    min_same_head_ms=_float(payload,'min_same_head_ms',50.,minimum=0.)
    if min_same_head_ms>200.:
        raise ValueError('min_same_head_ms 不能大于 200')
    intercept_tap_on_slide=bool(payload.get('intercept_tap_on_slide',False))
    touch_history_guard=_bool(payload,'touch_history_guard',True)
    touch_history_bound=_float(payload,'touch_history_bound',1.,minimum=0.)
    if touch_history_bound>4.:raise ValueError('touch_history_bound 不能大于 4')
    max_events=_int(payload,'max_events',0,minimum=0)
    device=_device(payload.get('device'))
    bpm=None
    if payload.get('bpm') not in (None,''):
        bpm=_float(payload,'bpm',120.,minimum=0.)
        if bpm<=0:raise ValueError('bpm 必须大于 0')
    task = runtime._create_task("composer_generate")
    output=requested_output.parent/task.task_id/requested_output.name
    output.parent.mkdir(parents=True,exist_ok=True)
    args = ["-m", "maichart.composer.generate"]
    for flag, value in (("--audio", audio), ("--checkpoint", checkpoint), ("--output", output)):
        _arg(args, flag, value)
    if bpm is not None:_arg(args,'--bpm',bpm)
    for flag,value in (('--first',first),('--level',condition.level),('--temperature',temperature),
                       ('--pointer-temperature',pointer_temperature),('--action-temperature',action_temperature),
                       ('--curve-strength',curve_strength),
                       ('--touch-layout-policy',touch_layout_policy),('--motif-strength',motif_strength),
                       ('--pointer-history-strength',pointer_history_strength),
                       ('--action-history-strength',action_history_strength),
                       ('--action-type-temperature',action_type_temperature),
                       ('--action-type-min-p',action_type_min_p),
                       ('--touch-history-bound',touch_history_bound),
                       ('--min-same-head-ms',min_same_head_ms),
                       ('--seed',seed),('--max-events',max_events),('--device',device)):
        _arg(args,flag,value)
    if intercept_tap_on_slide:args.append('--intercept-tap-on-slide')
    args.append('--touch-history-guard' if touch_history_guard else '--no-touch-history-guard')
    if seed_randomized:args.append('--seed-randomized')
    # Snapshot the effective condition so later UI/file edits cannot change
    # an already queued task. Explicit file keys override the UI; sparse keys
    # retain the UI values instead of silently resetting them to 0.5.
    task_directory=output.parent
    effective_file=task_directory/'condition.json'
    effective_file.write_text(json.dumps(condition.to_dict(),ensure_ascii=False,allow_nan=False),encoding='utf-8')
    _arg(args,'--condition-file',effective_file)
    if tempo:
        tempo_file=task_directory/'tempo.json'
        tempo_file.write_text(json.dumps(tempo,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        _arg(args,'--tempo-map',tempo_file)
    sidecar = output.with_suffix(".composer.json")
    task.result.update({"family": FAMILY, "checkpoint": str(checkpoint), "output": str(output),
        'requested_output':str(requested_output),'seed':seed,'seed_randomized':seed_randomized,
        'generation_contract':GENERATION_CONTRACT,'effective_settings':{
            'temperature':temperature,'pointer_temperature':pointer_temperature,
            'action_temperature':action_temperature,'curve_strength':curve_strength,
            'touch_layout_policy':touch_layout_policy,'motif_strength':motif_strength,
            'pointer_history_strength':pointer_history_strength,
            'action_history_strength':action_history_strength,
            'action_type_temperature':action_type_temperature,
            'action_type_min_p':action_type_min_p,
            'touch_history_guard':touch_history_guard,'touch_history_bound':touch_history_bound,
            'min_same_head_ms':min_same_head_ms,'intercept_tap_on_slide':intercept_tap_on_slide,
            'condition':condition.to_dict(),'condition_source':condition_source,
            'seed':seed,'seed_randomized':seed_randomized,'bpm':bpm,'first':first,
            'tempo_map':tempo,'max_events':max_events,'device':device}})
    runtime._run_subprocess_task(task, args, expected_epoch=None, result_paths=[output, sidecar])
    handler._json_response({"task_id": task.task_id, "family": FAMILY,'seed':seed,
        'seed_randomized':seed_randomized,'output':str(output)})


def inspect(handler, payload: dict[str, Any]) -> None:
    checkpoint = _path(payload.get("checkpoint"), "checkpoint", exists=True)
    try:
        from maichart.composer.checkpoint import inspect_payload,read_payload
        data = read_payload(checkpoint)
        if not isinstance(data, dict):
            raise ValueError("checkpoint payload 必须是对象")
        result = inspect_payload(data)
        groups=(data.get('optimizer_state_dict') or {}).get('param_groups') or []
        saved_lr=groups[0].get('lr') if groups else None
        if saved_lr is not None and math.isfinite(float(saved_lr)):
            result['optimizer_lr']=float(saved_lr)
        result.update({"path": str(checkpoint), "family": result.get("family") or data.get("family"),
                       "schema": result.get("schema") or data.get("schema")})
    except Exception as exc:  # inspection must return a useful UI error
        result = {"valid": False, "path": str(checkpoint), "family": None,
                  "schema": None, "error": str(exc)}
    result.setdefault('can_resume',False)
    result["can_finetune"] = bool(result.get("can_init"))
    result["can_generate"] = bool(result.get("valid"))
    handler._json_response(result)
