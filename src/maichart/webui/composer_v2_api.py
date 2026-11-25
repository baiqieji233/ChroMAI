"""Main-workbench adapter for the independent Composer v2 contract."""
from __future__ import annotations
from dataclasses import replace
import json
import hashlib
import math
import tempfile
import os
from datetime import datetime, timezone
from pathlib import Path
from maichart.portable_paths import material_path
from maichart.composer_v2 import spatial_policy
from .composer_api import _path, _int, _float, _arg

FAMILY = "chart_composer_v2"
SCHEMA = "maichart-composer-v2-checkpoint-v1"
API_CONTRACT = "composer_v2_continuous_events_v3"
from maichart.composer_v2.defaults import (
    DEFAULT_TEMPO_DETECTION, TEMPO_DETECTION_CHOICES,
    EVALUATION_DEFAULTS, FINETUNE_SCOPE_CHOICES, FUTURE_CONFIGURATION_CHOICES,
    FUTURE_CONFIGURATION_WEIGHT, GENERATION_DEFAULTS, MAX_BATCH_COUNT,
    MUSICAL_CONTEXT_CHOICES, MUSICAL_CONTEXT_PROBABILITY, MUSICAL_RELATION_CHOICES,
    MUSICAL_RELATION_FRACTION, TRAIN_DEFAULTS, resolve_auxiliary_policy,
    resolve_finetune_scope, resolve_musical_context,
)

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
    if result.get('can_resume'):
        try:
            _, manifest_data = _saved_manifest(result.get('args') or {})
            if 'auto_quality_manifests' in manifest_data:
                result['auto_quality_manifests'] = _manifest_paths(manifest_data['auto_quality_manifests'])
        except (OSError, ValueError):
            pass  # Checkpoint capabilities do not depend on a readable local training manifest.
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


def _single_line_text(payload, key):
    if key not in payload or payload[key] is None:
        return None
    value = payload[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} 必须是字符串")
    if "\n" in value or "\r" in value:
        raise ValueError(f"{key} 不能包含换行符")
    return value.strip()


def _manifest_paths(values):
    if values is None:
        return None
    if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError('正式体检和训练来源必须由训练清单路径组成')
    return list(dict.fromkeys(str(_path(material_path(value), '训练清单')) for value in values))


def _saved_manifest(saved):
    if not saved.get('manifest'):
        return None, {}
    path = _path(material_path(saved['manifest']), '父 checkpoint 训练清单')
    if not path.is_file():
        return path, {}
    return path, json.loads(path.read_text(encoding='utf-8-sig'))


def _requested_auto_quality_scope(payload, *, policy=None):
    scope = _manifest_paths(payload.get('auto_quality_manifests'))
    if scope == [] and (policy or payload.get('auto_quality_policy')) == 'on':
        raise ValueError('必须采用正式体检时，请至少勾选一个体检来源。')
    return scope


def _prepare_scoped_resume(payload, saved, *, policy=None):
    """Reuse the existing scoped composition, including its fixed split and sidecars."""
    scope = _requested_auto_quality_scope(payload, policy=policy)
    effective, data = _saved_manifest(saved)
    frozen_scope = _manifest_paths(data.get('auto_quality_manifests'))
    if frozen_scope is None and scope is None:
        return None
    checked_sources = _manifest_paths(payload.get('manifests')) or []
    # Selecting all sources is the same composition as legacy automatic adoption.
    if frozen_scope is None and checked_sources and set(scope) == set(checked_sources):
        return None
    if scope is not None and (frozen_scope is None or set(scope) != set(frozen_scope)):
        raise ValueError('续训必须沿用检查点的正式体检来源。要调整勾选，请在新目录微调。')
    if frozen_scope == [] and (policy or payload.get('auto_quality_policy')) == 'on':
        raise ValueError('检查点未采用任何来源的正式体检。要启用，请在新目录微调。')
    split = data.get('split_metadata') or {}
    origin = (_path(material_path(split['source_manifest']), '原始训练来源', exists=True)
              if split.get('source_manifest') else effective)
    source_paths = _manifest_paths(data.get('merged_from')) or [str(origin)]
    snapshot_paths = ({str(effective)}, {str(origin)})
    if checked_sources:
        selected = set(checked_sources)
    elif payload.get('manifest'):
        selected = {str(_path(material_path(payload['manifest']), '训练清单'))}
    else:
        selected = {str(effective)}
    if selected not in (*snapshot_paths, set(source_paths)):
        raise ValueError('续训必须沿用检查点的训练清单与数据集。要调整来源，请在新目录微调。')
    if split.get('source_sha256') and hashlib.sha256(origin.read_bytes()).hexdigest() != split['source_sha256']:
        raise ValueError('原始划分来源已变化，请核对来源或使用新目录微调。')
    saved_cache = saved.get('cache_dir') or data.get('cache_dir')
    prepared = {**payload, 'manifest': str(origin), 'auto_quality_manifests': frozen_scope,
                'cache_dir': payload.get('cache_dir') or (str(material_path(saved_cache)) if saved_cache else None)}
    selection = dict(manifest=str(origin), merged_from=source_paths, auto_quality_manifests=frozen_scope,
                     summary=data.get('source_manifest_summary', data.get('summary', {})))
    return prepared, selection


def _auto_quality_scope_summary(manifest, data, enabled):
    """Describe source adoption using the sidecar bound to the effective manifest."""
    from maichart.training.quality_annotations import resolve_auto_annotations, split_annotation_source
    scope = _manifest_paths(data.get('auto_quality_manifests'))
    applied = []
    if enabled:
        _, annotations = resolve_auto_annotations(manifest, data)
        if annotations is not None:
            composed = annotations.get('audit_metadata', {}).get('composed_sources') or []
            if composed:
                applied = _manifest_paths([row['source_manifest'] for row in composed if row.get('source_manifest')])
            elif scope is None:
                if data.get('merged_from'):
                    applied = _manifest_paths(data['merged_from'])
                else:
                    applied = [str(split_annotation_source(manifest, data) or manifest)]
    return dict(mode='selected' if scope is not None else 'all', selected_manifests=scope,
                applied_manifests=applied)


def _auxiliary_settings(payload, feature, choices, value_name, default, *, sequence_policy,
                        saved_config, resume):
    policy_key = f"{feature}_policy"
    value_key = f"{feature}_{value_name}"
    requested = _choice(payload, policy_key, "auto", choices)
    value = None
    if value_key in payload:
        if isinstance(payload[value_key], bool):
            raise ValueError(f"{value_key} 必须是数字")
        value = _float(payload, value_key, default)
    policy, value = resolve_auxiliary_policy(
        requested, value, feature=feature, choices=choices, default=default,
        sequence_policy=sequence_policy, saved_config=saved_config or {}, resume=resume)
    return requested, policy, value


def _estimate_model(payload, mode, scope):
    """Resolve and count the proposed model on the meta device."""
    import torch
    from maichart.composer_v2.defaults import ENCODER_DEFAULTS
    from maichart.composer_v2.intensity import resolve_policy as resolve_intensity
    from maichart.composer_v2.model import ComposerV2, ComposerV2Config
    from maichart.composer_v2.music_policy import resolve_policy as resolve_music
    from maichart.composer_v2.onset_policy import resolve_policy as resolve_onset
    from maichart.composer_v2.rhythm_structure import resolve_policy as resolve_structure
    from maichart.composer_v2.stream_codec import StreamCodec
    from maichart.composer_v2.train import configure_trainable_parameters

    saved = None
    if mode == "fresh":
        defaults = {**TRAIN_DEFAULTS, **ENCODER_DEFAULTS}
        dimensions = {
            key: _integer(payload, key, defaults[key], minimum=1)
            for key in ("d_model", "layers", "nhead", "dim_feedforward",
                        "audio_layers", "song_layers", "ticks_per_beat", "window_beats", "max_tokens")
        }
        dimensions.update({key: _integer(payload, key, defaults[key]) for key in ("history_beats", "future_beats")})
        config = ComposerV2Config(**dimensions,
            audio_backend=_choice(payload, "audio_backend", defaults["audio_backend"], ('native', 'mert')))
    else:
        saved = payload.get("model_config")
        if not isinstance(saved, dict):
            raise ValueError("请先检查 Composer v2 checkpoint")
        config = ComposerV2Config.from_dict(saved)
        if mode == "finetune":
            config = replace(config, audio_layers=_integer(
                payload, "audio_layers", config.audio_layers, minimum=config.audio_layers))

    if mode in ('fresh', 'finetune'):
        _, future_policy, future_weight = _auxiliary_settings(
            payload, "future_configuration", FUTURE_CONFIGURATION_CHOICES, "weight",
            FUTURE_CONFIGURATION_WEIGHT, sequence_policy=config.sequence_policy,
            saved_config=saved, resume=False)
        audio_block = _choice(payload, 'local_audio_block_beats', 'auto', ('auto', '1', '2'))
        config = replace(
            config,
            local_audio_block_beats=config.local_audio_block_beats if audio_block == 'auto' else int(audio_block),
            music_context_policy=resolve_music(payload.get("music_context", "auto"), saved)[0],
            structure_policy=resolve_structure(payload.get("rhythm_structure", "auto"), saved)[0],
            onset_readout_policy=resolve_onset(payload.get("onset_readout", "auto"), saved)[0],
            intensity_policy=resolve_intensity(payload.get("intensity_curve", "auto"), saved)[0],
            future_configuration_policy=future_policy,
            spatial_policy=spatial_policy.resolve(payload.get('spatial_policy','auto'), saved=saved,
                resume=False, sequence=config.sequence_policy, writing=future_policy),
            future_configuration_weight=future_weight,
        )

    # A small codec supplies valid token IDs for the model's buffers. Checkpoint
    # estimates account for their full vocabulary through the tied table and bias.
    ticks = config.ticks_per_beat if mode == "fresh" else TRAIN_DEFAULTS["ticks_per_beat"]
    if ticks > 8192:
        raise ValueError("词表网格基准过大，无法预估参数量")
    codec = StreamCodec.fit((), ticks_per_beat=ticks)
    vocab_size = codec.vocab_size if mode == "fresh" else _integer(payload, "vocab_size", 0, minimum=1)
    with torch.device("meta"):
        model = ComposerV2(replace(config, layers=2, audio_layers=1, song_layers=1), codec)
    configure_trainable_parameters(model, scope)
    # Each odd-indexed decoder block also reads whole-song audio.
    repetitions = (
        (model.decoder[0], (config.layers + 1) // 2),
        (model.decoder[1], config.layers // 2),
        (model.audio_blocks[0], config.audio_layers),
        (model.song_blocks[0], config.song_layers),
    )
    def count(trainable_only):
        def elements(module):
            return sum(parameter.numel() for parameter in module.parameters()
                       if not trainable_only or parameter.requires_grad)
        total = elements(model) + sum((copies - 1) * elements(block) for block, copies in repetitions)
        if not trainable_only or scope == 'all':
            total += (vocab_size - codec.vocab_size) * (config.d_model + 1)
        return total

    parameters = count(False)
    trainable = count(True) if scope != 'all' else parameters
    return config, vocab_size, parameters, trainable


def estimate_parameters(handler, payload):
    """Count Composer weights without loading training data, MERT or a checkpoint."""
    mode = _choice(payload, 'mode', 'fresh', ('fresh', 'resume', 'finetune'))
    _, vocab_size, parameters, _ = _estimate_model(payload, mode, 'all')
    return handler._json_response(dict(parameter_count=parameters, vocab_size=vocab_size))


def estimate_resources(handler, payload):
    from maichart.composer_v2.acceleration import AUDIO_DEVICES
    from .composer_v2_resources import estimate_memory

    mode = _choice(payload, 'mode', 'fresh', ('fresh', 'resume', 'finetune', 'generate'))
    contract = payload.get('training_contract') if mode == 'resume' else None
    contract = contract if isinstance(contract, dict) else {}
    scope = resolve_finetune_scope(_choice(payload, 'finetune_scope', 'auto', FINETUNE_SCOPE_CHOICES),
                                  contract, resume=mode == 'resume', initialized=mode == 'finetune')
    config, vocab_size, parameters, trainable = _estimate_model(payload, mode, scope)
    if mode == 'generate':
        requested = dict(
            requested_device=_choice(payload, 'device', 'auto', ('auto', 'cpu', 'cuda')),
            audio_device=_choice(payload, 'audio_device', 'auto', AUDIO_DEVICES),
            inference_backend=_choice(payload, 'inference_backend', 'auto', ('auto', 'torch', 'openvino-cpu', 'openvino-gpu')),
            count=_integer(payload, 'count', GENERATION_DEFAULTS['count'], minimum=1),
            parallel=_integer(payload, 'parallel', GENERATION_DEFAULTS['parallel'], minimum=1),
        )
        if max(requested['count'], requested['parallel']) > MAX_BATCH_COUNT:
            raise ValueError(f'生成张数和并行数须在 1 到 {MAX_BATCH_COUNT} 之间')
    else:
        device = _choice(payload, 'device', 'auto', ('auto', 'cpu', 'cuda', 'xpu'))
        precision = _choice(payload, 'precision', 'auto', ('auto', 'bf16', 'fp16', 'fp32'))
        requested = dict(
            requested_device=contract.get('device_type', device) if device == 'auto' else device,
            requested_precision=contract.get('precision', precision) if precision == 'auto' else precision,
            batch=_integer(payload, 'batch_size', contract.get('batch_size', TRAIN_DEFAULTS['batch_size']), minimum=1),
            activation_checkpointing=_choice(payload, 'activation_checkpointing', 'auto', ('auto', 'off', 'on')),
            decoder_flash_attention=_choice(payload, 'decoder_flash_attention', 'off', ('off', 'on')),
            audio_device=_choice(payload, 'prepare_device', 'auto', AUDIO_DEVICES),
            cuda_budget=_float(payload, 'cuda_memory_budget_gb', 0., minimum=0),
        )
        if requested['cuda_budget'] > 1024:
            raise ValueError('CUDA 显存预算不能超过 1024 GiB')
    memory = estimate_memory(config, vocab_size, parameters, trainable, mode=mode, **requested)
    return handler._json_response(dict(parameter_count=parameters, vocab_size=vocab_size,
        trainable_parameter_count=trainable if mode != 'generate' else None,
        model_weight_bytes=4 * parameters, memory=memory))

def _start(handler,runtime,kind,args,settings,paths,checkpoint_dir=None):
    task=runtime._create_task(kind,checkpoint_dir=checkpoint_dir)
    task.result.update(model_family=FAMILY,family=FAMILY,schema=SCHEMA,effective_settings=settings)
    if kind in ('composer_v2_curve','composer_v2_media'):
        task.result.update(audio=settings['audio'],audio_preview_url=f'/api/intensity/audio?id={task.task_id}')
    if kind.endswith("_train"): task.result["epoch_target"]=settings["epochs"]
    options = dict(expected_epoch=settings.get("epochs"), result_paths=paths)
    audit_lock = None
    if kind == "composer_v2_auto_quality_audit":
        from maichart.composer.training_lock import TrainingOutputBusyError, _directory_lock
        from maichart.training.dataset_catalog import _write_json
        from .material_audit_status import audit_state_path, audit_task_lock

        manifest = Path(settings["manifest"])
        preview = bool(settings["limit_charts"])
        state_path = audit_state_path(manifest, preview)
        audit_lock = _directory_lock(manifest.parent, audit_task_lock(preview))
        try:
            audit_lock.__enter__()
        except TrainingOutputBusyError as exc:
            task.status = "failed"
            task.error = "这个训练清单已有同类型体检正在运行"
            raise ValueError(task.error) from exc
        state = dict(manifest=str(manifest), task_id=task.task_id, status="queued", pid=os.getpid(),
                     started_at=datetime.now(timezone.utc).isoformat())

        def finish_audit():
            try:
                _write_json(state_path, dict(state, status=task.status, error=task.error,
                                             finished_at=datetime.now(timezone.utc).isoformat()))
            finally:
                audit_lock.__exit__(None, None, None)

        def audit_running():
            _write_json(state_path, dict(state, status="running"))

        options.update(on_start=audit_running, on_finish=finish_audit)
    try:
        if audit_lock is not None:
            _write_json(state_path, state)
        runtime._run_subprocess_task(task, args, **options)
    except Exception as exc:
        with task.lock:
            task.status="failed"
            task.error=str(exc)
        if audit_lock is not None:
            finish_audit()
        raise
    result = dict(task_id=task.task_id,model_family=FAMILY)
    if kind == 'composer_v2_auto_quality_audit':
        result.update(manifest=settings['manifest'], auto_scope='preview' if settings['limit_charts'] else 'formal')
    return handler._json_response(result)

def start_chartgate_setup(handler,payload,*,runtime):
    _family(payload)
    cache = _path(payload.get('cache_dir') or 'cache/chart_composer_v2','音频缓存目录')
    settings = {'cache_dir':str(cache)}
    args = ['-m','maichart.composer_v2.chartgate','--cache-dir',str(cache)]
    if payload.get('chartgate_dir'):
        directory = _path(payload['chartgate_dir'],'Chart;Gate 目录',exists=True,directory=True)
        settings['chartgate_dir'] = str(directory)
        _arg(args,'--chartgate-dir',directory)
    return _start(handler,runtime,'composer_v2_chartgate_setup',args,settings,[])


def start_train(handler,payload,*,runtime):
    _family(payload)
    if payload.get("auto_preprocess"): raise ValueError("Composer v2 请先准备训练 manifest")
    _requested_auto_quality_scope(payload)
    directory=_path(payload.get("checkpoint_dir") or "checkpoints/chart_composer_v2","checkpoint_dir")
    mode=_choice(payload,"mode",payload.get("training_mode","fresh"),{"fresh","resume","finetune"})
    parent=None
    scoped_resume=None
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
    sequence_policy = 'continuous_events_v1' if mode == 'fresh' else fixed.get('sequence_policy', 'window_v1')
    musical_context = _choice(payload, 'musical_context', 'auto', MUSICAL_CONTEXT_CHOICES)
    probability_key = 'musical_context_probability'
    musical_probability = None
    if probability_key in payload:
        if isinstance(payload[probability_key], bool):
            raise ValueError('音乐独立练习比例须在 0 至 1 之间')
        musical_probability = _float(payload, probability_key, MUSICAL_CONTEXT_PROBABILITY)
    musical_policy, musical_probability = resolve_musical_context(
        musical_context, musical_probability, sequence_policy=sequence_policy,
        parent_contract=checked.get('training_contract') if parent else None, resume=mode == 'resume')
    parent_contract = (checked.get("training_contract") or {}) if parent else {}
    from maichart.composer_v2 import generated_prefix
    saved_prefix = parent_contract.get('generated_prefix_experiment', {}) if mode == 'resume' else {}
    prefix_mode = _choice(payload, 'generated_prefix_mode', saved_prefix.get('mode', 'off'), set(generated_prefix.MODES))
    generated_prefix.validate_mode(prefix_mode, sequence_policy, mode == 'finetune', mode == 'resume')
    prefix_path = None
    prefix_contract = {}
    if prefix_mode != 'off':
        saved_bank = saved_args.get('generated_prefix_bank') if mode == 'resume' else None
        prefix_path = _path(payload.get('generated_prefix_bank') or saved_bank, '生成前文样本文件', exists=True)
        prefix_contract = generated_prefix.read_bank(prefix_path).contract(prefix_mode)
    elif payload.get('generated_prefix_bank'):
        raise ValueError('请先选择生成前文实验模式')
    if mode == 'resume' and prefix_contract != saved_prefix:
        raise ValueError('断点续训需要沿用保存时的生成前文模式与样本文件')
    relation_requested, relation_policy, relation_fraction = _auxiliary_settings(
        payload, "musical_relation", MUSICAL_RELATION_CHOICES, "fraction", MUSICAL_RELATION_FRACTION,
        sequence_policy=sequence_policy, saved_config=parent_contract, resume=mode == "resume")
    future_requested, future_policy, future_weight = _auxiliary_settings(
        payload, "future_configuration", FUTURE_CONFIGURATION_CHOICES, "weight",
        FUTURE_CONFIGURATION_WEIGHT, sequence_policy=sequence_policy,
        saved_config=fixed if parent else None, resume=mode == "resume")
    scope_requested = _choice(payload, "finetune_scope", "auto", FINETUNE_SCOPE_CHOICES)
    spatial_requested = _choice(payload, 'spatial_policy', 'auto', spatial_policy.CHOICES)
    spatial = spatial_policy.resolve(spatial_requested, saved=fixed if parent else None,
                                     resume=mode=='resume', sequence=sequence_policy, writing=future_policy)
    scope = resolve_finetune_scope(scope_requested, parent_contract, resume=mode == "resume",
                                   initialized=mode == "finetune")
    if scope == "music_reading" and sequence_policy != "continuous_events_v1":
        raise ValueError("音乐读取连接微调需要连续事件 checkpoint")
    if mode == 'resume':
        scoped_resume = _prepare_scoped_resume(payload, saved_args)
    if scoped_resume is not None:
        frozen_payload,selection=scoped_resume
        payload.update({key:frozen_payload[key] for key in ('manifest','auto_quality_manifests','cache_dir')})
    else:
        payload,selection=runtime._prepare_training_sources({**payload,"model_family":FAMILY})
    manifest=_path(payload.get("manifest"),"manifest",exists=True)
    cache=_path(payload.get("cache_dir") or "cache/chart_composer_v2","cache_dir")
    args=["-m","maichart.composer_v2.train","--manifest",str(manifest),"--checkpoint-dir",str(directory),"--cache-dir",str(cache)]
    _arg(args, '--musical-context', musical_context)
    _arg(args, '--musical-context-probability', musical_probability)
    settings = dict(musical_context=musical_context, musical_context_policy=musical_policy,
                    musical_context_probability=musical_probability)
    _arg(args, "--musical-relation-policy", relation_requested)
    _arg(args, "--musical-relation-fraction", relation_fraction)
    _arg(args, "--future-configuration-policy", future_requested)
    _arg(args, "--future-configuration-weight", future_weight)
    _arg(args, "--spatial-policy", spatial)
    _arg(args, "--finetune-scope", scope_requested)
    settings.update(musical_relation_policy=relation_policy, musical_relation_fraction=relation_fraction,
                    future_configuration_policy=future_policy, future_configuration_weight=future_weight,
                    spatial_policy=spatial, finetune_scope=scope)
    from maichart.composer_v2.difficulty import (
        DIFFICULTY_SLOTS, FITTED_SOURCES, LEGACY_DIFFICULTY_SLOTS, locate_scorer, resolve_selection, scorer_path_key,
    )
    difficulty=resolve_selection(payload.get('difficulty_source','auto'),payload.get('min_level'),
        payload.get('max_level'),min_declared_level=payload.get('min_declared_level'),
        difficulty_slots=payload.get('difficulty_slots'),parent=checked.get('training_contract') if parent else None,
        resume=mode=='resume')
    settings.update(difficulty_source=difficulty.source, **difficulty.dataset_options())
    for key in ('difficulty_source','min_level','max_level','min_declared_level'):
        _arg(args,'--'+key.replace('_','-'),settings[key])
    args.extend(['--difficulty-slots', *map(str, difficulty.difficulty_slots)])
    if difficulty.source in FITTED_SOURCES:
        key = scorer_path_key(difficulty.source)
        saved_directory = (checked.get('args') or {}).get(key) if mode=='resume' else None
        requested_directory = (_path(payload[key],'估分工具目录',exists=True,directory=True)
                               if payload.get(key) else None)
        root = locate_scorer(difficulty.source,cache,requested_directory,saved_directory)
        _arg(args,'--'+key.replace('_','-'),root)
        settings[key] = str(root)
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
    from maichart.composer_v2 import timing_recovery
    requested_recovery=_choice(payload,'timing_recovery','auto',timing_recovery.CHOICES)
    recovery_policy=timing_recovery.resolve(requested_recovery,sequence_policy=sequence_policy,
        saved=parent_contract.get('timing_recovery','off'),resume=mode=='resume')
    _arg(args,'--timing-recovery',requested_recovery)
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
    inherited_audio_block = (checked.get('model_config') or {}).get('local_audio_block_beats', 1) if parent else 1
    audio_block = _choice(payload, 'local_audio_block_beats', 'auto', {'auto', '1', '2'})
    audio_block = inherited_audio_block if audio_block == 'auto' else int(audio_block)
    if mode == 'resume' and audio_block != inherited_audio_block:
        raise ValueError('更改局部音频编码方式需要使用新目录微调。')
    if audio_block == 2 and sequence_policy != 'continuous_events_v1':
        raise ValueError('相邻两拍共享仅适用于连续事件检查点。')
    if audio_block != inherited_audio_block:
        _arg(args, '--local-audio-block-beats', audio_block)
    settings['local_audio_block_beats'] = audio_block
    from maichart.composer_v2.acceleration import AUDIO_DEVICES
    from maichart.composer_v2.defaults import LR_SCHEDULES
    from maichart.composer_v2.learning_rate import validate as validate_lr_plan
    choices=dict(lr_schedule=set(LR_SCHEDULES),audio_backend={"mert","native"},prepare_device=set(AUDIO_DEVICES),device={"auto","cuda","cpu","xpu"},precision={"bf16","fp16","fp32","auto"},activation_checkpointing={'auto','off','on'},attention_projection_packing={'auto','off','on'},decoder_flash_attention={'off','on'})
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
    from maichart.composer_v2.batching import CHOICES as BATCH_LAYOUTS, resolve_layout
    batch_layout=_choice(payload,'batch_layout','auto',BATCH_LAYOUTS)
    saved_contract=checked.get('training_contract',{}) if mode=='resume' else {}
    if mode!='resume' or 'batch_size' in saved_contract:
        resolved_layout,_,_=resolve_layout(batch_layout,settings['batch_size'],settings['grad_accumulation'],
            {'training_contract':saved_contract} if mode=='resume' else None)
        settings['resolved_batch_layout']=resolved_layout
    settings['batch_layout']=batch_layout
    _arg(args,'--batch-layout',batch_layout)
    if settings["d_model"]%settings["nhead"]: raise ValueError("d_model 必须能被 nhead 整除")
    if settings["lr"]<=0: raise ValueError("lr 必须大于0")
    if settings["dropout"]>=1 or settings["min_lr_ratio"]>1: raise ValueError("dropout/min_lr_ratio 超出范围")
    inherited_decay_end=parent_contract.get('lr_decay_end_step',0) if mode=='resume' else 0
    decay_end=_integer(payload,'lr_decay_end_step',inherited_decay_end,minimum=0)
    validate_lr_plan(settings['lr_schedule'],settings['warmup_steps'],decay_end)
    if mode=='resume' and decay_end!=inherited_decay_end:
        raise ValueError('更改衰减结束步数请在新目录加强训练（finetune）。')
    settings['lr_decay_end_step']=decay_end
    if decay_end:
        _arg(args,'--lr-decay-end-step',decay_end)
    if prefix_contract:
        if not settings['max_steps']:
            raise ValueError('生成前文短测请填写最大更新尝试数')
        _arg(args,'--generated-prefix-mode',prefix_mode)
        _arg(args,'--generated-prefix-bank',prefix_path)
        settings['generated_prefix_experiment']=prefix_contract
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
        saved_path=material_path if scoped_resume is not None else str
        manifest=_path(saved_path(saved["manifest"]) if saved.get("manifest") else None,"父 checkpoint 训练清单",exists=True)
        split_metadata=json.loads(manifest.read_text(encoding="utf-8-sig")).get("split_metadata",{})
        accepted={manifest}
        if split_metadata.get("source_manifest"):
            accepted.add(_path(saved_path(split_metadata["source_manifest"]),"原始训练来源"))
        if source_manifest not in accepted:raise ValueError("resume 必须使用父 checkpoint 的训练清单或其原始划分来源")
        expected=(split_metadata.get("source_sha256") if source_manifest!=manifest else
            ((checked.get("data_contract") or {}).get("train") or {}).get("manifest_sha256"))
        if expected and hashlib.sha256(source_manifest.read_bytes()).hexdigest()!=expected:
            raise ValueError("resume 训练清单内容已变化；请保留原 checkpoint 并使用新目录 finetune")
        validation=_path(saved_path(saved["validation_manifest"]),"父 checkpoint 验证清单",exists=True) if saved.get("validation_manifest") else None
        resolve_auto_quality_policy(auto_quality,manifest,checked.get('training_contract'),
                                    resume=True,quality_enabled=quality,sequence_policy=sequence_policy)
        source="inherited"
    elif source=="auto":
        split_directory=directory/"data_split"
        if split_directory.exists():raise ValueError("输出目录已有 data_split，请使用新的输出目录以保留原划分")
        from maichart.composer.evaluation import split_manifest
        extra_filters = difficulty.min_declared_level or difficulty.difficulty_slots != LEGACY_DIFFICULTY_SLOTS
        split_report=split_manifest(manifest,split_directory,fraction=fraction,seed=settings["seed"],cache_dir=cache,
                                    chart_eligibility=difficulty.split_exclusion if extra_filters else None)
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
    if 'auto_quality_manifests' in manifest_data:
        effective_data = manifest_data if manifest == source_manifest else json.loads(manifest.read_text(encoding='utf-8-sig'))
        adopted = resolve_auto_quality_policy(auto_quality,manifest,checked.get('training_contract') if parent else None,
            resume=mode=='resume',quality_enabled=quality,sequence_policy=sequence_policy) == 'auto_v1'
        settings['auto_quality_manifests'] = _manifest_paths(effective_data['auto_quality_manifests'])
        settings['auto_quality_scope'] = _auto_quality_scope_summary(manifest,effective_data,adopted)
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
    guidance=payload.get('radar_style_guidance',False)
    if not isinstance(guidance,bool):
        raise ValueError('radar_style_guidance 必须是布尔值')
    if guidance:
        from maichart.composer_v2.condition_guidance import validate as validate_guidance
        validate_guidance(True,checked.get('sequence_policy'),policy)
    audio=_path(payload.get("audio"),"audio",exists=True)
    output=_path(payload.get("output") or "outputs/composer_v2/maidata.txt","output")
    cache=_path(payload.get("cache_dir") or "cache/chart_composer_v2","cache_dir")
    settings=dict(checkpoint=str(checkpoint),audio=str(audio),output=str(output),cache_dir=str(cache))
    args=["-m","maichart.composer_v2.generate"]
    if condition: _arg(args, "--condition", json.dumps(condition, ensure_ascii=False))
    settings.update(condition=condition, condition_policy=policy, radar_style_guidance=guidance)
    if guidance:args.append('--radar-style-guidance')
    eos_suppression=payload.get('eos_suppression',False)
    if not isinstance(eos_suppression,bool):
        raise ValueError('提前结束抑制必须是布尔值')
    if eos_suppression and checked.get('sequence_policy')!='continuous_events_v1':
        raise ValueError('提前结束抑制仅适用于连续事件检查点')
    settings['eos_suppression']=eos_suppression
    if eos_suppression:args.append('--eos-suppression')
    record_probabilities=payload.get('record_candidate_probabilities',GENERATION_DEFAULTS['record_candidate_probabilities'])
    if not isinstance(record_probabilities,bool):
        raise ValueError('record_candidate_probabilities 必须是布尔值')
    if record_probabilities and checked.get('sequence_policy')=='window_v1':
        raise ValueError('记录候选评估数据仅支持连续事件模型；这个旧窗口模型不支持')
    settings['record_candidate_probabilities']=record_probabilities
    if record_probabilities:args.append('--record-candidate-probabilities')
    from maichart.composer_v2.phrase_search import PHRASE_MODES, RANK_POLICIES, resolve_search_options
    phrase_search, phrase_rank_policy=resolve_search_options(
        _choice(payload,'phrase_search','auto',('auto',*PHRASE_MODES)),
        _choice(payload,'phrase_rank_policy','auto',('auto',*RANK_POLICIES)),
        sequence_policy=checked.get('sequence_policy'),
        future_configuration_policy=(checked.get('model_config') or {}).get('future_configuration_policy','off'))
    phrase_beats=_float(payload,'phrase_beats',GENERATION_DEFAULTS['phrase_beats'],minimum=1.)
    phrase_commit_beats=_float(payload,'phrase_commit_beats',GENERATION_DEFAULTS['phrase_commit_beats'],minimum=0.)
    if isinstance(payload.get('phrase_beats'),bool) or isinstance(payload.get('phrase_commit_beats'),bool):
        raise ValueError('提前试写长度和每次保留长度必须是数字')
    if phrase_beats>16 or not 0<phrase_commit_beats<=phrase_beats:
        raise ValueError('提前试写长度须为1–16拍；每次保留长度须大于0，且不能超过提前试写长度')
    settings.update(phrase_search=phrase_search,phrase_rank_policy=phrase_rank_policy,phrase_beats=phrase_beats,phrase_commit_beats=phrase_commit_beats)
    for key in ('phrase_search','phrase_rank_policy','phrase_beats','phrase_commit_beats'):_arg(args,'--'+key.replace('_','-'),settings[key])
    settings['exhaustion_policy']=_choice(payload,'exhaustion_policy','auto',{'auto','report','stop','downgrade_or_skip'})
    if checked.get('sequence_policy')=='window_v1' and settings['exhaustion_policy'] in {'report','downgrade_or_skip'}:
        raise ValueError('旧窗口模型仅支持 auto 或 stop 耗尽策略')
    settings['tempo_detection']=_choice(payload,'tempo_detection',DEFAULT_TEMPO_DETECTION,TEMPO_DETECTION_CHOICES)
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
    continuous = checked.get('sequence_policy') == 'continuous_events_v1'
    recovery_default = GENERATION_DEFAULTS['configuration_recovery'] if continuous else False
    configuration_recovery = payload.get('configuration_recovery', recovery_default)
    if not isinstance(configuration_recovery, bool):
        raise ValueError('连续单键试写必须是布尔值')
    if configuration_recovery and not continuous:
        raise ValueError('连续单键试写仅支持连续事件模型')
    settings['configuration_recovery'] = configuration_recovery
    args.append('--configuration-recovery' if configuration_recovery else '--no-configuration-recovery')
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
    for key, label in (('reference_strength', '配置呼应强度'),
                       ('recent_configuration_strength', '近期配置强度')):
        strength = _float(payload, key, 1.)
        if isinstance(payload.get(key), bool) or not 0 <= strength <= 2:
            raise ValueError(f'{label}须在 0 到 2 之间')
        if strength != 1.:
            if checked.get('sequence_policy') != 'continuous_events_v1':
                raise ValueError(f'{label}适用于连续事件检查点')
            _arg(args, '--'+key.replace('_', '-'), strength)
        settings[key] = strength
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
    from maichart.composer_v2.audio_metadata import format_level_label
    settings['level_label']=format_level_label(settings['level'])
    for key in ('title','artist','designer'):
        text_value=_single_line_text(payload,key)
        if text_value is not None:
            settings[key]=text_value
            args.append(f'--{key}={text_value}')
    raw_difficulty=payload.get('difficulty')
    if raw_difficulty is None:
        difficulty=5
    else:
        difficulty=_integer({'difficulty':raw_difficulty},'difficulty',5,minimum=1)
        if difficulty>6:
            raise ValueError('difficulty 必须是 1 到 6 的整数')
    settings['difficulty']=difficulty
    _arg(args,'--difficulty',difficulty)
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
            argument = _tempo_map_argument(value, cache) if key == 'tempo_map' else json.dumps(value, ensure_ascii=False)
            _arg(args,"--"+key.replace("_","-"),argument)
            settings[key]=value
    count=_integer(payload,'count',GENERATION_DEFAULTS['count'],minimum=1)
    if count>MAX_BATCH_COUNT:raise ValueError(f'一次最多生成 {MAX_BATCH_COUNT} 张谱面')
    settings['count']=count
    parallel=_integer(payload,'parallel',GENERATION_DEFAULTS['parallel'],minimum=1)
    if parallel>MAX_BATCH_COUNT:raise ValueError(f'同时生成数不能超过 {MAX_BATCH_COUNT}')
    settings.update(parallel=parallel,effective_parallel=min(count,parallel))
    _arg(args,'--parallel',parallel)
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
    import uuid
    directory_tag = '_batch_' if count > 1 else '_'
    root = output.parent / (output.stem + directory_tag + uuid.uuid4().hex[:12])
    output = root / output.name
    args[args.index('--output')+1] = str(output)
    settings['output'] = str(output)
    paths = [output, output.with_suffix('.composer_v2.json')]
    if record_probabilities:
        paths.append(output.with_suffix('.candidate_audit.jsonl'))
    if count>1:
        manifest=root/'batch.composer_v2_batch.json'
        _arg(args,'--count',count);_arg(args,'--batch-manifest',manifest)
        settings['batch_manifest'] = str(manifest)
        paths=[manifest]  # Successful chart artifacts are collected from it.
    return _start(handler,runtime,"composer_v2_generate",args,settings,paths)


def _tempo_map_argument(value, cache_dir):
    content = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    # Leave room for Windows quoting, the interpreter path and other arguments.
    if len(content) <= 4096:
        return content
    directory = Path(cache_dir) / 'tempo_maps'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (hashlib.sha256(content.encode('utf-8')).hexdigest() + '.json')
    if not target.exists():
        temporary = tempfile.NamedTemporaryFile(dir=directory, delete=False)
        tmp = Path(temporary.name)
        try:
            with temporary:
                temporary.write(content.encode('utf-8'))
            if not target.exists():
                tmp.replace(target)
        finally:
            tmp.unlink(missing_ok=True)
    return '@' + str(target)


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
        tempo_detection=_choice(payload,'tempo_detection',DEFAULT_TEMPO_DETECTION,TEMPO_DETECTION_CHOICES),
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
        if value is None or value=='': continue
        if key == 'tempo_map':
            if isinstance(value, str): value = json.loads(value)
            value = _tempo_map_argument(value, settings['cache_dir'])
        elif isinstance(value, (dict,list)):
            value = json.dumps(value,ensure_ascii=False)
        _arg(args,'--'+key.replace('_','-'),value)
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
    if payload.get('evaluation_mode')=='stability':
        import uuid
        manifest=_path(payload.get('batch_manifest'),'批量生成清单',exists=True)
        if not manifest.is_file() or manifest.suffix.lower()!='.json':
            raise ValueError('请选择批量生成清单 JSON 文件')
        output=runtime.INTENSITY_ROOT / uuid.uuid4().hex / 'stability.json'
        args=['-m','maichart.composer_v2.evaluate','stability','--batch',str(manifest),'--output',str(output)]
        return _start(handler,runtime,'composer_v2_evaluate',args,dict(batch_manifest=str(manifest),output=str(output)),[output])
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
    writing_probe=payload.get('writing_probe',False)
    if type(writing_probe) is not bool: raise ValueError('写法对照必须是布尔值')
    settings['writing_probe']=writing_probe
    training = checkpoint.parent / 'data_split' / 'train_manifest.json'
    if overlap and not training.is_file():
        raise ValueError(f'无法审计训练重叠：缺少同检查点目录清单 {training}')
    import uuid
    output = runtime.INTENSITY_ROOT / uuid.uuid4().hex / 'quality_eval.json'
    settings['output'] = str(output)
    args = ['-m', 'maichart.composer_v2.evaluate', 'teacher-forced']
    if writing_probe: args.append('--writing-probe')
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
    from .dataset_api import _json_data, _pool_card
    from .material_audit_status import selection_audits
    from .quality_api import QualityAPI, auto_quality_view

    _family(payload)
    scope = _choice(payload, 'auto_scope', 'formal', {'formal', 'preview', 'auto'})
    policy = _choice(payload, 'auto_quality_policy', 'auto', {'auto', 'on', 'off'})
    selected_sources = _manifest_paths(payload.get('auto_quality_manifests'))
    quality_enabled = payload.get('quality_enabled', True)
    if not isinstance(quality_enabled, bool):
        raise ValueError('quality_enabled 必须是布尔值')
    resume = payload.get('mode') == 'resume'
    sequence = payload.get('sequence_policy', 'continuous_events_v1')
    empty = dict(available=False, manifest=None, auto_scope=scope, has_annotations=False,
                 has_preview_annotations=False, annotations_path=None, record_count=0, report=None,
                 preview_report=None, training_eligible=False, training_effective=None if resume else False)

    workspace = getattr(runtime, "WEBUI_ROOT", None)
    quality_api = QualityAPI(workspace)

    def status(value):
        manifest = _path(value, 'manifest')
        if not manifest.is_file():
            return {**empty, 'manifest': str(manifest)}
        data = _json_data(manifest)
        if data.get("candidate_pool"):
            return dict(empty, manifest=str(manifest), candidate_pool=_pool_card(manifest.parent, data))
        meta, _, _ = auto_quality_view(manifest, data, scope)
        report_key = "preview_report" if meta["auto_scope"] == "preview" else "report"
        if meta[report_key] and meta[report_key].get("reported_prompts") is not None:
            meta[report_key] = quality_api._catalog_view(manifest, scope).audit[report_key]
        eligible = meta['has_annotations'] and not (data.get('candidate_pool') or data.get('reviewed_training'))
        effective = bool(eligible and quality_enabled and policy != 'off' and sequence == 'continuous_events_v1')
        if selected_sources is not None and not (data.get('merged_from') or data.get('split_metadata')):
            effective = effective and str(manifest) in selected_sources
        return dict(meta, training_eligible=eligible, training_effective=None if resume else effective)

    manifests = payload.get('manifests') or []
    if not isinstance(manifests, list) or any(not isinstance(value, str) or not value.strip() for value in manifests):
        raise ValueError('训练数据集列表必须由有效路径组成')
    manifests = list(dict.fromkeys(manifests))
    if not manifests and payload.get("manifest"):
        return handler._json_response(status(payload["manifest"]))
    sources = [status(value) for value in manifests]
    if payload.get("sources_only") is True:
        return handler._json_response(dict(sources=sources))
    if len(sources) == 1 and not sources[0].get("candidate_pool"):
        return handler._json_response(sources[0])
    result = dict(empty, sources=sources)
    if workspace and sources:
        history = selection_audits(manifests, workspace)
        snapshot = history["current_snapshot"]
        if snapshot:
            result.update(status(snapshot["manifest"]))
        result.update(history)
    return handler._json_response(result)


def start_auto_quality_audit(handler, payload, *, runtime):
    """Inspect the selected corpus in a background CPU process."""
    _family(payload)
    limit = _integer(payload, 'limit_charts', 0, minimum=0)
    scoped_message = '这份训练快照只采用了指定来源的体检。请在素材列表选原始来源，再运行正式体检。'
    requested_sources = payload.get("manifests") or ([payload["manifest"]] if payload.get("manifest") else [])
    if isinstance(requested_sources, list):
        for value in requested_sources:
            requested = _path(value, "manifest", exists=True)
            requested_data = json.loads(requested.read_text(encoding="utf-8-sig"))
            if 'auto_quality_manifests' in requested_data:
                if not limit:
                    raise ValueError(scoped_message)
            elif requested_data.get("merged_from") or requested_data.get("split_metadata"):
                raise ValueError("请在素材列表勾选原始来源后体检，这份训练快照为只读")
            if requested_data.get("candidate_pool") and not payload.get("manifests"):
                payload = {**payload, "manifests": [str(requested)]}
    payload, _ = runtime._prepare_training_sources({**payload, 'model_family': FAMILY})
    manifest = _path(payload.get('manifest'), 'manifest', exists=True)
    if not limit and 'auto_quality_manifests' in json.loads(manifest.read_text(encoding='utf-8-sig')):
        raise ValueError(scoped_message)
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
    highlight_good_mode = _choice(payload, 'highlight_good_mode', 'off', {'auto', 'acoustic_and_flow_v1', 'chart_only_v1', 'off'})
    if highlight_good_mode == 'auto':
        highlight_good_mode = 'off'
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
    min_difficulty_index = _integer(payload, 'min_difficulty_index', 5, minimum=4)
    if min_difficulty_index > 6:
        raise ValueError('最低难度槽位须为 4–6')
    args = ['-m', 'maichart.composer_v2.auto_quality_audit', '--manifest', str(manifest),
            '--limit-charts', str(limit), '--tap-on-slide-mode', tail_mode,
            '--inner-muri-mode', inner_mode, '--tap-on-slide-ms', str(threshold)]
    if min_difficulty_index != 5:
        _arg(args, '--min-difficulty-index', min_difficulty_index)
    if touch_policy != 'palm_span_v1':
        _arg(args, '--touch-reachability-policy', touch_policy)
    if touch_radius != 180.:
        _arg(args, '--touch-cohold-radius', touch_radius)
    if domain_pathology_mode != 'maimai_dx_v1':
        _arg(args, '--domain-pathology-mode', domain_pathology_mode)
    if acoustic_audit_mode != 'auto':
        _arg(args, '--acoustic-audit-mode', acoustic_audit_mode)
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
                    min_difficulty_index=min_difficulty_index,
                    workers=workers,
                    device='cpu')
    return _start(handler, runtime, 'composer_v2_auto_quality_audit', args, settings, [report, annotations])
