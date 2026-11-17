"""Independent Composer v2 checkpoints with atomic writes and safe CPU reads."""
from __future__ import annotations
import math
import os
import logging
import shutil
from pathlib import Path
import tempfile
import time
import torch
from .codec import ComponentCodec
from .conditions import LEVEL_ONLY,RADAR_STYLE,condition_contract,policy_from_config
from .defaults import (
    FINETUNE_SCOPE_CHOICES, FUTURE_CONFIGURATION_CHOICES, configuration_view_fraction,
    FUTURE_CONFIGURATION_WEIGHT,
    MUSICAL_RELATION_FRACTION, MUSICAL_RELATION_POLICY,
)

FAMILY="chart_composer_v2"
SCHEMA="maichart-composer-v2-checkpoint-v1"

def _replace_checkpoint(temporary, path):
    """Keep atomic publication while tolerating short Windows file locks."""
    delays = (.05, .1, .2, .4, .8, 1.6, 2., 2.)
    for attempt in range(len(delays) + 1):
        try:
            os.replace(temporary, path)
            return
        except OSError as exc:
            if getattr(exc, 'winerror', None) not in (5, 32, 33) or attempt == len(delays):
                raise
            if attempt == 0:
                logging.getLogger(__name__).warning(
                    'Checkpoint temporarily locked (WinError %s); retrying atomic replacement: %s',
                    exc.winerror, path)
            time.sleep(delays[attempt])


def read(path):
    path=Path(path)
    try:
        payload=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    except RuntimeError as exc:
        if "mmap" not in str(exc).lower(): raise
        payload=torch.load(path,map_location="cpu",weights_only=True)
    if not isinstance(payload,dict): raise ValueError("Composer v2 checkpoint must be a dictionary")
    return payload

def save(path,payload):
    if payload.get("schema")!=SCHEMA or payload.get("family")!=FAMILY:
        raise ValueError("Refusing to save a foreign Composer v2 checkpoint")
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temporary=tempfile.mkstemp(prefix=path.name+".",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(descriptor,"wb") as stream:
            torch.save(payload,stream)
            stream.flush();os.fsync(stream.fileno())
        _replace_checkpoint(temporary,path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)

def save_epoch_snapshot(last_path, epoch, interval):
    """Archive a completed epoch without serializing GPU tensors a second time."""
    if interval <= 0 or epoch <= 0 or epoch % interval:
        return None
    last_path = Path(last_path)
    target = last_path.with_name(f"composer_v2_epoch_{epoch:03d}.pt")
    descriptor, temporary = tempfile.mkstemp(prefix=target.name+".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as destination, last_path.open("rb") as source:
            shutil.copyfileobj(source, destination, length=1024*1024)
            destination.flush()
            os.fsync(destination.fileno())
        _replace_checkpoint(temporary, target)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    return target


def inspect_payload(payload):
    errors=[]
    if payload.get("schema")!=SCHEMA: errors.append("独立 Composer v2 checkpoint schema 不匹配")
    if payload.get("family")!=FAMILY: errors.append("checkpoint 不属于 chart_composer_v2，不能加载 Composer v1 权重")
    config=payload.get("model_config") or payload.get("config")
    if not isinstance(config,dict): errors.append("缺少模型配置")
    training = payload.get('training_contract')
    saved_training = training if isinstance(training, dict) else {}
    model_fields = config if isinstance(config, dict) else {}
    continuous = model_fields.get('sequence_policy') == 'continuous_events_v1'
    relation_policy = saved_training.get('musical_relation_policy', 'off')
    relation_fraction = saved_training.get('musical_relation_fraction', MUSICAL_RELATION_FRACTION)
    future_policy = model_fields.get('future_configuration_policy', 'off')
    future_weight = model_fields.get('future_configuration_weight', FUTURE_CONFIGURATION_WEIGHT)
    future_train_view_fraction = configuration_view_fraction(future_policy)
    scope = saved_training.get('finetune_scope', 'all')
    if (relation_policy not in ('off', MUSICAL_RELATION_POLICY) or (relation_policy != 'off' and not continuous)
            or type(relation_fraction) not in (int, float) or not math.isfinite(relation_fraction)
            or not 0 <= relation_fraction < 1):
        errors.append('Invalid Composer v2 musical relation training contract')
    if (future_policy not in FUTURE_CONFIGURATION_CHOICES[1:] or (future_policy != 'off' and not continuous)
            or type(future_weight) not in (int, float) or not math.isfinite(future_weight)
            or not 0 <= future_weight <= 1):
        errors.append('Invalid Composer v2 ordered future configuration')
    if isinstance(training, dict) and (
            saved_training.get('future_configuration_policy', 'off') != future_policy
            or saved_training.get('future_configuration_weight', FUTURE_CONFIGURATION_WEIGHT) != future_weight
            or saved_training.get('future_configuration_train_view_fraction', future_train_view_fraction)
            != future_train_view_fraction):
        errors.append('Ordered future training contract does not match model configuration')
    if scope not in FINETUNE_SCOPE_CHOICES[1:] or (scope == 'music_reading' and not continuous):
        errors.append('Invalid Composer v2 fine-tuning scope')
    if isinstance(config,dict):
        if config.get('attention_normalization', 'off') not in ('off', 'qk_rms_v1'):
            errors.append('Unknown attention normalization policy')
        from .music_policy import POLICIES as MUSIC_POLICIES
        if config.get('music_context_policy','off') not in MUSIC_POLICIES:
            errors.append('Unknown Composer v2 music context policy')
        if config.get('music_context_policy')=='song_paired_v2' and config.get('sequence_policy')!='continuous_events_v1':
            errors.append('Paired music context requires continuous events')
        audio_block = config.get('local_audio_block_beats', 1)
        if (type(audio_block) is not int or audio_block not in (1, 2)
                or (audio_block == 2 and config.get('sequence_policy') != 'continuous_events_v1')):
            errors.append('局部音频编码配置与连续事件模型不匹配')
        for key in ('audio_layers','song_layers'):
            value=config.get(key,2)
            if not isinstance(value,int) or isinstance(value,bool) or value<1:
                errors.append('Invalid encoder depth: '+key)
        from .intensity import POLICY as CURVE_POLICY
        curve_policy=config.get('intensity_policy','off')
        if curve_policy not in ('off',CURVE_POLICY) or (curve_policy!='off' and config.get('sequence_policy')!='continuous_events_v1'):
            errors.append('强度曲线配置与连续事件合同不匹配')
        from .onset_policy import POLICY as ONSET_POLICY
        onset_policy = config.get('onset_readout_policy','off')
        if onset_policy not in ('off',ONSET_POLICY) or (onset_policy != 'off' and config.get('sequence_policy') != 'continuous_events_v1'):
            errors.append('独立声学读出配置与连续事件合同不匹配')
        from .rhythm_structure import POLICIES
        structure_policy = config.get('structure_policy','off')
        if structure_policy not in POLICIES or (structure_policy != 'off' and config.get('sequence_policy') != 'continuous_events_v1'):
            errors.append('节拍结构与对齐参考配置与连续事件合同不匹配')
        try:
            policy=policy_from_config(config)
            expected_condition_contract=condition_contract(policy)
            if payload.get('condition_contract') is not None and payload['condition_contract'] != expected_condition_contract:
                errors.append('Composer v2 condition contract does not match its model configuration')
            if policy == RADAR_STYLE and payload.get('condition_contract') != expected_condition_contract:
                errors.append('Radar / Style checkpoint requires an explicit condition contract')
        except ValueError as exc:
            errors.append(str(exc))
        for key in ("d_model","layers","nhead","dim_feedforward","max_tokens"):
            if not isinstance(config.get(key),int) or isinstance(config.get(key),bool) or config[key]<1:
                errors.append("Invalid model config: "+key)
        if config.get("audio_backend") not in {"mert","native"}: errors.append("Unknown audio backend")
        for key in ("window_beats","history_beats","future_beats"):
            value=config.get(key)
            if not isinstance(value,int) or isinstance(value,bool) or value<(1 if key=="window_beats" else 0):
                errors.append("Window lengths must be integral beats: "+key)
    codec=payload.get("codec")
    from .stream_codec import StreamCodec
    expected_codec=StreamCodec.schema if continuous else ComponentCodec.schema
    if not isinstance(codec,dict) or codec.get("schema")!=expected_codec: errors.append("缺少或不兼容的组件词表")
    if isinstance(codec,dict):
        tokens=codec.get("tokens")
        if not isinstance(tokens,list) or not tokens or any(not isinstance(token,str) for token in tokens) or len(set(tokens))!=len(tokens):
            errors.append("Invalid component vocabulary tokens")
    if not isinstance(payload.get("model_state_dict"),dict): errors.append("缺少模型权重")
    state=payload.get("training_state") or {}
    policy=config.get('condition_policy',LEVEL_ONLY) if isinstance(config,dict) else LEVEL_ONLY
    capabilities_contract=condition_contract(policy) if policy in (LEVEL_ONLY,RADAR_STYLE) else {}
    resume=not errors and all(isinstance(payload.get(key),dict) for key in ("optimizer_state_dict","rng_state","training_state","data_identity","training_contract"))
    if resume:
        resume=all(isinstance(state.get(key),int) and state[key]>=0 for key in ("epoch","batch_cursor","optimizer_steps","update_attempts"))
        resume=resume and all(key in payload["rng_state"] for key in ("python","numpy","torch","cuda"))
    return dict(schema=payload.get("schema"),family=payload.get("family"),model_family=payload.get("family"),
        valid=not errors,can_generate=not errors,can_init=not errors,can_train=not errors,can_resume=bool(resume),
        model_config=config or {},config=config or {},args=payload.get("args") or {},
        vocab_size=len(tokens) if not errors else None,
        training_contract=payload.get("training_contract") or {},
        training_capabilities=dict(musical_relation_policy=relation_policy, musical_relation_fraction=relation_fraction,
            future_configuration_policy=future_policy, future_configuration_weight=future_weight,
            future_configuration_train_view_fraction=future_train_view_fraction, finetune_scope=scope),
        parameter_audit=payload.get('parameter_audit') or {},
        supervision_audit=payload.get('supervision_audit') or {},
        data_contract=payload.get("data_identity") or {},epoch=state.get("epoch",0),
        batch_cursor=state.get("batch_cursor",0),optimizer_steps=state.get("optimizer_steps",0),
        inference_only=bool(payload.get("inference_only")),best_basis=payload.get("best_basis"),
        condition_contract=capabilities_contract,
        sequence_policy=model_fields.get('sequence_policy','window_v1'),
        generation_capabilities=dict(component_sequence=True,fine_note_strength=False,local_onset_guidance=False,
            local_writing_condition=future_policy=='audio_writing_v2',
            intensity_curve=model_fields.get('intensity_policy','off')=='rate_curve_v1',
            onset_readout=model_fields.get('onset_readout_policy','off')=='phase_audio_v1',
            rhythm_structure=model_fields.get('structure_policy','off')!='off',
            music_context=model_fields.get('music_context_policy','off')!='off',
            paired_music=model_fields.get('music_context_policy')=='song_paired_v2',
            entry_preview=model_fields.get('music_context_policy')=='song_paired_v2',
            continuous_time=continuous,
            condition_policy=policy,condition_keys=capabilities_contract.get('supported_keys',[]),
            radar_style=policy==RADAR_STYLE,automatic_conditions=policy==RADAR_STYLE),
        errors=errors,error="; ".join(errors) if errors else None)

def inspect(path):
    result=inspect_payload(read(path));result["path"]=str(Path(path).resolve());return result
