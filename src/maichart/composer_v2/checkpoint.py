"""Independent Composer v2 checkpoints with atomic writes and safe CPU reads."""
from __future__ import annotations
import os
import logging
import shutil
from pathlib import Path
import tempfile
import time
import torch
from .codec import ComponentCodec
from .conditions import LEVEL_ONLY,RADAR_STYLE,condition_contract,policy_from_config

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
    if isinstance(config,dict):
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
    expected_codec=StreamCodec.schema if (config or {}).get('sequence_policy')=='continuous_events_v1' else ComponentCodec.schema
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
        data_contract=payload.get("data_identity") or {},epoch=state.get("epoch",0),
        batch_cursor=state.get("batch_cursor",0),optimizer_steps=state.get("optimizer_steps",0),
        inference_only=bool(payload.get("inference_only")),best_basis=payload.get("best_basis"),
        condition_contract=capabilities_contract,
        sequence_policy=(config or {}).get('sequence_policy','window_v1'),
        generation_capabilities=dict(component_sequence=True,fine_note_strength=False,local_onset_guidance=False,
            intensity_curve=(config or {}).get('intensity_policy','off')=='rate_curve_v1',
            onset_readout=(config or {}).get('onset_readout_policy','off')=='phase_audio_v1',
            rhythm_structure=(config or {}).get('structure_policy','off')!='off',
            music_context=(config or {}).get('music_context_policy','off')!='off',
            paired_music=(config or {}).get('music_context_policy')=='song_paired_v2',
            entry_preview=(config or {}).get('music_context_policy')=='song_paired_v2',
            continuous_time=(config or {}).get('sequence_policy')=='continuous_events_v1',
            condition_policy=policy,condition_keys=capabilities_contract.get('supported_keys',[]),
            radar_style=policy==RADAR_STYLE,automatic_conditions=policy==RADAR_STYLE),
        errors=errors,error="; ".join(errors) if errors else None)

def inspect(path):
    result=inspect_payload(read(path));result["path"]=str(Path(path).resolve());return result
