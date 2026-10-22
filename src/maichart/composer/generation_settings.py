"""Resolve generation controls consistently for the CLI and WebUI."""
from __future__ import annotations

import json
import secrets
from pathlib import Path

from maichart.generation.conditions import GenerationCondition,GENERATION_DEFAULTS


def resolve_seed(value):
    if value is None or (isinstance(value,str) and value.strip().lower() in {'','random','rand','randomize'}):
        return secrets.randbelow(2**31),True
    try:
        seed=int(value)
    except (TypeError,ValueError,OverflowError):
        raise ValueError('seed 必须是非负整数，或留空表示随机') from None
    if isinstance(value,bool) or (isinstance(value,float) and value!=seed) or not 0<=seed<2**63:
        raise ValueError('seed 必须是 0 至 2^63-1 的整数，或留空表示随机')
    return seed,False


def resolve_condition(level,condition=None,condition_file=None):
    result=GenerationCondition(level=level,radar=GENERATION_DEFAULTS,style=GENERATION_DEFAULTS).to_dict()
    sources=[]
    for name,raw in (('ui',condition),('file',json.loads(Path(condition_file).read_text(encoding='utf-8-sig'))
                                     if condition_file else None)):
        if raw is None:
            continue
        if not isinstance(raw,dict):
            raise ValueError(f'{name} condition 必须是 JSON 对象')
        for key in ('radar','style'):
            values=raw.get(key)
            if values is not None:
                if not isinstance(values,dict):
                    raise ValueError(f'condition.{key} 必须是对象')
                result[key].update(values)
        for key in ('level','difficulty'):
            if key in raw:
                result[key]=raw[key]
        sources.append(name)
    return GenerationCondition.from_dict(result),'+'.join(sources) or 'defaults'
