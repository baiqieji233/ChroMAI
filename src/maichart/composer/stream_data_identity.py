"""Content identity for safe Composer stream checkpoint continuation.

This supplements the training recipe fingerprint. It reads the manifest's
actual source assets and the prepared stream supervision without rebuilding
audio features or changing preparation cache keys.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from maichart.form.cache_support import source_digests
from maichart.training.dataset_catalog import _source_asset


SCHEMA = 'stream_content_identity_v1'


def _canonical(value):
    """Keep float and tensor values stable without relying on object reprs."""
    if hasattr(value, 'detach'):
        value = value.detach().cpu().tolist()
    elif hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f'Unsupported stream identity value: {type(value).__name__}')


def _append(hasher, field, value):
    record=json.dumps([field,_canonical(value)],ensure_ascii=False,sort_keys=True,
                      separators=(',',':')).encode('utf-8')
    hasher.update(len(record).to_bytes(8,'big'))
    hasher.update(record)


def _annotation_path(manifest, payload):
    """Follow FormDataset's local-first, split-source fallback selection."""
    path=manifest.with_name('quality_annotations.json')
    if not path.exists() and isinstance(payload.get('split_metadata'),dict):
        source=payload['split_metadata'].get('source_manifest')
        if source:
            from maichart.portable_paths import material_path
            path=material_path(source).with_name('quality_annotations.json')
    return path.resolve() if path.exists() else None


def dataset_content_identity(manifest, dataset, *, cache_dir='cache/chart_composer_v1'):
    """Return a schema-tagged SHA256 of sources and prepared supervision.

    ``dataset`` is an already prepared StreamDataset. Paths resolve exactly as
    FormDataset resolves manifest assets. Each unique asset is passed to the
    shared source-digest cache once, so repeated chart rows do not reread the
    same audio file. Missing referenced assets fail closed.
    """
    manifest=Path(manifest).resolve()
    payload=json.loads(manifest.read_text(encoding='utf-8-sig'))
    digests=source_digests(str(Path(cache_dir).resolve()))
    seen={}

    def digest(path):
        path=Path(path).resolve()
        if path not in seen:
            seen[path]=digests.digest(path)
        return seen[path]

    h=hashlib.sha256()
    _append(h,'schema',SCHEMA)
    _append(h,'manifest_sha256',digest(manifest))
    entries=payload.get('songs',[])
    if not isinstance(entries,list):
        raise ValueError('Stream manifest songs must be a list')
    assets=set()
    for index,entry in enumerate(entries):
        for role,key in (('maidata','maidata_path'),('audio','audio_path')):
            if key not in entry:
                raise ValueError(f'Stream manifest song {index} lacks {key}')
            source=_source_asset(entry[key],manifest,payload)
            assets.add(source)
            _append(h,'source',[index,role,digest(source)])
    annotation=_annotation_path(manifest,payload)
    _append(h,'quality_annotations',None if annotation is None else digest(annotation))

    streams=dataset.streams
    conditions=dataset.conditions
    chart_weights=dataset.chart_weights
    if not len(streams)==len(conditions)==len(chart_weights):
        raise ValueError('Prepared stream, condition and chart-weight counts differ')
    _append(h,'prepared_song_count',len(streams))
    for song,(stream,condition,chart_weight) in enumerate(zip(streams,conditions,chart_weights)):
        _append(h,'chart_weight',[song,chart_weight])
        _append(h,'condition',[song,condition])
        _append(h,'stream_length',[song,len(stream)])
        for step in stream:
            _append(h,'step',[step.kind,str(step.start),str(step.end),
                              step.start_second,step.end_second,step.text,step.weight])
    if hasattr(dataset,'target_window_weights'):
        _append(h,'target_window_weights',dataset.target_window_weights)
    return {'schema':SCHEMA,'fingerprint':h.hexdigest(),
            'manifest_sha256':digest(manifest),'source_file_count':len(assets),
            'prepared_song_count':len(streams),
            'quality_annotations_path':None if annotation is None else str(annotation)}
