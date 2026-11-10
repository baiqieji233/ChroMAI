"""Portable, explicitly approved regions with complete chart/audio context.

Packages never contain model weights, caches, or an automatically trainable
whole chart. Import publishes a new immutable masked source, not a live edit.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
import time
import zipfile

from maichart.training.dataset_catalog import _write_json
from maichart.training.quality_annotations import (
    _validate_regions, audio_sha256, chart_sha256, load_annotations,
    region_component_weights, save_annotations,
)

SCHEMA = 'maichart-reviewed-fragments-v1'
CHUNK = 1024 * 1024
MAX_BYTES = 20 * 1024**3
MAX_MANIFEST_BYTES = 64 * 1024**2


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _stat(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def positive_regions(regions):
    """Last review wins; a later exclusion must not resurrect an older praise."""
    if not isinstance(regions,list):raise ValueError('片段范围必须是数组')
    # Validate individual records, then partition overlap in review order.
    regions = [item for row in regions for item in _validate_regions([row])]
    bounds = sorted({float(row[key]) for row in regions for key in ('start_sec', 'end_sec')})
    result = []
    for start, end in zip(bounds, bounds[1:]):
        selected = next((row for row in reversed(regions) if row['start_sec'] <= start and end <= row['end_sec']), None)
        if not selected or selected.get('label') != 'good':continue
        row = copy.deepcopy(selected)
        timing, components = region_component_weights(row)
        if not (timing > 0 or components > 0):continue
        row.update(start_sec=start, end_sec=end)
        if result and result[-1]['end_sec'] == start and {k:v for k,v in result[-1].items() if k not in ('start_sec','end_sec')} == {k:v for k,v in row.items() if k not in ('start_sec','end_sec')}:
            result[-1]['end_sec'] = end
        else:result.append(row)
    return result


def _source_regions(regions):
    result=positive_regions(regions)
    for row in result:
        # Ordinary source praise boosts the named component but retains the
        # other component's baseline. That baseline is not an approval to export.
        if row.get('reason')=='timing':row['component_weight']=0.
        elif row.get('reason') in ('layout','comfort'):row['timing_weight']=0.;row.pop('entry_start_sec',None)
    return result


def source_examples(manifests, progress=None):
    from maichart.training.candidate_pool import _charts, candidate_record
    examples = []
    for number, manifest in enumerate(dict.fromkeys(manifests), 1):
        manifest = Path(manifest).resolve()
        data = _read(manifest)
        annotation_file = manifest.with_name('quality_annotations.json')
        records = load_annotations(manifest)['records']
        # Unmarked ordinary material is not an approval, even at weight 1.
        selected = {(str(Path(row.get('source_path','')).resolve()).casefold(), int(row.get('difficulty_index',-1))):row
                    for row in records if _source_regions(row.get('regions',[])) and float(row.get('weight',1)) > 0
                    and row.get('label') not in ('excluded','negative')}
        if not selected:continue
        before = _stat(manifest), _stat(annotation_file)
        pool = data.get('candidate_pool')
        metadata = _read(manifest.with_name('pool.json')) if pool else None
        if pool:
            state_path = manifest.with_name('scan_state.json')
            state = _read(state_path) if state_path.exists() else {}
            if state.get('status') in ('queued','running') or not metadata.get('scan_complete'):
                raise ValueError('候选库尚未完成检查，请完成扫描后再导出')
        for song, difficulty, index in _charts(manifest, data):
            record = selected.get((str(Path(song['maidata_path']).resolve()).casefold(), index))
            if not record:continue
            if pool and candidate_record(manifest, song, difficulty, [record], _metadata=metadata) is None:
                raise ValueError('候选片段已失效，请同步并重新确认后导出')
            if record.get('chart_sha256') != chart_sha256(song['maidata_path'], index):
                raise ValueError('已标注的源谱发生变化，请重新确认后导出')
            if not record.get('audio_sha256'):
                raise ValueError('旧标注缺少音频身份，请重新试听并保存片段后导出')
            examples.append(dict(path=song['maidata_path'],audio_path=song.get('audio_path'),
                difficulty_index=index,chart_sha256=record['chart_sha256'],audio_sha256=record.get('audio_sha256'),
                regions=_source_regions(record['regions']),weight=float(record.get('weight',1))))
        if before != (_stat(manifest), _stat(annotation_file)):
            raise ValueError('导出期间标注发生变化，请重新导出')
        if progress:progress(dict(stage='collecting',completed=number,total=len(manifests)))
    return examples


def generation_examples(candidates, root):
    from maichart.training.reviewed_generation import export_reviewed
    collection = export_reviewed(candidates, root, inspect_only=True)
    groups = {}
    for item in collection['items']:
        key = item['path'], item['audio_path'], item['difficulty_index']
        value = groups.setdefault(key, dict(path=item['path'],audio_path=item['audio_path'],
            chart_sha256=item['chart_sha256'],file_sha256=item['file_sha256'],audio_sha256=item['audio_sha256'],
            difficulty_index=item['difficulty_index'],weight=1.,regions=[]))
        value['regions'].append({k:v for k,v in item.items() if k in {
            'start_sec','end_sec','label','reason','weight','timing_weight','component_weight','entry_start_sec'}})
    return list(groups.values())


def _checked_example(example, hashes):
    from maichart.maidata import parse_maidata_file
    from maichart.event_timing import parse_event_chart
    from maichart.tempo_clock import TempoClock
    value = copy.deepcopy(example)
    path, audio = Path(value['path']), Path(value.get('audio_path') or '')
    if not path.is_file() or not audio.is_file():raise ValueError('片段包需要对应的完整谱面和音频')
    for asset, field in ((path,'file_sha256'),(audio,'audio_sha256')):
        key = str(asset.resolve()), _stat(asset)
        actual = hashes.setdefault(key, None)
        if actual is None:actual = hashes[key] = audio_sha256(asset)
        if value.get(field) and value[field] != actual:raise ValueError('谱面或音频已变化，请重新试听确认')
        value[field] = actual
    slot = int(value['difficulty_index'])
    if chart_sha256(path,slot) != value['chart_sha256']:raise ValueError('片段对应的谱面内容已变化')
    raw = parse_maidata_file(path)
    difficulty = next(d for d in raw.difficulties if d.index == slot)
    if not difficulty.level:raise ValueError('请先补齐谱面等级再导出')
    chart = parse_event_chart(path,slot,strict_notes=False)
    times = TempoClock(chart.tempo_changes,chart.audio_first).at_floats([float(p.beat) for p in chart.points])
    regions = positive_regions(value['regions'])
    if not any(r['start_sec'] <= t < r['end_sec'] for r in regions for t in times):
        raise ValueError('已确认范围内没有音符，请包含休止两端的音符后再收集')
    value.update(regions=regions,title=raw.title or path.parent.name,level=difficulty.level)
    return value


def export_bundle(request, output, progress=None):
    examples = (generation_examples(request['candidates'],Path(output).parent) if request.get('candidates')
                else source_examples(request.get('manifests',[]),progress))
    if not examples:raise ValueError('所选内容没有已确认的亮眼片段，请先圈选并确认学习内容')
    output = Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    hashes, assets, entries = {}, {}, {}
    for index, example in enumerate(examples,1):
        value = _checked_example(example,hashes)
        chart_name = 'charts/'+value['file_sha256']+'.txt'
        audio_name = 'audio/'+value['audio_sha256']+Path(value['audio_path']).suffix.lower()
        assets[chart_name] = (value['path'],value['file_sha256'])
        assets[audio_name] = (value['audio_path'],value['audio_sha256'])
        key = value['chart_sha256'],value['audio_sha256'],int(value['difficulty_index'])
        prior = entries.get(key,{})
        entries[key] = dict(chart=chart_name,audio=audio_name,chart_sha256=value['chart_sha256'],
            file_sha256=value['file_sha256'],audio_sha256=value['audio_sha256'],difficulty_index=value['difficulty_index'],
            title=value['title'],level=value['level'],weight=value.get('weight',1.),
            regions=positive_regions(prior.get('regions',[])+value['regions']))
        if progress:progress(dict(stage='collecting',completed=index,total=len(examples)))
    manifest = dict(schema=SCHEMA,entries=list(entries.values()))
    manifest['bundle_id'] = _digest(manifest['entries'])
    manifest_bytes=json.dumps(manifest,ensure_ascii=False).encode('utf-8')
    if len(manifest_bytes)>MAX_MANIFEST_BYTES:raise ValueError('单包标注过多，请按来源分别导出')
    if sum(_stat(path)[0] for path,_ in assets.values())+len(manifest_bytes)>MAX_BYTES:
        raise ValueError('单包超过 20 GiB，请按来源分别导出')
    temporary = output.with_suffix('.partial')
    try:
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
            for number,(name,(path,expected)) in enumerate(assets.items(),1):
                before=_stat(path);digest=hashlib.sha256()
                with open(path,'rb') as source,archive.open(name,'w',force_zip64=True) as destination:
                    while chunk:=source.read(CHUNK):destination.write(chunk);digest.update(chunk)
                if digest.hexdigest()!=expected or _stat(path)!=before:raise ValueError('导出期间文件变化，请重试')
                if progress:progress(dict(stage='packing',completed=number,total=len(assets)))
            archive.writestr('reviewed_fragments.json',manifest_bytes)
        if temporary.stat().st_size>MAX_BYTES:raise ValueError('单包超过 20 GiB，请按来源分别导出')
        os.replace(temporary,output)
    finally:temporary.unlink(missing_ok=True)
    return dict(kind='export',path=str(output),charts=len(entries),regions=sum(len(v['regions']) for v in entries.values()),
                bytes=output.stat().st_size,bundle_id=manifest['bundle_id'])


def _member(name):
    if not isinstance(name,str) or '\\' in name or ':' in name:raise ValueError('片段包包含无效文件路径')
    path=PurePosixPath(name)
    if not path.parts or path.is_absolute() or '..' in path.parts or str(path)!=name:raise ValueError('片段包包含越界文件路径')
    if path.parts[0] not in ('charts','audio') or len(path.parts)!=2:raise ValueError('片段包资源路径无效')
    return name


def _metadata(archive):
    names=archive.namelist()
    if len(names)!=len(set(names)):raise ValueError('片段包包含重复文件名')
    if sum(item.file_size for item in archive.infolist())>MAX_BYTES:raise ValueError('解包后超过 20 GiB，请拆分导出')
    info=archive.getinfo('reviewed_fragments.json')
    if info.file_size>MAX_MANIFEST_BYTES:raise ValueError('片段清单过大，请按来源分别导出')
    data=json.loads(archive.read(info))
    if data.get('schema')!=SCHEMA or not isinstance(data.get('entries'),list) or not data['entries']:
        raise ValueError('请选择“导出已确认片段”生成的 ZIP 包')
    assets={}
    for entry in data['entries']:
        regions=positive_regions(entry['regions'])
        if regions!=entry['regions'] or not regions:raise ValueError('片段包必须只包含明确确认的学习范围')
        slot=entry['difficulty_index']
        if isinstance(slot,bool) or not isinstance(slot,int) or slot<0:raise ValueError('难度槽位无效')
        if not math.isfinite(float(entry.get('weight',1))) or not 0<float(entry.get('weight',1))<=4:raise ValueError('片段学习权重无效')
        for key in ('chart_sha256','file_sha256','audio_sha256'):
            value=entry[key]
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('片段包文件校验值无效')
        for key,hash_key in (('chart','file_sha256'),('audio','audio_sha256')):
            name=_member(entry[key]);archive.getinfo(name)
            suffix=PurePosixPath(name).suffix.lower()
            if key=='chart' and name!='charts/'+entry['file_sha256']+'.txt':raise ValueError('谱面资源名与校验值不匹配')
            if key=='audio' and name!='audio/'+entry['audio_sha256']+suffix:
                raise ValueError('音频资源名与校验值不匹配')
            if name in assets and assets[name]!=entry[hash_key]:raise ValueError('同一资源的校验值不一致')
            assets[name]=entry[hash_key]
    if data.get('bundle_id')!=_digest(data['entries']):raise ValueError('片段清单校验失败')
    return data,assets


def inspect_bundle(path,progress=None):
    path=Path(path).resolve();before=_stat(path)
    with zipfile.ZipFile(path) as archive:
        metadata,assets=_metadata(archive)
        for number,(name,expected) in enumerate(assets.items(),1):
            digest=hashlib.sha256()
            with archive.open(name) as source:
                while chunk:=source.read(CHUNK):digest.update(chunk)
            if digest.hexdigest()!=expected:raise ValueError('片段包文件损坏：'+name)
            if progress:progress(dict(stage='verifying',completed=number,total=len(assets)))
    if _stat(path)!=before:raise ValueError('校验期间片段包发生变化，请重新导入')
    return dict(kind='preview',archive=str(path),archive_stat=list(before),bundle_id=metadata['bundle_id'],
        charts=len(metadata['entries']),regions=sum(len(e['regions']) for e in metadata['entries']),
        items=[dict(title=e.get('title',''),level=e.get('level',''),difficulty_index=e['difficulty_index'],regions=e['regions']) for e in metadata['entries']])


def import_bundle(preview, workspace_root, progress=None):
    path=Path(preview['archive']).resolve()
    if list(_stat(path))!=preview['archive_stat']:raise ValueError('片段包已变化，请重新预览')
    root=Path(workspace_root).resolve()/'reviewed_imports';root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        metadata,assets=_metadata(archive)
        if metadata['bundle_id']!=preview['bundle_id']:raise ValueError('片段包已变化，请重新预览')
        for number,(name,expected) in enumerate(assets.items(),1):
            destination=root/name;destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.is_file():
                if audio_sha256(destination)!=expected:raise ValueError('已导入资源发生变化，请保留原文件并重新核对')
            else:
                fd,temp=tempfile.mkstemp(dir=destination.parent,suffix='.partial');digest=hashlib.sha256()
                try:
                    with os.fdopen(fd,'wb') as target,archive.open(name) as source:
                        while chunk:=source.read(CHUNK):target.write(chunk);digest.update(chunk)
                    if digest.hexdigest()!=expected:raise ValueError('片段包资源校验失败')
                    os.replace(temp,destination)
                finally:Path(temp).unlink(missing_ok=True)
            if progress:progress(dict(stage='importing',completed=number,total=len(assets)))
    # One logical import collection. Re-importing identical material cannot
    # multiply its training weight; newer overlapping approvals replace older.
    from maichart.training.reviewed_generation import library_file_lock
    with library_file_lock(root/'imports.lock'):
        return _publish_import(metadata,preview,root,workspace_root)


def _publish_import(metadata,preview,root,workspace_root):
    index_path=root/'index.json';index=_read(index_path) if index_path.exists() else dict(bundles={})
    index['bundles'][metadata['bundle_id']]=metadata['entries']
    merged={}
    for entries in index['bundles'].values():
        for entry in entries:
            key=entry['chart_sha256'],entry['audio_sha256'],entry['difficulty_index']
            old=merged.get(key,{})
            merged[key]={**entry,'regions':positive_regions(old.get('regions',[])+entry['regions'])}
    identity=_digest(list(merged.values()));manifest=root/'collections'/identity/'training_manifest.json'
    songs=[];records=[];checked_hashes={}
    for number,entry in enumerate(merged.values()):
        example=_checked_example(dict(path=str(root/entry['chart']),audio_path=str(root/entry['audio']),
            difficulty_index=entry['difficulty_index'],chart_sha256=entry['chart_sha256'],file_sha256=entry['file_sha256'],
            audio_sha256=entry['audio_sha256'],regions=entry['regions'],weight=entry.get('weight',1.)),checked_hashes)
        song_id='imported_'+entry['chart_sha256'][:20]+'_'+entry['audio_sha256'][:12]
        songs.append(dict(song_id=song_id,title=example['title'],maidata_path=example['path'],audio_path=example['audio_path'],
            reviewed_only=True,difficulties=[dict(difficulty_index=entry['difficulty_index'],level_raw=example['level'],
            event_usable_for_training=True,usable_for_training=True)]))
        records.append(dict(song_id=song_id,source_path=example['path'],audio_path=example['audio_path'],
            chart_sha256=entry['chart_sha256'],audio_sha256=entry['audio_sha256'],difficulty_index=entry['difficulty_index'],
            label='normal',weight=entry.get('weight',1.),unmarked_weight=0.,regions=entry['regions']))
    if not manifest.exists():
        save_annotations(manifest,records)
        _write_json(manifest,dict(schema='maichart-training-manifest-v1',source_root=str(root),songs=songs,
            preprocessing=dict(target_schema='event_v1'),reviewed_training=dict(policy='approved_regions_only_v1',unmarked='context_only')))
    from maichart.training.reviewed_generation import update_library
    collection=dict(manifest=str(manifest),collection_id=identity,charts=len(songs),regions=sum(len(r['regions']) for r in records))
    _write_json(index_path,index)
    result=update_library(collection,'imported-approved-fragments',Path(workspace_root)/'reviewed_training')
    return dict(result,kind='import',bundle_id=metadata['bundle_id'],imported_charts=preview['charts'],imported_regions=preview['regions'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('export','preview','import'))
    parser.add_argument('--request',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--workspace-root',required=True)
    args=parser.parse_args(argv);request=_read(args.request);last=[0.]
    def progress(value):
        now=time.monotonic()
        if now-last[0]>=.5 or value.get('completed')==value.get('total'):
            print('review_bundle '+json.dumps(value,ensure_ascii=False),flush=True);last[0]=now
    try:
        if args.operation=='export':result=export_bundle(request,Path(args.output).with_name('已确认片段.zip'),progress)
        elif args.operation=='preview':result=inspect_bundle(request['path'],progress)
        else:result=import_bundle(request,args.workspace_root,progress)
    except (ValueError,KeyError,TypeError,OSError,zipfile.BadZipFile) as exc:
        print('片段包处理失败：'+str(exc),flush=True)
        raise SystemExit(1) from None
    _write_json(Path(args.output),result)


if __name__=='__main__':main()
