"""Background, cached source-share estimates. No training or source writes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

POLICY = 'source_supervision_budget_v3'


def fingerprint(manifests):
    from .dataset_catalog import _absolutize_song_paths
    inputs = {}
    for value in manifests:
        manifest = Path(value).resolve()
        data = json.loads(manifest.read_text(encoding='utf-8-sig'))
        paths = [manifest, manifest.with_name('quality_annotations.json'), manifest.with_name('pool.json')]
        if data.get('candidate_pool') and paths[-1].exists():
            metadata = json.loads(paths[-1].read_text(encoding='utf-8-sig'))
            paths += [Path(metadata['source_manifest']), manifest.with_name('scan_state.json')]
        sidecar=manifest.with_name('quality_annotations.json')
        records=json.loads(sidecar.read_text(encoding='utf-8-sig')).get('records',[]) if sidecar.is_file() else []
        for record in records:
            # Unconfirmed candidate rows contribute zero without reading their
            # source/audio. Whole unannotated sources use preprocessing counts.
            if data.get('candidate_pool') and not record.get('regions'):continue
            paths += [Path(record[key]) for key in ('source_path','audio_path') if record.get(key)]
        if not data.get('candidate_pool'):
            for original in data.get('songs', []):
                if all(d.get('note_count') is not None for d in original.get('difficulties',[])):continue
                song=dict(original)
                if not Path(str(song.get('maidata_path',''))).is_absolute():
                    _absolutize_song_paths(song,manifest,data)
                if song.get('maidata_path'):paths.append(Path(song['maidata_path']))
        for path in paths:
            key=str(path)
            if key in inputs:continue
            try:
                stat=path.stat();inputs[key]=(stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
            except OSError:inputs[key]=None
    raw=json.dumps(dict(policy=POLICY,manifests=manifests,inputs=inputs),sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def estimate(manifests, cache_dir, progress=None):
    from .dataset_catalog import _write_json
    if progress:progress(dict(stage='checking',completed=0,total=0,source=''))
    key=fingerprint(manifests)
    path=Path(cache_dir)/(key+'.json')
    if path.is_file():
        result=json.loads(path.read_text(encoding='utf-8'))
        result['cache_hit']=True
        return result
    started=time.monotonic()
    from .candidate_pool import supervision_budget
    result=supervision_budget(manifests,progress=progress)
    if fingerprint(manifests)!=key:
        raise ValueError('资料或标注在估算期间发生变化，请重新估算')
    result.update(cache_hit=False,policy=POLICY,input_fingerprint=key,elapsed_sec=time.monotonic()-started)
    _write_json(path,result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--cache-dir',required=True)
    args=parser.parse_args(argv)
    values=json.loads(Path(args.request).read_text(encoding='utf-8'))['manifests']
    last=[float('-inf')]
    def emit(value):
        now=time.monotonic()
        if now-last[0]<.5 and value.get('completed')!=value.get('total'):return
        last[0]=now
        print('candidate_budget '+json.dumps(value,ensure_ascii=False),flush=True)
    result=estimate(values,args.cache_dir,emit)
    from .dataset_catalog import _write_json
    _write_json(Path(args.output),result)
    emit(dict(stage='complete',completed=1,total=1,source='',cache_hit=result.get('cache_hit',False)))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
