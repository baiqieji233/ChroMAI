"""Precompute audio-only composer metrical evidence for a chart manifest."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--cache-dir', required=True)
    p.add_argument('--backend', choices=('auto','native','beat_this'), default='auto')
    p.add_argument('--structure-policy',choices=('metrical_context_v1','metrical_context_v2'),default='metrical_context_v2')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--device', default='cpu')
    p.add_argument('--report', help='Write inspectable candidate segments as JSON lines')
    p.add_argument('--download-beat-this', action='store_true', help='Explicitly download the official final0 weights into the torch hub cache')
    a=p.parse_args(argv)
    if a.workers<1: p.error('--workers must be positive')
    os.environ['MAICHART_BEAT_THIS_DEVICE']=a.device
    import torch
    torch.set_num_threads(1)
    if a.download_beat_this:
        from beat_this.inference import load_checkpoint
        load_checkpoint('final0', device='cpu')
    from maichart.form.data import FormDataset
    from maichart.composer.structure import structure_fingerprint, structure_features, structure_segments, resolve_structure_backend
    a.backend = resolve_structure_backend(a.backend)
    ds=FormDataset(a.manifest, cache_dir=a.cache_dir, max_tokens=1_000_000,
                   max_region_seconds=999., preparation_workers=a.workers,
                   audio_profile='rational_audio_queries_v1', quality_profile='musical_evidence_v1',
                   preserve_excluded_context=True)
    songs=[record[0] for record in ds.songs]
    def prepare(song):
        try:
            fingerprint=structure_fingerprint(song,a.cache_dir,a.backend,policy=a.structure_policy)
            structure_features(song,[0.],[float(song.first)],cache_dir=a.cache_dir,backend=a.backend,policy=a.structure_policy)
            report = structure_segments(song,cache_dir=a.cache_dir,backend=a.backend,policy=a.structure_policy) if a.report else None
            return fingerprint, report, None
        except Exception as exc:
            return None,None,{'source':str(song.audio.source),'error':f'{type(exc).__name__}: {exc}'}
    report_file = open(a.report, 'w', encoding='utf8') if a.report else None
    try:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            failures=[]
            for i, (key, report, error) in enumerate(pool.map(prepare, songs), 1):
                if error:
                    failures.append(error)
                    print('structure_prepare_error '+json.dumps({'index':i,**error}),flush=True)
                    continue
                print('structure_prepare '+json.dumps({'index':i, 'total':len(songs), 'fingerprint':key, 'backend':a.backend,'structure_policy':a.structure_policy}), flush=True)
                if report_file:
                    report_file.write(json.dumps({'index':i,'fingerprint':key,'backend':a.backend,
                        'source':str(songs[i-1].audio.source),'segments':report}, ensure_ascii=False)+'\n')
    finally:
        if report_file: report_file.close()
    print('structure_prepare_summary '+json.dumps({'charts':len(songs),'completed':len(songs)-len(failures),
        'failed':len(failures),'errors':failures}),flush=True)
    return 1 if failures else 0

if __name__=='__main__':
    raise SystemExit(main())
