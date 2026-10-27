"""Benchmark warm structure queries on songs from a real training manifest."""
from __future__ import annotations

import argparse
import cProfile
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import statistics
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--backend', choices=('auto', 'native', 'beat_this'), default='native')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--profile', help='Optional cProfile output file')
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('--iterations must be positive')
    import numpy as np
    import torch
    from maichart.form.data import FormDataset
    from maichart.composer.structure import structure_features, structure_features_and_transition
    torch.set_num_threads(1)
    with redirect_stdout(sys.stderr):
        ds = FormDataset(args.manifest, cache_dir=args.cache_dir, max_tokens=1_000_000,
                         max_region_seconds=999., preparation_workers=1,
                         audio_profile='rational_audio_queries_v1', quality_profile='musical_evidence_v1',
                         preserve_excluded_context=True)
    songs = []
    seen = set()
    for record in ds.songs:
        song = record[0]
        source = str(song.audio.source)
        if source not in seen:
            songs.append(song)
            seen.add(source)
        if len(songs) == 3:
            break
    if len(songs) != 3:
        raise ValueError(f'manifest must yield at least three distinct songs; got {len(songs)}')
    kwargs = dict(policy='metrical_context_v1', cache_dir=args.cache_dir, backend=args.backend)
    beats = np.linspace(8., 16., 64)
    # Local beat/second conversion uses the song's actual tempo map.
    from maichart.composer.structure import _seconds_at_beats
    queries = [(song, _seconds_at_beats(beats, song), float(_seconds_at_beats(np.array([7.5]), song)[0])) for song in songs]
    def old(song, seconds, prev_sec):
        current = structure_features(song, beats, seconds, **kwargs)
        previous = structure_features(song, [7.5], [prev_sec], **kwargs)
        return current, current - previous
    def combined(song, seconds, prev_sec):
        return structure_features_and_transition(song, beats, seconds, 7.5, prev_sec, **kwargs)
    for query in queries:
        left, right = old(*query), combined(*query)
        for a, b in zip(left, right):
            np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
    samples = {'separate': [], 'combined': []}
    def measure():
        # Alternate order to reduce systematic drift between variants.
        for i in range(args.iterations):
            for name, fn in ((('separate', old), ('combined', combined)) if i % 2 == 0
                             else (('combined', combined), ('separate', old))):
                for query in queries:
                    start = time.perf_counter_ns()
                    fn(*query)
                    samples[name].append((time.perf_counter_ns() - start) / 1e6)
    if args.profile:
        profiler = cProfile.Profile()
        profiler.runcall(measure)
        profiler.dump_stats(args.profile)
    else:
        measure()
    result = {'manifest': str(Path(args.manifest).absolute()), 'cache_dir': str(Path(args.cache_dir).absolute()),
              'backend': args.backend, 'song_sources': list(seen), 'candidates': len(beats),
              'iterations_per_song': args.iterations, 'unit': 'ms per sample',
              'timings': {name: {'count': len(values), 'median': statistics.median(values),
                                 'p95': float(np.percentile(values, 95))}
                          for name, values in samples.items()}, 'profile': args.profile}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
