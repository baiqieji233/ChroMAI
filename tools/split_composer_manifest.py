"""Create audio-disjoint train/dev/final manifests; fraction is total holdout."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from maichart.composer.evaluation import split_manifest

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest'); p.add_argument('output_dir')
    p.add_argument('--fraction', type=float, default=.05, help='total holdout group fraction, split between dev and final'); p.add_argument('--seed', type=int, default=0)
    a = p.parse_args(argv)
    print(json.dumps(split_manifest(a.manifest, a.output_dir, fraction=a.fraction, seed=a.seed), ensure_ascii=False, indent=2))
    return 0

if __name__ == '__main__': raise SystemExit(main())
