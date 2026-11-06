"""Descriptive paired-scene EPE intervals from an already completed result."""
import argparse
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    contrasts = {}
    for name, contrast in summary['contrasts'].items():
        values = list(contrast['per_scene_epe_delta'].values())
        rng = random.Random(20261104)
        estimates = sorted(sum(rng.choices(values, k=len(values))) / len(values)
                           for _ in range(10000))
        contrasts[name] = dict(mean_delta=sum(values)/len(values),
                              percentile_95_interval=[estimates[249], estimates[9749]],
                              scene_count=len(values))
    output = dict(method='paired scene resampling; 10000 bootstrap replicates; '
                  'percentile order statistics 250 and 9750; exploratory descriptive interval',
                  seed=20261104, contrasts=contrasts)
    args.summary.with_name('scene-bootstrap.json').write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
