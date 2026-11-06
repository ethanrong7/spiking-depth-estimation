#!/usr/bin/env python3
"""Freeze only the T50 model, exact-loss dependencies and this experiment."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

PROJECT = Path(__file__).resolve().parents[3]
REPO = PROJECT.parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    launcher = PROJECT / 'experiments/sew-t50-training/slurm/train.sbatch'
    if '#SBATCH --time=05:00:00' not in launcher.read_text().splitlines():
        raise ValueError('this experiment requires an explicit five-hour walltime limit')
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    source = destination / 'projects/snn-depth-estimation'
    names = ['core/' + name for name in ['__init__.py', 'ALIF.py', 'corr.py',
        'extractor.py', 'loss.py', 'sew_encoder.py', 'spikestereonet.py', 'update.py',
        'utils/__init__.py', 'utils/reader.py', 'utils/utils.py']]
    names += ['experiments/sew-t50-feasibility/scripts/' + name
              for name in ['run_feasibility.py', 'exact_batch_loss.py']]
    names += ['experiments/sew-t50-memory-stability/scripts/' + name
              for name in ['run_memory_stability.py', 'training_lifecycle.py']]
    names += ['tests/test_sew_t50_training_state.py']
    for path in (PROJECT / 'experiments/sew-t50-training').rglob('*'):
        if path.is_file() and 'results' not in path.relative_to(PROJECT).parts:
            names.append(str(path.relative_to(PROJECT)))
    digests = {}
    for name in sorted(set(names)):
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / name, target)
        digests[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    provenance = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        dirty_status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True),
        upstream_commit='8c5e87750f25a035bca41bd37ae44b042a405f9a',
        source_files=digests)
    patch = subprocess.check_output(['git', 'diff', '--binary', 'HEAD', '--',
        *[str((PROJECT / name).relative_to(REPO)) for name in names]], cwd=REPO)
    (destination / 'dirty.patch').write_bytes(patch)
    provenance['dirty_patch_sha256'] = hashlib.sha256(patch).hexdigest()
    (destination / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    archive = destination.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(destination, arcname=destination.name)
    print(json.dumps(dict(source=str(destination), archive=str(archive),
        source_files=len(digests), archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
