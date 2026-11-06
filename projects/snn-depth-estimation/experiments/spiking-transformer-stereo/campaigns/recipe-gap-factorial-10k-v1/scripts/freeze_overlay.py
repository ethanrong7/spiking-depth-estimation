"""Freeze the small campaign overlay and bind both immutable dependencies."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

CAMPAIGN = Path(__file__).resolve().parents[1]
REPO = next(p for p in CAMPAIGN.parents if (p / 'projects/AGENTS.md').exists())


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for arg in ('output', 'base-source', 'parent-source'):
        parser.add_argument('--' + arg, type=Path, required=True)
    args = parser.parse_args()
    registration = json.loads((CAMPAIGN / 'configs/registration.json').read_text())
    assert registration['max_simultaneous_gpus'] == 2
    assert len(registration['cases']) == len({c['id'] for c in registration['cases']}) == 12
    dependencies = {}
    for role in ('base', 'parent'):
        root = getattr(args, role + '_source')
        record = json.loads((root / 'source-registry.json').read_text())
        assert record['source_identity_sha256'] == registration[role + '_identity']
        assert sha((root / 'source.tar.gz').read_bytes()) == record['archive_sha256']
        with tarfile.open(root / 'source.tar.gz') as archive:
            for rel, expected in record['source_files'].items():
                assert sha(archive.extractfile(rel).read()) == expected, (role, rel)
        dependencies[role] = dict(snapshot=root.name, source_identity_sha256=record['source_identity_sha256'], archive_sha256=record['archive_sha256'])
    args.output.mkdir(parents=True, exist_ok=False)
    files = sorted(p for folder in ('scripts', 'configs', 'slurm', 'analysis') for p in (CAMPAIGN / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files += [CAMPAIGN / 'README.md', CAMPAIGN / 'results/astra-design-review.md']
    hashes = {}
    with tarfile.open(args.output / 'source.tar.gz', 'w:gz') as archive:
        for path in files:
            rel = str(path.relative_to(CAMPAIGN))
            payload = path.read_bytes(); hashes[rel] = sha(payload)
            entry = tarfile.TarInfo(rel); entry.size = len(payload); entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(payload))
    record = dict(campaign_id=registration['campaign_id'], git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        dirty_status=subprocess.check_output(['git', 'status', '--porcelain', '--', str(CAMPAIGN.relative_to(REPO))], cwd=REPO, text=True),
        source_files=hashes, source_identity_sha256=sha(json.dumps(hashes, sort_keys=True).encode()),
        archive_sha256=sha((args.output / 'source.tar.gz').read_bytes()), dependencies=dependencies)
    (args.output / 'source-registry.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'source_files'}, indent=2))


if __name__ == '__main__':
    main()
