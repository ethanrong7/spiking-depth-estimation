"""Freeze new correspondence source and its exact historical recipe adapter."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[3]
RECIPE = 'projects/snn-depth-estimation/experiments/spiking-transformer-stereo/campaigns/recipe-aligned-three-way-v1/scripts/recipe.py'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def freeze(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Snapshot must be a new immutable directory')
    project = ROOT / 'projects/snn-depth-estimation'
    files = [ROOT / 'AGENTS.md', ROOT / 'projects/AGENTS.md', project / 'AGENTS.md',
             project / 'LICENSE', project / 'UPSTREAM.md', project / 'requirements.txt', ROOT / RECIPE, project / 'core/spiking_stereo_correspondence.py']
    # Compact source only. Raw results, data and caches never enter a snapshot.
    for directory in (EXPERIMENT,):
        files.extend(p for p in directory.rglob('*') if p.is_file()
            and not any(x in p.relative_to(directory).parts for x in ('results', '__pycache__', '.pytest_cache'))
            and p.suffix not in ('.pyc', '.png', '.pt', '.pth') and p.name != '.DS_Store')
    payloads = {str(p.relative_to(ROOT)): p.read_bytes() for p in sorted(set(files)) if p.is_file()}
    git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT)
    commit = git('rev-parse', 'HEAD').decode().strip()
    overlay, base_hashes = {}, {}
    for name, raw in payloads.items():
        before = subprocess.run(['git', 'show', f'{commit}:{name}'], cwd=ROOT, capture_output=True)
        base_hashes[name] = sha(before.stdout) if before.returncode == 0 else None
        if before.returncode or before.stdout != raw:
            overlay[name] = raw
    output.mkdir(parents=True, exist_ok=False)
    def archive(filename, entries):
        with tarfile.open(output / filename, 'w:gz') as tar:
            for name, raw in sorted(entries.items()):
                info = tarfile.TarInfo(name)
                info.mode, info.size, info.mtime = 0o444, len(raw), 0
                tar.addfile(info, io.BytesIO(raw))
    archive('source.tar.gz', payloads)
    archive('overlay.tar.gz', overlay)
    hashes = {name: sha(raw) for name, raw in payloads.items()}
    registry = {'schema': 'spiking-stereo-correspondence-source-v1', 'git_commit': commit,
        'dirty_status': git('status', '--porcelain').decode(), 'source_files': hashes,
        'base_file_sha256': base_hashes, 'overlay_files': {name: hashes[name] for name in overlay},
        'source_identity_sha256': sha(json.dumps(hashes, sort_keys=True).encode()),
        'archive_sha256': sha((output / 'source.tar.gz').read_bytes()),
        'overlay_sha256': sha((output / 'overlay.tar.gz').read_bytes()),
        'historical_recipe': {'member': RECIPE, 'sha256': hashes[RECIPE]},
        'policy': 'Exact Git base and full immutable source overlay; external historical capsule verified by recipe.load_frozen.'}
    (output / 'source-registry.json').write_text(json.dumps(registry, indent=2) + '\n')
    for p in output.iterdir():
        p.chmod(0o444)
    return {'snapshot': str(output), 'source_identity': registry['source_identity_sha256'], 'members': len(hashes)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(freeze(parser.parse_args().output)))
