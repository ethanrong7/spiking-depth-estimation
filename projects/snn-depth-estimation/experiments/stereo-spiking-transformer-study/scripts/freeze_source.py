"""Freeze exact study source over a recorded Git base without creating a commit."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def freeze(output):
    root = Path(__file__).resolve().parents[5]
    project = root / 'projects/snn-depth-estimation'
    study = project / 'experiments/stereo-spiking-transformer-study'
    output.mkdir(parents=True, exist_ok=False)
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root)
    base = git('rev-parse', 'HEAD').decode().strip()
    files = [root / 'AGENTS.md', root / 'projects/AGENTS.md', project / 'AGENTS.md',
             project / 'LICENSE', project / 'UPSTREAM.md', project / 'requirements.txt']
    # These existing immutable adapters are reused, never rewritten by this study.
    prep = project / 'experiments/flyingthings3d-native-v1'
    files.extend([prep / 'scripts/ema_adapter.py', prep / 'scripts/source_io.py',
                  prep / 'scripts/range_archive.py', prep / 'scripts/download_selected.py',
                  prep / 'configs/protocol.json'])
    for directory in (project / 'core', project / 'tests', study):
        files.extend(p for p in directory.rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix != '.pyc'
                     and 'results' not in p.relative_to(directory).parts)
    payloads = {str(p.relative_to(root)): p.read_bytes() for p in sorted(set(files))}
    overlay = {}
    base_files = {}
    for name, payload in payloads.items():
        old = subprocess.run(['git', 'show', f'{base}:{name}'], cwd=root,
                             capture_output=True)
        base_files[name] = digest(old.stdout) if old.returncode == 0 else None
        if old.returncode or old.stdout != payload:
            overlay[name] = payload
    def archive(path, entries):
        with tarfile.open(path, 'w:gz') as tar:
            for name, payload in sorted(entries.items()):
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(payload))
    archive(output / 'source.tar.gz', payloads)
    archive(output / 'overlay.tar.gz', overlay)
    hashes = {name: digest(value) for name, value in payloads.items()}
    registry = {'schema': 1, 'git_commit': base,
                'dirty_status': git('status', '--porcelain').decode(),
                'source_files': hashes, 'base_file_sha256': base_files,
                'overlay_files': {name: hashes[name] for name in overlay},
                'source_identity_sha256': digest(json.dumps(hashes, sort_keys=True).encode()),
                'archive_sha256': digest((output / 'source.tar.gz').read_bytes()),
                'overlay_sha256': digest((output / 'overlay.tar.gz').read_bytes()),
                'policy': 'Exact Git base plus immutable full source and overlay; no commit created.'}
    (output / 'source-registry.json').write_text(json.dumps(registry, indent=2) + '\n')
    for path in output.iterdir():
        path.chmod(0o444)
    print(json.dumps({'snapshot': str(output), 'source_identity': registry['source_identity_sha256']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    freeze(parser.parse_args().output)
