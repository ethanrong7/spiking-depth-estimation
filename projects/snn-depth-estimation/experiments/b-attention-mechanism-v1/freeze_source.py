"""Package this standalone checkout; optionally overlay an old CUDA capsule."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[3]
PROJECT = EXP.parents[1]
REFERENCE_ID = '17f577a9bf16f6cface49e9134be085f0aac4301958c97212b5b5c9783430c0f'


def sha(value):
    return hashlib.sha256(value).hexdigest()


def freeze_checkout(output):
    """Self-contained fresh-study source, without datasets or old run stores."""
    git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    names = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT
    ).decode().split('\0')
    excluded = {'outputs', 'results', 'storage', 'scratch', 'logs', 'checkpoints',
                'datasets', '.venv', '__pycache__', '.git'}
    payloads = {}
    for name in sorted(set(filter(None, names))):
        path = ROOT / name
        if excluded.intersection(Path(name).parts) or not path.is_file():
            continue
        if path.suffix not in {'.py', '.md', '.json', '.yaml', '.yml', '.txt',
                               '.toml', '.cfg', '.ini', '.sh', '.sbatch'} and path.name not in {'LICENSE', '.gitignore'}:
            continue
        payloads[name] = path.read_bytes()
    hashes = {name: sha(value) for name, value in payloads.items()}
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(output / 'source.tar.gz', 'w:gz') as archive:
        for name, value in sorted(payloads.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(value), 0o644
            archive.addfile(info, io.BytesIO(value))
    registry = dict(
        study_mode='fresh_standalone_npu_port', historical_equivalence=False,
        git_commit=git('rev-parse', 'HEAD'), dirty_status=git('status', '--porcelain'),
        source_files=hashes, source_identity_sha256=sha(json.dumps(hashes, sort_keys=True).encode()),
        archive_sha256=sha((output / 'source.tar.gz').read_bytes()))
    (output / 'source-registry.json').write_text(json.dumps(registry, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), source_identity=registry['source_identity_sha256'],
                          mode=registry['study_mode'], files=len(hashes))))


def freeze(reference, output):
    old = json.loads((reference / 'source-registry.json').read_text())
    assert old['source_identity_sha256'] == REFERENCE_ID
    assert sha((reference / 'source.tar.gz').read_bytes()) == old['archive_sha256']
    with tarfile.open(reference / 'source.tar.gz') as archive:
        payloads = {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
    assert {n: sha(v) for n, v in payloads.items()} == old['source_files']
    files = [PROJECT / 'core/attention_encoder_controls.py',
             EXP.parent / 'spiking-attention-placement-phase2/protocol.py',
             EXP.parent / 'conventional-attention-placement-v1/worker.py',
             EXP.parent / 'conventional-attention-placement-v1/run_on_gpu.py']
    files += [p for p in EXP.rglob('*') if p.is_file()
              and not {'__pycache__', 'results', 'outputs'}.intersection(p.relative_to(EXP).parts)]
    overlays = {str(p.relative_to(ROOT)): p.read_bytes() for p in files}
    allowed_replacements = {str((EXP.parent / 'conventional-attention-placement-v1' / n).relative_to(ROOT))
                            for n in ('worker.py', 'run_on_gpu.py')}
    replacements = set(overlays) & set(payloads)
    assert replacements == allowed_replacements, replacements
    payloads.update(overlays)
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(output / 'source.tar.gz', 'w:gz') as archive:
        for name, value in sorted(payloads.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(value), 0o644
            archive.addfile(info, io.BytesIO(value))
    hashes = {n: sha(v) for n, v in payloads.items()}
    git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    registry = dict(
        git_commit=old['git_commit'], control_plane_git_commit=git('rev-parse', 'HEAD'),
        dirty_status=git('status', '--porcelain'),
        inherited_source_identity=REFERENCE_ID, inherited_archive_sha256=old['archive_sha256'],
        source_files=hashes, source_identity_sha256=sha(json.dumps(hashes, sort_keys=True).encode()),
        archive_sha256=sha((output / 'source.tar.gz').read_bytes()),
        additions=sorted(set(overlays) - replacements),
        replacements={n: dict(before=old['source_files'][n], after=hashes[n]) for n in sorted(replacements)})
    (output / 'source-registry.json').write_text(json.dumps(registry, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), source_identity=registry['source_identity_sha256'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path,
                        help='Optional legacy CUDA overlay mode; not needed for the fresh NPU study')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.reference:
        freeze(args.reference, args.output)
    else:
        freeze_checkout(args.output)
