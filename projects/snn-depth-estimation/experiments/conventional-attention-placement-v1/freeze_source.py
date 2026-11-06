"""Inherit every historical scientific dependency, adding only this control."""
import argparse, hashlib, io, json, subprocess, tarfile
from pathlib import Path
EXP=Path(__file__).resolve().parent
ROOT=EXP.parents[3]
PROJECT=EXP.parents[1]
REFERENCE=ROOT/'storage/experiment-runs/snn-depth-estimation/spiking-transformer-stereo/source-spiking-attention-placement-v1-r2'
def sha(value): return hashlib.sha256(value).hexdigest()
def freeze(output):
    old=json.loads((REFERENCE/'source-registry.json').read_text())
    assert old['source_identity_sha256']=='606dcfe39ee26b7eecb4f52768376db76a80c5a4243929795cec2d233ba82f56'
    assert sha((REFERENCE/'source.tar.gz').read_bytes())==old['archive_sha256']
    with tarfile.open(REFERENCE/'source.tar.gz') as tar:
        payloads={m.name:tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    assert {n:sha(v) for n,v in payloads.items()}==old['source_files']
    files=[PROJECT/'core/conventional_attention_placement.py']+[p for p in EXP.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'results' not in p.relative_to(EXP).parts]
    additions={str(p.relative_to(ROOT)):p.read_bytes() for p in files}
    assert not set(additions)&set(payloads)
    payloads.update(additions);output.mkdir(parents=True,exist_ok=False)
    with tarfile.open(output/'source.tar.gz','w:gz') as tar:
        for name,value in sorted(payloads.items()):
            info=tarfile.TarInfo(name);info.size=len(value);info.mode=0o644;tar.addfile(info,io.BytesIO(value))
    git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT).decode().strip()
    hashes={n:sha(v) for n,v in payloads.items()}
    registry=dict(git_commit=old['git_commit'],control_plane_git_commit=git('rev-parse','HEAD'),dirty_status=git('status','--porcelain'),inherited_source_identity=old['source_identity_sha256'],inherited_archive_sha256=old['archive_sha256'],source_files=hashes,source_identity_sha256=sha(json.dumps(hashes,sort_keys=True).encode()),archive_sha256=sha((output/'source.tar.gz').read_bytes()),additions=sorted(additions))
    (output/'source-registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),source_identity=registry['source_identity_sha256'],files=len(hashes))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);freeze(p.parse_args().output)
