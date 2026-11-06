"""Freeze full source and dirty overlay over the recorded Git base; no commit."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path
EXP=Path(__file__).resolve().parent
ROOT=EXP.parents[3]
PROJECT=EXP.parents[1]
def digest(value):return hashlib.sha256(value).hexdigest()
def freeze(output):
    output.mkdir(parents=True,exist_ok=False)
    git=lambda *args:subprocess.check_output(['git',*args],cwd=ROOT)
    base=git('rev-parse','HEAD').decode().strip()
    files=[ROOT/'AGENTS.md',ROOT/'projects/AGENTS.md',PROJECT/'AGENTS.md',PROJECT/'LICENSE',PROJECT/'UPSTREAM.md',PROJECT/'requirements.txt']
    for directory in [PROJECT/'core',EXP,EXP.parent/'spiking-attention-placement-v1',EXP.parent/'spiking-transformer-stereo/scripts',EXP.parent/'shared-gpu-30k-v1',PROJECT/'vendor']:
        files += [p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and 'results' not in p.relative_to(directory).parts and p.suffix not in ('.pt','.pth')]
    files += [EXP.parent/'spiking-transformer-stereo/configs/manifest.json']
    payloads={str(p.relative_to(ROOT)):p.read_bytes() for p in sorted(set(files))}
    # Use the exact native Original forward from the matched reference capsule.
    reference=ROOT/'storage/experiment-runs/snn-depth-estimation/spiking-transformer-stereo/source-shared-gpu-30k-v1/source.tar.gz'
    native='projects/snn-depth-estimation/core/spikestereonet.py'
    with tarfile.open(reference) as tar:payloads[native]=tar.extractfile(native).read()
    old={name:subprocess.run(['git','show',base+':'+name],cwd=ROOT,capture_output=True) for name in payloads}
    overlay={name:value for name,value in payloads.items() if old[name].returncode or old[name].stdout!=value}
    def archive(path,entries):
        with tarfile.open(path,'w:gz') as tar:
            for name,value in entries.items():
                info=tarfile.TarInfo(name);info.size=len(value);info.mode=0o644;tar.addfile(info,io.BytesIO(value))
    archive(output/'source.tar.gz',payloads);archive(output/'overlay.tar.gz',overlay)
    hashes={name:digest(value) for name,value in payloads.items()}
    registry=dict(source_override={native:'archived shared-gpu-30k-v1 native Original forward'},reference_archive_sha256=digest(reference.read_bytes()),git_commit=base,dirty_status=git('status','--porcelain').decode(),source_files=hashes,source_identity_sha256=digest(json.dumps(hashes,sort_keys=True).encode()),archive_sha256=digest((output/'source.tar.gz').read_bytes()),overlay_sha256=digest((output/'overlay.tar.gz').read_bytes()),base_file_sha256={name:digest(r.stdout) if r.returncode==0 else None for name,r in old.items()})
    (output/'source-registry.json').write_text(json.dumps(registry,indent=2)+'\n');print(json.dumps(dict(source_identity=registry['source_identity_sha256'],files=len(payloads),output=str(output))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);freeze(p.parse_args().output)
