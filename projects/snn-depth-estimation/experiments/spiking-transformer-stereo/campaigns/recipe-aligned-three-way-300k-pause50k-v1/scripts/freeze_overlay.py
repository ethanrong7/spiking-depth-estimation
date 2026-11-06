"""Freeze the300k-stage overlay and bind the exact immutable runtime/recipe/core."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path
C=Path(__file__).resolve().parents[1];REPO=next(p for p in C.parents if (p/'projects/AGENTS.md').exists())
p=argparse.ArgumentParser();p.add_argument('--runtime-overlay',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);runtime=json.loads((a.runtime_overlay/'source-registry.json').read_text());registration=C/'configs/registration.json'
files=sorted(path for folder in ('scripts','configs','slurm','analysis') for path in (C/folder).rglob('*') if path.is_file() and '__pycache__' not in path.parts)+[C/'README.md'];hashes={}
with tarfile.open(a.output/'source.tar.gz','w:gz') as tar:
 for path in files:
  name=str(path.relative_to(C));payload=path.read_bytes();hashes[name]=hashlib.sha256(payload).hexdigest();info=tarfile.TarInfo(name);info.size=len(payload);info.mode=0o644;tar.addfile(info,io.BytesIO(payload))
registry=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),dirty_status=subprocess.check_output(['git','status','--porcelain','--',str(C.relative_to(REPO))],cwd=REPO,text=True),source_files=hashes,
 source_identity_sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),archive_sha256=hashlib.sha256((a.output/'source.tar.gz').read_bytes()).hexdigest(),registration_sha256=hashlib.sha256(registration.read_bytes()).hexdigest(),
 runtime_snapshot=a.runtime_overlay.name,runtime_source_identity_sha256=runtime['source_identity_sha256'],runtime_archive_sha256=runtime['archive_sha256'])
for label in ('parent','base'):
 for key in ('snapshot','source_identity_sha256','archive_sha256'):registry[label+'_'+key]=runtime[label+'_'+key]
(a.output/'source-registry.json').write_text(json.dumps(registry,indent=2)+'\n');print(json.dumps({k:v for k,v in registry.items() if k!='source_files'},indent=2))
