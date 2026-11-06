"""Freeze only the fresh 50k prefetched overlay and bind both immutable dependencies."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path
CAMPAIGN=Path(__file__).resolve().parents[1]
REPO=next(path for path in CAMPAIGN.parents if (path/'projects/AGENTS.md').exists())
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--parent-overlay',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
parent=json.loads((a.parent_overlay/'source-registry.json').read_text());registration=CAMPAIGN/'configs/registration.json'
files=sorted(path for folder in ('scripts','configs','slurm','analysis') for path in (CAMPAIGN/folder).rglob('*') if path.is_file() and '__pycache__' not in path.parts)+[CAMPAIGN/'README.md']
hashes={}
with tarfile.open(a.output/'source.tar.gz','w:gz') as archive:
 for path in files:
  name=str(path.relative_to(CAMPAIGN));payload=path.read_bytes();hashes[name]=hashlib.sha256(payload).hexdigest()
  info=tarfile.TarInfo(name);info.size=len(payload);info.mode=0o644;archive.addfile(info,io.BytesIO(payload))
registry=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
 dirty_status=subprocess.check_output(['git','status','--porcelain','--',str(CAMPAIGN.relative_to(REPO))],cwd=REPO,text=True),source_files=hashes,
 source_identity_sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),archive_sha256=hashlib.sha256((a.output/'source.tar.gz').read_bytes()).hexdigest(),
 registration_sha256=hashlib.sha256(registration.read_bytes()).hexdigest(),parent_snapshot=a.parent_overlay.name,parent_source_identity_sha256=parent['source_identity_sha256'],parent_archive_sha256=parent['archive_sha256'],
 base_snapshot=parent['base_snapshot'],base_source_identity_sha256=parent['base_source_identity_sha256'],base_archive_sha256=parent['base_archive_sha256'])
(a.output/'source-registry.json').write_text(json.dumps(registry,indent=2)+'\n');print(json.dumps({key:value for key,value in registry.items() if key!='source_files'},indent=2))
