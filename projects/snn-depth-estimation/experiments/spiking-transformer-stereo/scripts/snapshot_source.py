"""Archive source, optionally retaining exact core files from an earlier snapshot."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--reuse-core-from',type=Path);p.add_argument('--exact-core',action='store_true',help='Exclude local core additions absent from the reused immutable core');a=p.parse_args()
if a.exact_core and not a.reuse_core_from:p.error('--exact-core requires --reuse-core-from')
a.output.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[5];project='projects/snn-depth-estimation'
def git(*args):return subprocess.check_output(['git',*args],cwd=root)
files=sorted(set(git('ls-files',project).decode().splitlines()+git('ls-files','--others','--exclude-standard',project).decode().splitlines()))
files=[f for f in files if (root/f).is_file()]
patch=git('diff','HEAD','--',project);(a.output/'dirty.patch').write_bytes(patch)
overrides={};parent=None
if a.reuse_core_from:
    parent=json.loads((a.reuse_core_from/'source-registry.json').read_text())
    with tarfile.open(a.reuse_core_from/'source.tar.gz') as old:
        for name,digest in parent['source_files'].items():
            if '/core/' not in name:continue
            payload=old.extractfile(name).read()
            if hashlib.sha256(payload).hexdigest()!=digest:raise RuntimeError('parent core identity mismatch '+name)
            overrides[name]=payload
    files=sorted(set(files)|set(overrides))
    if a.exact_core:files=[name for name in files if '/core/' not in name or name in overrides]
hashes={}
with tarfile.open(a.output/'source.tar.gz','w:gz') as tar:
    for name in files:
        payload=overrides.get(name)
        if payload is None:payload=(root/name).read_bytes()
        hashes[name]=hashlib.sha256(payload).hexdigest()
        info=tarfile.TarInfo(name);info.size=len(payload);info.mode=(root/name).stat().st_mode&0o777 if (root/name).exists() else 0o644
        tar.addfile(info,io.BytesIO(payload))
registry={'git_commit':git('rev-parse','HEAD').decode().strip(),'dirty_status':git('status','--porcelain',project).decode(),'dirty_patch_sha256':hashlib.sha256(patch).hexdigest(),'source_files':hashes,'source_identity_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),'archive_sha256':hashlib.sha256((a.output/'source.tar.gz').read_bytes()).hexdigest()}
if parent:registry['source_override']={'scope':'all core files from earlier immutable source; current driver/configs/docs retained','parent_source_identity_sha256':parent['source_identity_sha256'],'parent_archive_sha256':parent['archive_sha256'],'overridden_files':sorted(overrides)}
(a.output/'source-registry.json').write_text(json.dumps(registry,indent=2)+'\n');print(registry['source_identity_sha256'])
