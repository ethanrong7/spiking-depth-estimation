"""Deploy a new immutable overlay, inspect readiness, or explicitly submit its 12 cases."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

REMOTE = r'''
import datetime, hashlib, json, os
from pathlib import Path
import subprocess, sys, tarfile
cfg, name, action = json.loads(sys.argv[1]), sys.argv[2], sys.argv[3]
assert subprocess.check_output(['id','-un'],text=True).strip() == cfg['expected_user']
assert subprocess.check_output(['hostname'],text=True).strip() == cfg['expected_hostname']
overlay = Path(cfg['source_root']) / name
root = Path(cfg['run_root']) / 'recipe-gap-factorial-10k-v1'
base = Path(cfg['source_root']) / 'source-alif-mechanism-screen-v1'
parent = Path(cfg['source_root']) / 'source-recipe-aligned-three-way-v1'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
source = json.loads((overlay/'source-registry.json').read_text())
assert sha(overlay/'source.tar.gz') == source['archive_sha256']
if action == 'extract':
    with tarfile.open(overlay/'source.tar.gz') as archive:
        assert all(not p.name.startswith('/') and '..' not in Path(p.name).parts and p.isfile() for p in archive.getmembers())
        archive.extractall(overlay, filter='data')
    action = 'verify'
for rel, expected in source['source_files'].items(): assert sha(overlay/rel) == expected, rel
reg = json.loads((overlay/'configs/registration.json').read_text())
assert reg['max_simultaneous_gpus'] == 2
assert len(reg['cases']) == len({c['id'] for c in reg['cases']}) == 12
for role, folder in [('base',base),('parent',parent)]:
    record = json.loads((folder/'source-registry.json').read_text())
    assert record['source_identity_sha256'] == reg[role+'_identity'] == source['dependencies'][role]['source_identity_sha256']
    assert sha(folder/'source.tar.gz') == record['archive_sha256'] == source['dependencies'][role]['archive_sha256']
    for rel, expected in record['source_files'].items(): assert sha(folder/rel) == expected, (role,rel)
assert Path(cfg['data_root']).is_dir() and (Path(cfg['environment'])/'bin/python').is_file()
queue = subprocess.check_output(['squeue','-h','-r','-o','%i|%u|%j|%T|%b|%l'],text=True)
assert not any('stereo-gap10k' in line for line in queue.splitlines()), 'Existing campaign allocation; inspect before submitting'
receipt = dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_identity_sha256=source['source_identity_sha256'],
    source_archive_sha256=source['archive_sha256'], git_commit=source['git_commit'], dirty_status=source['dirty_status'],
    dependencies=source['dependencies'], registration=reg, queue_before=queue,
    partition=subprocess.check_output(['scontrol','show','partition','debug'],text=True),
    node=subprocess.check_output(['scontrol','show','node',cfg['expected_hostname']],text=True),
    gpu_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total','--format=csv'],text=True),
    environment=subprocess.check_output([str(Path(cfg['environment'])/'bin/python'),'-c','import sys,torch; print(sys.version); print(torch.__version__); print(torch.version.cuda)'],text=True),
    source_root=str(overlay), run_root=str(root), resources=dict(partition='debug',gpus_per_case=1,max_simultaneous_gpus=2,cpus_per_case=8,mem_per_case='32G',time_per_case='08:00:00',array='0-11%2',requeue=False))
envs=dict(STEREO_OVERLAY=str(overlay),STEREO_PARENT_OVERLAY=str(parent),STEREO_BASE=str(base),STEREO_ENV=cfg['environment'],STEREO_DATA=cfg['data_root'],STEREO_RUN_ROOT=str(root))
command=['sbatch','--parsable','--chdir='+str(overlay),'--output='+str(root/'logs/%A_%a.out'),'--error='+str(root/'logs/%A_%a.err'),
    '--export=ALL,'+','.join(k+'='+v for k,v in envs.items()),str(overlay/'slurm/train.sbatch')]
receipt['command']=command
if action == 'submit':
    root.mkdir(exist_ok=False)
    (root/'logs').mkdir(); (root/'failures').mkdir()
    (root/'submission-intent.json').write_text(json.dumps(receipt,indent=2)+'\n')
    job=subprocess.check_output(command,text=True).strip()
    assert job.split(';')[0].isdigit(), job
    receipt['job_id']=job.split(';')[0]; receipt['status']='SUBMITTED'
    (root/'submission.json').write_text(json.dumps(receipt,indent=2)+'\n')
else:
    assert action == 'verify'
    assert not root.exists(), 'Campaign run root exists; no duplicate launch permitted'
    receipt['scheduler_test']=subprocess.check_output(command[:1]+['--test-only']+command[1:],stderr=subprocess.STDOUT,text=True)
    receipt['status']='VERIFIED_NOT_SUBMITTED'
print(json.dumps(receipt,indent=2))
'''


def remote(config, snapshot, action):
    command = 'python3 - ' + ' '.join(shlex.quote(x) for x in (json.dumps(config), snapshot.name, action))
    return subprocess.check_output(['ssh', config['ssh_host'], command], input=REMOTE, text=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster-config', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--action', choices=('deploy','verify','submit'), required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    config=json.loads(args.cluster_config.read_text())
    source=json.loads((args.snapshot/'source-registry.json').read_text())
    assert hashlib.sha256((args.snapshot/'source.tar.gz').read_bytes()).hexdigest() == source['archive_sha256']
    assert args.snapshot.name.startswith('source-recipe-gap-factorial-10k-v1')
    if args.action == 'deploy':
        destination=str(Path(config['source_root'])/args.snapshot.name)
        subprocess.run(['ssh',config['ssh_host'],'mkdir '+shlex.quote(destination)],check=True)
        subprocess.run(['scp',str(args.snapshot/'source.tar.gz'),str(args.snapshot/'source-registry.json'),config['ssh_host']+':'+destination+'/'],check=True)
        payload=remote(config,args.snapshot,'extract')
    else:
        payload=remote(config,args.snapshot,args.action)
    receipt=json.loads(payload)
    assert receipt['source_identity_sha256'] == source['source_identity_sha256']
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    with args.receipt.open('x') as stream: stream.write(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('status','source_identity_sha256','run_root')},indent=2))
    if 'job_id' in receipt: print('Slurm array '+receipt['job_id'])


if __name__ == '__main__':
    main()
