"""Read completed pilot checkpoint metadata on the cluster; no model execution."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex

from cluster_control import remote


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cluster-config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    study=Path(__file__).resolve().parents[1];repository=study.parents[3]
    if args.output.exists() or not any(args.output.resolve().is_relative_to(repository/name) for name in ('storage','scratch')):
        raise ValueError('Fresh checkpoint receipt under ignored storage/ or scratch/ required')
    cluster=json.loads(args.cluster_config.read_text())
    if remote(cluster,'id -un; hostname').splitlines()!=[cluster['expected_user'],cluster['expected_hostname']]:
        raise RuntimeError('Verified remote identity mismatch')
    queue=json.loads((study/'configs/pilot-first-seed-queue-v1.json').read_text())['runs']
    payload=dict(root=cluster['run_root'].rstrip('/')+'/stereo-spiking-transformer-study',
                 runs=[row['run'] for row in queue])
    code='''import hashlib,json,pathlib,sys,torch
p=json.loads(PAYLOAD);root=pathlib.Path(p['root']);completed=[];pending=[]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
for run in p['runs']:
 out=root/run;status_path=out/'status.json'
 if not status_path.exists():pending.append(dict(run=run,status='NOT_STARTED'));continue
 status=json.loads(status_path.read_text())
 if status.get('status')!='COMPLETED_FROZEN_HORIZON' or status.get('update')!=1000:
  pending.append(dict(run=run,status=status.get('status'),update=status.get('update')));continue
 paths={name:out/filename for name,filename in [('checkpoint','best-val.pt'),('metadata','metadata.json'),('selection_validation_log','validation.jsonl')]}
 hashes={name:sha(path) for name,path in paths.items()}
 metadata=json.loads(paths['metadata'].read_text())
 checkpoint=torch.load(paths['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
 state=checkpoint['state'];binding=checkpoint['binding']
 assert metadata['binding']==binding,'metadata/checkpoint binding differs'
 assert binding['total_updates']==1000 and binding['effective_batch']==8,'frozen exposure differs'
 assert state['update']==state['best_update'] and state['update'] in (250,500,1000),'invalid trained selection'
 visits=[json.loads(line) for line in paths['selection_validation_log'].read_text().splitlines() if line.strip()]
 trained=[row for row in visits if row['update']>0]
 assert [row['update'] for row in trained]==[250,500,1000],'incomplete final selection schedule'
 best=min(trained,key=lambda row:row['mean_scene_epe'])
 assert best['update']==state['update'] and best['mean_scene_epe']==state['best_val_epe'],'checkpoint is not final best trained Val'
 assert all(sha(paths[name])==value for name,value in hashes.items()),'checkpoint metadata changed during receipt collection'
 completed.append(dict(run=run,status=status,checkpoint_state=state,checkpoint_binding=binding,
  checkpoint_logs=checkpoint['logs'],checkpoint_sha256=hashes['checkpoint'],metadata_sha256=hashes['metadata'],metadata=metadata,
  selection_validation_log_sha256=hashes['selection_validation_log'],selection_validation_log_bytes=paths['selection_validation_log'].stat().st_size))
 del checkpoint
print(json.dumps(dict(completed=completed,pending=pending,python=sys.version,torch=torch.__version__)))
'''.replace('PAYLOAD',repr(json.dumps(payload)))
    command=shlex.quote(cluster['environment']+'/bin/python')+' -c '+shlex.quote(code)
    result=json.loads(remote(cluster,command))
    result.update(schema='stereo-study-checkpoint-receipts-v1',
        collected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='read-only checkpoint/state/log metadata; no model construction, sensor/GT arrays or inference')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(output=str(args.output),completed=[dict(run=row['run'],update=row['checkpoint_state']['update'],
        epe=row['checkpoint_state']['best_val_epe']) for row in result['completed']],pending=result['pending'])))


if __name__=='__main__':main()
