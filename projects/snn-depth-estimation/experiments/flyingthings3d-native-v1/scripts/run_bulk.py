"""Registered serial149-sequence conversion; immutable pilot reuse and atomic resume."""
import argparse,datetime,fcntl,json,os,shutil,subprocess,sys,time
from pathlib import Path
from bulk_contract import sha,verify_sequence,centers_for_split
from source_io import validate_index


def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.part');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def main():
 p=argparse.ArgumentParser()
 for n in ['registration','protocol','source-index','source-root','derived-root','ema-source','checkpoint','pilot-results','pilot-source-registry','source-registry','output']:p.add_argument('--'+n,required=True)
 p.add_argument('--resume',action='store_true');a=p.parse_args();started=time.monotonic();out=Path(a.output)
 if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('bulk requires one SlurmGPU allocation')
 out.parent.mkdir(parents=True,exist_ok=True);lock=(out.parent/'bulk-operation.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if out.exists() and not a.resume:raise RuntimeError('bulk output exists; inspect explicit resume, never blind rerun')
 out.mkdir(exist_ok=True);reg=json.loads(Path(a.registration).read_text());index=json.loads(Path(a.source_index).read_text());sequences=validate_index(index);source=json.loads(Path(a.source_registry).read_text());pilot_source=json.loads(Path(a.pilot_source_registry).read_text());pilot=Path(a.pilot_results);derived=Path(a.derived_root)
 if reg['status']!='registered' or reg['mode']!='bulk-conversion' or reg['max_allocated_gpus']!=1 or reg['new_sequences']!=149 or reg['bulk_time_limit_seconds']!=7200:raise RuntimeError('wrong bulk authorization')
 if reg['source_index_sha256']!=sha(a.source_index) or reg['protocol_sha256']!=sha(a.protocol):raise RuntimeError('sourceindex/protocol mismatch')
 for name,value in source['source_files'].items():
  if sha(Path(a.source_registry).parent/name)!=value:raise RuntimeError('bulk source changed:'+name)
 reuse=reg['reuse_pilot'];seq=next(s for s in sequences if s['source_id']==reuse['source_id']);pilot_result=json.loads((pilot/'result.json').read_text())
 if sha(pilot/'result.json')!=reuse['result_sha256'] or pilot_source['source_identity_sha256']!=reuse['source_identity_sha256'] or pilot_source['archive_sha256']!=reuse['archive_sha256'] or sha(derived/seq['source_id']/'manifest.json')!=reuse['sequence_manifest_sha256']:raise RuntimeError('audited pilot identity changed')
 verify_sequence(seq,derived,pilot_result,reg['source_index_sha256'])
 if reg['new_source_ids']!=[s['source_id'] for s in sequences if s['source_id']!=reuse['source_id']]:raise RuntimeError('bulk selection differs from frozen cohort/encounter order')
 status=dict(status='RUNNING',job_id=os.environ['SLURM_JOB_ID'],started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_sequences_completed=0,reused_pilot_sequences=1,total_selected_sequences=150,max_allocated_gpus=1,bulk_registration_sha256=sha(a.registration),bulk_source_identity_sha256=source['source_identity_sha256'],training_started=False)
 save(out/'status.json',status)
 for seq in sequences:
  sid=seq['source_id']
  if sid==reuse['source_id']:continue
  folder=out/'sequences'/sid;destination=derived/sid;staging=derived/(sid+'.conversion-staging')
  if destination.exists():
   resultpath=folder/'result.json'
   if not resultpath.exists():
    recover=destination/'conversion-result.json'
    if not recover.exists():raise RuntimeError('committed sequence lacks recoverable result')
    save(resultpath,json.loads(recover.read_text()))
   result=json.loads(resultpath.read_text());verify_sequence(seq,derived,result,reg['source_index_sha256'])
   if result['registration_sha256']!=sha(a.registration):raise RuntimeError('existing output belongs to another registration')
  else:
   if folder.exists() or staging.exists():
    if not a.resume:raise RuntimeError('partial sequence retained; explicit inspected resume required')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for old in [folder,staging]:
     if old.exists():
      quarantine=out/'quarantined-partials'/sid/(old.name+'.incomplete-'+stamp);quarantine.parent.mkdir(parents=True,exist_ok=True);old.rename(quarantine)
      with (out/'resume-quarantines.jsonl').open('a') as f:f.write(json.dumps(dict(source_id=sid,preserved=str(quarantine),original=str(old),time=stamp))+'\n')
   remaining=reg['bulk_time_limit_seconds']-(time.monotonic()-started)
   if remaining<60:raise TimeoutError('bulk wall budget exhausted; no automaticextension')
   logs=out/'logs';logs.mkdir(exist_ok=True);prefix=sid.replace('/','-')
   command=[sys.executable,str(Path(__file__).parent/'convert_sequence.py'),'--registration',a.registration,'--protocol',a.protocol,'--source-index',a.source_index,'--source-id',sid,'--source-root',a.source_root,'--derived-root',a.derived_root,'--ema-source',a.ema_source,'--checkpoint',a.checkpoint,'--output',str(folder)]
   status.update(current_source_id=sid,elapsed_seconds=time.monotonic()-started);save(out/'status.json',status)
   with (logs/(prefix+'.out')).open('w') as stdout,(logs/(prefix+'.err')).open('w') as stderr:
    child=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=min(reg['time_limit_seconds'],remaining-15))
   if child.returncode:raise RuntimeError('sequenceworkerfailed '+sid+' returncode'+str(child.returncode)+'; inspect retainedlogs, no automaticretry')
   result=json.loads((folder/'result.json').read_text());verify_sequence(seq,derived,result,reg['source_index_sha256'])
  status['new_sequences_completed']+=1;status.update(elapsed_seconds=time.monotonic()-started);save(out/'status.json',status);print('COMPLETE',sid,status['new_sequences_completed'],flush=True)
 rows={'train':[],'dev':[]};provenance={};total_dat=0
 for seq in sequences:
  reused=seq['source_id']==reuse['source_id'];folder=pilot if reused else out/'sequences'/seq['source_id'];result=json.loads((folder/'result.json').read_text());sequence_rows=verify_sequence(seq,derived,result,reg['source_index_sha256']);rows[seq['split']].extend(sequence_rows);total_dat+=len(result['windows'])
  provenance[seq['source_id']]=dict(reused_audited_pilot=reused,result_sha256=sha(folder/'result.json'),sequence_manifest_sha256=sha(derived/seq['source_id']/'manifest.json'),source_identity_sha256=pilot_source['source_identity_sha256'] if reused else source['source_identity_sha256'],registration_sha256=result['registration_sha256'])
 if len(rows['train'])!=960 or len(rows['dev'])!=60 or total_dat!=2040 or len({r['sample_id'] for r in rows['train']+rows['dev']})!=1020:raise RuntimeError('final dataset cardinality mismatch')
 manifest=dict(protocol_id='flyingthings3d-centered-native-v1',source_index_sha256=reg['source_index_sha256'],generation=dict(bulk_source_identity_sha256=source['source_identity_sha256'],bulk_registration_sha256=sha(a.registration),source_roots='runtime-configcanonicalFlyingThings3D/native-v1/source-dataandderived',no_Test=True,all450samples_per_view=True,checkpoint_sha256=reg['checkpoint_sha256']),sequence_provenance=provenance,**rows)
 encoded=json.dumps(manifest,indent=2,allow_nan=False)+'\n';manifestpath=derived/'manifest.json'
 if manifestpath.exists() and manifestpath.read_text()!=encoded:raise RuntimeError('existing finalmanifest differs; preserve andinspect')
 if not manifestpath.exists():save(manifestpath,manifest)
 status.update(status='COMPLETED_PENDING_INDEPENDENT_DATASET_AUDIT',new_sequences_completed=149,reused_pilot_sequences=1,train_samples=960,dev_samples=60,DAT_files=2040,total_spike_bytes=6609600000,final_manifest_sha256=sha(manifestpath),final_manifest_path=str(manifestpath),elapsed_seconds=time.monotonic()-started,training_started=False)
 save(out/'result.json',status);save(out/'status.json',status)

if __name__=='__main__':
 try:main()
 except Exception as exc:
  # Preserve the committed/partial data and all logs. No job retries here.
  try:
   if '--output' in sys.argv:
    path=Path(sys.argv[sys.argv.index('--output')+1]);save(path/'failure.json',dict(status='FAILED_STOPPED',error=str(exc),automatic_retry=False,training_started=False))
  finally:raise
