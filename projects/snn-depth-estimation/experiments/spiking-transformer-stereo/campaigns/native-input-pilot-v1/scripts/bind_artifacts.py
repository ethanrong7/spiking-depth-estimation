"""Bind reviewed data/geometry to external immutable registrations, never code edits."""
import argparse,copy,json
from pathlib import Path
from run_native import digest,validate_registration


def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['preflight','training'],required=True)
 for k in ['source-registry','manifest','data-audit','output']:p.add_argument('--'+k,required=True)
 p.add_argument('--geometry-receipt');p.add_argument('--template-root',default=str(Path(__file__).parents[1]/'configs'));a=p.parse_args();source=json.loads(Path(a.source_registry).read_text());audit=json.loads(Path(a.data_audit).read_text());sha=digest(a.manifest);out=Path(a.output)
 if out.exists() or out.with_name(out.name+'.binding-staging').exists():raise RuntimeError('binding alreadyexists/partial; inspect, do not overwrite')
 if audit['status']!='AUDIT_PASS' or audit['manifest_sha256']!=sha or (audit['train_samples'],audit['dev_samples'])!=(960,60):raise RuntimeError('fullpaireddata audit notpassed')
 if a.mode=='preflight':
  record=dict(status='registered',mode='native-memory-preflight',manifest_sha256=sha,data_audit_sha256=digest(a.data_audit),source_identity_sha256=source['source_identity_sha256'],code_archive_sha256=source['archive_sha256'],models=['original','raft_stereo','selective_raft'],seed=1234,max_allocated_account_gpus=1,optimizer_updates=0,train_refinements=16,eval_refinements=32,physical_batch=1,resident_accumulated_examples=2,optimizer_memory='FP32AdamW2moments_step0+oneparameterforeachbuffer+64MiB',candidate_geometries=['native','common_crop384x768_only_if_ANY_native_train_OOM'],fullnative_eval_required=True,extra_fallbacks=False)
  out.parent.mkdir(parents=True,exist_ok=True);save(out,record)
 else:
  if not a.geometry_receipt:raise RuntimeError('geometryreceipt required')
  geometry=json.loads(Path(a.geometry_receipt).read_text());templates=Path(a.template_root);reg=json.loads((templates/'registration.json').read_text());reg=copy.deepcopy(reg)
  if geometry['status']!='PASS' or geometry['source_identity_sha256']!=source['source_identity_sha256'] or geometry['manifest_sha256']!=sha:raise RuntimeError('preflight/code/data mismatch')
  selected=geometry['selected_training_geometry']
  for run in reg['runs']:
   config=run['config'];config['training_geometry']=selected;config['manifest_sha256']=sha
  reg.update(status='registered',manifest_sha256=sha,training_geometry=selected,source_identity_sha256=source['source_identity_sha256'],code_archive_sha256=source['archive_sha256'],data_audit_sha256=digest(a.data_audit),geometry_receipt_sha256=digest(a.geometry_receipt),preflight_registration_sha256=geometry['registration_sha256'],binding='Externalconfigsresolvedafterengineeringcheck;immutablecodepackageunchanged',input_cohort='960Train/60Devactualnativepaireddata')
  for run in reg['runs']:validate_registration(run['config'],reg,sha,audit,geometry,source['source_identity_sha256'])
  staging=out.with_name(out.name+'.binding-staging');staging.mkdir(parents=True)
  for run in reg['runs']:save(staging/run['config_file'],run['config'])
  save(staging/'registration.json',reg);staging.replace(out)
 print(json.dumps(dict(mode=a.mode,output=str(out),source_identity_sha256=source['source_identity_sha256'],manifest_sha256=sha),indent=2))
if __name__=='__main__':main()
