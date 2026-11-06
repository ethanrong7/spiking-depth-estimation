"""Freeze all planned historical best checkpoints before confirmation access."""
import argparse,hashlib,json,shlex,subprocess
from pathlib import Path

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def config_identity(meta):
    path=meta['command'][meta['command'].index('--config')+1]
    relative='projects/'+path.split('/projects/',1)[1]
    return {'config':meta['config'],'canonical_sha256':hashlib.sha256(json.dumps(meta['config'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),'source_relative_file':relative,'file_sha256':meta['source']['source_files'][relative]}

def freeze(queue,raw,manifest,remote_host,output,evaluation_config_dir):
    identity=subprocess.check_output(['ssh','-o','BatchMode=yes',remote_host,'id -un; hostname'],text=True).strip().splitlines()
    if identity!=['rongliang','cccl3']:raise RuntimeError('unexpected remote identity')
    plan=json.loads(queue.read_text());cohort=json.loads(manifest.read_text());manifest_sha=digest(manifest)
    assert len(plan['runs'])==10
    remote_root=Path(plan['runs'][0]['parent_checkpoint']).parent.parent
    paths=[str(remote_root/r['output']/'best.pt') for r in plan['runs']]
    command='sha256sum '+' '.join(shlex.quote(p) for p in paths)
    hashes={name:sha for sha,name in (line.split(None,1) for line in subprocess.check_output(['ssh','-o','BatchMode=yes',remote_host,command],text=True).splitlines())}
    entries=[]
    for row,path in zip(plan['runs'],paths):
        directory=raw/row['output'];result=json.loads((directory/'result.json').read_text());meta=json.loads((directory/'metadata.json').read_text());history=json.loads((directory/'history.json').read_text())
        for key,default in [('model',None),('seed',1234),('normalize_residual_current',False)]:
            if row[key]!=meta['config'].get(key,default):raise RuntimeError('queue/metadata configuration mismatch: '+row['output']+' '+key)
        if result['updates_completed']!=3000:raise RuntimeError('incomplete final run '+row['output'])
        if meta['manifest_sha256']!=manifest_sha:raise RuntimeError('manifest differs '+row['output'])
        best=min(history,key=lambda r:r['mean_scene_epe'])
        if best['update']!=result['best_update'] or abs(best['mean_scene_epe']-result['selected']['mean_scene_epe'])>1e-5:raise RuntimeError('historical selection mismatch '+row['output'])
        origin=meta;inherited=False
        inherited_path=directory/'inherited-best.json'
        if inherited_path.exists():
            record=json.loads(inherited_path.read_text())
            if record['sha256']==hashes[path]:
                inherited=True;origin=json.loads((raw/Path(record['checkpoint']).parent.name/'metadata.json').read_text())
        entries.append({'run':row['output'],'model':row['model'],'variant':row['variant'],'seed':row['seed'],'normalize_residual_current':row['normalize_residual_current'],'completed_updates':3000,'cumulative_samples_seen':6000,'selected_update':result['best_update'],'selected_checkpoint_samples_seen':result['best_update']*2,'checkpoint':path,'checkpoint_sha256':hashes[path],'checkpoint_source_identity_sha256':origin['source']['source_identity_sha256'],'run_source_identity_sha256':meta['source']['source_identity_sha256'],'source_override':origin['source'].get('source_override'),'checkpoint_config_identity':config_identity(origin),'run_config_identity':config_identity(meta),'parent_lineage':meta.get('parent_lineage'),'inherited_best':inherited,'manifest_sha256':manifest_sha,'prior_sha256':digest(directory/'priors.npz'),'prior_path':str(Path(path).parent/'priors.npz'),'api_evaluation_iters':32,'actual_refinements':None if row['model']=='ssde_inspired' else 32,'actual_inference_passes':1 if row['model']=='ssde_inspired' else None,'development':{k:result['selected'][k] for k in ['mean_scene_epe','pooled_epe','bad1','bad3','bad5','valid_pixels','valid_fraction']},'slurm_job_id':meta['slurm']['SLURM_JOB_ID'],'confirmation_output':str(Path(path).parent.parent/('confirmation-'+row['output']))})
    record={'schema_version':1,'status':'frozen_pending_root_review','selection':'historical minimum mean-scene development EPE within each of all ten registered final runs; no best-seed selection','manifest_sha256':manifest_sha,'confirmation_count':len(cohort['confirmation']),'confirmation_scenes':len({r['scene'] for r in cohort['confirmation']}),'preselected_case_ids':['scene_0150-0001','scene_0170-0001','scene_0189-0001'],'primary_api_iters':32,'inspired_actual_inference_passes':1,'queue_sha256':digest(queue),'runs':entries}
    assert len(entries)==10 and record['confirmation_count']==80
    evaluation_config_dir.mkdir(parents=True,exist_ok=True)
    for entry in entries:
        config={'stage':'evaluate','model':entry['model'],'seed':entry['seed'],'normalize_residual_current':entry['normalize_residual_current'],'updates':0,'schedule_updates':3000,'lr':.0002,'train_split':'train','eval_split':'confirmation','eval_iters':32,'physical_batch':2,'effective_batch':2,'train_refinements':16,'eval_refinements':32}
        name='confirmation-'+entry['variant']+'-seed'+str(entry['seed'])+'.json';path=evaluation_config_dir/name
        payload=json.dumps(config,indent=2)+'\n'
        if path.exists() and path.read_text()!=payload:raise RuntimeError('existing evaluation configuration differs: '+str(path))
        path.write_text(payload)
        entry['evaluation_config_identity']={'file':name,'config':config,'file_sha256':digest(path)}
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'path':str(output),'sha256':digest(output),'runs':len(entries)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--queue',type=Path,required=True);p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--remote-host',default='cccl-5090b');p.add_argument('--output',type=Path,required=True);p.add_argument('--evaluation-config-dir',type=Path,required=True);a=p.parse_args();freeze(a.queue,a.raw_root,a.manifest,a.remote_host,a.output,a.evaluation_config_dir)
