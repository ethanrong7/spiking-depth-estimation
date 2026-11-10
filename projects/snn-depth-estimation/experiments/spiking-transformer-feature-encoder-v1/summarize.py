"""Export audited final endpoints, complete curves and paired scene changes."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
from run_on_npu import VARIANTS,SEEDS,EVALS

BASE=Path(__file__).resolve().parent.parent/'b-attention-mechanism-v1'
spec=importlib.util.spec_from_file_location('encoder_metric_audit',BASE/'summarize_ablation_npu.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
rows,table,checked_evaluation=base.rows,base.table,base.checked_evaluation


def audit_training(path,seed,metadata):
    data=rows(path/'train.jsonl')
    order=metadata['order'];config=metadata['config']
    assert sorted(order)==list(range(960))
    digest=hashlib.sha256()
    for update,r in enumerate(data,1):
        assert r['update']==update and r['samples_seen']==8*update
        assert math.isfinite(r['loss']) and math.isfinite(r['gradient_norm'])
        positions=range((update-1)*8,update*8)
        assert r['indices']==[order[p%960] for p in positions]
        flags=[]
        for p in positions:
            h=hashlib.sha256(f'recipe-hv-v1:{seed}:{p}'.encode()).digest()
            flags.append([h[0]<128,h[1]<128])
        assert r['augmentation_hv']==flags
        assert (r['physical_batch'],r['accumulation'],r['world_size'])==(config['physical_batch'],config['accumulation'],1)
        def lr(s):return 8e-6+(2e-4-8e-6)*s/299 if s<=299 else 2e-4+(8e-10-2e-4)*(s-299)/29700
        assert math.isclose(r['lr_used'],lr(update-1),rel_tol=1e-7,abs_tol=1e-12)
        assert math.isclose(r['lr_next'],lr(update),rel_tol=1e-7,abs_tol=1e-12)
        digest.update(json.dumps({k:r[k] for k in ('update','indices','augmentation_hv','lr_used')},sort_keys=True).encode())
    return data,digest.hexdigest()


def summarize(root):
    root=Path(root);dest=root/'summaries';dest.mkdir(parents=True,exist_ok=True)
    curves=[];endpoints=[];scenes=[];audits=[];values={}
    for seed in SEEDS:
        for variant in VARIANTS:
            path=root/f'seed-{seed}'/variant
            record=dict(seed=seed,variant=variant,status='pending')
            if not (path/'metadata.json').exists():
                audits.append(record);continue
            try:
                meta=json.loads((path/'metadata.json').read_text());config=meta['config']
                assert config['seed']==seed and config['variant']==variant
                training,identity=audit_training(path,seed,meta)
                record.update(updates=len(training),stream_identity=identity,shared_state=config['shared_state_sha256'],
                              source_identity=meta['source_registry']['source_identity_sha256'])
                evaluations=rows(path/'eval.jsonl')
                assert len({e['update'] for e in evaluations})==len(evaluations)
                cell_curves=[];cell_scenes=[]
                for e in evaluations:
                    assert e['update'] in EVALS and e['samples_seen']==e['update']*8
                    a,b,per_scene=checked_evaluation(e)
                    header=dict(study=root.name,backend='npu',seed=seed,variant=variant,update=e['update'],samples_seen=e['update']*8)
                    for metric,value,unit in [('mean_scene_epe',a,'px'),('pooled_bad3_percent',b,'percent')]:
                        cell_curves.append(dict(**header,metric=metric,value=value,unit=unit))
                    for scene,error in per_scene.items():
                        frames=[f for f in e['per_frame'] if f['scene']==scene]
                        count=sum(f['valid_pixels'] for f in frames)
                        bad=sum(f.get('bad3_pixels',f['bad3']*f['valid_pixels']/100) for f in frames)
                        row=dict(seed=seed,variant=variant,update=e['update'],scene_id=scene,frame_count=len(frames),
                                 mean_scene_epe=error,valid_pixels=count,bad3_pixels=bad)
                        cell_scenes.append(row)
                result=None
                if (path/'result.json').exists():
                    result=json.loads((path/'result.json').read_text())
                    assert result['status']=='COMPLETED' and result['updates']==len(training)==30000
                    assert result['finite_model_optimizer'] and result['scheduler_last_epoch']==30000
                    assert sorted(e['update'] for e in evaluations)==EVALS
                    record['status']='completed'
                else:record['status']='partial'
                for r in training:
                    if r['update']%25 and r['update']!=1:continue
                    header=dict(study=root.name,backend='npu',seed=seed,variant=variant,update=r['update'],samples_seen=r['samples_seen'])
                    for metric,key,unit,scale in [('train_loss','loss','px',1),('learning_rate','lr_used','',1),('step_seconds','seconds','seconds',1),('input_wait_seconds','consumer_wait_seconds','seconds',1),('peak_memory_bytes','peak_allocated_mib','bytes',2**20)]:
                        cell_curves.append(dict(**header,metric=metric,value=r[key]*scale,unit=unit))
                curves.extend(cell_curves);scenes.extend(cell_scenes)
                for row in cell_scenes:values[seed,variant,row['update'],row['scene_id']]=row
                if result:
                    e=next(e for e in evaluations if e['update']==30000);model=config['encoder']
                    endpoints.append(dict(seed=seed,variant=variant,update=30000,mean_scene_epe=e['mean_scene_epe'],
                        pooled_bad3_percent=e['bad3'],encoder_params=model['encoder_params'],readout_params=model['readout_params'],
                        total_params=model['total_params'],train_seconds=sum(r['update_wall_seconds'] for r in training),
                        inference_ms_per_pair=result.get('inference_ms_per_pair'),peak_memory_bytes=result['peak_allocated_mib']*2**20))
            except (AssertionError,KeyError,ValueError) as e:
                record.update(status='audit_failed',error=repr(e))
            audits.append(record)
    for seed in SEEDS:
        present=[a for a in audits if a['seed']==seed and 'shared_state' in a]
        if len({a['shared_state'] for a in present})>1 or len({a['source_identity'] for a in present})>1:
            for a in present:a.update(status='audit_failed',error='Shared solver initialization/source mismatch')
        completed=[a for a in present if a['status']=='completed']
        if len({a['stream_identity'] for a in completed})>1:
            for a in completed:a.update(status='audit_failed',error='Paired data streams differ')
    valid={(a['seed'],a['variant']) for a in audits if a['status']!='audit_failed'}
    curves=[r for r in curves if (r['seed'],r['variant']) in valid]
    endpoints=[r for r in endpoints if (r['seed'],r['variant']) in valid]
    scenes=[r for r in scenes if (r['seed'],r['variant']) in valid]
    deltas=[]
    for (seed,variant,update,scene),row in values.items():
        ref=values.get((seed,'swin',update,scene))
        if ref is None or variant=='swin' or (seed,variant) not in valid or (seed,'swin') not in valid:continue
        deltas.append(dict(seed=seed,variant=variant,reference='swin',update=update,scene_id=scene,
            epe_delta=row['mean_scene_epe']-ref['mean_scene_epe'],
            bad3_percentage_point_delta=100*(row['bad3_pixels']/row['valid_pixels']-ref['bad3_pixels']/ref['valid_pixels'])))
    table(dest/'curves.csv',['study','backend','seed','variant','update','samples_seen','metric','value','unit'],curves)
    table(dest/'endpoints.csv',['seed','variant','update','mean_scene_epe','pooled_bad3_percent','encoder_params','readout_params','total_params','train_seconds','inference_ms_per_pair','peak_memory_bytes'],endpoints)
    table(dest/'scenes.csv',['seed','variant','update','scene_id','frame_count','mean_scene_epe','valid_pixels','bad3_pixels'],scenes)
    table(dest/'paired-deltas.csv',['seed','variant','reference','update','scene_id','epe_delta','bad3_percentage_point_delta'],deltas)
    status='failed' if any(a['status']=='audit_failed' for a in audits) else ('completed' if all(a['status']=='completed' for a in audits) else 'partial')
    (dest/'completion-audit.json').write_text(json.dumps(dict(status=status,cells=audits),indent=2)+'\n')
    lines=['# Encoder results','',f'Status: {status}. Fixed 30k endpoints; lower is better.','',
           '| Seed | Encoder | EPE (px) | bad3 (%) | Scene wins vs Swin |','| --- | --- | ---: | ---: | ---: |']
    for r in endpoints:
        ds=[d for d in deltas if d['seed']==r['seed'] and d['variant']==r['variant'] and d['update']==30000]
        wins=f"{sum(d['epe_delta']<0 for d in ds)}/{len(ds)}" if ds else '—'
        lines.append(f"| {r['seed']} | {r['variant']} | {r['mean_scene_epe']:.6f} | {r['pooled_bad3_percent']:.4f} | {wins} |")
    lines+=['','Comparisons are encoder-package effects, not an isolated spike-activation ablation.','Historical Original is not a matched current-batch control.']
    (dest/'results.md').write_text('\n'.join(lines)+'\n')
    if status=='failed':raise RuntimeError(f'Audit failed: {dest}/completion-audit.json')
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,required=True)
    print(summarize(p.parse_args().output_root))
