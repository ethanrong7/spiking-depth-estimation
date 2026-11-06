"""Read-only log audit and compact diagnostic evidence; no model inference."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def lr(total, index):
    peak=.01*total-1
    if index<=peak:return 8e-6+(2e-4-8e-6)*index/peak
    return 2e-4+(8e-10-2e-4)*(index-peak)/(total-1-peak)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    reg=json.loads((a.source/'configs/registration.json').read_text())
    summaries=[];audits=[];results={};initial={};incomplete=[]
    for config in reg['cases']:
        directory=a.raw/'cases'/config['id']
        if not (directory/'result.json').exists():incomplete.append(config['id']);continue
        result=json.loads((directory/'result.json').read_text())
        meta=json.loads((directory/'metadata.json').read_text())
        rows=[json.loads(line) for line in (directory/'train.jsonl').read_text().splitlines()]
        assert config==meta['config']==result['config']
        assert [r['update'] for r in rows]==list(range(1,481))
        order=meta['order'];batch=config['batch'];horizon=config['schedule_total']
        assert sorted(order)==list(range(960))
        for u,r in enumerate(rows):
            assert r['indices']==[order[(u*batch+i)%960] for i in range(batch)]
            flags=[]
            for i in range(batch):
                h=hashlib.sha256(f'recipe-hv-v1:1234:{u*batch+i}'.encode()).digest()
                flags.append([h[0]<128,h[1]<128] if config['augmentation']=='hv' else [False,False])
            assert r['augmentation_hv']==flags
            assert r['samples_seen']==(u+1)*batch
            assert all(math.isfinite(r[k]) for k in ('loss','train_epe','gradient_norm','lr_used','lr_next'))
            assert r['gradient_norm']>0
            assert math.isclose(r['lr_used'],lr(horizon,u),rel_tol=1e-12,abs_tol=1e-16)
            assert math.isclose(r['lr_next'],lr(horizon,u+1),rel_tol=1e-12,abs_tol=1e-16)
        history=result['history'];assert [r['update'] for r in history]==[0,120,480]
        assert all(len(r['per_frame'])==60 and len(r['scene_epe'])==30 for r in history)
        assert result['optimizer_steps']==[480,480] and result['scheduler_last_epoch']==480
        assert result['finite_model_optimizer']
        arm=config['model'];initial.setdefault(arm,meta['initial_state_sha256'])
        assert initial[arm]==meta['initial_state_sha256']
        reference=a.raw.parent/'recipe-aligned-three-way-v1'/(arm.lower()+'-seed1234-10k-v1')/'initialization.json'
        assert reference.exists(), 'Missing saved parent initialization proof'
        assert meta['initial_state_sha256']==json.loads(reference.read_text())['complete_initial_state_sha256']
        final=history[-1];scenes=sorted(final['scene_epe'].values(),reverse=True)
        row=dict(case=config['id'],model=arm,batch=batch,augmentation=config['augmentation'],schedule_total=horizon,
            dev0=history[0]['mean_scene_epe'],dev120=history[1]['mean_scene_epe'],dev480=final['mean_scene_epe'],
            presentations=480*batch,bad1=final['bad1'],bad3=final['bad3'],bad5=final['bad5'],
            worst_five_mean=statistics.mean(scenes[:5]),last120_loss=result['last120_loss'],
            gradient_norm_median=statistics.median(r['gradient_norm'] for r in rows),
            clip_fraction=result['clip_fraction'],process_seconds=result['process_seconds'],
            mean_update_seconds=statistics.mean(r['seconds'] for r in rows),
            mean_wait_seconds=statistics.mean(r['consumer_wait_seconds'] for r in rows),
            peak_allocated_mib=max(r['peak_allocated_mib'] for r in rows))
        summaries.append(row);results[(arm,batch,config['augmentation'],horizon)]=result
        audits.append(dict(case=config['id'],updates=480,presentations=480*batch,
            config_order_HV_LR_finite_counts='PASS',initial_state_sha256=meta['initial_state_sha256'],
            result_sha256=hashlib.sha256((directory/'result.json').read_bytes()).hexdigest()))
    def e(model,batch,aug,total=10100,at=480):
        return next(r['mean_scene_epe'] for r in results[(model,batch,aug,total)]['history'] if r['update']==at)
    contrasts=[]
    for model in ('original','original_I','spiking_normalized'):
        if not all((model,b,g,h) in results for b,g,h in [(2,'none',10100),(2,'hv',10100),(8,'none',10100),(8,'hv',10100),(8,'hv',300100)]):continue
        baseline=e(model,2,'none');b=e(model,8,'none');aug=e(model,2,'hv');both=e(model,8,'hv')
        contrasts.append(dict(model=model,batch_effect_without_flips=b-baseline,
            flips_effect_batch2=aug-baseline,flips_effect_batch8=both-b,
            batch_effect_with_flips=both-aug,interaction=(both-b)-(aug-baseline),
            combined_recipe_effect=both-baseline,long_schedule_effect=e(model,8,'hv',300100)-both,
            equal960presentation_batch8_minus_batch2=e(model,8,'hv',at=120)-e(model,2,'hv')))
    evidence=dict(complete=not incomplete,incomplete=incomplete,cases=summaries,contrasts=contrasts,audits=audits,
        interpretation_boundary='One seed,480 updates; causal recipe contrasts for these trajectories, not explanation of final10k convergence or architecture ranking.')
    (a.output/'summary.json').write_text(json.dumps(evidence,indent=2)+'\n')
    for name,rows in [('cases.csv',summaries),('contrasts.csv',contrasts)]:
        if rows:
            with (a.output/name).open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(dict(complete=evidence['complete'],finished=len(summaries),pending=len(incomplete),cases=[{k:r[k] for k in ('case','dev480','gradient_norm_median')} for r in summaries],contrasts=contrasts),indent=2))


if __name__=='__main__':main()
