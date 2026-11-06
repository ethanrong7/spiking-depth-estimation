"""Reproduce compact measurements for the predeclared32-update health gate."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,default=Path(__file__).resolve().parents[1])
    a=p.parse_args();results=a.study/'results'
    jobs=json.loads((results/'jobs.json').read_text());out=[]
    for seed in (9471,9472):
        for clip in (1,1000):
            run=f'health-A-spike-clip{clip}-seed{seed}-v1'
            job=next(j for j in jobs if j['run']==run)
            receipt=json.loads((results/run/'training-summary.json').read_text());rows=receipt['rows']
            if job['state']!='COMPLETED' or [r['update'] for r in rows]!=list(range(1,33)):
                raise ValueError('Four completed32-update receipts required: '+run)
            targets=[r['readout_optimizer_health']['targets']['decoder.residual.2.weight'] for r in rows]
            first=targets[0];tail=[t['raw_energy_weighted_denominator_attenuation'] for t in targets[24:32]]
            row=dict(run=run,job_id=job['job_id'],state=job['state'],updates=len(rows),seed=seed,clip=clip,
                initial_hash=receipt['metadata']['fresh_seed_model_sha256'],
                ratio_step1=first['raw_energy_weighted_denominator_attenuation'],
                ratio_median25_32=statistics.median(tail),
                adaptive_rms_step1=first['adaptive_delta_excluding_ideal_decay']['rms'],
                changed_energy_step1=first['raw_energy_on_changed_stored_coordinates'],
                all_moments_finite=all(r['optimizer_moments_all_finite'] for r in rows),
                all_target_metrics_available=all(t['available'] for t in targets),
                first_loss=rows[0]['loss'],last_loss=rows[-1]['loss'],
                last8_loss_ratio_to_initial=statistics.median(r['loss'] for r in rows[-8:])/rows[0]['loss'],
                all50sensor_gradients_finite=all(all(r['sensor_gradient'][v]['all_finite'] and r['sensor_gradient'][v]['bins']==50 for v in ('left','right')) for r in rows))
            row['predeclared_kernel_numerical_screen_pass']=(row['ratio_step1']>.1 and row['ratio_median25_32']>.1
                and row['adaptive_rms_step1']>0 and row['changed_energy_step1']>=.9
                and row['all_moments_finite'] and row['all_target_metrics_available'] and row['all50sensor_gradients_finite'])
            out.append(row)
    for seed in (9471,9472):
        if len({r['initial_hash'] for r in out if r['seed']==seed})!=1:
            raise ValueError('Paired clipping arms changed initialized model at seed'+str(seed))
    (results/'clipping-health-summary-v1.json').write_text(json.dumps(out,indent=2)+'\n')
    with (results/'clipping-health-summary-v1.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
