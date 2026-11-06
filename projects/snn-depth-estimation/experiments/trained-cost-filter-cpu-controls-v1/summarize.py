"""Describe the two fixed CPU cases; no seed/scene population inference."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary',type=Path);args=parser.parse_args()
    result=json.loads(args.summary.read_text())
    assert result['status']=='COMPLETED_CPU_DIAGNOSTIC' and result['scope'].startswith('separate CPU-only')
    frames=sorted({(r['scene'],r['frame']) for r in result['per_frame']})
    metrics=('epe','bad1','bad3','bad5')
    descriptions=[]
    for scene,frame in frames:
        rows={r['condition']:r for r in result['per_frame'] if (r['scene'],r['frame'])==(scene,frame)}
        assert set(rows)==set(result['conditions'])
        baseline=rows['baseline']
        assert baseline['CPU_baseline_full_tensor_exact_repeat']
        assert baseline['CPU_baseline_repeat_max_difference']==0
        for row in rows.values():
            assert row['target_support_sha256']==baseline['target_support_sha256'] and row['state_unchanged']
            for stratum in ('full','inview','oov'):
                assert row['metrics'][stratum]['valid_pixels']==baseline['metrics'][stratum]['valid_pixels']
        for name,row in rows.items():
            deltas={s:{m:row['metrics'][s][m]-baseline['metrics'][s][m] for m in metrics}
                    for s in ('full','inview','oov')}
            descriptions.append(dict(scene=scene,frame=frame,condition=name,metrics=row['metrics'],delta_from_CPU_baseline=deltas,
                inside_prospective_descriptive_band=abs(deltas['full']['epe'])<=.05 and abs(deltas['full']['bad3'])<=.5))
    averages={name:{m:sum(r['metrics']['full'][m] for r in descriptions if r['condition']==name)/len(frames)
                    for m in metrics} for name in result['conditions']}
    for name in averages:
        averages[name]['delta_epe']=averages[name]['epe']-averages['baseline']['epe']
        averages[name]['delta_bad3_pp']=averages[name]['bad3']-averages['baseline']['bad3']
    output=dict(scope='fixed two-case CPU within-backend diagnostic, not historical GPU reproduction or population inference',
        cases=len(frames),checkpoint_sha256=result['checkpoint_sha256'],conditions=averages,per_case=descriptions,
        cumulative_compute_seconds=result['cumulative_compute_seconds'],checkpoint_unchanged=result['checkpoint_unchanged'],
        state_unchanged=result['state_unchanged'],CPU_exact_repeat_both_cases=True,
        failed_GPU_admission_preserved='../native-first2-20261104/summary.json')
    destination=args.summary.with_name('contrasts.json');destination.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(averages,indent=2))


if __name__=='__main__':main()
