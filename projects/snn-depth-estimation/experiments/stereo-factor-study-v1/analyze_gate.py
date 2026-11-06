"""Official scene-paired two-axis gates and optional measured efficiency/interaction."""
import argparse,json
from pathlib import Path


def endpoint(path,update):
    text=path.read_text()
    if not text.endswith('\n'):text=text.rsplit('\n',1)[0]
    rows=[json.loads(s) for s in text.splitlines() if s.strip()]
    rows=[r for r in rows if r['update']==update]
    if len(rows)!=1:raise ValueError(f'{path}: exact {update} endpoint missing/duplicated')
    return rows[0]


def analyze(paths,update,efficiency=None):
    rows={k:endpoint(p,update) for k,p in paths.items()}
    o=rows['O'];cohort=lambda r:{(f['scene'],str(f['frame']),f['valid_pixels']) for f in r['per_frame']}
    support=cohort(o)
    if len(support)!=60 or len(o['scene_epe'])!=30:raise ValueError('Incomplete Original cohort')
    original=sum(o['scene_epe'].values())/30
    results={}
    for arm,row in rows.items():
        if cohort(row)!=support or row['scene_epe'].keys()!=o['scene_epe'].keys():raise ValueError(f'{arm}: valid-support mismatch')
        epe=sum(row['scene_epe'].values())/30
        delta={s:row['scene_epe'][s]-o['scene_epe'][s] for s in sorted(o['scene_epe'])}
        wins=sum(v<0 for v in delta.values());accuracy=False;efficient=False
        if update==2500:accuracy=epe<=4.14779 and row['bad3']<=18.442 and wins>=15
        elif update==30000:
            accuracy=epe<=original*.95 and wins>=16 and row['bad3']<=o['bad3']+1
            if efficiency is not None and arm in efficiency:
                measure=efficiency[arm]
                required=('fixed_batch_resolution','isolated_gpu','baseline_inference_ms','arm_inference_ms','baseline_peak_memory_mib','arm_peak_memory_mib')
                if not all(k in measure for k in required) or not measure['fixed_batch_resolution'] or not measure['isolated_gpu']:raise ValueError('Efficiency evidence must use isolated GPU and fixed batch/resolution')
                improvement=(measure['arm_inference_ms']<=.8*measure['baseline_inference_ms'] or measure['arm_peak_memory_mib']<=.8*measure['baseline_peak_memory_mib'])
                efficient=epe<=original*1.02 and row['bad3']<=o['bad3']+1 and improvement
        else:raise ValueError('Only registered2500/30000 gates')
        results[arm]=dict(epe=epe,bad1=row.get('bad1'),bad3=row['bad3'],bad5=row.get('bad5'),wins_vs_original=wins,scene_minus_original=delta,accuracy_gate=bool(accuracy),efficiency_gate=bool(efficient),advance=bool(accuracy or efficient),efficiency_measured=efficiency is not None and arm in efficiency)
    results['O'].update(accuracy_gate=None,efficiency_gate=None,advance=None)
    interaction=None
    if all(k in rows for k in ('A','B','AB')):
        if update!=30000 or not all(results[k]['advance'] for k in ('A','B')):raise ValueError('AB permitted only after both singles qualify at30000')
        interaction={s:rows['AB']['scene_epe'][s]-rows['A']['scene_epe'][s]-rows['B']['scene_epe'][s]+o['scene_epe'][s] for s in sorted(o['scene_epe'])}
    return dict(update=update,arms=results,scene_interaction_AB_minus_A_minus_B_plus_O=interaction,leaderboard_eligible=False,note='Operational gates; paired seeds2345/3456 required before reproducibility claim. No missing efficiency measurements inferred.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--original',type=Path,required=True)
    for k in ('a','b','ab'):p.add_argument('--'+k,type=Path)
    p.add_argument('--update',type=int,choices=(2500,30000),required=True);p.add_argument('--efficiency',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    paths={'O':args.original};paths.update({k.upper():getattr(args,k) for k in ('a','b','ab') if getattr(args,k)})
    if len(paths)==1:p.error('Supply at least one measured single intervention')
    result=analyze(paths,args.update,json.loads(args.efficiency.read_text()) if args.efficiency else None)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
