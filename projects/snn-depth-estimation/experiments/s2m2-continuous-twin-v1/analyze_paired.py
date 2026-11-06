"""Audit saved E1 records against archived2736 controls; never synthesize endpoints."""
import argparse,json
from pathlib import Path


def records(path):
    text=path.read_text() if path.exists() else ''
    if not text.endswith('\n'):text=text.rsplit('\n',1)[0] if '\n' in text else ''
    return [json.loads(s) for s in text.splitlines() if s.strip()]


def analyze(twin,controls):
    spike=controls/'s2m2_v2b';original=controls/'original'
    meta=[json.loads((p/'metadata.json').read_text()) for p in (twin,spike)]
    identity=meta[0]['initial_state_sha256']==meta[1]['initial_state_sha256']
    trains=[{r['update']:r for r in records(p/'train.jsonl')} for p in (twin,spike)]
    common=sorted(trains[0].keys()&trains[1].keys())
    keys=('samples_seen','indices','augmentation_hv','lr_used','lr_next')
    mismatches=[dict(update=u,fields=[k for k in keys if trains[0][u].get(k)!=trains[1][u].get(k)]) for u in common if any(trains[0][u].get(k)!=trains[1][u].get(k) for k in keys)]
    evaluations=[{r['update']:r for r in records(p/'evaluations.jsonl')} for p in (twin,spike,original)]
    endpoints=[]
    for update in (2500,5000,7500,10000):
        if not all(update in e for e in evaluations):
            endpoints.append(dict(update=update,status='NOT_AVAILABLE'));continue
        rows=[e[update] for e in evaluations]
        frames=[{(r['scene'],str(r['frame']),r['valid_pixels']) for r in row['per_frame']} for row in rows]
        if not frames[0]==frames[1]==frames[2] or len(frames[0])!=60:raise ValueError('Official frame/valid-support mismatch')
        scenes=[r['scene_epe'] for r in rows]
        if not scenes[0].keys()==scenes[1].keys()==scenes[2].keys() or len(scenes[0])!=30:raise ValueError('Scene mismatch')
        deltas={s:dict(twin_minus_spike=scenes[0][s]-scenes[1][s],twin_minus_original=scenes[0][s]-scenes[2][s]) for s in sorted(scenes[0])}
        difference=sum(d['twin_minus_spike'] for d in deltas.values())/30
        win=sum(d['twin_minus_spike']<0 for d in deltas.values());loss=sum(d['twin_minus_spike']>0 for d in deltas.values())
        summaries={}
        for label,row in zip(('continuous','spiking','original'),rows):
            summaries[label]={k:row.get(k) for k in ('mean_scene_epe','initial_mean_scene_epe','global_mean_scene_epe','bad3')}
        endpoints.append(dict(update=update,status='SAVED_OFFICIAL',summaries=summaries,paired_scene_deltas=deltas,twin_minus_spike=difference,wins_vs_spike=win,losses_vs_spike=loss,material_difference=abs(difference)>=.5 and (win if difference<0 else loss)>=20))
    config_differences={k:[meta[0]['config'].get(k),meta[1]['config'].get(k)] for k in meta[0]['config'].keys()|meta[1]['config'].keys() if meta[0]['config'].get(k)!=meta[1]['config'].get(k)}
    return dict(config_differences=config_differences,order_metadata_equal=meta[0]['order']==meta[1]['order'],initial_state_equal=identity,initial_state_sha256=meta[0]['initial_state_sha256'],matched_training_updates=len(common),last_matched_update=max(common,default=0),train_record_fields=list(keys),train_record_mismatches=mismatches,metadata_config={label:m['config'] for label,m in zip(('continuous','spiking'),meta)},endpoints=endpoints,leaderboard_eligible=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--twin',type=Path,required=True);p.add_argument('--controls',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze(a.twin,a.controls);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('initial_state_equal','matched_training_updates','last_matched_update')},indent=2))
    for e in result['endpoints']:print(e['update'],e['status'],e.get('twin_minus_spike'),e.get('material_difference'))
