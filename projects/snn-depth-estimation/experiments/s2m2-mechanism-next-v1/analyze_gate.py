"""Exact saved endpoint/stage contrasts for the tied-MRT intervention."""
import argparse,json
from pathlib import Path


def endpoint(path,update):
    text=path.read_text()
    if not text.endswith('\n'):text=text.rsplit('\n',1)[0]
    rows=[json.loads(s) for s in text.splitlines() if s.strip()]
    found=[r for r in rows if r['update']==update]
    if len(found)!=1:raise ValueError(f'{path}: exact endpoint missing or duplicated')
    return found[0]


def analyze(paths,update,ranking=None):
    rows={k:endpoint(p,update) for k,p in paths.items()}
    cohort=lambda r:{(f['scene'],str(f['frame']),f['valid_pixels']) for f in r['per_frame']}
    t=rows['two_pass'];support=cohort(t)
    if len(support)!=60 or len(t['scene_epe'])!=30:raise ValueError('Incomplete cohort')
    summaries={};deltas={}
    for label,row in rows.items():
        if cohort(row)!=support or row['scene_epe'].keys()!=t['scene_epe'].keys():raise ValueError('Mismatched official support')
        summaries[label]={k:row.get(k) for k in ('mean_scene_epe','initial_mean_scene_epe','global_mean_scene_epe','bad3')}
        deltas[label]={s:t['scene_epe'][s]-row['scene_epe'][s] for s in sorted(t['scene_epe'])}
    gap=sum(deltas['one_pass'].values())/30;wins=sum(v<0 for v in deltas['one_pass'].values())
    rank_available=False;rank_gain=None
    if ranking is not None:
        # Separate frozen checkpoint evidence must explicitly establish matched anchor identity.
        a,b=ranking['two_pass'],ranking['one_pass']
        if a['update']!=update or b['update']!=update:raise ValueError('Ranking checkpoint mismatch')
        if not a.get('support_identity') or a['support_identity']!=b.get('support_identity') or a['count']!=b['count']:raise ValueError('Ranking anchor support mismatch')
        rank_gain=b['raw_top1_epe']-a['raw_top1_epe'];rank_available=True
    if update==2500:passed=(rank_available and rank_gain>=.5 and gap<=-.25) or (gap<=-.5 and wins>=20)
    elif update==10000:passed=gap<=-.5 and wins>=20 and summaries['two_pass']['mean_scene_epe']<=4.388473
    elif update==30000:passed=sum(deltas['original'].values())<0 and sum(v<0 for v in deltas['original'].values())>=20 and t['bad3']<=rows['original']['bad3']
    else:raise ValueError('Unregistered gate endpoint')
    return dict(update=update,gate_pass=bool(passed),summaries=summaries,paired_scene_deltas=deltas,two_minus_one=gap,wins_vs_one_pass=wins,ranking_available=rank_available,raw_top1_gain=rank_gain,ranking_evidence=ranking,leaderboard_eligible=False,
                note='Absent checkpoint ranking evidence disables branch(a); official stage EPE and anchor metrics retain distinct support.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('two-pass','one-pass','original'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--ranking',type=Path);p.add_argument('--update',type=int,choices=(2500,10000,30000),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze({k:getattr(a,k) for k in ('two_pass','one_pass','original')},a.update,json.loads(a.ranking.read_text()) if a.ranking else None)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
