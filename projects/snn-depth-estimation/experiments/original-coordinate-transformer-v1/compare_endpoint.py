"""Compare exact recorded E4 endpoints; no nearest/best endpoint substitution."""
import argparse,json
from pathlib import Path


def endpoint(path,update):
    text=path.read_text()
    if not text.endswith('\n'):text=text.rsplit('\n',1)[0]
    found=[json.loads(line) for line in text.splitlines() if line.strip() and json.loads(line)['update']==update]
    if len(found)!=1:raise ValueError(f'{path}: expected exactly one saved endpoint {update}')
    row=found[0]
    support={(f['scene'],str(f['frame']),f['valid_pixels']) for f in row['per_frame']}
    if len(row['scene_epe'])!=30 or len(support)!=60:raise ValueError('Expected complete 30-scene / 60-frame official cohort')
    return row,support


def compare(paths,update):
    loaded={k:endpoint(p,update) for k,p in paths.items()}
    challenger,support=loaded['challenger']
    results={}
    for label,(row,cohort) in loaded.items():
        if cohort!=support or row['scene_epe'].keys()!=challenger['scene_epe'].keys():raise ValueError(f'{label}: cohort mismatch')
        deltas={s:challenger['scene_epe'][s]-row['scene_epe'][s] for s in sorted(row['scene_epe'])}
        results[label]=dict(mean_scene_epe=sum(row['scene_epe'].values())/30,bad3=row['bad3'],
                            initial_mean_scene_epe=row.get('initial_mean_scene_epe'),global_mean_scene_epe=row.get('global_mean_scene_epe'),
                            challenger_minus_control=sum(deltas.values())/30,challenger_wins=sum(d<0 for d in deltas.values()),paired_scene_deltas=deltas)
    c=results['challenger'];control=results['matched_control'];original=results['original']
    paired=control['challenger_minus_control']<=-.25 and control['challenger_wins']>=20
    if update==2500:passed=c['mean_scene_epe']<=4.450275 and paired
    elif update==10000:passed=c['mean_scene_epe']<=2.961262 and c['bad3']<=original['bad3']+1 and paired
    elif update==30000:passed=c['mean_scene_epe']<original['mean_scene_epe'] and c['bad3']<=original['bad3'] and original['challenger_wins']>=20 and control['challenger_minus_control']<0
    else:passed=None
    return dict(gate_pass=passed,update=update,comparisons=results,paths={k:str(p) for k,p in paths.items()},leaderboard_eligible=False,
                note='Exact official saved endpoint comparisons; Gate follows registered E4 STUDY.md. Prior lookup comparison only where identical endpoints exist.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('challenger','matched-control','original'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prior-lookup',type=Path)
    p.add_argument('--update',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    paths={k:getattr(a,k) for k in ('challenger','matched_control','original')}
    if a.prior_lookup:paths['prior_lookup']=a.prior_lookup
    result=compare(paths,a.update)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
