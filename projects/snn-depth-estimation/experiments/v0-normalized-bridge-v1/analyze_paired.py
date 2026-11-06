"""Paired v0 package/control evidence from exact official saved endpoints."""
import argparse,json
from pathlib import Path


def endpoint(path,update):
    text=path.read_text()
    if not text.endswith('\n'):text=text.rsplit('\n',1)[0]
    rows=[json.loads(s) for s in text.splitlines() if s.strip()]
    found=[r for r in rows if r['update']==update]
    if len(found)!=1:raise ValueError(f'{path}: exact endpoint {update} unavailable or duplicated')
    return found[0]


def compare(paths,update):
    rows={k:endpoint(v,update) for k,v in paths.items()}
    original=rows['original'];cohort=lambda r:{(f['scene'],str(f['frame']),f['valid_pixels']) for f in r['per_frame']}
    support=cohort(original)
    if len(support)!=60 or len(original['scene_epe'])!=30:raise ValueError('Incomplete Original cohort')
    result={}
    for label,row in rows.items():
        if cohort(row)!=support or row['scene_epe'].keys()!=original['scene_epe'].keys():raise ValueError('Cohort/support mismatch')
        delta={s:row['scene_epe'][s]-original['scene_epe'][s] for s in sorted(original['scene_epe'])}
        result[label]=dict(mean_scene_epe=sum(row['scene_epe'].values())/30,bad3=row['bad3'],wins_vs_original=sum(v<0 for v in delta.values()),scene_minus_original=delta,
                           stage_metrics={k:v for k,v in row.items() if k.endswith('_mean_scene_epe')},
                           mechanism={k:row[k] for k in ('coordinate','coding','correspondence','diagnostics') if k in row})
    paired={s:rows['full']['scene_epe'][s]-rows['no_attention']['scene_epe'][s] for s in sorted(original['scene_epe'])}
    v0=result['full'];gate=None
    if update==30000:gate=v0['mean_scene_epe']<result['original']['mean_scene_epe'] and v0['wins_vs_original']>=20 and v0['bad3']<=result['original']['bad3']
    return dict(update=update,arms=result,v0_minus_no_attention=paired,v0_wins_vs_no_attention=sum(v<0 for v in paired.values()),package_gate_pass=gate,leaderboard_eligible=False,
                note='Earlier endpoints descriptive; missing mechanism fields mean unmeasured, not zero. Package and attention contrasts remain distinct.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('full','no-attention','original'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--update',type=int,choices=(2500,5000,10000,20000,30000),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=compare({k:getattr(a,k) for k in ('full','no_attention','original')},a.update)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
