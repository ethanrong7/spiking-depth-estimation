"""Exact Candidate C screen and paired final comparisons from saved evaluations."""
import argparse,json
from pathlib import Path


def endpoint(path,update):
    text=path.read_text()
    if not text.endswith('\n'):text=text.rsplit('\n',1)[0]
    rows=[json.loads(s) for s in text.splitlines() if s.strip()]
    found=[r for r in rows if r['update']==update]
    if len(found)!=1:raise ValueError(f'{path}: exact endpoint unavailable or duplicated')
    return found[0]


def analyze(paths,update):
    rows={k:endpoint(p,update) for k,p in paths.items()};c=rows['C']
    cohort=lambda r:{(f['scene'],str(f['frame']),f['valid_pixels']) for f in r['per_frame']}
    support=cohort(c)
    if len(support)!=60 or len(c['scene_epe'])!=30:raise ValueError('Incomplete official cohort')
    summaries={}
    ce=sum(c['scene_epe'].values())/30
    for label,row in rows.items():
        if cohort(row)!=support or row['scene_epe'].keys()!=c['scene_epe'].keys():raise ValueError('Cohort mismatch')
        delta={s:c['scene_epe'][s]-row['scene_epe'][s] for s in sorted(c['scene_epe'])}
        summaries[label]=dict(epe=sum(row['scene_epe'].values())/30,bad1=row.get('bad1'),bad3=row['bad3'],bad5=row.get('bad5'),c_minus_reference=sum(delta.values())/30,c_wins=sum(v<0 for v in delta.values()),scene_deltas=delta)
    if update==7500:
        decision='continue_same_state_to_30000' if ce<=6.229221447308858 else 'stop'
        reference_band='at_or_better_than_original' if ce<=2.974869481722514 else ('between_original_and_v2b' if ce<=6.229221447308858 else 'worse_than_v2b')
    elif update==30000:
        if not all(k in summaries for k in ('original','full','no_attention')):raise ValueError('Final requires all three controls')
        decision='train_positional_key_control' if ce<summaries['no_attention']['epe'] else 'report_no_attention_win_or_tie'
        reference_band=None
    else:raise ValueError('Unregistered gate endpoint')
    return dict(update=update,decision=decision,reference_band=reference_band,comparisons=summaries,leaderboard_eligible=False,note='No early gate or additional scene/bad3 hurdle at7500; no reproducibility claim from one seed.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True)
    for name in ('original','full','no-attention','v2b'):p.add_argument('--'+name,type=Path)
    p.add_argument('--update',type=int,choices=(7500,30000),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    paths={'C':a.candidate};paths.update({k:getattr(a,k) for k in ('original','full','no_attention','v2b') if getattr(a,k)})
    result=analyze(paths,a.update);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
