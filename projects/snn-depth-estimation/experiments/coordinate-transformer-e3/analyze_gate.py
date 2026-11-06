"""Read exact saved endpoints and apply the preregistered E3 paired gates."""
import argparse
import json
from pathlib import Path


def endpoint(path, update):
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    matches = [r for r in rows if r['update'] == update]
    if len(matches) != 1:
        raise ValueError(f'{path}: need exactly one official evaluation at {update}')
    row = matches[0]
    if len(row['scene_epe']) != 30:
        raise ValueError('Gate requires the complete 30-scene cohort')
    frames = {(r['scene'], str(r['frame'])) for r in row['per_frame']}
    if len(frames) != 60:
        raise ValueError('Gate requires 60 distinct official frames')
    return row, frames


def analyze(challenger, archived, original, update):
    (t, tf), (o, of) = [endpoint(f, update) for f in (challenger, original)]
    p, pf = endpoint(archived, 2500)

    if not tf == pf == of or not t['scene_epe'].keys() == p['scene_epe'].keys() == o['scene_epe'].keys():
        raise ValueError('Official frame/scene cohorts differ')
    delta = {s: dict(challenger_minus_archived=t['scene_epe'][s]-p['scene_epe'][s],
                     challenger_minus_original=t['scene_epe'][s]-o['scene_epe'][s]) for s in sorted(t['scene_epe'])}
    wins_p = sum(d['challenger_minus_archived'] < 0 for d in delta.values())
    wins_o = sum(d['challenger_minus_original'] < 0 for d in delta.values())
    mean = lambda row: sum(row['scene_epe'].values())/30
    te, pe, oe = map(mean, (t, p, o))
    mixing = pe-te >= .5 and wins_p >= 20
    if update == 2500:
        passed = mixing and te <= 5.940870
        action = 'advance_to_10000' if passed else 'stop_pair'
    elif update == 10000:
        passed = te <= oe+.25 and t['bad3'] <= o['bad3']+1
        action = 'advance_to_30000' if passed else 'stop_pair'
    else:
        passed = te < oe and wins_o >= 20 and t['bad3'] <= o['bad3']
        action = 'replicate_with_paired_seeds_and_confirmation' if passed else 'no_superiority_gate'
    result = dict(archived_comparison_available=update==2500,update=update, decision=action, gate_pass=passed, gain_half_pixel_and_20_scenes=mixing,
                challenger_epe=te, archived_epe=pe, original_epe=oe,
                wins_vs_archived=wins_p, wins_vs_original=wins_o, paired_scene_deltas=delta,
                bad3_percent={k:r['bad3'] for k,r in [('challenger',t),('archived_e2_2500',p),('original',o)]},
                leaderboard_eligible=False, note='Official saved evaluations only; diagnostics do not replace rankings.')
    if update != 2500:
        result['archived_epe']=None
        result['wins_vs_archived']=None
        result['gain_half_pixel_and_20_scenes']=None
        for delta_row in result['paired_scene_deltas'].values():delta_row.pop('challenger_minus_archived')
        result['bad3_percent'].pop('archived_e2_2500',None)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('challenger','archived','original','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--update',type=int,choices=(2500,10000,30000),required=True)
    a=p.parse_args();result=analyze(a.challenger,a.archived,a.original,a.update)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
