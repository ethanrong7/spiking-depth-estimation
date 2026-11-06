"""Read exact saved endpoints and apply the preregistered E2 paired gates."""
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


def analyze(transformer, pointwise, original, update):
    (t, tf), (p, pf), (o, of) = [endpoint(f, update) for f in (transformer, pointwise, original)]
    if not tf == pf == of or not t['scene_epe'].keys() == p['scene_epe'].keys() == o['scene_epe'].keys():
        raise ValueError('Official frame/scene cohorts differ')
    delta = {s: dict(transformer_minus_pointwise=t['scene_epe'][s]-p['scene_epe'][s],
                     transformer_minus_original=t['scene_epe'][s]-o['scene_epe'][s]) for s in sorted(t['scene_epe'])}
    wins_p = sum(d['transformer_minus_pointwise'] < 0 for d in delta.values())
    wins_o = sum(d['transformer_minus_original'] < 0 for d in delta.values())
    mean = lambda row: sum(row['scene_epe'].values())/30
    te, pe, oe = map(mean, (t, p, o))
    mixing = pe-te >= .25 and wins_p >= 20
    if update == 2500:
        passed = mixing or min(te, pe) <= oe+.5
        action = 'advance_both_unchanged_to_10000' if passed else 'stop_pair'
    elif update == 10000:
        passed = mixing and te <= oe+.25 and t['bad3'] <= o['bad3']+1
        action = 'advance_both_unchanged_to_30000' if passed else 'stop_pair'
    else:
        passed = te < min(oe, pe) and wins_o >= 20 and t['bad3'] <= o['bad3']
        action = 'replicate_with_paired_seeds_and_confirmation' if passed else 'no_superiority_gate'
    return dict(update=update, decision=action, gate_pass=passed, mixing_benefit=mixing,
                transformer_epe=te, pointwise_epe=pe, original_epe=oe,
                wins_vs_pointwise=wins_p, wins_vs_original=wins_o, paired_scene_deltas=delta,
                bad3_percent={k:r['bad3'] for k,r in [('transformer',t),('pointwise',p),('original',o)]},
                leaderboard_eligible=False, note='Official saved evaluations only; diagnostics do not replace rankings.')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('transformer','pointwise','original','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--update',type=int,choices=(2500,10000,30000),required=True)
    a=p.parse_args();result=analyze(a.transformer,a.pointwise,a.original,a.update)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
