"""Apply predeclared functionality margins to completed supplied pilot receipts."""
import csv
import hashlib
import json
from pathlib import Path


def main():
    study = Path(__file__).resolve().parents[1]
    results = study/'results'
    sources = {}
    def read(path):
        raw = path.read_bytes()
        sources[str(path.relative_to(study))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    queue = read(study/'configs/pilot-first-seed-queue-v1.json')['runs']
    prior = read(results/'pilot-train-gt-priors-v1/result.json')
    constant = next(r['mean_scene_epe'] for r in prior['scene_equal_summary'] if r['method']=='train_constant_median')
    rows = []
    for item in queue:
        path = results/('functionality-'+item['run'])/'result.json'
        if not path.exists():
            continue
        receipt = read(path)
        if receipt['status'] != 'COMPLETED_REGISTERED_DIAGNOSTIC_NO_SCIENTIFIC_PASS_THRESHOLD':
            continue
        if receipt['test_access'] is not False or receipt['config']['training_run'] != item['run']:
            raise ValueError('Receipt scope/run mismatch')
        values = {r['condition']:r for r in receipt['scene_equal_summary']}
        if any(values[k]['scenes']!=10 or values[k]['mean_scene_coverage']!=1 for k in ('correct','shuffled','masked')):
            raise ValueError('Primary ten-scene full-GT coverage changed')
        correct = values['correct']['mean_scene_epe']
        shuffled = values['shuffled']['mean_scene_epe']
        shift = receipt['shift_scene_response_summary']
        if shift['scene_count']!=10:
            raise ValueError('Incomplete registered shift population')
        constant_gain = (constant-correct)/constant
        right_gain = (shuffled-correct)/shuffled
        response = shift['median_scene_mean_response_ratio']
        row = dict(run=item['run'],sensor=item['sensor'],model='ALIF+I' if item['family']=='baseline' else item['architecture']+('_spiking' if item['spiking'] else '_continuous'),
            selected_update=receipt['checkpoint']['state']['update'],correct_scene_epe=correct,
            shuffled_scene_epe=shuffled,masked_scene_epe=values['masked']['mean_scene_epe'],
            train_constant_scene_epe=constant,constant_relative_improvement=constant_gain,
            correct_right_relative_improvement=right_gain,median_scene_mean_shift_response=response,
            positive_shift_scenes=shift['positive_mean_response_scenes'],
            constant_margin_pass=constant_gain>=.10,right_margin_pass=right_gain>=.05,
            shift_margin_pass=response is not None and .5<=response<=1.5 and shift['majority_scenes_mean_positive'],
            checkpoint_sha256=receipt['checkpoint']['checkpoint_sha256'])
        row['all_functionality_margins_pass']=all(row[k] for k in ('constant_margin_pass','right_margin_pass','shift_margin_pass'))
        rows.append(row)
    if not rows:
        raise ValueError('No completed registered checkpoint functionality receipts')
    output = results/'pilot-functionality-summary-v1'
    with output.with_suffix('.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    output.with_suffix('.json').write_text(json.dumps(dict(rows=rows,sources_sha256=sources,
        margins_source='analysis/pilot-budget-24gpu-hours-v1.md',registered_cells=len(queue),completed_cells=len(rows),
        interpretation='Preliminary functionality only; one seed, historical Val; no architectural ranking or final generalization claim'),indent=2)+'\n')
    print(json.dumps({'completed_cells':len(rows),'passing_cells':sum(r['all_functionality_margins_pass'] for r in rows)}))


if __name__=='__main__':
    main()
