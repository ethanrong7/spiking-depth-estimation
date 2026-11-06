"""Audit saved first-panel arithmetic without decoding data or running a model."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    inventory = json.loads(args.inventory.read_text())
    config = json.loads(args.config.read_text())
    member = inventory['files']['report.json']
    assert sha(args.report) == member['sha256']
    assert args.report.stat().st_size == member['bytes']
    assert inventory['exact_file_set_verified'] and inventory['remote_before_after_equal']
    assert report['status'] == 'COMPLETED_FIXED_FIVE_FAMILY100_PREDICTION_DEVELOPMENT_PANEL'
    assert report['config_sha256'] == sha(args.config) and report['config'] == config
    assert report['sample_ids'] == config['dsec']['sample_ids']
    assert report['gt_target_attempt_ids'] == report['sample_ids']
    assert report['gt_targets_attempted'] == report['gt_targets_opened'] == 20
    assert report['predictions_completed'] == 100
    for key in ('test_access', 'synthetic_observations_opened', 'synthetic_GT_opened',
                'adaptation', 'confidence_filter'):
        assert report[key] is False, key
    assert [m['case_id'] for m in report['models']] == config['models']
    for model, case in zip(report['models'], config['cases']):
        assert model['predictions'] == 20 and model['weights_and_buffers_unchanged']
        assert model['tensor_state_sha256_before'] == model['tensor_state_sha256_after']
        assert model['overrides'] == {}
        assert model['provenance']['checkpoint_sha256'] == case['checkpoint']['path_sha256']
        assert model['inference_model_config'] == model['provenance']['model']
    metrics = ('epe', 'mae', 'rmse', 'one_pe', 'two_pe', 'coverage')
    sums = ('valid_pixels', 'total_pixels', 'absolute_error_sum', 'squared_error_sum',
            'bad1_pixels', 'bad2_pixels', 'nonpositive_prediction_pixels')
    methods = config['models'] + ['zero_disparity']
    frames = report['frame_metrics']
    assert len(frames) == 120
    reference_support = None
    sequences = []
    def close(a, b):
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)
    for method in methods:
        rows = [r for r in frames if r['method'] == method]
        assert [r['sample_id'] for r in rows] == report['sample_ids']
        support = [(r['sample_id'], r['valid_pixels'], r['total_pixels']) for r in rows]
        if reference_support is None:
            reference_support = support
        assert support == reference_support
        for scene in config['scenes']:
            selected = [r for r in rows if r['scene_id'] == scene]
            assert len(selected) == 10
            totals = {key: math.fsum(r[key] for r in selected) for key in sums}
            n = totals['valid_pixels']
            computed = dict(epe=totals['absolute_error_sum']/n,
                            mae=totals['absolute_error_sum']/n,
                            rmse=math.sqrt(totals['squared_error_sum']/n),
                            one_pe=totals['bad1_pixels']/n,
                            two_pe=totals['bad2_pixels']/n,
                            coverage=n/totals['total_pixels'])
            saved = next(r for r in report['sequence_metrics']
                         if r['method'] == method and r['scene_id'] == scene)
            for key, value in {**totals, **computed}.items():
                close(value, saved[key])
            sequences.append(dict(method=method, scene_id=scene, **computed))
    summaries = []
    for method in methods:
        rows = [r for r in sequences if r['method'] == method]
        summary = dict(method=method, **{key: math.fsum(r[key] for r in rows)/2 for key in metrics})
        saved = next(r for r in report['equal_sequence_summary'] if r['method'] == method)
        for key in metrics:
            close(summary[key], saved[key])
        summaries.append(summary)
    result = dict(schema='stereo-study-dsec-saved-report-analysis-v1',
                  status='PASS_SAVED_ARITHMETIC_AND_FIXED_PANEL_BINDINGS',
                  report_sha256=sha(args.report), inventory_sha256=sha(args.inventory),
                  config_sha256=sha(args.config), analyzer_sha256=sha(Path(__file__)),
                  job_id=report['slurm_job_id'], predictions=100, targets=20,
                  same_support=True, weights_and_buffers_unchanged=True,
                  aggregation=report['aggregation'], native_conversion=report['native_conversion'],
                  thresholds=report['thresholds'], claim=report['claim'],
                  equal_sequence_summary=summaries, sequence_metrics=sequences,
                  scope='Saved sums reconstructed; no independent prediction or GT replay; two exposed development sequences')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    with args.output.with_suffix('.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['method', *metrics])
        writer.writeheader(); writer.writerows(summaries)
    print(json.dumps(dict(status=result['status'], predictions=100, targets=20,
                          report_sha256=result['report_sha256'])))


if __name__ == '__main__':
    main()
