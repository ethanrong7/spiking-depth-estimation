"""Configuration, saved-scalar baseline parity, and cluster boundary checks."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

PROJECT=Path(__file__).resolve().parents[1]
STUDY=PROJECT/'experiments/spiking-stereo-correspondence'
spec=importlib.util.spec_from_file_location('correspondence_runner',STUDY/'scripts/run_experiment.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class RunnerTest(unittest.TestCase):
    def test_registered_configs_keep_one_scientific_geometry(self):
        paths=list((STUDY/'configs').glob('*.json'))
        paths=[p for p in paths if p.name!='run-queue.json']
        self.assertEqual(len(paths),15)
        for path in paths:
            config=json.loads(path.read_text());runner.validate_config(config)
            self.assertEqual(config['run_id'],path.stem)
            self.assertEqual(config['scheduler_resources']['gpus'],1)
            self.assertEqual(config['allocated_gpu_hour_cap'],24)

    def test_historical_manifest_identity_and_train_dev_only(self):
        path=PROJECT/'experiments/spiking-transformer-stereo/configs/manifest.json'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),runner.MANIFEST_SHA)
        manifest=json.loads(path.read_text())
        self.assertEqual(len(manifest['train']),960);self.assertEqual(len(manifest['dev']),60)
        for row in manifest['train']+manifest['dev']:
            for name in ('spike_l','spike_r','depth'):
                self.assertTrue(row[name].startswith('Train/'))

    def test_saved_historical_aggregates_match_current_metric_adapter(self):
        audited=PROJECT/'experiments/spiking-transformer-stereo/campaigns/recipe-aligned-three-way-v1/results/audited'
        with (audited/'dev-frames.csv').open() as f:rows=list(csv.DictReader(f))
        with (audited/'dev-trajectories.csv').open() as f:references=list(csv.DictReader(f))
        checked=0
        for reference in references:
            if reference['run_id']!='original_i-seed1234-10k-v1':continue
            group=[dict(r,valid_pixels=int(r['valid_pixels']),**{k:float(r[k]) for k in ('epe','bad1','bad3','bad5')})
                for r in rows if r['run_id']==reference['run_id'] and r['update']==reference['update']]
            result=runner.aggregate_metrics(group)
            self.assertAlmostEqual(result['mean_scene_epe'],float(reference['mean_scene_epe']),places=10)
            # Raw FP32 error_sum was retained in the historical trajectory but not frame CSV.
            self.assertAlmostEqual(result['pooled_epe'],float(reference['pooled_epe']),places=5)
            for key in ('bad1','bad3','bad5'):
                self.assertAlmostEqual(result[key],float(reference[key]),places=5)
            self.assertEqual(len(result['per_frame']),60);checked+=1
        self.assertEqual(checked,6)

    def test_saved_forward_diagnostics_keep_full_mask_and_branch_supports(self):
        import math
        import torch
        target=torch.tensor([-1.,0.,1.,2.]).reshape(1,1,1,4)
        valid=torch.ones(1,1,4,dtype=torch.bool)
        output=dict(coarse_flow=torch.zeros_like(target),null_probability=torch.tensor([.8,.1,.2,.9]).reshape_as(target),
            null_flow=torch.full_like(target,2.),residual=torch.full_like(target,.25),
            temporal_weights=torch.tensor([.25,.75]),probabilities=torch.full((1,1,4,4),.25),
            query_valid=torch.ones(1,1,4,dtype=torch.bool))
        row=runner.output_diagnostics(output,target,valid)
        summary=runner.aggregate_diagnostics([row])
        self.assertEqual(summary['supports']['valid_pixels'],4)
        self.assertEqual(summary['supports']['inframe_pixels'],2)
        self.assertEqual(summary['supports']['outframe_pixels'],2)
        self.assertEqual(summary['supports']['correspondence_pixels'],0)
        self.assertAlmostEqual(summary['match_inframe_epe'],.5)
        self.assertAlmostEqual(summary['null_inframe_mean'],.15,places=6)
        self.assertAlmostEqual(summary['null_outframe_mean'],.85,places=6)
        self.assertEqual(summary['null_inframe_above_half_fraction'],0)
        self.assertEqual(summary['null_outframe_above_half_fraction'],1)
        self.assertAlmostEqual(summary['mean_abs_weighted_null_contribution'],1.,places=6)
        self.assertEqual(summary['mean_abs_residual_contribution'],.25)
        self.assertAlmostEqual(summary['correspondence_entropy_nats'],math.log(4),places=6)
        self.assertEqual(summary['temporal_weights'],[.25,.75])

    def test_empty_outframe_support_remains_missing(self):
        import torch
        target=torch.zeros(1,1,1,4);valid=torch.ones(1,1,4,dtype=torch.bool)
        output=dict(coarse_flow=target,null_probability=target,null_flow=target,residual=target,
            temporal_weights=torch.ones(1),probabilities=torch.full((1,1,4,4),.25),
            query_valid=torch.ones(1,1,4,dtype=torch.bool))
        summary=runner.aggregate_diagnostics([runner.output_diagnostics(output,target,valid)])
        self.assertIsNone(summary['null_outframe_mean'])
        self.assertIsNone(summary['null_outframe_above_half_fraction'])
        self.assertEqual(summary['supports']['correspondence_pixels'],1)

    def test_fixed_panel_selection_uses_scene_order_and_manifest_first_target(self):
        rows=[dict(scene=f'scene_{scene:04d}',frame=str(frame)) for frame in (9,1) for scene in reversed(range(10))]
        panel=runner.fixed_diagnostic_panel(rows)
        self.assertEqual([r['scene'] for r in panel],[f'scene_{i:04d}' for i in range(10)])
        self.assertTrue(all(r['frame']=='9' for r in panel))
        self.assertEqual([r['position'] for r in panel if r['save_map']],[1,5,10])
        self.assertEqual(panel[-1]['shuffled_right_scene'],'scene_0000')
        self.assertTrue(all(r['scene']!=r['shuffled_right_scene'] for r in panel))

    def test_shift_has_correct_sign_zero_fill_and_common_gt_support(self):
        import torch
        right=torch.arange(8.).reshape(1,1,1,8).expand(1,3,1,8)
        shifted=runner.shift_right_zero_fill(right,2)
        self.assertTrue(torch.equal(shifted[0,0,0],torch.tensor([0.,0.,0.,1.,2.,3.,4.,5.])))
        target=torch.full((1,1,1,8),-1.);valid=torch.ones(1,1,8,dtype=torch.bool)
        common=runner.common_shift_support(target,valid,2)
        self.assertTrue(torch.equal(common[0,0,0],torch.tensor([False,True,True,True,True,True,True,False])))
        response=runner.shift_response(target,target+2,common,2)
        self.assertEqual(response['mean'],1.);self.assertEqual(response['pixels'],6)
        # A predicted invalid boundary cannot change the fixed GT support.
        self.assertTrue(torch.equal(common,runner.common_shift_support(target,valid,2)))

    def test_branch_sum_and_shift_panel_equal_scene_aggregation(self):
        import torch
        target=torch.zeros(1,1,1,8)
        output=dict(flow=target+5,coarse_flow=target+4,null_probability=target+.5,null_flow=target+2,residual=target+2)
        branches=runner.branch_fields(output)
        self.assertTrue(torch.equal(branches['final'],branches['weighted_match']+branches['weighted_null']+branches['residual']))
        records=[]
        for response,delta in ((1.,2.),(.5,4.)):
            records.append(dict(shift_response={'final':dict(mean=response)},interventions={
                'intact':dict(full_gt={'final':dict(epe=1.)}),
                'shuffle_right':dict(full_gt={'final':dict(epe=1.+delta)})}))
        summary=runner.summarize_panel(records)
        self.assertEqual(summary['shift_response']['final']['equal_scene_mean'],.75)
        self.assertEqual(summary['shift_response']['final']['positive_scenes'],2)
        self.assertEqual(summary['shuffled_minus_correct']['final']['equal_scene_epe_difference'],3.)

    def test_cpu_cannot_execute_experiment(self):
        import torch
        if torch.cuda.is_available():self.skipTest('CPU-boundary check is for local unit environment')
        config=STUDY/'configs/preflight-native-t50-v1.json'
        with self.assertRaisesRegex(RuntimeError,'Slurm CUDA'):
            runner.setup(SimpleNamespace(config=config))


if __name__=='__main__':unittest.main()
