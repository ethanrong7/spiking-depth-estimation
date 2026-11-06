"""Scalar-only comparison checks; no models, checkpoints or dataset arrays."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PROJECT=Path(__file__).resolve().parents[1]
path=PROJECT/'experiments/spiking-stereo-correspondence/scripts/compare_saved_scalars.py'
spec=importlib.util.spec_from_file_location('comparison',path);comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)


class ComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.baselines,cls.references,cls.pins=comparison.load_baselines()

    def test_exact_historical_endpoint_and_cohort_pins(self):
        self.assertEqual(self.baselines['original-seed1234-10k-v1'][2500]['mean_scene_epe'],4.722502601146698)
        self.assertEqual(self.baselines['original_i-seed1234-10k-v1'][2500]['mean_scene_epe'],3.6731657087802887)
        self.assertEqual(self.pins['cohort_scenes'],30)
        self.assertEqual(len(self.references[0]),60)
        self.assertEqual({r['valid_pixels'] for r in self.references[0].values()},{100000})

    def test_missing_final_keeps_primary_ratios_auc_and_scene_differences_missing(self):
        observed={0:self.baselines['original_i-seed1234-10k-v1'][0]}
        report=comparison.build_comparison(self.baselines,observed,False)
        self.assertEqual(report['status'],'PENDING_FINAL_2500')
        new=report['methods'][-1]
        for key in ('mean_scene_epe','best_trained_scheduled_epe','normalized_common_grid_auc_including_zero'):
            self.assertIsNone(new[key])
        self.assertTrue(all(v['new_over_reference_epe'] is None for v in report['endpoint_ratios'].values()))
        self.assertTrue(all(row['new_epe'] is None for row in report['paired_scene_differences']))
        curve=[r for r in report['curves'] if r['identity']==comparison.P2_RUN]
        self.assertIsNotNone(curve[0]['mean_scene_epe']);self.assertIsNone(curve[-1]['mean_scene_epe'])

    def test_common_grid_auc_and_best_exclude_later_history_and_initialization(self):
        # Reusing historical scalars as a mathematical fixture is not a new result.
        fixture=copy.deepcopy(self.baselines['original_i-seed1234-10k-v1'])
        fixture[0]['mean_scene_epe']=0.
        fixture[10000]=dict(fixture[2500],update=10000,mean_scene_epe=-100.)
        report=comparison.build_comparison(self.baselines,fixture,True)
        new=report['methods'][-1]
        self.assertEqual(new['best_trained_scheduled_update'],2500)
        expected=((fixture[0]['mean_scene_epe']+fixture[1000]['mean_scene_epe'])/2*1000+(fixture[1000]['mean_scene_epe']+fixture[2500]['mean_scene_epe'])/2*1500)/2500
        self.assertEqual(new['normalized_common_grid_auc_including_zero'],expected)
        self.assertEqual(report['endpoint_ratios']['original_i-seed1234-10k-v1']['new_over_reference_epe'],1.)
        self.assertEqual(len(report['paired_scene_differences']),60)
        self.assertFalse(report['practical_10k_target_applicable'])

    def test_p1_and_engineering_visits_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'only exact P2'):
            comparison.validate_metadata(dict(config=dict(stage='sanity')))
        row=dict(update=0,split='train_diagnostic',scientific_training_updates=0)
        with self.assertRaisesRegex(ValueError,'ineligible'):comparison.validate_new_visit(row,self.references,False)

    def test_frame_id_and_valid_count_mismatch_are_rejected(self):
        reference=self.baselines['original-seed1234-10k-v1'][0]
        row=copy.deepcopy(reference);row['scientific_training_updates']=0
        row['per_frame'][0]['valid_pixels']-=1
        with self.assertRaisesRegex(ValueError,'valid count differs'):comparison.validate_new_visit(row,self.references)
        row=copy.deepcopy(reference);row['scientific_training_updates']=0;row['per_frame'].pop()
        with self.assertRaisesRegex(ValueError,'frame cohort differs'):comparison.validate_new_visit(row,self.references)

    def test_saved_terminal_json_alone_cannot_fill_primary(self):
        metadata=dict(config=dict(stage='main',run_id=comparison.P2_RUN,model='correspondence',seed=1234,updates=2500,effective_batch=8,time_steps=50),
            manifest_sha256=comparison.MANIFEST_SHA,base_source=dict(source_identity_sha256=comparison.BASE_ID),source=dict(source_identity_sha256=comparison.P2_ID))
        observation=dict(self.baselines['original-seed1234-10k-v1'][0],scientific_training_updates=0)
        observation.pop('per_frame')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'metadata.json').write_text(json.dumps(metadata))
            (root/'validation.jsonl').write_text(json.dumps(observation)+'\n')
            (root/'result.json').write_text(json.dumps(dict(status='COMPLETE_MAIN',updates=2500)))
            visits,complete,_=comparison.load_new(root,self.references)
        self.assertEqual(set(visits),{0});self.assertFalse(complete)


if __name__=='__main__':unittest.main()
