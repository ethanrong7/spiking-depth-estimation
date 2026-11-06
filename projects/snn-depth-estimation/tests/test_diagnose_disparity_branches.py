"""Tiny CPU geometry/mocked wiring tests; no real arrays or neural experiment."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'experiments/stereo-spiking-transformer-study/scripts/diagnose_disparity_branches.py'
spec=importlib.util.spec_from_file_location('branch_diagnostic_test',SCRIPT)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def config():
    return json.loads((SCRIPT.parents[1]/'configs/branch-shift-AB-spike-v1.json').read_text())


def sample():
    return dict(sample_id='fixture/0',scene_id='fixture',target=torch.full((1,5,13),4.),
        valid=torch.ones((1,5,13),dtype=torch.bool),left=torch.zeros((50,1,5,13)),right=torch.ones((50,1,5,13)))


class BranchDiagnosticTests(unittest.TestCase):
    def test_selected_bindings_caps_and_no_extra_conditions(self):
        c=config();self.assertTrue(runner.validate_config(c))
        for key,value in [('shift_pixels',3),('maximum_forwards',81),('max_seconds',301),
                          ('conditions',['correct','shifted','masked']),('inference_variant','changed'),
                          ('identity_tolerance',{'atol':1.,'rtol':1.})]:
            bad=copy.deepcopy(c);bad[key]=value
            with self.assertRaises(ValueError):runner.validate_config(bad)
        for key,value in [('path_sha256','f'*64),('update',500),('selection','latest.pt'),
                          ('selection_validation_log_sha256','a'*64)]:
            bad=copy.deepcopy(c);bad['cases'][0]['checkpoint'][key]=value
            with self.assertRaises(ValueError):runner.validate_config(bad)
        bad=copy.deepcopy(c);bad['test_access']='open'
        with self.assertRaises(ValueError):runner.validate_config(bad)

    def test_exact_pad16_then_crop_geometry_and_negative_values(self):
        aux=torch.arange(16,dtype=torch.float32).reshape(1,1,4,4)/4
        padded=F.interpolate(aux,size=(16,16),mode='bilinear',align_corners=False)
        main=padded[...,:5,:13]-10
        output={'disparity':main,'aux_disparities_native':{4:aux}}
        branches=runner.extract_branches(output,(5,13))
        torch.testing.assert_close(branches['matcher'],padded[0,:,:5,:13],rtol=0,atol=0)
        torch.testing.assert_close(branches['residual'],torch.full((1,5,13),-10.),rtol=0,atol=0)
        self.assertTrue(bool((branches['main']<0).all()))
        # Direct resize to unpadded dimensions gives different coordinates.
        wrong=F.interpolate(aux,size=(5,13),mode='bilinear',align_corners=False)[0]
        self.assertFalse(torch.allclose(branches['matcher'],wrong))
        bad={'disparity':main,'aux_disparities_native':{4:torch.zeros((1,1,2,4))}}
        with self.assertRaises(ValueError):runner.extract_branches(bad,(5,13))

    def test_affine_shift_response_decomposition_on_common_GT_only_support(self):
        s=sample();x=torch.arange(13)[None,None].expand(1,5,13).float()
        matcher=8+x*.1;main=4+x*.2
        correct=dict(main=main,matcher=matcher,residual=main-matcher)
        shifted=dict(main=main-.8,matcher=matcher-2.4)
        shifted['residual']=shifted['main']-shifted['matcher']
        changed,common=runner.shifted_disparity_support(s['target'],s['valid'],2)
        self.assertEqual(int(common.sum()),5*9)
        self.assertTrue(bool((changed[common]==2).all()))
        row=runner.response_row('A-spike',s,correct,shifted,common,2)
        self.assertAlmostEqual(row['branches']['main']['response_ratio'],.4,places=6)
        self.assertAlmostEqual(row['branches']['matcher']['response_ratio'],1.2,places=6)
        self.assertAlmostEqual(row['branches']['residual']['response_ratio'],-.8,places=6)
        self.assertLess(row['mean_identity']['absolute_error'],1e-6)
        self.assertEqual(row['common_trusted_GT_coverage'],9/13)
        self.assertTrue(bool((runner.shift_right_input(s['right'],2)[...,:2]==0).all()))
        # Predictions cannot change support; finite negative predictions score.
        metric=runner.disparity_metrics(torch.full_like(s['target'],-1),s['target'],common)
        self.assertEqual(metric['valid_pixels'],45);self.assertEqual(metric['epe'],5)
        self.assertEqual(metric['bad5'],0.)  # strict >5, not >=5
        self.assertEqual(metric['bad3'],1.)
        with self.assertRaises(ValueError):runner.check_mean_identity({'main':1.,'matcher':1.,'residual':1.})

    def test_frame_to_scene_pooling_equal_scene_and_only_main_median(self):
        rows=[]
        for scene in range(10):
            for frame,n in [(0,1),(1,3)]:
                main=0. if frame==0 else 1.;matcher=main+.25;residual=-.25
                branches={name:dict(response_sum=value*n,response_pixels=n,response_ratio=value,
                    positive_response_pixels=n if value>0 else 0) for name,value in
                    [('main',main),('matcher',matcher),('residual',residual)]}
                rows.append(dict(case_id='A-spike',scene_id=str(scene),sample_id=f'{scene}/{frame}',
                    response_pixels=n,total_pixels=4,trusted_original_pixels=4,branches=branches))
        scenes,equal=runner.summarize_responses(rows)
        self.assertEqual(len(scenes),10)
        self.assertEqual(scenes[0]['branches']['main']['response_ratio'],.75)
        self.assertEqual(equal[0]['mean_of_scene_mean_response_ratios'],{'main':.75,'matcher':1.,'residual':-.25})
        self.assertEqual(equal[0]['main_median_of_scene_mean_response_ratios'],.75)
        self.assertTrue(equal[0]['unchanged_main_shift_screen']['median_in_registered_range'])
        self.assertEqual(equal[0]['positive_main_response_scenes'],10)
        self.assertNotIn('matcher_median',equal[0])
        with self.assertRaises(ValueError):runner.summarize_responses(rows[:-1])

    def test_loaded_model_keeps_exact_architecture_and_polarity(self):
        c=config();case=c['cases'][0]
        training=json.loads((SCRIPT.parents[1]/'configs'/case['training_config']).read_text())
        provenance={'binding':{key:case[value] for key,value in [('source_identity_sha256','training_source_identity_sha256'),
            ('train_sample_ids_sha256','train_sample_ids_sha256'),('val_sample_ids_sha256','val_sample_ids_sha256')]}}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'metadata.json';path.write_text(json.dumps({'fresh_seed_model_sha256':case['training_initial_weights_sha256']}))
            with mock.patch.dict(runner.os.environ,{case['checkpoint']['metadata_env']:str(path)}):
                runner.validate_loaded_case(case,training,provenance)
                training['model']['config']['widths']=[8,8,8,8]
                with self.assertRaises(ValueError):runner.validate_loaded_case(case,training,provenance)

    def test_partial_forward_failure_preserves_counts_and_state_check(self):
        c=config();report=dict(forwards_completed=0,models=[],frame_metrics=[],frame_responses=[])
        training={'model':{'family':'study','config':{'unchanged_fixture':True}}};original=copy.deepcopy(training)
        fake=mock.Mock(side_effect=[{'disparity':torch.zeros((1,1,5,13)),
            'aux_disparities_native':{4:torch.zeros((1,1,4,4))}},RuntimeError('mock forward failure')])
        fake.state_dict.return_value={'weight':torch.zeros(1)}
        dataset=mock.Mock();dataset.rows=[dict(scene_id=str(i//2)) for i in range(20)]
        dataset.__len__=mock.Mock(return_value=20);dataset.__getitem__=mock.Mock(return_value=sample())
        with mock.patch.object(runner,'PairedStereoDataset',return_value=dataset), \
                mock.patch.object(runner.functionality,'selected_cases',return_value=c['preselected_case_ids']), \
                mock.patch.object(runner.functionality,'load_checkpoint',return_value=(fake,training,{})), \
                mock.patch.object(runner,'validate_loaded_case'):
            with self.assertRaisesRegex(RuntimeError,'mock forward failure'):
                runner.run(c,Path('unused'),Path('unused'),{'output_scale':.25},Path('unused'),report,torch.device('cpu'))
        self.assertEqual(report['forwards_completed'],1);self.assertEqual(len(report['frame_responses']),0)
        self.assertTrue(report['models'][0]['weights_and_buffers_unchanged'])
        self.assertEqual(training,original);self.assertEqual(report['models'][0]['overrides'],{})
        self.assertNotIn('scene_equal_response_summary',report)

    def test_weight_mutation_blocks_completion_even_on_partial_failure(self):
        c=config();report=dict(forwards_completed=0,models=[],frame_metrics=[],frame_responses=[])
        weight=torch.zeros(1)
        def changed_forward(left,right):
            weight.add_(1)
            raise RuntimeError('mock interrupted forward')
        fake=mock.Mock(side_effect=changed_forward);fake.state_dict.return_value={'weight':weight}
        dataset=mock.Mock();dataset.rows=[dict(scene_id=str(i//2)) for i in range(20)]
        dataset.__len__=mock.Mock(return_value=20);dataset.__getitem__=mock.Mock(return_value=sample())
        with mock.patch.object(runner,'PairedStereoDataset',return_value=dataset), \
                mock.patch.object(runner.functionality,'selected_cases',return_value=c['preselected_case_ids']), \
                mock.patch.object(runner.functionality,'load_checkpoint',return_value=(fake,{'model':{}},{})), \
                mock.patch.object(runner,'validate_loaded_case'):
            with self.assertRaisesRegex(RuntimeError,'Frozen weights or persistent buffers changed'):
                runner.run(c,Path('unused'),Path('unused'),{'output_scale':.25},Path('unused'),report,torch.device('cpu'))
        self.assertFalse(report['models'][0]['weights_and_buffers_unchanged'])
        self.assertEqual(report['models'][0]['status'],'FAILED_WEIGHTS_OR_BUFFERS_MUTATED')
        self.assertNotIn('scene_equal_response_summary',report)


if __name__=='__main__':unittest.main()
