"""Tiny toy control-flow tests, not study training/validation experiments."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from core.study_training import (append_json, collate_samples, evaluate_validation,
                                 optimizer_and_scheduler, restore_checkpoint, sample_indices,
                                 save_checkpoint, stable_clip_grad_norm, train_update, validate_config)
from core.study_training import training_sample_indices, validation_due


class ToyDataset:
    def __len__(self):
        return 5

    def __getitem__(self, i):
        return {'left': torch.full((50, 1, 2, 2), .1 + i / 20),
                'right': torch.full((50, 1, 2, 2), .2 + i / 20),
                'target': torch.ones(1, 2, 2), 'valid': torch.ones(1, 2, 2, dtype=torch.bool),
                'sample_id': str(i), 'scene_id': 'scene' + str(i), 'metadata': {'physical_fps': None}}


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)
        self.dropout = nn.Dropout2d(.2)

    def forward(self, left, right):
        x = .5 * (left.mean(1) + right.mean(1))
        return {'disparity': self.conv(self.dropout(x)), 'diagnostics': {}}


def toy_config():
    return {'seed': 9471, 'model': {'family': 'study', 'config': {'max_disparity': 12}},
            'optimizer': {'lr': .001, 'weight_decay': 1e-4, 'warmup_fraction': .05, 'minimum_lr_ratio': .1},
            'training': {'effective_batch': 8, 'microbatch': 2, 'clip_grad': 1.}, 'neuron_regularization': None}


class TrainingControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_example_configs_remain_nonlaunchable_and_contracts_guarded(self):
        directory = Path(__file__).resolve().parents[1] / 'experiments/stereo-spiking-transformer-study/configs'
        examples = list(directory.glob('training-*.example.json'))
        self.assertEqual(len(examples), 3)
        for path in examples:
            config = json.loads(path.read_text())
            validate_config(config, require_registered=False)
            with self.assertRaisesRegex(ValueError, 'unregistered'):
                validate_config(config)
        config = json.loads((directory / 'training-engineering-A-spike-v1.example.json').read_text())
        config['registered'] = True
        config['data']['manifest_sha256'] = '0' * 64
        validate_config(config)
        config['model']['config']['time_steps'] = 49
        with self.assertRaisesRegex(ValueError, '50 sensor'):
            validate_config(config)

    def test_sample_order_resumes_across_epochs_and_metadata_none(self):
        whole = sample_indices(9471, 0, 24, 5)
        self.assertEqual(whole, sample_indices(9471, 0, 8, 5) + sample_indices(9471, 8, 16, 5))
        self.assertEqual(sorted(whole[:5]), list(range(5)))
        batch = collate_samples([ToyDataset()[0], ToyDataset()[1]])
        self.assertEqual(batch['left'].shape, (2, 50, 1, 2, 2))
        self.assertIsNone(batch['metadata'][0]['physical_fps'])

    def test_explicit_validation_boundaries_prevent_extra_selection_checkpoints(self):
        config={'total_updates':1000,'training':{'eval_interval':250,'evaluation_updates':[250,500,1000]}}
        self.assertEqual([u for u in range(1,1001) if validation_due(config,u)],[250,500,1000])
        del config['training']['evaluation_updates']
        self.assertEqual([u for u in range(1,1001) if validation_due(config,u)],[250,500,750,1000])

    def test_fp64_clipping_finite_large_elements(self):
        model = nn.Linear(2, 1, bias=False)
        model.weight.grad = torch.tensor([[1e30, -1e30]])
        self.assertFalse(torch.isfinite(model.weight.grad.norm()))
        audit = {}
        norm, coefficient = stable_clip_grad_norm(model, 1., diagnostics=audit)
        self.assertTrue(torch.isfinite(norm))
        self.assertLess(float(coefficient), 1e-29)
        self.assertAlmostEqual(model.weight.grad.double().norm().item(), 1., places=6)
        self.assertTrue(audit['is_extreme'])
        self.assertEqual(audit['top10_named_preclip_gradient_l2'][0]['name'], 'weight')
        self.assertGreater(audit['_raw_parameter_direction_statistics']['weight']['gradient_l2'], 1e30)
        model.weight.grad[0, 0] = torch.inf
        with self.assertRaises(FloatingPointError):
            stable_clip_grad_norm(model, 1.)

    def test_readout_logging_preserves_optimizer_and_fixed_panel_order(self):
        class Panel(ToyDataset):
            def __len__(self):
                return 8
        class HeadModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = nn.Module()
                self.decoder.residual = nn.Sequential(nn.Identity(),nn.Identity(),nn.Conv2d(1,1,1))
            def forward(self, left, right):
                return {'disparity': self.decoder.residual((left.mean(1)+right.mean(1))*.5), 'diagnostics': {}}
        config = toy_config()
        config.update(scope='engineering',data={'train_sample_ids':[str(i) for i in range(8)]})
        config['training'].update(sampling='repeated_panel_fixed_order')
        self.assertEqual(training_sample_indices(config,8,8,8),list(range(8)))
        self.assertEqual(training_sample_indices(config,6,4,8),[6,7,0,1])
        torch.manual_seed(1); a=HeadModel(); b=copy.deepcopy(a)
        oa,sa=optimizer_and_scheduler(a,config['optimizer'],1000)
        ob,sb=optimizer_and_scheduler(b,config['optimizer'],1000)
        logged=copy.deepcopy(config);logged['training']['readout_optimizer_health']=True
        with tempfile.TemporaryDirectory() as tmp:
            for position in (0,8):
                ra=train_update(a,oa,sa,Panel(),config,position,'cpu',Path(tmp)/'fail-a.json')
                rb=train_update(b,ob,sb,Panel(),logged,position,'cpu',Path(tmp)/'fail-b.json')
                self.assertEqual(ra['sample_indices'],list(range(8)))
                self.assertEqual(ra['loss'],rb['loss'])
                self.assertTrue(rb['optimizer_moments_all_finite'])
                row=rb['readout_optimizer_health']['targets']['decoder.residual.2.weight']
                self.assertTrue(row['available']);self.assertEqual(row['step'],position//8+1)
        for name,tensor in a.state_dict().items():
            self.assertTrue(torch.equal(tensor,b.state_dict()[name]),name)
        for left,right in zip(oa.state.values(),ob.state.values()):
            for key in left:self.assertTrue(torch.equal(left[key],right[key]),key)
        self.assertEqual(sa.state_dict(),sb.state_dict())

    def test_finite_extreme_first_update_records_named_norms_and_separate_raw_directions(self):
        class Gain(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, gradient):
                return gradient * 1e20

        class GainModel(ToyModel):
            def forward(self, left, right):
                out = super().forward(left, right)
                out['disparity'] = Gain.apply(out['disparity'])
                return out

        torch.manual_seed(13)
        model = GainModel()
        model.dropout = nn.Identity()
        config = toy_config()
        optimizer, scheduler = optimizer_and_scheduler(model, config['optimizer'], 6)
        with tempfile.TemporaryDirectory() as temp:
            record = train_update(model, optimizer, scheduler, ToyDataset(), config, 0, 'cpu', Path(temp)/'failure.json')
            audit = record['first_update_extreme_gradient_audit']
            self.assertTrue(audit['is_extreme'])
            self.assertTrue(audit['top10_named_preclip_gradient_l2'])
            self.assertNotIn('_raw_parameter_direction_statistics', audit)
            raw = json.loads(Path(audit['raw_parameter_direction_path']).read_text())
            self.assertTrue(raw['before_clipping'])
            self.assertIn('gradient_parameter_cosine', raw['parameters']['conv.weight'])

    def test_effective_batch_matches_microbatch_accumulation(self):
        config = toy_config()
        torch.manual_seed(5)
        a = ToyModel()
        a.dropout = nn.Identity()
        b = copy.deepcopy(a)
        oa, sa = optimizer_and_scheduler(a, config['optimizer'], 10)
        ob, sb = optimizer_and_scheduler(b, config['optimizer'], 10)
        with tempfile.TemporaryDirectory() as temp:
            ra = train_update(a, oa, sa, ToyDataset(), config, 0, 'cpu', Path(temp) / 'a.json')
            other = copy.deepcopy(config)
            other['training']['microbatch'] = 8
            rb = train_update(b, ob, sb, ToyDataset(), other, 0, 'cpu', Path(temp) / 'b.json')
        self.assertEqual(ra['sample_indices'], rb['sample_indices'])
        self.assertAlmostEqual(ra['loss'], rb['loss'], places=6)
        for pa, pb in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.allclose(pa, pb, atol=1e-7, rtol=1e-6))

    def test_exact_resume_model_optimizer_rng_order_and_log_tails(self):
        config = toy_config()
        torch.manual_seed(44)
        random.seed(44)
        np.random.seed(44)
        a = ToyModel()
        oa, sa = optimizer_and_scheduler(a, config['optimizer'], 6)
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            logs = {'training': temp / 'training.jsonl', 'validation': temp / 'validation.jsonl'}
            one = train_update(a, oa, sa, ToyDataset(), config, 0, 'cpu', temp / 'failure.json')
            append_json(logs['training'], {'update': 1, 'loss': one['loss']})
            binding = {'effective_batch': 8, 'total_updates': 6, 'source': 'tiny-control-test'}
            save_checkpoint(temp / 'latest.pt', a, oa, sa, {'update': 1, 'presentations': 8}, binding, logs)
            two = train_update(a, oa, sa, ToyDataset(), config, 8, 'cpu', temp / 'failure.json')
            expected_rng = (random.random(), float(np.random.random()), torch.rand(3))
            append_json(logs['training'], {'uncommitted_after_checkpoint': True})
            b = ToyModel()
            ob, sb = optimizer_and_scheduler(b, config['optimizer'], 6)
            state = restore_checkpoint(temp / 'latest.pt', b, ob, sb, binding, logs)
            self.assertEqual(state['presentations'], 8)
            self.assertEqual(len(logs['training'].read_text().splitlines()), 1)
            self.assertEqual(len(list(temp.glob('training.jsonl.uncommitted-tail-*'))), 1)
            again = train_update(b, ob, sb, ToyDataset(), config, 8, 'cpu', temp / 'failure.json')
            actual_rng = (random.random(), float(np.random.random()), torch.rand(3))
            self.assertEqual(two['sample_indices'], again['sample_indices'])
            self.assertEqual(two['loss'], again['loss'])
            self.assertEqual(expected_rng[:2], actual_rng[:2])
            self.assertTrue(torch.equal(expected_rng[2], actual_rng[2]))
            self.assertEqual(sa.state_dict(), sb.state_dict())
            for pa, pb in zip(a.parameters(), b.parameters()):
                self.assertTrue(torch.equal(pa, pb))
                for key, value in oa.state[pa].items():
                    self.assertTrue(torch.equal(value, ob.state[pb][key]))
            with self.assertRaisesRegex(RuntimeError, 'horizon'):
                restore_checkpoint(temp / 'latest.pt', b, ob, sb, {**binding, 'total_updates': 7}, logs)

    def test_nonfinite_backward_keeps_forward_and_gradient_evidence(self):
        class Poison(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, gradient):
                return gradient * float('inf')

        class PoisonModel(ToyModel):
            def forward(self, left, right):
                out = super().forward(left, right)
                out['disparity'] = Poison.apply(out['disparity'])
                return out

        config = toy_config()
        model = PoisonModel()
        optimizer, scheduler = optimizer_and_scheduler(model, config['optimizer'], 6)
        old = copy.deepcopy(model.state_dict())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'failure.json'
            with self.assertRaises(FloatingPointError):
                train_update(model, optimizer, scheduler, ToyDataset(), config, 0, 'cpu', path)
            failure = json.loads(path.read_text())
            record = failure['microbatches'][0]
            self.assertTrue(record['loss_all_finite'])
            self.assertFalse(record['gradients']['left']['per_sensor_step'][0]['all_finite'])
            self.assertEqual(failure['status'], 'FAILED_UPDATE_NOT_COMMITTED')
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, old[name]))

    def test_validation_scene_equal_and_coverage(self):
        class Data:
            def __len__(self):
                return 3

            def __getitem__(self, index):
                row = ToyDataset()[index]
                row['left'].fill_((1, 3, 8)[index] + 1)
                row['scene_id'] = 'A' if index < 2 else 'B'
                if index == 2:
                    row['valid'].fill_(False)
                    row['valid'][0, 0, 0] = True
                return row

        class Predict(nn.Module):
            def forward(self, left, right):
                return {'disparity': left[:, 0]}

        result = evaluate_validation(Predict(), Data(), 'cpu')
        self.assertEqual(result['mean_scene_epe'], 5.)
        self.assertAlmostEqual(result['pooled_epe'], 24 / 9)
        self.assertEqual(result['coverage'], .75)
        self.assertEqual(result['bad_pixel_units'], 'fraction')


if __name__ == '__main__':
    unittest.main()
