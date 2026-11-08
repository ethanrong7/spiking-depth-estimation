import tempfile
import unittest
from pathlib import Path

import torch

from benchmark_nmrf_selective_updates import load_benchmark_weights
from nmrf_runtime import UPSTREAM_REVISION, build_native_nmrf, prepare_batch, weighted_loss
from nmrf_selective_updates import ALTERNATING_SELECTOR
from run_nmrf_finetune import (crop_batch, install_architecture, load_initial_weights,
                               matched_batches, restore_architecture, save_checkpoint)


class MatchedFinetuneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_matched_input_plan_is_independent_of_global_rng(self):
        batch = (torch.arange(48*64).reshape(1, 1, 48, 64),) * 4 + (torch.tensor([0]),)
        orders = matched_batches(9, 4, 1234, 8)
        origins = [crop_batch(batch, (32, 40), 1234, i)[1] for i in range(8)]
        torch.rand(4096)
        self.assertEqual(orders, matched_batches(9, 4, 1234, 8))
        self.assertEqual(origins[5:], [crop_batch(batch, (32, 40), 1234, i)[1] for i in range(5, 8)])
        self.assertNotEqual(orders, matched_batches(9, 4, 2345, 8))
        self.assertNotEqual(origins, [crop_batch(batch, (32, 40), 2345, i)[1] for i in range(8)])
        crop, (y, x) = crop_batch(batch, (32, 40), 1234, 0)
        for array in crop[:4]:
            self.assertEqual(int(array[0, 0, 0, 0]), 64*y+x)
        self.assertEqual(len({index for micro in orders[:2] for index in micro}), 8)

    def test_shared_weights_and_ternary_native_training_checkpoint(self):
        original, _, _ = build_native_nmrf(1234)
        state = dict(model=original.state_dict(), model_name='nmrf-native-50', step=2000,
                     upstream_revision=UPSTREAM_REVISION, optimizer='ignored', scheduler='ignored')
        parent = dict(model='nmrf-native-50', steps=2000)
        for variant in ('original', 'continuous', 'ternary'):
            model, criterion, _ = build_native_nmrf(9)
            architecture = install_architecture(model, variant)
            load_initial_weights(model, state, architecture, parent, UPSTREAM_REVISION)
            for name, weight in model.state_dict().items():
                self.assertTrue(torch.equal(weight, state['model'][name]), name)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=5e-5, total_steps=500, pct_start=.05, cycle_momentum=False)
        self.assertFalse(optimizer.state)
        self.assertEqual(schedule.total_steps, 500)
        left = torch.randint(2, (1, 50, 40, 64)).float()
        flow = torch.full((1, 1, 40, 64), -6.)
        mask = torch.ones((1, 40, 64), dtype=torch.bool)
        batch, _ = crop_batch((left, 1-left, flow, mask, torch.tensor([0])), (32, 64), 1234, 0)
        left, right, target, valid, _ = prepare_batch(batch, 'cpu')
        self.assertTrue((target == 6.).all())
        block = model.parent.inference.layers[1].nmp
        initial = block.qkv.weight.detach().clone()
        loss, _ = weighted_loss(criterion, model(left, right), target, valid)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        schedule.step()
        self.assertFalse(torch.equal(initial, block.qkv.weight))

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            config = dict(device='cpu', architecture=architecture, upstream_revision=UPSTREAM_REVISION,
                          identity={'recipe': 'test'})
            save_checkpoint(output, model, optimizer, schedule, 1, config)
            saved = torch.load(output/'checkpoint-last.pt', map_location='cpu', weights_only=True)
        restored, _, _ = build_native_nmrf(9)
        restore_architecture(restored, saved)
        new_block = restored.parent.inference.layers[1].nmp
        self.assertEqual(new_block.coding, 'ternary')
        self.assertEqual(new_block.keep_fraction, .5)
        self.assertTrue(torch.equal(block.qkv.weight, new_block.qkv.weight))
        self.assertEqual(saved['step'], 1)
        self.assertEqual(saved['scheduler']['last_epoch'], 1)

    def test_alternate_selector_checkpoint_reloads_with_its_coverage(self):
        model, _, _ = build_native_nmrf(1234)
        config = install_architecture(model, 'continuous', selector=ALTERNATING_SELECTOR)
        state = dict(format='nmrf-matched-finetune-v1', architecture=config, model=model.state_dict(),
                     step=500, run_config={'recipe': 'test'})
        restored, _, _ = build_native_nmrf(81)
        restore_architecture(restored, state)
        for i in (1, 2, 3):
            self.assertTrue(torch.equal(restored.parent.inference.layers[i].nmp.selected_pixels('cpu'),
                                        model.parent.inference.layers[i].nmp.selected_pixels('cpu')))
        benchmark, _, _ = build_native_nmrf(81)
        receipt = dict(architecture=config, steps=500, identity=state['run_config'])
        with self.assertRaisesRegex(ValueError, 'variant'):
            load_benchmark_weights(benchmark, state, receipt, 'ternary')
        with self.assertRaisesRegex(ValueError, 'retention'):
            load_benchmark_weights(benchmark, state, receipt, 'continuous', .75)
        with self.assertRaisesRegex(ValueError, 'completed'):
            load_benchmark_weights(benchmark, state, dict(receipt, steps=250), 'continuous')
        self.assertEqual(load_benchmark_weights(benchmark, state, receipt, 'continuous'), config)
        self.assertEqual(benchmark.parent.inference.layers[2].nmp.phase, 1)


if __name__ == '__main__':
    unittest.main()
