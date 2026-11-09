"""CPU checks for the NPU batch's routing, admission, data and result semantics."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

EXP = Path(__file__).resolve().parent
sys.path[:0] = [str(EXP), str(EXP.parents[1]), str(EXP.parent/'shared-gpu-30k-v1')]
import run_ablation_on_npu as runner
import summarize_ablation_npu as summary
from input_pipeline_npu import prepare_rank_batch, RankPreparedPrefetch
from settings import load


class AblationNPUTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_cell_routing_and_worker_arguments(self):
        for devices in ([0, 1, 2, 3, 4, 5, 6, 7], [3, 6]):
            cells = runner.cells(devices)
            self.assertEqual(len(cells), 8)
            self.assertEqual(len({(c['seed'], c['variant']) for c in cells}), 8)
            args = SimpleNamespace(output_root=Path('/runs'), data_root=Path('/data'), manifest=Path('/manifest'))
            for cell in cells:
                root, cmd = runner.worker_command(args, cell, preflight=True, pause=True, resume=True)
                self.assertIn('preflight', root.parts)
                self.assertEqual(cmd[cmd.index('--seed')+1], str(cell['seed']))
                self.assertEqual(cmd[cmd.index('--variant')+1], cell['variant'])
                self.assertEqual(cmd[cmd.index('--stop-at')+1], '3')
                self.assertIn('--resume', cmd)
                _, production = runner.worker_command(args, cell)
                self.assertEqual(production[production.index('--stop-at')+1], '30000')
                self.assertNotIn('--pause-at', production)
                self.assertNotIn('--resume', production)
                env = runner.environment(cell['device'], cell['seed'])
                self.assertEqual(env['WORLD_SIZE'], '1')
                self.assertEqual(env['ASCEND_RT_VISIBLE_DEVICES'], str(cell['device']))

    def test_native_preparation_and_resume_addressing(self):
        contract = load('test_npu_contract', EXP.parent/'spiking-attention-placement-phase2/protocol.py')
        class Dataset:
            def __getitem__(self, index):
                x = torch.arange(12).reshape(3, 4)+index
                return x.byte(), (x+1).byte(), x.float()[None], x.bool()
        dataset = Dataset()
        order = list(range(32))
        for seed in runner.SEEDS:
            reference = contract.Prefetch(dataset, order, 5, 6, seed)
            try:
                expected = reference.get(5)
            finally:
                reference.close()
            prepared = prepare_rank_batch(dataset, order, 5, seed, 0, 1, 8, 4)
            self.assertEqual(prepared['indices'], expected['indices'])
            self.assertEqual(prepared['flags'], expected['flags'])
            for i, micro in enumerate(prepared['microbatches']):
                for j, actual in enumerate(micro['tensors']):
                    wanted = torch.stack([item[j] for item in expected['items'][4*i:4*i+4]])
                    self.assertTrue(torch.equal(actual, wanted))
            resumed = RankPreparedPrefetch(dataset, order, 5, 6, seed, 0, 1)
            try:
                actual = resumed.get(5)
            finally:
                resumed.close()
            self.assertEqual(actual['flags'], prepared['flags'])
            self.assertEqual(actual['valid_count'], prepared['valid_count'])

    def test_production_blocked_when_preflight_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(output_root=Path(folder)/'run', execute=True, preflight_only=False,
                                   devices=list(range(8)))
            with patch.object(runner, 'parse_args', return_value=args), \
                 patch.object(runner, 'prepare'), patch.object(runner, 'verify_source', return_value='id'), \
                 patch.object(runner, 'on_devices', side_effect=RuntimeError('bad VJP')) as run:
                with self.assertRaisesRegex(RuntimeError, 'bad VJP'):
                    runner.main([])
                run.assert_called_once_with(args, runner.preflight_cell)
            self.assertFalse((args.output_root/'port-validation.json').exists())

    def test_preflight_requires_pause_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(output_root=Path(folder), data_root=Path('/data'), manifest=Path('/manifest'))
            cell = runner.cells([0])[0]
            root, _ = runner.worker_command(args, cell, preflight=True)
            root.mkdir(parents=True)
            runner.write_json(root/'encoder-validation.json', {'status': 'passed'})
            calls = []
            def worker(args, cell, **kwargs):
                calls.append(kwargs)
                if kwargs.get('pause'):
                    (root/'last.pt').touch()
                    runner.write_json(root/'checkpoint.json', {'update': 2})
                else:
                    runner.write_json(root/'restore-audit-2.json', {'status': 'passed'})
            with patch.object(runner, 'run_worker', side_effect=worker):
                runner.preflight_cell(args, cell)
            self.assertEqual(calls, [dict(preflight=True, pause=True), dict(preflight=True, resume=True)])

    def test_metric_recomputation_and_factorial_signs(self):
        frames = [dict(scene=str(i), frame=str(j), valid_pixels=j+1, epe=i+j/2, bad3=i/2)
                  for i in range(30) for j in range(2)]
        scenes = {str(i): i+.25 for i in range(30)}
        row = dict(per_frame=frames, scene_epe=scenes, mean_scene_epe=14.75, bad3=7.25)
        self.assertEqual(summary.checked_evaluation(row)[:2], (14.75, 7.25))
        row['mean_scene_epe'] += .1
        with self.assertRaises(AssertionError):
            summary.checked_evaluation(row)
        effects = summary.contrasts(3, 1, 2.5, 2)
        self.assertEqual(effects['hybrid_package'], 2)
        self.assertEqual(effects['forward_at_surrogate'], 1.5)
        self.assertEqual(effects['interaction'], .5)

    def test_preflight_matrix_rejects_backend_or_initialization_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(output_root=Path(folder), data_root=Path('/data'),
                                   manifest=Path('/manifest'), devices=list(range(8)))
            for cell in runner.cells(args.devices):
                root, _ = runner.worker_command(args, cell, preflight=True)
                runner.write_json(root/'result.json', dict(status='PREFLIGHT_COMPLETED', updates=3,
                    source_identity='source', finite_model_optimizer=True))
                backend = dict(python='3.11', torch='2.9', torch_npu='2.9', npu_name='Ascend', precision={'dtype': 'float32'})
                runner.write_json(root/'metadata.json', dict(config=cell, initial_state_sha256=str(cell['seed']),
                                                            rank_environments=[backend]))
                runner.write_json(root/'restore-audit-2.json', dict(status='passed'))
                runner.write_json(root/'encoder-validation.json', dict(status='passed'))
            self.assertEqual(len(runner.validate_preflight_matrix(args, 'source')), 8)
            meta = runner.read_json(root/'metadata.json')
            meta['initial_state_sha256'] = 'wrong'
            runner.write_json(root/'metadata.json', meta)
            with self.assertRaisesRegex(RuntimeError, 'Initial weights'):
                runner.validate_preflight_matrix(args, 'source')
            meta['initial_state_sha256'] = str(cell['seed'])
            meta['rank_environments'][0]['npu_name'] = 'different-device'
            runner.write_json(root/'metadata.json', meta)
            with self.assertRaisesRegex(RuntimeError, 'different NPU'):
                runner.validate_preflight_matrix(args, 'source')

    def test_partial_summary_keeps_all_cells_and_rejects_bad_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for seed in runner.SEEDS:
                for index, variant in enumerate(runner.VARIANTS):
                    path = root/f'seed-{seed}'/variant
                    forward, backward = summary.RULES[variant]
                    runner.write_json(path/'metadata.json', dict(order=list(range(960)),
                        initial_state_sha256=str(seed), source_registry={'source_identity_sha256': 'source'},
                        config=dict(seed=seed, variant=variant, forward_rule=forward, backward_rule=backward)))
                    frames = [dict(scene=str(i), frame=str(j), valid_pixels=10,
                                   epe=1.+index, bad3=5.) for i in range(30) for j in range(2)]
                    row = dict(update=0, samples_seen=0, per_frame=frames, mean_scene_epe=1.+index,
                               bad3=5., scene_epe={str(i): 1.+index for i in range(30)})
                    (path/'evaluations.jsonl').write_text(json.dumps(row)+'\n')
            self.assertEqual(summary.summarize(root), 'partial')
            audit = runner.read_json(root/'summaries/completion-audit.json')
            self.assertEqual(len(audit['cells']), 8)
            self.assertEqual(len((root/'summaries/factorial-contrasts.csv').read_text().splitlines()), 13)
            self.assertEqual(len((root/'summaries/endpoints.csv').read_text().splitlines()), 1)
            row['mean_scene_epe'] += 1
            (path/'evaluations.jsonl').write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(RuntimeError, 'audit failed'):
                summary.summarize(root)

    def test_training_audit_checks_actual_onecycle_and_detects_corruption(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([parameter], lr=2e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4, total_steps=30000,
            pct_start=.01, cycle_momentum=False, anneal_strategy='linear')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runner.write_json(root/'metadata.json', dict(order=list(range(960))))
            records = []
            for update in range(1, 301):
                positions = range((update-1)*8, update*8)
                flags = []
                for p in positions:
                    digest = hashlib.sha256(f'recipe-hv-v1:2345:{p}'.encode()).digest()
                    flags.append([digest[0] < 128, digest[1] < 128])
                lr = optimizer.param_groups[0]['lr']
                optimizer.step(); scheduler.step()
                records.append(dict(update=update, samples_seen=update*8, indices=[p%960 for p in positions],
                    augmentation_hv=flags, lr_used=lr, lr_next=scheduler.get_last_lr()[0], loss=1.,
                    gradient_norm=1., physical_batch=4, accumulation=2, world_size=1))
            path = root/'train.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in records))
            self.assertEqual(summary.audit_training(root, 2345)['updates'], 300)
            records[2]['indices'][0] += 1
            path.write_text(''.join(json.dumps(r)+'\n' for r in records))
            with self.assertRaises(AssertionError):
                summary.audit_training(root, 2345)


if __name__ == '__main__':
    unittest.main()
