"""Tiny deterministic loader/control tests; no performance claims on the Mac."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import torch

from core.study_prefetch import OrderedBatchStream, prefetch_options
from core.study_training import (evaluate_validation, optimizer_and_scheduler,
                                 restore_checkpoint, sample_indices, save_checkpoint,
                                 train_update)
from test_study_training import ToyDataset, ToyModel, toy_config


class PrefetchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_queue_is_bounded_ordered_and_keeps_metadata(self):
        data = ToyDataset()
        with OrderedBatchStream(data, seed=9471, position=0, microbatch=2, end_position=24,
                                device='cpu', options={'workers': 2, 'batches': 2}) as stream:
            for position in range(0, 24, 2):
                self.assertLessEqual(len(stream.queue), 2)
                batch, indices, timing, events = stream.take(position)
                self.assertEqual(indices, sample_indices(9471, position, 2, len(data)))
                self.assertEqual(batch['sample_id'], [str(i) for i in indices])
                self.assertIsNone(batch['metadata'][0]['physical_fps'])
                self.assertEqual(batch['left'].shape, (2, 50, 1, 2, 2))
                self.assertEqual(events, (None, None))
                self.assertFalse(stream.stage_next())
            with self.assertRaises(RuntimeError):
                stream.take(24)
        self.assertFalse(stream.queue)

    def test_exception_surfaces_in_consumption_order_and_closes(self):
        class Broken(ToyDataset):
            def __getitem__(self, i):
                if i == 2:
                    raise ValueError('sample2_decode_failed')
                return super().__getitem__(i)
        stream = OrderedBatchStream(Broken(), seed=0, position=0, microbatch=1, end_position=5,
                                    device='cpu', options={'workers': 2},
                                    indices_fn=lambda seed, pos, count, length: [pos])
        with stream:
            self.assertEqual(stream.take(0)[1], [0])
            self.assertEqual(stream.take(1)[1], [1])
            with self.assertRaisesRegex(ValueError, 'sample2_decode_failed'):
                stream.take(2)
        self.assertTrue(stream.closed)

    def test_optimizer_rng_and_resume_parity_with_speculative_batches(self):
        data, config = ToyDataset(), toy_config()
        torch.manual_seed(27)
        original = ToyModel()
        binding = {'effective_batch': 8, 'total_updates': 3}
        def make():
            model = copy.deepcopy(original)
            optimizer, scheduler = optimizer_and_scheduler(model, config['optimizer'], 3)
            return model, optimizer, scheduler
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            reference, ro, rs = make()
            torch.manual_seed(61)
            reference_records = [train_update(reference, ro, rs, data, config, pos, 'cpu', folder/'fail.json')
                                 for pos in (0, 8, 16)]
            pipelined, po, ps = make()
            torch.manual_seed(61)
            with OrderedBatchStream(data, seed=config['seed'], position=0, microbatch=2, end_position=24,
                                    device='cpu', options={'workers': 2}) as stream:
                first = train_update(pipelined, po, ps, data, config, 0, 'cpu', folder/'fail.json', stream)
                save_checkpoint(folder/'latest.pt', pipelined, po, ps, {'presentations': 8, 'update': 1}, binding, {})
                # Queue already contains uncommitted later samples; discard it.
            resumed, so, ss = make()
            state = restore_checkpoint(folder/'latest.pt', resumed, so, ss, binding, {})
            records = [first]
            with OrderedBatchStream(data, seed=config['seed'], position=state['presentations'], microbatch=2,
                                    end_position=24, device='cpu', options={'workers': 2}) as stream:
                for pos in (8, 16):
                    records.append(train_update(resumed, so, ss, data, config, pos, 'cpu', folder/'fail.json', stream))
            self.assertEqual([r['loss'] for r in reference_records], [r['loss'] for r in records])
            self.assertEqual([r['sample_indices'] for r in reference_records], [r['sample_indices'] for r in records])
            for key, tensor in reference.state_dict().items():
                self.assertTrue(torch.equal(tensor, resumed.state_dict()[key]), key)
            for left, right in zip(ro.state.values(), so.state.values()):
                for key in left:
                    self.assertTrue(torch.equal(left[key], right[key]), key)
            self.assertEqual(rs.state_dict(), ss.state_dict())

    def test_speculative_decode_failure_does_not_abort_current_update(self):
        class Broken(ToyDataset):
            def __getitem__(self, i):
                if i == 1:
                    raise ValueError('next_position_failed')
                return super().__getitem__(i)
        with OrderedBatchStream(Broken(), seed=0, position=0, microbatch=1, end_position=3,
                                device='cpu', options={'workers': 2},
                                indices_fn=lambda seed, pos, count, length: [pos]) as stream:
            stream.take(0)
            # Exercise the device-staging decision without CUDA on the Mac.
            stream.copy_stream = Mock()
            stream.queue[0][1].exception()
            self.assertFalse(stream.stage_next())
            self.assertEqual(stream.position, 1)
            with self.assertRaisesRegex(ValueError, 'next_position_failed'):
                stream.take(1)

    def test_validation_parity_and_option_bounds(self):
        torch.manual_seed(2)
        model = ToyModel()
        reference = evaluate_validation(model, ToyDataset(), 'cpu')
        prefetched = evaluate_validation(model, ToyDataset(), 'cpu', prefetch={'workers': 2})
        self.assertEqual(reference, prefetched)
        for values in ({'workers': 5}, {'batches': 0}, {'pin_memory': 1}, {'unknown': True}):
            with self.assertRaises(ValueError):
                prefetch_options({'prefetch': values})


if __name__ == '__main__':
    unittest.main()
