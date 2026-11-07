"""CPU objective/gradient/optimizer equivalence and invalid-input checks."""
import ast
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from speed_runtime import SpeedRuntime


def legacy_loss():
    source = HERE.parent/'spiking-transformer-stereo/scripts/run_experiment.py'
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'disparity_loss')
    scope = {'torch': torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), scope)
    return scope['disparity_loss']


class SpeedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.runtime = SpeedRuntime()
        self.old = legacy_loss()

    def test_loss_and_gradient_match_with_invalid_targets(self):
        for dtype, tolerance in ((torch.float32, 2e-6), (torch.float64, 1e-12)):
            target = torch.randn(4, 1, 8, 12, dtype=dtype)
            target[0, 0, 0, :3] = torch.tensor([float('nan'), float('inf'), 800.], dtype=dtype)
            valid = torch.rand(4, 8, 12) > .25
            old = [torch.randn_like(target, requires_grad=True) for _ in range(16)]
            new = [p.detach().clone().requires_grad_() for p in old]
            expected = self.old(old, target, valid)
            actual, okay = self.runtime.loss(new, target, valid)
            self.assertTrue(okay)
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
            expected.backward(); actual.backward()
            for a, b in zip(old, new):
                torch.testing.assert_close(a.grad, b.grad, atol=tolerance, rtol=tolerance)

    def test_adamw_update_and_clipping_match(self):
        a = torch.nn.Conv2d(2, 1, 1)
        b = torch.nn.Conv2d(2, 1, 1); b.load_state_dict(a.state_dict())
        opts = [torch.optim.AdamW(m.parameters(), lr=2e-4) for m in (a, b)]
        for step in range(3):
            x = torch.randn(8, 2, 8, 12)
            target = torch.randn(8, 1, 8, 12)
            valid = torch.rand(8, 8, 12) > .3
            for model, opt, fast in zip((a, b), opts, (False, True)):
                opt.zero_grad(); losses=[]; checks=[]
                for offset in (0, 4):
                    output = model(x[offset:offset+4])
                    preds = [output * (i+1)/16 for i in range(16)]
                    if fast:
                        loss, okay = self.runtime.loss(preds, target[offset:offset+4], valid[offset:offset+4]); checks.append(okay)
                    else:loss = self.old(preds, target[offset:offset+4], valid[offset:offset+4])
                    loss = loss * int(valid[offset:offset+4].sum())/int(valid.sum())
                    loss.backward(); losses.append(loss.detach())
                branch = {n:p.grad.norm() for n,p in model.named_parameters()}
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if fast:self.runtime.summarize(losses, checks, norm, branch)
                opt.step()
            for p, q in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-6)

    def test_rejects_empty_mask_and_nonfinite_predictions(self):
        target = torch.ones(1, 1, 2, 2)
        for empty, invalid in ((True, False), (False, True)):
            pred = target.clone()
            if invalid:pred[0, 0, 0, 0] = float('nan')
            valid = torch.full((1, 2, 2), not empty)
            loss, okay = self.runtime.loss([pred]*16, target, valid)
            with self.assertRaises(FloatingPointError):
                self.runtime.summarize([loss], [okay], torch.tensor(1.), {})
        with self.assertRaises(FloatingPointError):
            self.runtime.summarize([torch.tensor(1.)], [torch.tensor(True)], torch.tensor(float('inf')), {})

    def test_instrumentation_preserves_rng_and_gradient_values(self):
        before = torch.get_rng_state().clone()
        result = self.runtime.summarize([torch.tensor(2.),torch.tensor(3.)],
                                       [torch.tensor(True)]*2, torch.tensor(4.), {'q':torch.tensor(5.)})
        self.assertEqual(result, (5., 4., {'q':5.}))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_optional_host_trace_finishes_after_twelve_updates(self):
        with patch.dict('os.environ', {'B_ATTENTION_PROFILE_CPU':'1'}), tempfile.TemporaryDirectory() as folder:
            runtime = SpeedRuntime()
            runtime.start_profile(Path(folder))
            for _ in range(15):
                with runtime.span('synthetic_forward'):
                    torch.ones(4).square().sum()
                runtime.profile_step()
            runtime.stop_profile()
            self.assertIsNone(runtime.profiler)
            self.assertTrue((Path(folder)/'cpu-trace.json').exists())
            self.assertIn('synthetic_forward', (Path(folder)/'cpu-profile.txt').read_text())


if __name__ == '__main__': unittest.main()
