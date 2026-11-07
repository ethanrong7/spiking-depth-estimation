import unittest
import torch
from torch import nn
from nmrf_inference_runtime import CUDAGraphInference


class GraphRuntimeTests(unittest.TestCase):
    def test_rejects_cpu_and_training_model(self):
        model = nn.Linear(3, 2)
        with self.assertRaisesRegex(ValueError, 'eval-mode'):
            CUDAGraphInference(model, (torch.ones(1, 3),))
        with self.assertRaisesRegex(ValueError, 'CUDA tensor'):
            CUDAGraphInference(model.eval(), (torch.ones(1, 3),))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA Graph needs GPU')
    def test_replay_fresh_inputs_owned_outputs_and_shape_guard(self):
        class Model(nn.Module):
            def forward(self, x):
                return {'value': x*2+1, 'nested': [x.square()]}
        model = Model().cuda().eval()
        x = torch.arange(10, device='cuda').float()
        graph = CUDAGraphInference(model, (x,))
        with torch.no_grad():
            first = graph(x)
            saved = first['value'].clone()
            second = graph(x+2)
            self.assertTrue(torch.equal(second['value'], model(x+2)['value']))
            self.assertTrue(torch.equal(first['value'], saved))
            with self.assertRaisesRegex(ValueError, 'shape/dtype/device'):
                graph(x[:5])
        with self.assertRaisesRegex(RuntimeError, 'inference-only'):
            graph.train()
        with self.assertRaisesRegex(RuntimeError, 'no_grad'):
            graph(x)


if __name__ == '__main__':
    unittest.main()
