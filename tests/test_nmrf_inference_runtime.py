import unittest
import torch
from torch import nn
from nmrf_inference_runtime import CUDAGraphInference, pack_qkv_for_inference


class GraphRuntimeTests(unittest.TestCase):
    def test_packed_qkv_keeps_outputs_activity_and_checkpoint(self):
        from spiking_nmrf import SpikingStereoBackbone
        torch.set_num_threads(2)
        torch.manual_seed(1234)
        model = SpikingStereoBackbone().eval()
        packed = pack_qkv_for_inference(model)
        model.collect_activity(True)
        packed.collect_activity(True)
        with torch.no_grad():
            for x in [torch.zeros(2, 50, 32, 64), torch.ones(2, 50, 32, 64)*255,
                      torch.randint(0, 2, (2, 50, 32, 64)).float()*255]:
                expected, actual = model(x), packed(x)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual)))
                self.assertEqual(model.activity(), packed.activity())
        self.assertEqual(model.state_dict().keys(), packed.state_dict().keys())
        self.assertTrue(all(torch.equal(v, packed.state_dict()[k]) for k, v in model.state_dict().items()))
        with self.assertRaisesRegex(RuntimeError, 'inference-only'):
            packed(x)

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
