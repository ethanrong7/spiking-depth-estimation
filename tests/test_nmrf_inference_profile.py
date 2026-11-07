import unittest
import torch
from torch import nn
from profile_nmrf_inference import active_connections, contraction_counts, ActivityAudit


class InferenceProfileTests(unittest.TestCase):
    def test_synaptic_fanout_stride_padding_groups(self):
        for stride, padding, dilation, groups in [(1, 1, 1, 1), (2, 1, 1, 2), (1, 2, 2, 2)]:
            conv = nn.Conv2d(2, 4, 3, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
            x = torch.zeros(2, 2, 5, 7)
            x[0, 0, 0, 0] = 1
            x[0, 1, 2, 3] = -1
            x[1, 1, 4, 6] = 1
            h, w = conv(x).shape[-2:]
            expected = 0
            for n, c, y, z in x.nonzero().tolist():
                for oy in range(h):
                    for ox in range(w):
                        for ky in range(3):
                            for kx in range(3):
                                if oy*stride-padding+ky*dilation == y and ox*stride-padding+kx*dilation == z:
                                    expected += 4//groups
            self.assertEqual(active_connections(conv, x), expected)

    def test_full_model_hooks_preserve_prediction_and_match_conv_counts(self):
        from spiking_nmrf import build_spiking_nmrf
        from nmrf_runtime import build_native_nmrf
        torch.set_num_threads(2)
        left = torch.randint(0, 2, (1, 50, 32, 64)).float()
        for kind, build in [('ann', build_native_nmrf), ('spiking', build_spiking_nmrf)]:
            model, _, _ = build()
            model.eval()
            with torch.no_grad():
                reference = model(left, left)['disp']
                output, counts = contraction_counts(model, left, left)
                audit = ActivityAudit(model, kind)
                try:
                    observed = model(left, left)['disp']
                finally:
                    audit.close()
            self.assertTrue(torch.equal(reference, output))
            self.assertTrue(torch.equal(reference, observed))
            self.assertGreater(counts['continuous_rest_flops'], 0)
            for name, row in audit.rows.items():
                path = next(k for k in counts['by_module'] if k.endswith('.backbone.'+name))
                self.assertEqual(counts['by_module'][path], 2*row['dense_macs'])
                self.assertLessEqual(row['active_acs'], row['dense_macs'])
                self.assertLessEqual(row['active_macs'], row['dense_macs'])


if __name__ == '__main__':
    unittest.main()
