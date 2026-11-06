import unittest

import torch

from core.corr import CorrBlock1D, PytorchAlternateCorrBlock1D


class CorrelationEquivalenceTest(unittest.TestCase):
    def test_regular_and_alternate_outputs_and_feature_gradients_match(self):
        torch.manual_seed(23)
        fmap1_regular = torch.randn(1, 8, 2, 32, requires_grad=True)
        fmap2_regular = torch.randn(1, 8, 2, 32, requires_grad=True)
        fmap1_alternate = fmap1_regular.detach().clone().requires_grad_(True)
        fmap2_alternate = fmap2_regular.detach().clone().requires_grad_(True)

        x = torch.linspace(8.0, 23.0, 32).view(1, 1, 1, 32).expand(1, 1, 2, 32)
        y = torch.arange(2.0).view(1, 1, 2, 1).expand(1, 1, 2, 32)
        coords = torch.cat([x, y], dim=1)

        regular = CorrBlock1D(
            fmap1_regular, fmap2_regular, num_levels=4, radius=4
        )(coords)
        alternate = PytorchAlternateCorrBlock1D(
            fmap1_alternate, fmap2_alternate, num_levels=4, radius=4
        )(coords)

        self.assertTrue(torch.allclose(regular, alternate, atol=2e-6, rtol=2e-5))
        regular.square().mean().backward()
        alternate.square().mean().backward()
        self.assertTrue(
            torch.allclose(
                fmap1_regular.grad,
                fmap1_alternate.grad,
                atol=2e-6,
                rtol=2e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                fmap2_regular.grad,
                fmap2_alternate.grad,
                atol=2e-6,
                rtol=2e-5,
            )
        )


if __name__ == "__main__":
    unittest.main()
