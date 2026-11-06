"""Tiny CPU-only intervention math tests, separate from the real decoder run."""
import unittest
import torch
from controls import fit_control


class ControlsTest(unittest.TestCase):
    def test_positive_affine_exact_fit_and_moment_controls(self):
        x=torch.tensor([[1.,3.,2.,5.],[4.,2.,8.,1.]])
        y=3*x+2
        result,stats=fit_control(x,y,'positive_affine_LS')
        torch.testing.assert_close(result,y,atol=1e-6,rtol=0)
        self.assertLess(stats['affine_fit_MSE'],1e-20)
        raw,_=fit_control(x,y,'raw_match_Y_moments')
        torch.testing.assert_close(raw,y,atol=1e-6,rtol=0)
        learned,_=fit_control(x,y,'Y_match_X_moments')
        torch.testing.assert_close(learned,x,atol=1e-6,rtol=0)

    def test_degenerate_fallback_preserves_learned_shape(self):
        x=torch.ones(1,7)
        y=torch.arange(7).float()[None]
        learned,stats=fit_control(x,y,'Y_match_X_moments')
        self.assertTrue(torch.equal(learned,y))
        self.assertEqual(stats['fallback_fraction'],1.)
        raw,_=fit_control(x,y,'raw_match_Y_moments')
        self.assertTrue(torch.equal(raw,torch.full_like(raw,3)))

    def test_smoothing_boundary_excludes_absent_neighbors_and_negative_affine(self):
        x=torch.arange(7).float()[None]
        y=-x
        result,stats=fit_control(x,y,'positive_affine_LS')
        self.assertEqual(stats['affine_floor_fraction'],1.)
        self.assertTrue(torch.all(result[:,1:]>=result[:,:-1]))
        smooth,_=fit_control(x,y,'smooth5_match_Y_moments')
        self.assertAlmostEqual(smooth.mean().item(),y.mean().item(),places=6)
        self.assertAlmostEqual(smooth.std(unbiased=False).item(),y.std(unbiased=False).item(),places=6)


if __name__=='__main__':unittest.main()
