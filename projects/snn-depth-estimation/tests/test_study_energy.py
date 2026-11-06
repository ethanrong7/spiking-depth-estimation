"""Tiny arithmetic boundary checks; no GPU/energy experiment on Mac."""
import unittest
from core.study_energy import integrate_power


class EnergyIntegrationTest(unittest.TestCase):
    def test_constant_and_linear_interpolated_endpoints(self):
        self.assertAlmostEqual(integrate_power([(0, 150), (60, 150)], 0, 60), 9000)
        self.assertAlmostEqual(integrate_power([(0, 100), (10, 200)], 2, 8), 900)

    def test_refuse_missing_coverage_and_invalid_samples(self):
        with self.assertRaises(ValueError):
            integrate_power([(1, 100), (10, 100)], 0, 10)
        with self.assertRaises(ValueError):
            integrate_power([(0, 100), (10, float('nan'))], 0, 10)


if __name__ == '__main__':
    unittest.main()
