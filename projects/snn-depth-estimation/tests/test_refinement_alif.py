"""Tiny CPU checks for causal adaptation, temporal gradients and reset."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.refinement_alif import RefinementALIF
from core.spiking_attention_update import RefinementLIF


class AdaptiveNeuronTests(unittest.TestCase):
    def test_first_step_matches_lif_and_next_threshold_uses_previous_spike(self):
        lif, alif = RefinementLIF(), RefinementALIF()
        current = torch.tensor([0.8, 0.1])
        for a, b in zip(lif(current), alif(current)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        alif(current)
        torch.testing.assert_close(alif.threshold, torch.tensor([0.55, 0.5]))
        self.assertEqual(sum(p.numel() for p in alif.parameters()), 0)

    def test_gradient_through_adaptation_and_reset(self):
        neuron = RefinementALIF()
        first = torch.tensor([0.8], requires_grad=True)
        initial = tuple(x.detach().clone() for x in neuron(first))
        neuron(torch.tensor([0.2]))
        neuron.adaptation.sum().backward()
        self.assertGreater(first.grad.abs().item(), 0)
        neuron.reset()
        self.assertIsNone(neuron.adaptation)
        self.assertIsNone(neuron.previous_spike)
        self.assertIsNone(neuron.membrane)
        for a, b in zip(neuron(first.detach()), initial):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_zero_adaptation_strength_matches_lif_through_multiple_steps(self):
        lif, alif = RefinementLIF(), RefinementALIF(adaptation_strength=0)
        for current in [torch.tensor([0.8]), torch.tensor([0.2]), torch.tensor([0.7])]:
            for a, b in zip(lif(current), alif(current)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
