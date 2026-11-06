"""Parameter-matched adaptive-threshold neurons for the attention ablation.

These are refinement-time ALIF dynamics, not the upstream convolutional ALIF
updater. Adaptation follows previous spikes with full temporal gradients.
"""
import torch
from torch import nn

from core.spiking_attention_update import _Spike


class RefinementALIF(nn.Module):
    def __init__(self, decay=0.5, threshold=0.5, adaptation_decay=0.9,
                 adaptation_strength=0.5):
        super().__init__()
        self.decay = decay
        self.base_threshold = threshold
        self.adaptation_decay = adaptation_decay
        self.adaptation_strength = adaptation_strength
        self.reset()

    def reset(self):
        self.membrane = None
        self.adaptation = None
        self.previous_spike = None
        # The existing attention diagnostics read the threshold used this step.
        self.threshold = self.base_threshold

    def forward(self, current):
        previous = 0.0 if self.membrane is None else self.membrane
        if self.adaptation is None:
            adaptation = torch.zeros_like(current)
        else:
            adaptation = (self.adaptation_decay * self.adaptation
                          + (1 - self.adaptation_decay) * self.previous_spike)
        threshold = self.base_threshold + self.adaptation_strength * adaptation
        voltage = self.decay * previous + current
        spike = _Spike.apply(voltage - threshold)
        self.membrane = voltage - threshold * spike
        self.adaptation = adaptation
        self.previous_spike = spike
        self.threshold = threshold.detach()
        return spike, self.membrane


def install_alif_neurons(model):
    """Keep all learned tensors and replace only the four neurons per scale."""
    for cell in model.update_block.cells:
        for name in ('q_lif', 'k_lif', 'v_lif', 'output_lif'):
            old = getattr(cell, name)
            setattr(cell, name, RefinementALIF(decay=old.decay, threshold=old.threshold))
    return model
