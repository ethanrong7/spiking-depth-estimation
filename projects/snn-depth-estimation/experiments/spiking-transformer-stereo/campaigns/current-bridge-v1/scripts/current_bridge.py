"""Parameter-identical signed-current bridge for the frozen original ALIF."""
from types import MethodType
import torch


def install_current_bridge(model, strength=1.0):
    """Keep all original tensors/gates; share signed q preactivation as drive.

    This changes direct access to normalized signed current, not normalization
    alone. q_pre remains the input to the existing gamma reset gate as well.
    """
    for name in ('snn32', 'snn16', 'snn08'):
        cell = getattr(model.update_block, name)
        if hasattr(cell, 'bridge_strength'):
            raise ValueError('bridge already installed')
        cell.bridge_strength = float(strength)
        original_charge = cell.snn.neuronal_charge
        def make_charge(base_charge):
            def charge(neuron, s, h, alpha):
                base_charge(s, h, alpha)
                if neuron.bridge_strength != 0.0:
                    neuron.v = neuron.v + (1.0 - alpha) * neuron.bridge_strength * neuron._bridge_current
            return charge
        cell.snn.bridge_strength = float(strength)
        cell.snn.neuronal_charge = MethodType(make_charge(original_charge), cell.snn)
        def forward(this, s, h, cz, cr, cq, *x_list):
            x = torch.cat(x_list, dim=1)
            hx = torch.cat([h, x], dim=1)
            z = this.parametric_sigmoid(this.ln_z(this.convz(hx) + cz))
            r = this.parametric_sigmoid(this.ln_r(this.convr(hx) + cr))
            q_pre = this.ln_q(this.convq(hx) + cq)
            q = this.parametric_sigmoid(q_pre)
            this.snn._bridge_current = q_pre
            try:
                return this.snn(s, h, z, r, q)
            finally:
                del this.snn._bridge_current
        cell.forward = MethodType(forward, cell)
    return model
