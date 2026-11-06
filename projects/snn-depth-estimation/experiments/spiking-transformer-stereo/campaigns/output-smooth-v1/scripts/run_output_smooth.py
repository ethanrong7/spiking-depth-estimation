"""Selective continuous output signaling; Q/K/V retain the frozen hard operator."""
import importlib.util
import math
from pathlib import Path
import sys
from types import MethodType
import torch

EXPERIMENT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(EXPERIMENT / 'scripts'))
spec = importlib.util.spec_from_file_location('output_smooth_driver', EXPERIMENT / 'scripts/run_experiment.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
from core import spiking_attention_update as attention

base_build = driver.build_model
base_training_contract = driver.training_contract
base_diagnostics = attention.LocalAttentionRefinement.diagnostics
HARD_SPIKE = attention._Spike
SIGNAL_IDENTITY = dict(name='output_smooth_hard_qkv',
    changed_sites=['output_lif'], changed_scales=[0,1,2], unchanged_hard_sites=['q_lif', 'k_lif', 'v_lif'],
    output_activation='0.5+atan(pi*x)/pi', output_derivative='1/(1+(pi*x)^2)',
    qkv_activation='frozen _Spike.apply hard binary forward with original surrogate',
    membrane_decay=.5, threshold=.5, charge='u=0.5*previous_membrane+current',
    reset='membrane=u-0.5*site_signal; full gradients; output reset trajectories change',
    attention='unchanged nonnegative hard-QKV coincidence divided by 1+neighbor score sum; no softmax',
    global_spike_override=False)


class SmoothOutputLIF(attention.RefinementLIF):
    """Parameter-free selective output subclass; inherited reset/state interface."""
    def forward(self, current):
        previous = 0.0 if self.membrane is None else self.membrane
        voltage = self.decay * previous + current
        signal = 0.5 + (math.pi * (voltage - self.threshold)).atan() / math.pi
        self.membrane = voltage - self.threshold * signal
        return signal, self.membrane


def mixed_diagnostics(cell):
    values = base_diagnostics(cell)
    rename = {'output_rate': 'output_activation_mean',
              'output_surrogate_derivative_mean': 'output_signal_derivative_mean'}
    # Q/K/V remain binary: their rate labels retain their original meaning.
    return {rename.get(key, key): value for key, value in values.items()}


def install_output_smooth(model):
    if attention._Spike is not HARD_SPIKE or HARD_SPIKE.__name__ != '_Spike' or not issubclass(HARD_SPIKE, torch.autograd.Function):
        raise RuntimeError('QKV hard operator was changed by another process-local builder')
    for cell in model.update_block.cells:
        if not all(type(lif) is attention.RefinementLIF for lif in (cell.q_lif,cell.k_lif,cell.v_lif,cell.output_lif)):
            raise ValueError('requires unchanged full hard-LIF cells')
        old = cell.output_lif
        cell.output_lif = SmoothOutputLIF(decay=old.decay, threshold=old.threshold)
        cell.diagnostics = MethodType(mixed_diagnostics, cell)
    return model


def build_model(name, seed, normalize_residual_current=False):
    if name != 'output_smooth_hard_qkv' or not normalize_residual_current:
        raise ValueError('requires registered normalized output_smooth_hard_qkv model')
    return install_output_smooth(base_build('spiking', seed, True))


def training_contract(*args, **kwargs):
    return {**base_training_contract(*args, **kwargs), 'signal_identity': SIGNAL_IDENTITY}


driver.build_model = build_model
driver.training_contract = training_contract
if __name__ == '__main__':
    driver.main()
