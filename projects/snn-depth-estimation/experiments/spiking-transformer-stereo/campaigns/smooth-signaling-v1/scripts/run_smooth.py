"""Experiment-local smooth signal with the frozen LIF dynamics and coincidence rule."""
import importlib.util
import math
from pathlib import Path
import sys
from types import MethodType

EXPERIMENT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(EXPERIMENT / 'scripts'))
spec = importlib.util.spec_from_file_location('smooth_training_driver', EXPERIMENT / 'scripts/run_experiment.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
from core import spiking_attention_update as attention

base_build = driver.build_model
hard_spike = attention._Spike
base_training_contract = driver.training_contract
base_diagnostics = attention.LocalAttentionRefinement.diagnostics
SIGNAL_IDENTITY = dict(name='smooth_atan_coincidence', activation='0.5+atan(pi*x)/pi',
    derivative='1/(1+(pi*x)^2)', changed_sites=['q_lif', 'k_lif', 'v_lif', 'output_lif'],
    reset='unchanged voltage-threshold*continuous_signal; continuous reset trajectories differ from hard spikes',
    attention='unchanged nonnegative coincidence divided by 1+neighbor score sum; no softmax')


class SmoothSignal:
    @staticmethod
    def apply(x):
        return 0.5 + (math.pi * x).atan() / math.pi


def activation_diagnostics(cell):
    values = base_diagnostics(cell)
    rename = {'q_rate': 'q_activation_mean', 'k_rate': 'k_activation_mean',
              'v_rate': 'v_activation_mean', 'output_rate': 'output_activation_mean',
              'nonzero_attention_fraction': 'positive_attention_weight_fraction',
              'output_surrogate_derivative_mean': 'output_signal_derivative_mean'}
    return {rename.get(key, key): value for key, value in values.items()}


def build_model(name, seed, normalize_residual_current=False):
    if name != 'smooth_coincidence' or not normalize_residual_current:
        raise ValueError('Requires registered normalized smooth_coincidence model')
    model = base_build('spiking', seed, True)
    # The unchanged RefinementLIF calls this global at its four signal sites.
    # Override only in this experiment's process; source/core files stay frozen.
    attention._Spike = SmoothSignal
    for cell in model.update_block.cells:
        cell.diagnostics = MethodType(activation_diagnostics, cell)
    return model


def training_contract(*args, **kwargs):
    return {**base_training_contract(*args, **kwargs), 'signal_identity': SIGNAL_IDENTITY}


driver.build_model = build_model
driver.training_contract = training_contract
if __name__ == '__main__':
    driver.main()
