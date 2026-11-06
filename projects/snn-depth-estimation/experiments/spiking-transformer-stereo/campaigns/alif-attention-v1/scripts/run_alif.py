"""Use the unchanged paired driver with a parameter-matched ALIF neuron hook."""
import importlib.util
import sys
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(EXPERIMENT / 'scripts'))
spec = importlib.util.spec_from_file_location('paired_driver', EXPERIMENT / 'scripts/run_experiment.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
from core.refinement_alif import install_alif_neurons

base_build = driver.build_model


def build_model(name, seed, normalize_residual_current=False):
    if name != 'alif_attention':
        raise ValueError('this entry point only runs the registered ALIF-attention model')
    return install_alif_neurons(base_build('spiking', seed, normalize_residual_current))


driver.build_model = build_model

if __name__ == '__main__':
    driver.main()
