"""Frozen original driver with an isolated optional original-ALIF current bridge."""
import importlib.util
import sys
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(EXPERIMENT / 'scripts'))
spec = importlib.util.spec_from_file_location('bridge_driver', EXPERIMENT / 'scripts/run_experiment.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
from current_bridge import install_current_bridge
base_build = driver.build_model


def build_model(name, seed, normalize_residual_current=False):
    if normalize_residual_current or name not in ('original_low_lr', 'original_current_bridge'):
        raise ValueError('requires registered original-ALIF arm')
    model = base_build('original', seed, False)
    return install_current_bridge(model) if name == 'original_current_bridge' else model


driver.build_model = build_model
if __name__ == '__main__':
    driver.main()
