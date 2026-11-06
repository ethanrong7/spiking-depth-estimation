"""Bounded recipe changes around immutable historical scientific source."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ARMS = ('original', 'original_I', 'spiking_normalized')
DRIVER_SHA = '078d3342b88c8674c03910260aa8fc1aaad210430131822b9eee600e1605ab7c'
MECHANISM_SHA = 'f8a4147f682a4794ee7f814defcb3684d14764958d61b6606e527de9865f9ae0'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen(base):
    """Import only from the registered historical capsule, never the checkout core."""
    base = Path(base).resolve()
    registry = json.loads((base / 'source-registry.json').read_text())
    project = base / 'projects/snn-depth-estimation'
    experiment = project / 'experiments/spiking-transformer-stereo'
    for name, expected in registry['source_files'].items():
        if '/core/' in name or name.endswith(('/scripts/run_experiment.py', '/scripts/run_provenance.py', '/configs/manifest.json', '/scripts/mechanism.py')):
            if digest(base / name) != expected:
                raise RuntimeError('historical source mismatch: ' + name)
    driver_path = experiment / 'scripts/run_experiment.py'
    mechanism_path = experiment / 'campaigns/alif-mechanism-screen-v1/scripts/mechanism.py'
    if digest(driver_path) != DRIVER_SHA or digest(mechanism_path) != MECHANISM_SHA:
        raise RuntimeError('unexpected frozen driver or I adapter')
    sys.path[:0] = [str(experiment / 'scripts'), str(project), str(project / 'core')]
    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        obj = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(obj)
        return obj
    driver = module('historical_recipe_driver', driver_path)
    mechanism = module('historical_recipe_mechanism', mechanism_path)
    for name, obj in list(sys.modules.items()):
        if name == 'core' or name.startswith('core.'):
            location = getattr(obj, '__file__', None)
            if location and not Path(location).resolve().is_relative_to(project / 'core'):
                raise RuntimeError('noncapsule core module: ' + name)
    return driver, mechanism, registry


def build_model(driver, mechanism, arm, seed):
    if arm not in ARMS:
        raise ValueError(arm)
    if arm == 'spiking_normalized':
        return driver.build_model('spiking', seed, True)
    model = driver.build_model('original', seed, False)
    return mechanism.adapt(model, 'I', seed) if arm == 'original_I' else model


def paired_flags(seed, presentation):
    """Stable per-presentation p=.5 draws independent of model and mutable RNG."""
    value = hashlib.sha256(f'recipe-hv-v1:{seed}:{presentation}'.encode()).digest()
    return value[0] < 128, value[1] < 128


def augment(item, horizontal, vertical):
    left, right, target, valid, index = item
    if horizontal:
        left, right, target, valid = [x.flip(-1) for x in (left, right, target, valid)]
        target = -target
    if vertical:
        left, right, target, valid = [x.flip(-2) for x in (left, right, target, valid)]
    return left, right, target, valid, index


def batch_indices(order, step, effective_batch=8):
    return [order[(step * effective_batch + i) % len(order)] for i in range(effective_batch)]


def validate_config(config):
    expected = dict(seed=1234, updates=10000, schedule_updates=10000, effective_batch=8,
                    train_refinements=16, eval_refinements=32, eval_batch=2, lr=.0002,
                    weight_decay=1e-5, epsilon=1e-8, clip_grad=1.,
                    eval_updates=[0, 1000, 2500, 5000, 7500, 10000],
                    augmentation={'horizontal_probability': .5, 'vertical_probability': .5,
                                  'horizontal': 'mirror_both_no_swap_negate_signed_flow',
                                  'vertical': 'mirror_both_keep_signed_flow'})
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError('recipe mismatch: ' + key)
    if config['model'] not in ARMS or config['physical_batch'] not in (4, 8):
        raise RuntimeError('unsupported arm or physical batch')
    if config['physical_batch'] != 8 and not config.get('common_fallback_authorized', False):
        raise RuntimeError('physical4 requires prospective common-arm fallback registration')
