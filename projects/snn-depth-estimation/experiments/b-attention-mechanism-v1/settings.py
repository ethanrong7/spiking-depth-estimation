"""Registered run identity and file-based imports; no CUDA initialization."""
import importlib.util
from pathlib import Path

EXP = Path(__file__).resolve().parent
LEGACY = EXP.parent / 'conventional-attention-placement-v1'
SEEDS = (1234, 2345)
VARIANTS = ('linear-identity', 'signed-surrogate', 'linear-surrogate', 'signed-identity')


def validate(seed, variant):
    if seed not in SEEDS or variant not in VARIANTS:
        raise ValueError(f'Unregistered B control: {seed}/{variant}')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_folder(root, seed, variant):
    validate(seed, variant)
    return Path(root) / f'seed-{seed}' / variant
