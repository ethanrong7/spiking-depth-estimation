"""Registered run identity and file-based imports; no CUDA initialization."""
import importlib.util
import os
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


def add_input_arguments(parser):
    parser.add_argument('--input-mode', choices=('legacy', 'fast', 'preload'),
                        default=os.environ.get('B_ATTENTION_INPUT_MODE', 'legacy'))
    parser.add_argument('--preload-gib', type=float,
                        default=float(os.environ.get('B_ATTENTION_PRELOAD_GIB', '12')))
    parser.add_argument('--input-workers', type=int,
                        default=int(os.environ.get('B_ATTENTION_INPUT_WORKERS', '2')))
    parser.add_argument('--prefetch-depth', type=int,
                        default=int(os.environ.get('B_ATTENTION_PREFETCH_DEPTH', '2')))
