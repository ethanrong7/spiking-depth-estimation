"""Explicit FP32 Ascend adapter; no global CUDA redirection or model rewrite."""
from contextlib import contextmanager, nullcontext
import os
import random
import numpy as np
import torch


def configure(device_index=0):
    os.environ['HCCL_DETERMINISTIC'] = 'false'
    import torch_npu
    torch.set_num_threads(4)
    torch.npu.set_device(device_index)
    torch.npu.conv.allow_hf32 = False
    torch.npu.matmul.allow_hf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    torch.npu.set_compile_mode(jit_compile=False)
    import core.spikestereonet as native
    def fp32_context(enabled=False, **kwargs):
        if enabled:
            raise ValueError('This study permits FP32 only')
        return nullcontext()
    native.autocast = fp32_context
    return torch.device(f'npu:{device_index}')


def environment():
    import sys, torch_npu
    return dict(python=sys.version, torch=torch.__version__, torch_npu=torch_npu.__version__,
                npu_name=torch.npu.get_device_name(), device_count=torch.npu.device_count(),
                visible_devices=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
                precision=dict(dtype='float32', conv_allow_hf32=False,
                               matmul_allow_hf32=False, mixed_precision=False, jit_compile=False,
                               deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                               cudnn_deterministic=torch.backends.cudnn.deterministic,
                               cudnn_benchmark=torch.backends.cudnn.benchmark,
                               hccl_deterministic=os.environ['HCCL_DETERMINISTIC']))


def rng_state(loader=None):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), npu=torch.npu.get_rng_state().cpu(),
                loader=loader.generator.get_state() if loader is not None else None)


def restore_rng(state, loader=None):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.npu.set_rng_state(state['npu'])
    if loader is not None:
        loader.generator.set_state(state['loader'])


@contextmanager
def preserve_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value
