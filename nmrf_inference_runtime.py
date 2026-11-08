"""Opt-in fixed-shape CUDA Graph inference; load the checkpoint before wrapping.

Keeps FP32 arithmetic, weights, time steps and kernels unchanged. Capture has a
one-time setup cost and retains GPU buffers. The wrapped model must stay frozen
on its captured device; one wrapper serves one input shape/dtype and stream at
a time. No training support. Source: PyTorch CUDA semantics, CUDA Graphs.
https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs
"""
import copy
from types import MethodType

import torch
from torch import nn
from torch.utils._pytree import tree_map


def _packed_sdsa_forward(self, x):
    if self.training or torch.is_grad_enabled():
        raise RuntimeError('Packed QKV is inference-only; use the original model to train')
    t, b, c, h, w = x.shape
    # Each LIF element has an independent membrane. Equal neuron parameters
    # allow Q/K/V to share a kernel without sharing membrane state.
    packed = self.q_lif(self.qkv(self.input_lif(x)))
    q, k, v = packed.chunk(3, 2)
    for neuron, output in [(self.q_lif, q), (self.k_lif, k), (self.v_lif, v)]:
        if neuron.collect_activity:
            neuron.activity = (output.detach().sum(), output.numel())
    q = q.reshape(t, b, self.heads, c//self.heads, h*w)
    k = k.reshape_as(q)
    v = v.reshape_as(q)
    gate = self.gate_lif((q*k).sum(-1, keepdim=True))
    message = (v*gate).reshape(t, b, c, h, w)
    x = x+self.proj(message)
    return x+self.fc2(self.ffn_hidden_lif(self.fc1(self.ffn_in_lif(x))))


def pack_qkv_for_inference(model):
    """Return an eval-only copy batching the independent Q/K/V LIF updates.

    Same state-dict keys and values; training model is left untouched. This
    preserves per-neuron activity values, but module hooks on separate K/V LIF
    calls are bypassed. Use the original model for per-module profiling.
    """
    from spiking_nmrf import SDSABlock
    if any(m.training for m in model.modules()):
        raise ValueError('Set model.eval() before packing QKV')
    result = copy.deepcopy(model)
    count = 0
    for module in result.modules():
        if isinstance(module, SDSABlock):
            settings = {(n.tau, n.threshold) for n in (module.q_lif, module.k_lif, module.v_lif)}
            if len(settings) != 1:
                raise ValueError('Packing requires identical Q/K/V neuron parameters')
            module.forward = MethodType(_packed_sdsa_forward, module)
            count += 1
    if not count:
        raise ValueError('No SDSA blocks found')
    return result


class CUDAGraphInference(nn.Module):
    """Replay a frozen module with tensor arguments and tensor-tree outputs.

    Output tensors are cloned so callers can retain results across replays.
    Input copying and output cloning are included in forward timings. Create
    a new wrapper after changing weights/device or the input shape. Do not
    call concurrently from multiple threads/streams.
    """
    def __init__(self, module, example_inputs, warmup=3):
        super().__init__()
        if module.training or any(m.training for m in module.modules()):
            raise ValueError('CUDA Graph inference requires an eval-mode model')
        if not example_inputs or any(not isinstance(x, torch.Tensor) or not x.is_cuda for x in example_inputs):
            raise ValueError('CUDA tensor example inputs are required')
        if warmup < 1:
            raise ValueError('Warmup must be positive')
        self.module = module
        self.training = False
        self.signature = tuple((tuple(x.shape), x.dtype, x.device) for x in example_inputs)
        with torch.cuda.device(example_inputs[0].device), torch.inference_mode(False), torch.no_grad():
            self.static_inputs = tuple(x.detach().clone() for x in example_inputs)
            stream = torch.cuda.Stream(device=example_inputs[0].device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(warmup):
                    self.static_outputs = module(*self.static_inputs)
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.static_outputs = module(*self.static_inputs)
            torch.cuda.current_stream().wait_stream(stream)

    def train(self, mode=True):
        if mode:
            raise RuntimeError('CUDA Graph wrapper is inference-only; use the original model to train')
        return super().train(False)

    def forward(self, *inputs):
        if torch.is_grad_enabled() or self.module.training:
            raise RuntimeError('Use eval mode and torch.no_grad()/inference_mode()')
        signature = tuple((tuple(x.shape), x.dtype, x.device) for x in inputs)
        if signature != self.signature:
            raise ValueError('Input shape/dtype/device changed; construct a new graph wrapper')
        for saved, current in zip(self.static_inputs, inputs):
            saved.copy_(current)
        self.graph.replay()
        return tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, self.static_outputs)
