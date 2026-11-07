"""Hierarchical spike-driven feature backbone for the continuous NMRF matcher.

Independent PyTorch implementation inspired by Yao et al., Spike-driven
Transformer, NeurIPS 2023 (https://arxiv.org/abs/2307.01694). Uses the paper's
SDSA version 1: gate = SN(sum_tokens(Q*K)); output = V*gate. This is a custom
stereo adaptation, not an upstream-model reproduction. No upstream code copied.

Five real sensor chunks, stateful LIF within each sample, membrane residuals,
binary synaptic inputs, and continuous descriptor readouts. Dense GPU training
simulates spike arithmetic; it does not provide sparse inference acceleration.
"""
import math

import torch
from torch import nn

from nmrf_runtime import build_native_nmrf

MODEL_NAME = 'nmrf-sdsa-backbone-t5-v1'
ARCHITECTURE = dict(model_name=MODEL_NAME, sensor_bins=50, time_steps=5,
                    bins_per_step=10, widths=[64,96,128], depths=[2,2],
                    heads=4, mlp_ratio=2, output_channels=256,
                    lif_tau=2., lif_threshold=1., gate_threshold=.5,
                    reset='hard zero, detached spike reset; local state per forward',
                    surrogate='atan alpha=2', normalization='batch, joint time/batch',
                    attention='SDSA v1: V * LIF(sum_spatial(Q*K))',
                    readout='mean output spikes over five steps, linear 1x1 projection',
                    input='raw 0/1 bins, no signed intensity normalization',
                    matcher='unchanged continuous NMRF')


class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        # Derivative of atan(pi*x)/pi + 1/2; alpha=2 convention.
        return grad/(1+(math.pi*x).square())


class LIF(nn.Module):
    """Time-first tensor; membrane history never leaks to another sample/call."""
    def __init__(self, threshold=1., tau=2.):
        super().__init__()
        self.threshold, self.tau = threshold, tau
        self.collect_activity = False
        self.activity = None

    def forward(self, x):
        potential = torch.zeros_like(x[0])
        spikes = []
        for current in x.unbind(0):
            potential = potential + (current-potential)/self.tau
            emitted = Spike.apply(potential-self.threshold)
            potential = potential*(1-emitted.detach())
            spikes.append(emitted)
        result = torch.stack(spikes)
        if self.collect_activity:
            self.activity = (result.detach().sum(), result.numel())
        return result


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, normalize=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel,stride,kernel//2,bias=not normalize)
        self.norm = nn.BatchNorm2d(out_channels) if normalize else nn.Identity()

    def forward(self,x):
        t,b=x.shape[:2]
        y=self.norm(self.conv(x.flatten(0,1)))
        return y.reshape(t,b,*y.shape[1:])


class SDSABlock(nn.Module):
    def __init__(self, channels, heads=4, expansion=2):
        super().__init__()
        if channels % heads: raise ValueError('Channels must divide heads')
        self.heads=heads
        self.input_lif=LIF()
        self.qkv=TemporalConv(channels,3*channels)
        self.q_lif,self.k_lif,self.v_lif=LIF(),LIF(),LIF()
        self.gate_lif=LIF(threshold=.5)
        self.proj=TemporalConv(channels,channels)
        self.ffn_in_lif=LIF()
        self.fc1=TemporalConv(channels,channels*expansion)
        self.ffn_hidden_lif=LIF()
        self.fc2=TemporalConv(channels*expansion,channels)

    def forward(self,x):
        t,b,c,h,w=x.shape
        q,k,v=self.qkv(self.input_lif(x)).chunk(3,2)
        q=self.q_lif(q).reshape(t,b,self.heads,c//self.heads,h*w)
        k=self.k_lif(k).reshape_as(q)
        v=self.v_lif(v).reshape_as(q)
        gate=self.gate_lif((q*k).sum(-1,keepdim=True))
        message=(v*gate).reshape(t,b,c,h,w)
        x=x+self.proj(message)
        return x+self.fc2(self.ffn_hidden_lif(self.fc1(self.ffn_in_lif(x))))


class SpikingStereoBackbone(nn.Module):
    output_dim=256

    def __init__(self):
        super().__init__()
        self.stem1=TemporalConv(10,64,3,2)
        self.stem_lif=LIF()
        self.stem2=TemporalConv(64,96,3,2)
        self.fine=nn.Sequential(SDSABlock(96),SDSABlock(96))
        self.fine_lif=LIF()
        self.down=TemporalConv(96,128,3,2)
        self.coarse=nn.Sequential(SDSABlock(128),SDSABlock(128))
        self.coarse_lif=LIF()
        self.fine_readout=nn.Conv2d(96,self.output_dim,1)
        self.coarse_readout=nn.Conv2d(128,self.output_dim,1)

    @staticmethod
    def sensor_chunks(x):
        if x.ndim != 4 or x.shape[1] != 50:
            raise ValueError('Expected B,50,H,W raw sensor input')
        return x.reshape(x.shape[0],5,10,*x.shape[-2:]).transpose(0,1)

    def forward(self,x):
        # NMRF's adapter passes 255*s; restore raw sensor bits without 2*s-1.
        sequence=self.sensor_chunks(x/255.)
        fine=self.fine(self.stem2(self.stem_lif(self.stem1(sequence))))
        fine_spikes=self.fine_lif(fine)
        coarse_spikes=self.coarse_lif(self.coarse(self.down(fine_spikes)))
        return [self.fine_readout(fine_spikes.mean(0)),
                self.coarse_readout(coarse_spikes.mean(0))]

    def collect_activity(self,enabled=True):
        for module in self.modules():
            if isinstance(module,LIF):
                module.collect_activity=enabled
                module.activity=None

    def activity(self):
        return {name:float(m.activity[0])/m.activity[1]
                for name,m in self.named_modules() if isinstance(m,LIF) and m.activity is not None}


def build_spiking_nmrf(seed=1234):
    model,criterion,cfg=build_native_nmrf(seed)
    # Preserve the matcher initialization from the native model. Use a separate
    # deterministic seed for the new backbone; no pretrained weights imported.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+10000)
        model.parent.backbone=SpikingStereoBackbone()
    return model,criterion,cfg
