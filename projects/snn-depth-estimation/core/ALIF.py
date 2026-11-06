import torch
import torch.nn as nn
from spikingjelly.activation_based.neuron import surrogate, base
from typing import Callable
from abc import abstractmethod



class ALIFNode32(base.MemoryModule):
    def __init__(self, v_peak: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.ATan(), detach_reset: bool = False,
                 step_mode='s', backend='torch', store_v_seq: bool = False):  
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
        else:
            self.register_memory('v', v_reset)

        self.register_memory('spike', 0.)

        self.v_peak = v_peak
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function
        self.step_mode = step_mode
        self.backend = backend
        self.store_v_seq = store_v_seq

        self.forward_kernel = None
        self.backward_kernel = None

        self.convw = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)  # [B, C, H, W] -> [B, C, H, W]

    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value:
            if not hasattr(self, 'v_seq'):
                self.register_memory('v_seq', None)

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: torch.Tensor, gamma: torch.Tensor):
        v = v - spike * gamma * v_threshold
        return v
    
    @abstractmethod
    def neuronal_charge(self, s: torch.Tensor, h: torch.Tensor, alpha: torch.Tensor):
        self.v = alpha * self.v + (1. - alpha) * (self.convw(h))

    def threshold_adaptation(self, beta):
        self.v_threshold = beta * self.v_peak

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike, gamma):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        # soft reset
        self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold, gamma)

    def single_step_forward(self, s: torch.Tensor, h: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor):
        self.v_float_to_tensor(alpha)
        self.threshold_adaptation(beta)
        self.neuronal_charge(s, h, alpha)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike, gamma)

        return spike

    def v_float_to_tensor(self, alpha: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(alpha.data, v_init)
    

class ALIFNode16(ALIFNode32):
    def __init__(self, v_peak: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.ATan(), detach_reset: bool = False,
                 step_mode='s', backend='torch', store_v_seq: bool = False):  
        super().__init__(v_peak, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)

        self.convw = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)  # [B, C, H, W] -> [B, C, H, W]

        self.convf = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
    
    @abstractmethod
    def neuronal_charge(self, s: torch.Tensor, h: torch.Tensor, alpha: torch.Tensor):
        self.v = alpha * self.v + (1. - alpha) * (self.convf(s) + self.convw(h))


class ALIFNode08(ALIFNode16):
    def __init__(self, v_peak: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.ATan(), detach_reset: bool = False,
                 step_mode='s', backend='torch', store_v_seq: bool = False):  
        super().__init__(v_peak, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)

        self.convw = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)

        self.convf = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

    @abstractmethod
    def neuronal_charge(self, s: torch.Tensor, h: torch.Tensor, alpha: torch.Tensor):
        self.v = alpha * self.v + (1. - alpha) * (self.convf(s)[:, :, :self.v.size(2), :] + self.convw(h))
        
    def single_step_forward(self, s: torch.Tensor, h: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor):
        self.v_float_to_tensor(alpha)
        self.threshold_adaptation(beta)
        self.neuronal_charge(s, h, alpha)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike, gamma)

        return spike, self.v
