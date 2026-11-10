"""Full pinned encoder backbones with chronological T5 stereo feature adapters."""
import math
import types
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from upstream import load_definitions, SOURCES
from neuron import MultiStepLIFNode

VARIANTS = ('spikformer', 'qkformer', 'meta-spikeformer', 'swin')


def dynamic_swin(ns):
    """Keep author attention/MLP/merging; pad windows for rectangular stereo."""
    parent = ns['SwinTransformerBlock']
    partition, reverse = ns['window_partition'], ns['window_reverse']

    class RectangularBlock(parent):
        def __init__(self, *args, **kwargs):
            # Original 224-pixel constructor has divisible grids. Restore intended
            # shifted windows for the last stage when actual stereo grids are >7.
            shift = kwargs.get('shift_size', 0)
            super().__init__(*args, **kwargs)
            self.requested_shift = shift
            self._mask_cache = {}

        def forward(self, x):
            h, w = self.input_resolution
            b, length, c = x.shape
            assert length == h*w
            ws = self.window_size
            shift = self.requested_shift if min(h, w) > ws else 0
            hp, wp = math.ceil(h/ws)*ws, math.ceil(w/ws)*ws
            key = (h, w, str(x.device), x.dtype, shift)
            if key not in self._mask_cache:
                # Geometry only, constructed once per shape on CPU then uploaded.
                regions = torch.zeros(1, hp, wp, 1)
                if shift:
                    slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
                    for i, hs in enumerate(slices):
                        for j, vs in enumerate(slices):
                            regions[:, hs, vs] = i*3+j
                ids = partition(regions, ws).reshape(-1, ws*ws)
                mask = (ids[:, :, None] != ids[:, None, :]).float() * -100.
                valid = torch.zeros(1, hp, wp, 1)
                valid[:, :h, :w] = 1
                if shift:
                    valid = torch.roll(valid, (-shift, -shift), (1, 2))
                keys = partition(valid, ws).reshape(-1, ws*ws)
                mask = mask.masked_fill(keys[:, None, :] == 0, -100.)
                self._mask_cache[key] = mask.to(device=x.device, dtype=x.dtype)
            shortcut = x
            x = F.pad(self.norm1(x).reshape(b,h,w,c), (0,0,0,wp-w,0,hp-h))
            if shift:
                x = torch.roll(x, (-shift,-shift), (1,2))
            windows = partition(x,ws).reshape(-1,ws*ws,c)
            x = reverse(self.attn(windows,mask=self._mask_cache[key]).reshape(-1,ws,ws,c),ws,hp,wp)
            if shift:
                x = torch.roll(x, (shift,shift), (1,2))
            x = shortcut + self.drop_path(x[:,:h,:w].reshape(b,h*w,c))
            return x + self.drop_path(self.mlp(self.norm2(x)))

    ns['SwinTransformerBlock'] = RectangularBlock


class StereoEncoder(nn.Module):
    def __init__(self, variant, source_dir, neuron_class=None):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        ns = load_definitions(variant, source_dir, neuron_class)
        common = dict(img_size_h=256, img_size_w=416, patch_size=16, in_channels=10,
                      num_classes=0, mlp_ratios=4, qkv_bias=False, depths=8,
                      drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                      norm_layer=partial(nn.LayerNorm, eps=1e-6), sr_ratios=1)
        self._taps = []
        if variant == 'spikformer':
            self.backbone = ns['Spikformer'](embed_dims=384,num_heads=8,**common)
            modules = [self.backbone.patch_embed.maxpool1,self.backbone.patch_embed.maxpool2,self.backbone.block[-1]]
            channels = [96,192,384]
        elif variant == 'qkformer':
            common['depths'] = 10
            self.backbone = ns['hierarchical_spiking_transformer'](T=5,embed_dims=384,num_heads=6,**common)
            modules = [self.backbone.stage1[-1],self.backbone.stage2[-1],self.backbone.stage3[-1]]
            channels = [96,192,384]
        elif variant == 'meta-spikeformer':
            self.backbone = ns['Spiking_vit_MetaFormer'](embed_dim=[96,192,384,480],num_heads=8,**common)
            modules = [self.backbone.ConvBlock1_2[-1],self.backbone.ConvBlock2_2[-1],self.backbone.block4[-1]]
            channels = [96,192,480]
        else:
            dynamic_swin(ns)
            self.backbone = ns['SwinTransformer'](img_size=224,patch_size=4,in_chans=10,num_classes=0,
                embed_dim=96,depths=[2,2,6,2],num_heads=[3,6,12,24],window_size=7,
                mlp_ratio=4.,qkv_bias=True,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.,
                ape=False,patch_norm=True,use_checkpoint=False,fused_window_process=False)
            modules = []
            channels = [96,192,768]
        for module in modules:
            module.register_forward_hook(self._capture)
        self.lateral = nn.ModuleList([nn.Conv2d(c,64,1) for c in channels])
        self.readout = nn.Sequential(nn.Conv2d(960,256,1),nn.GroupNorm(32,256),nn.GELU(),nn.Conv2d(256,256,3,padding=1))

    def _capture(self, module, inputs, output):
        self._taps.append(output[0] if isinstance(output,tuple) else output)

    def _swin_features(self, x):
        b, _, h, w = x.shape
        model = self.backbone
        x = model.patch_embed.proj(x)
        h,w = x.shape[-2:]
        x = x.flatten(2).transpose(1,2)
        if model.patch_embed.norm is not None:
            x = model.patch_embed.norm(x)
        x = model.pos_drop(x)
        taps = []
        for i,stage in enumerate(model.layers):
            for block in stage.blocks:
                block.input_resolution = (h,w)
                x = block(x)
            if i in (0,1,3):
                value = model.norm(x) if i==3 else x
                taps.append(value.transpose(1,2).reshape(b,-1,h,w))
            if stage.downsample is not None:
                stage.downsample.input_resolution=(h,w)
                x=stage.downsample(x)
                h,w=h//2,w//2
        return taps

    def forward(self, pair):
        if not isinstance(pair,(list,tuple)) or len(pair)!=2 or pair[0].shape!=pair[1].shape:
            raise ValueError('Expected equally sized left/right tensors')
        b,c,h,w = pair[0].shape
        if c != 50 or h%4 or w%4:
            raise ValueError('Expected 50 sensor bins on native solver-padded grid')
        x = (torch.cat(pair,0)+1.)/2.
        hp,wp=math.ceil(h/32)*32,math.ceil(w/32)*32
        x=F.pad(x,(0,wp-w,0,hp-h))
        x=x.reshape(2*b,5,10,hp,wp).permute(1,0,2,3,4).contiguous()
        self._taps=[]
        try:
            if self.variant=='swin':
                taps=self._swin_features(x.flatten(0,1))
            else:
                self.backbone.forward_features(x)
                taps=self._taps
            if len(taps)!=3:
                raise RuntimeError(f'Expected three encoder taps, got {len(taps)}')
            values=[]
            for tap,projection in zip(taps,self.lateral):
                if tap.ndim==5:
                    tap=tap.flatten(0,1)
                tap=projection(tap)
                tap=F.interpolate(tap,size=(hp//4,wp//4),mode='bilinear',align_corners=False)
                values.append(tap)
            x=torch.cat(values,1).reshape(5,2*b,192,hp//4,wp//4)
            x=x.permute(1,0,2,3,4).reshape(2*b,960,hp//4,wp//4)
            x=self.readout(x)[...,:h//4,:w//4]
            return tuple(x.split(b,0))
        finally:
            self._taps=[]  # no graph/state retained between examples

    def metadata(self):
        return dict(variant=self.variant,upstream=SOURCES[self.variant],time_steps=5,bins_per_step=10,
                    compatibility_corrections=['restore T,B,H,W before final projection LIF'] if self.variant=='qkformer' else (['rectangular window padding and masks'] if self.variant=='swin' else []),
                    encoder_params=sum(p.numel() for p in self.backbone.parameters()),
                    readout_params=sum(p.numel() for p in self.lateral.parameters())+sum(p.numel() for p in self.readout.parameters()),
                    neurons={n:dict(tau=m.tau,decay_input=m.decay_input,threshold=m.v_threshold,
                                      reset=m.v_reset,detach_reset=m.detach_reset,surrogate='sigmoid_alpha_4')
                             for n,m in self.named_modules() if isinstance(m,MultiStepLIFNode)},
                    batchnorm={n:dict(eps=m.eps,momentum=m.momentum) for n,m in self.named_modules()
                               if isinstance(m,nn.modules.batchnorm._BatchNorm)})


def install(model, variant, seed, source_dir):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+783)
        model.fnet=StereoEncoder(variant,source_dir)
    def freeze_native_bn(self):
        # Keep original BatchNorm2d freeze semantics only outside the new fnet.
        for name,module in self.named_modules():
            if not name.startswith('fnet.') and isinstance(module,nn.BatchNorm2d):
                module.eval()
        if self.training:
            self.fnet.train()
    model.freeze_bn=types.MethodType(freeze_native_bn,model)
    return model
