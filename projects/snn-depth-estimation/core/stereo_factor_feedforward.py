"""Frozen arm B: one native lookup and a T4 candidate Transformer."""
from types import MethodType
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from core.corr import CorrBlock1D
class Threshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        return grad / (1 + ctx.saved_tensors[0].abs()).square()


def signed_rate(current, steps=4):
    voltage = torch.zeros_like(current)
    total = torch.zeros_like(current)
    for _ in range(steps):
        voltage = .9 * voltage + current
        spike = Threshold.apply(voltage - 1) - Threshold.apply(-voltage - 1)
        voltage = voltage - spike
        total = total + spike
    return total / steps


class FeedforwardHead(nn.Module):
    def __init__(self, context_channels=128, levels=4, radius=4, stride=4):
        super().__init__()
        self.levels, self.radius, self.stride = levels, radius, stride
        self.input = nn.Linear(context_channels + 3, 32)
        self.norm1 = nn.LayerNorm(32)
        self.q, self.k, self.v = [nn.Linear(32, 32) for _ in range(3)]
        self.proj = nn.Linear(32, 32)
        self.norm2 = nn.LayerNorm(32)
        self.ff = nn.Sequential(nn.Linear(32,64))
        self.ffout = nn.Linear(64,32)
        self.score = nn.Linear(32,1)
        self.residual = nn.Linear(32,1)
        nn.init.zeros_(self.residual.weight); nn.init.zeros_(self.residual.bias)
        self.mask = nn.Sequential(nn.Conv2d(32,128,3,padding=1),nn.ReLU(),nn.Conv2d(128,9*stride*stride,1))
        self.native = None
        self.override = None
        self.row_chunk = 4
        self.use_checkpoint = True

    def set_native_geometry(self, shape, pad):
        self.native = (tuple(shape[-2:]), tuple(pad))

    def forward(self, context, corr):
        b,_,h,w = corr.shape
        offsets = torch.cat([torch.arange(-self.radius,self.radius+1,device=corr.device)*2**i for i in range(self.levels)]).to(corr.dtype)
        xgrid = torch.arange(w,device=corr.device)[None,None,:,None]
        valid = torch.cat([((xgrid/2**i + torch.arange(-self.radius,self.radius+1,device=corr.device))>=0) & ((xgrid/2**i + torch.arange(-self.radius,self.radius+1,device=corr.device))<=w//2**i-1) for i in range(self.levels)],-1).expand(b,h,w,-1)
        chunks = []
        step = self.row_chunk or h
        for start in range(0, h, step):
            args = (context[:,:,start:start+step], corr[:,:,start:start+step], valid[:,start:start+step], offsets)
            if self.training and self.use_checkpoint:
                chunks.append(checkpoint(self._chunk, *args, use_reentrant=False))
            else:
                chunks.append(self._chunk(*args))
        pooled = torch.cat([x[0] for x in chunks], 1)
        coordinate = torch.cat([x[1] for x in chunks], 2)
        residual = torch.cat([x[2] for x in chunks], 2)
        weights = torch.cat([x[3] for x in chunks], 1)
        disp = torch.cat([coordinate+residual,torch.zeros_like(coordinate)],1)
        # Apply the spatial mask CNN once, preserving cross-chunk neighbors.
        return disp, .25*self.mask(pooled.permute(0,3,1,2)), dict(coordinate=coordinate,residual=residual,weights=weights,valid=valid,offsets=offsets)

    def _chunk(self, context, corr, valid, offsets):
        b,_,h,w = corr.shape
        n = offsets.numel()
        ctx = context.permute(0,2,3,1)[...,None,:].expand(-1,-1,-1,n,-1)
        tokens = torch.cat([ctx,corr.permute(0,2,3,1)[...,None],(offsets/(self.radius*2**(self.levels-1))).expand(b,h,w,n)[...,None],valid[...,None].to(corr.dtype)],-1)
        x = self.input(tokens); z = self.norm1(x)
        q,k,v = [signed_rate(layer(z),4).reshape(b,h,w,n,4,8).transpose(-3,-2) for layer in (self.q,self.k,self.v)]
        logits = (q @ k.transpose(-1,-2))/8**.5
        logits = logits.masked_fill(~valid[...,None,None,:],-1e4)
        mix = (logits.softmax(-1) @ v).transpose(-3,-2).reshape(b,h,w,n,32)
        x = x + self.proj(mix)
        x = x + self.ffout(signed_rate(self.ff(self.norm2(x)),4))
        score = self.score(x).squeeze(-1).masked_fill(~valid,-1e4)
        weights = score.softmax(-1)*valid
        weights = weights/weights.sum(-1,keepdim=True).clamp_min(1e-9)
        if self.override == 'uniform':
            weights = valid.to(corr.dtype)/valid.sum(-1,keepdim=True).clamp_min(1)
        pooled = (x*weights[...,None]).sum(-2)
        coordinate = (weights*offsets).sum(-1)[:,None]
        residual = self.residual(pooled).permute(0,3,1,2)
        if self.override == 'zero_residual': residual = residual*0
        return pooled, coordinate, residual, weights


def _forward(self, spike1, spike2, iters=16, disp_init=None, test_mode=False):
    if disp_init is not None and torch.count_nonzero(disp_init).item():
        raise ValueError('Arm B requires an undisplaced initial grid')
    spike1,spike2 = (2*spike1-1).contiguous(),(2*spike2-1).contiguous()
    if self.args.shared_backbone:
        *contexts,x = self.cnet(torch.cat([spike1,spike2]),dual_inp=True,num_layers=self.args.n_snn_layers)
        fmap1,fmap2 = self.conv2(x).chunk(2)
    else:
        contexts = self.cnet(spike1,num_layers=self.args.n_snn_layers)
        fmap1,fmap2 = self.fnet([spike1,spike2])
    context = torch.relu(contexts[0][1])
    coords,_ = self.initialize_disp(context)
    volume = CorrBlock1D(fmap1.float(),fmap2.float(),num_levels=self.args.corr_levels,radius=self.args.corr_radius)
    disp,mask,self.last_components = self.feedforward_head(context.float(),volume(coords))
    up = self.upsample_disp(disp,mask)[:,:1]
    if not self.training:
        for component in ('coordinate', 'residual'):
            value = self.last_components[component]
            vector = torch.cat([value, torch.zeros_like(value)], 1)
            self.last_components[component + '_native'] = self.upsample_disp(vector, mask)[:, :1]
    return (disp,up,[]) if test_mode else ([up],[],[])


def install_feedforward(model, seed=2017):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = FeedforwardHead(model.context_zqr_convs[0].in_channels,model.args.corr_levels,model.args.corr_radius,2**model.args.n_downsample)
    del model.update_block
    del model.context_zqr_convs
    model.feedforward_head = head
    model.set_native_geometry = head.set_native_geometry
    model.forward = MethodType(_forward,model)
    model.last_components = {}
    return model
