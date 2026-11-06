"""S2M2 v2: scoped spike coding, global OT regression, residual refinement."""
import torch
from torch import nn
from torch.nn import functional as F
from core.s2m2_spiking import SpikingS2M2, SpikingOT, SpikeProjection, SpikeHidden, fixed_native_initial
from vendor.s2m2.model.refinenet import GlobalRefiner


def global_correspondence(probability):
    keys = torch.arange(probability.shape[-1], device=probability.device,
                        dtype=probability.dtype)
    mass = probability.sum(-1)
    return (probability * keys).sum(-1) / mass.clamp_min(1e-6), mass


class GlobalSpikingOT(SpikingOT):
    def scores(self, features):
        left, right = self.layer_norm(features.permute(0, 2, 3, 1)).chunk(2)
        q = F.normalize(self.q(left), dim=-1, eps=1e-6)
        k = F.normalize(self.k(right), dim=-1, eps=1e-6)
        # Bounded cosine logits decouple matching temperature from firing rate.
        gain = F.softplus(self.log_gain).clamp(max=20.)
        return torch.einsum('bhic,bhjc->bhij', q, k) * gain

    def forward(self, features):
        _, _, occ, cv, down = super().forward(features)
        position, mass = global_correspondence(self.last_probability)
        keys = torch.arange(position.shape[-1], device=position.device,
                            dtype=position.dtype)
        disp = (keys - position)[:, None]
        # Concentration, independent of the selected argmax window.
        conditional = self.last_probability / mass[..., None].clamp_min(1e-6)
        conf = conditional.square().sum(-1) * mass
        self.last_initial = disp
        return disp, conf[:, None], occ, cv, down


class ResidualGlobalRefiner(GlobalRefiner):
    def __init__(self, channels):
        super().__init__(channels)
        nn.init.zeros_(self.out_feat[0].weight)
        nn.init.zeros_(self.out_feat[0].bias)

    def forward(self, ctx, disp, conf):
        feature = self.init_feat(torch.cat(
            [disp / 100, conf.logit(eps=.01), ctx], dim=1).to(disp.dtype))
        # Coarse-pixel correction; no hard confidence bypass or x100 gain.
        result = disp + self.out_feat(self.refine_unet(feature)[0])
        self.last_output = result
        return result


class SpikingS2M2V2(SpikingS2M2):
    def __init__(self, feature_channels=32, num_transformer=1, refine_iter=3):
        super().__init__(feature_channels, num_transformer, refine_iter)
        # Undo the v1 recursive conversion outside the named stereo Transformer.
        for root in (self.feat_pyramid, self.global_refiner, self.refiner):
            for module in list(root.modules()):
                for name, child in list(module.named_children()):
                    if isinstance(child, SpikeProjection):
                        setattr(module, name, child.linear)
                    elif isinstance(child, SpikeHidden):
                        setattr(module, name, nn.GELU())
        self.disp_init = GlobalSpikingOT(feature_channels)
        self.global_refiner = ResidualGlobalRefiner(feature_channels)

    def forward(self, left, right):
        output = super().forward(left, right)
        output['global_flow'] = -fixed_native_initial(
            self.global_refiner.last_output, output['padding'], *left.shape[-2:])
        return output


def s2m2_v2_loss(output, target, valid, normalization_counts=None):
    from core.ssde_direct_attention import direct_loss
    _, parts = direct_loss(output, target, valid, normalization_counts)
    mask = valid[:, None] if valid.ndim == 3 else valid
    x = torch.arange(target.shape[-1], device=target.device)[None, None, None]
    inside = mask.bool() & torch.isfinite(target) & (target.abs() < 512)
    inside = inside & (x + target >= 0) & (x + target <= target.shape[-1] - 1)
    counts = normalization_counts or {}
    initial = F.smooth_l1_loss(output['initial_flow'], target, reduction='none', beta=4.)
    initial = initial[inside].sum() / max(1, counts.get('initial', int(inside.sum())))
    parts['initial_l1'] = initial  # legacy log key; now in-frame SmoothL1(beta4)
    # Stronger global matching supervision; retain full-valid final depth L1.
    loss = parts['full_l1'] / 32 + parts['correspondence_ce'] + initial / 320
    return loss, parts
