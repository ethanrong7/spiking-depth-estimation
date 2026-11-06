"""Opt-in spatial attention model retaining the original 50-channel CNN paths."""
from core.spikestereonet import SpikeStereoNet
from core.spiking_attention_update import SpikingAttentionMultiUpdateBlock


class SpikingTransformerStereo(SpikeStereoNet):
    def __init__(self, args, mode="spiking", spatial_mixing=True,
                 heads=4, head_dim=8, window=3, normalize_residual_current=False):
        if getattr(args, "feature_encoder", "cnn") != "cnn" or getattr(args, "context_encoder", "cnn") != "cnn":
            raise ValueError("this experiment preserves both original 50-channel CNN encoders")
        super().__init__(args)
        original = self.update_block
        self.update_block = SpikingAttentionMultiUpdateBlock(
            args, args.hidden_dims, mode, spatial_mixing, heads, head_dim, window,
            normalize_residual_current)
        # Retain exact already initialized shared modules, not just matching seeds.
        self.update_block.encoder = original.encoder
        self.update_block.disp_head = original.disp_head
        self.update_block.mask = original.mask

    def load_common_state(self, reference):
        """Copy all common modules from an explicit original-model reference."""
        modules = ["cnet", "context_zqr_convs", "update_block.encoder",
                   "update_block.disp_head", "update_block.mask"]
        modules.append("conv2" if self.args.shared_backbone else "fnet")
        for name in modules:
            self.get_submodule(name).load_state_dict(reference.get_submodule(name).state_dict())

    def forward(self, *args, **kwargs):
        self.update_block.reset_state()
        try:
            return super().forward(*args, **kwargs)
        finally:
            # No cross-batch state or retained graph, including failed forwards.
            self.update_block.reset_state(clear_diagnostics=False)
