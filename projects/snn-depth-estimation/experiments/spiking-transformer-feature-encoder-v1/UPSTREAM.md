# Upstream encoder sources

Pinned author sources are fetched into an ignored cache by `upstream.py`. They are not redistributed or relicensed in this repository. The effective source copy and checksums are recorded in each batch.

| Encoder | Repository | Commit | Source | License record |
| --- | --- | --- | --- | --- |
| spikformer | https://github.com/ZK-Zhou/spikformer | `621268981827af47aa666c1b5f774de4281d90ad` | `imagenet/model.py` | upstream file available |
| qkformer | https://github.com/zhouchenlin2096/QKFormer | `4c7e0356dd535fec683b6a33214ddd2c5b45e56e` | `imagenet/qkformer.py` | No top-level license found; downloaded for local research, not redistributed here |
| meta-spikeformer | https://github.com/BICLab/Spike-Driven-Transformer-V2 | `35f245f4f79acf1a2b3ef22698c58ab05c1ede33` | `classification/models.py` | No top-level license found; downloaded for local research, not redistributed here |
| swin | https://github.com/microsoft/Swin-Transformer | `f82860bfb5225915aca09c3227159ee9e1df874d` | `models/swin_transformer.py` | upstream file available |

`upstream-sources.json` records SHA-256 for each source and available license. Missing top-level licenses are not treated as an MIT grant. Retain downloaded attribution and consult author terms before distributing their source or a derived release.

The loader keeps author class/function bodies and supplies explicit torch dependencies. It omits module-level CUDA/CuPy imports, timm registration and demonstration code. LIF is replaced by the tested torch port; zero-probability DropPath is identity. Swin window handling is adapted for rectangular maps. No attention equation is replaced by the old Hybrid B operator.

Adapter changes: input channels 10 over five chronological chunks; remove classification heads/pooling from the used features; expose fine/deep feature taps; common continuous dense readout; independent left/right state; source-specific BN training within fnet. See the protocol for exact widths/depths and common recipe.

QKFormer compatibility correction: in final-stage `Spiking_Self_Attention`, restore `[T,B,C,H,W]` **before** `proj_lif`, instead of feeding it `[T*B,C,N]` and reshaping to `[T,B,C,W,H]` afterward. The pinned classifier code otherwise swaps rectangular axes and interprets batch entries as neuron time steps. This is an explicit shape/state correction, not an attention-equation replacement. Reference block tests apply the same correction with the original torch neuron, and separate tests verify sample independence and rectangular output shape.
