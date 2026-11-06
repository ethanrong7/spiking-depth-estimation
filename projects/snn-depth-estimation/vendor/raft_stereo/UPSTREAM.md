# raft_stereo

Source: https://github.com/princeton-vl/RAFT-Stereo
Exact commit: `6e93ed2169bd858dbb43033988563f3b0bb49506`
Imported subtree: `core` (all Python source)
License: MIT; full upstream LICENSE retained. No upstreamweights or datasets imported.

Mechanical adaptations: namespace core imports to avoid collision with originalprojectcore; change only BasicEncoder/MultiBasicEncoder firstconv inputchannels3→50; replace RGB255 normalization with2*spike−1. Architecture bodies and all update/attention/correlation modules are retained. IMPORT.json records each upstream/adapted file digest and archive identity. Official launchers/loaders are not used; commonpilot driver controls data, FP32/frozenBN, losses andSlurmvisibility.

The upstream `core/` directory is stored as `net/` under its isolated vendor namespace; source-freeze tooling reserves `/core/` for the original project core. This is a packaging-only path change.
