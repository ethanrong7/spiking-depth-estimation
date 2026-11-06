# Source provenance

Extracted from the current Research working tree on 6 November 2026, including uncommitted model implementations. No Git history or experiment results were imported. `provenance/source-files.json` records the source HEAD and exact imported file hashes; that HEAD alone does not describe the dirty working tree.

The SpikeStereoNet implementation originated from:

- https://github.com/Criticality-Cognitive-Computation-Lab/SpikeStereoNet
- Commit `8c5e87750f25a035bca41bd37ae44b042a405f9a`
- MIT license in `projects/snn-depth-estimation/LICENSE`.

The imported current implementation includes later local extensions and is not a byte-identical upstream checkout. The portable `runtime.py` / `run.py` wrapper is new. Copied model files retain their original bytes.

Vendored baselines retain their individual LICENSE and UPSTREAM files under `projects/snn-depth-estimation/vendor/`. Their licenses apply separately, including the noncommercial S2M2 license. Inspired/reconstructed model names do not imply official authorship or published-system reproduction.
