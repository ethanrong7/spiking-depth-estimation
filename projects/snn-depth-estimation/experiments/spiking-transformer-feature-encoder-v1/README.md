# Stereo Transformer encoder screen on eight NPUs

Start with **[the experiment protocol and server prompt](spiking-transformer-feature-encoder-NPU.md)**. This batch replaces only SpikeStereoNet's matching encoder with Spikformer, QKFormer, Meta-SpikeFormer or Swin, each at two seeds. The model code is implemented; NPU validation runs on the target server.

```bash
python projects/snn-depth-estimation/experiments/spiking-transformer-feature-encoder-v1/run_on_npu.py \
  --data-root /path/to/synthetic-dataset --devices 0 1 2 3 4 5 6 7 --execute
```

Use the allocated device IDs and real dataset path. Omit `--execute` for CPU preparation only. `--preflight-only --execute` performs the four 200-update NPU preflights; `--resume --execute` continues the same batch. Eight fresh 30k production cells follow successful preflights. Old Hybrid B workers are not changed.

| File | Responsibility |
| --- | --- |
| `upstream-sources.json`, `upstream.py` | Immutable source pins, automatic download/checksums, explicit torch dependency bindings |
| `neuron.py` | Torch/NPU LIF implementation preserving forward/reset/surrogate semantics |
| `encoders.py` | Full author backbone blocks, rectangular Swin adaptation, chronological input and common dense readout |
| `run_on_npu.py` | Source preparation, paired initialization, preflight admission, per-device queues and resume |
| `worker_npu.py` | Native NPU data/loss/evaluation with replacement fnet, checkpointing and inference timing |
| `validate_npu.py`, `test_encoders.py` | Small reference equivalence tests and stereo integration checks |
| `summarize.py` | Audited curves, endpoints, scenes and paired deltas for later plots |

Author code is downloaded into ignored output/cache directories rather than redistributed under this repository's license. See [UPSTREAM.md](UPSTREAM.md). Preparation needs internet once, or a pre-populated cache with `--offline`. Runtime workers are offline and verify pinned sources.

Local validation covers tiny CPU inputs only. Native NPU accuracy, memory and throughput are unverified until server preflights complete. This compares encoder packages, not an isolated spiking-activation effect.
