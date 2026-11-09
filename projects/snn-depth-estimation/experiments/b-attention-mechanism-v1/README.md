# Placement B: forward encoding versus learning rule

For the current eight-run NPU batch, start with
**[hybrid-b-ablation-911-NPU.md](hybrid-b-ablation-911-NPU.md)**. It contains the
execution command, matched completed-NPU recipe, interpretation and server-agent
prompt. [b-attention-verification.md](b-attention-verification.md) retains the
earlier staged-screen plan; its budgets do not apply to this named batch.

## Current status

- The four encoder variants and seven CPU checks are implemented.
- The single-NPU worker/backend, eight-cell launcher and CSV summarizer are implemented.
- CPU checks cover encoder derivatives, checkpoint recomputation, input pairing,
  launch admission and result auditing. NPU validation runs on the server before training.
- No jobs for this new batch have been launched locally. Historical NPU results
  are included as context in the current protocol.
- The experiment uses this standalone checkout as a fresh scientific baseline.
  No old repository, source capsule, checkpoint or result log is required.
- Dataset files and an assigned Ascend environment must be supplied externally.

| Variant | Forward Q/K/V values | Local derivative |
| --- | --- | --- |
| `linear-identity` | Unbounded linear projection | Identity |
| `signed-surrogate` | Signed T4 code | Original surrogate through recurrence/reset |
| `linear-surrogate` | Unbounded linear projection | Original T4 surrogate |
| `signed-identity` | Signed T4 code | Identity / straight-through |

The current runner executes all four variants at seeds 1234/2345 to 30k, one NPU
per model. It uses the completed single-device NPU recipe: physical batch 4,
accumulation 2, native masked loss and FP32. The older
[configs/matrix-npu.json](configs/matrix-npu.json) and
[configs/protocol-npu.json](configs/protocol-npu.json) describe the earlier port;
do not use their staged/multi-device settings for the 911 batch.
Historical CUDA results are context, not reused NPU control cells.
`configs/matrix.json` is retained only as the earlier CUDA planning record.

## Local checks and source packaging

From the repository root, in an environment with the root requirements installed:

```bash
python projects/snn-depth-estimation/experiments/b-attention-mechanism-v1/test_controls.py -v
python projects/snn-depth-estimation/experiments/b-attention-mechanism-v1/test_ablation_npu.py -v
```

The current launcher packages this checkout automatically. For manual packaging:

```bash
python projects/snn-depth-estimation/experiments/b-attention-mechanism-v1/freeze_source.py \
  --output "$B_ATTENTION_SOURCE"
```

Supply a new output directory. Default packaging requires only this checkout and
Git, and excludes raw outputs. It records current source identity and marks the
study as fresh, without historical equivalence. The optional `--reference` flag
exists for legacy CUDA archive overlays; it is not needed for NPU work.

The old `run_on_gpu.py` / `slurm/run.sbatch` paths remain CUDA implementation
references. They are not ready-to-run NPU commands and retain old deployment
assumptions. Use `run_ablation_on_npu.py` for the current NPU batch.

## Files and outputs

The reusable model implementation is
[core/attention_encoder_controls.py](../../core/attention_encoder_controls.py).
It keeps native B parameter names/shapes and attention calculations unchanged;
the two diagonal cells execute the original linear or signed encoder directly.
The crossed cells specify first-order surrogate derivatives independently of
forward values.

Write logs and later plots under this directory's ignored `outputs/`, separated
by study revision, backend, seed and variant. Use the JSONL/CSV contracts in the
handoff. Keep checkpoints outside Git. The current batch's primary endpoint is
fixed-30k Dev mean-scene EPE, with bad3, complete curves and paired scene differences.
The launcher supports preparation-only, preflight-only and explicit batch resume.

## Input preloading

**Legacy CUDA speed defaults:** compact `fast` input preparation and the reduced-synchronization
training loop are enabled in the CUDA worker. They do not apply to the current
NPU batch, which explicitly uses preload and the historical native loss.
See [SPEED.md](SPEED.md) for the CPU diagnostic,
server profiling command, changed checkpoint cadence and remaining bottlenecks.

See [INPUT_PIPELINE.md](INPUT_PIPELINE.md) for the optional RAM preload and compact
batch-preparation path. It preserves paired training inputs and can feed the
existing accelerator compatibility layer. Enable with `B_ATTENTION_INPUT_MODE=preload`
and a configured host-memory budget; NPU throughput is not yet measured.

## Device determinism

The B worker disables strict device determinism for all four variants:
`torch.use_deterministic_algorithms(False)` and
`torch.backends.cudnn.deterministic=False`. Seeds, initialization, sample order,
augmentation, precision and the learning-rate schedule remain unchanged.
cuDNN benchmarking remains off so this change isolates the determinism setting.
The execution settings are recorded in run configuration and checkpoints.

Use the updated source package for new runs; this does not modify running jobs.
Existing checkpoint configuration/source checks remain in place, so an old run
cannot silently resume under the changed execution policy. On Huawei, confirm
that the installed compatibility layer honors the PyTorch setting and that the
site launcher does not separately force deterministic NPU kernels. No NPU
throughput improvement has been measured locally. Compare steady-state seconds
per step with the same policy across all variants.
