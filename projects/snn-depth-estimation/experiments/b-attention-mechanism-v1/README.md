# Placement B: forward encoding versus learning rule

Start with **[b-attention-verification.md](b-attention-verification.md)**. This is
the self-contained agent handoff: historical endpoint/curve summaries, experiment
questions, Huawei NPU porting tasks, result interpretation and output schemas.

## Current status

- The four encoder variants and seven CPU checks are implemented.
- The NPU worker/backend and site launcher still need implementation and validation.
- No NPU jobs have been submitted and no NPU EPE results exist.
- The experiment uses this standalone checkout as a fresh scientific baseline.
  No old repository, source capsule, checkpoint or result log is required.
- Dataset files and an assigned Ascend environment must be supplied externally.

| Variant | Forward Q/K/V values | Local derivative |
| --- | --- | --- |
| `linear-identity` | Unbounded linear projection | Identity |
| `signed-surrogate` | Signed T4 code | Original surrogate through recurrence/reset |
| `linear-surrogate` | Unbounded linear projection | Original T4 surrogate |
| `signed-identity` | Signed T4 code | Identity / straight-through |

Use [configs/matrix-npu.json](configs/matrix-npu.json): all four variants trained
fresh at both seeds 1234 and 2345. Use
[configs/protocol-npu.json](configs/protocol-npu.json) for the matched 30k settings.
Historical CUDA results are context, not reused NPU control cells.
`configs/matrix.json` is retained only as the earlier CUDA planning record.

## Local checks and source packaging

From the repository root, in an environment with the root requirements installed:

```bash
python projects/snn-depth-estimation/experiments/b-attention-mechanism-v1/test_controls.py -v
```

Package this checkout, including the reviewed NPU port once implemented:

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
assumptions. The handoff identifies the exact changes needed.

## Files and outputs

The reusable model implementation is
[core/attention_encoder_controls.py](../../core/attention_encoder_controls.py).
It keeps native B parameter names/shapes and attention calculations unchanged;
the two diagonal cells execute the original linear or signed encoder directly.
The crossed cells specify first-order surrogate derivatives independently of
forward values.

Write logs and later plots under this directory's ignored `outputs/`, separated
by study revision, backend, seed and variant. Use the JSONL/CSV contracts in the
handoff. Keep checkpoints outside Git. The primary metric is fixed-30k Dev
mean-scene EPE; include bad3, complete curves and per-scene paired differences.
