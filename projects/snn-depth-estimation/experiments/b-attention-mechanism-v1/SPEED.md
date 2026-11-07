# Fast B-mechanism diagnostics

The purpose is to distinguish the forward-code and backward-rule effects quickly.
Use the same speed settings for all four variants and the existing staged 5k
plan. This update does not increase batch size, change precision, drop refinement
iterations, compress the 30k learning-rate horizon or change the evaluation split.

## Defaults for new worker launches

- `B_ATTENTION_SPEED_MODE=1` (default): fixed-size masked loss reductions, one
  packed pre-optimizer metric read, attention-gradient diagnostics at update 1
  and every 100 updates, status writes every 25 updates, recovery checkpoints
  every 500 updates instead of 50. Preflight keeps detailed gradient checks.
- `B_ATTENTION_INPUT_MODE=fast` (new default): compact decoding and CPU-prepared
  microbatches. Use `preload` with sufficient RAM to eliminate repeated file
  reads; see [INPUT_PIPELINE.md](INPUT_PIPELINE.md). No corpus-sized cache is
  allocated by default.
- Strict deterministic device execution remains disabled. Seeds, initialization,
  order and augmentations remain fixed.
- Safety checks still reject nonfinite predictions/loss/gradient norms and empty
  masks before the optimizer step. Gradient clipping still runs every update.
- Per-update loss/LR/norm records remain in `train.jsonl`; an empty
  `placement_gradients` object means detailed diagnostics were not sampled.
- Checkpoints are also saved after validation, at graceful pause/exit and at the
  original retained endpoints. An abrupt failure can lose up to 499 updates
  between recovery checkpoints. Evaluation dates/counts are unchanged.

`B_ATTENTION_SPEED_MODE=0` selects the old loss/diagnostics/checkpoint loop for a
matched speed comparison. Also set `B_ATTENTION_INPUT_MODE=legacy` only if testing
old loading as well. Neither switch re-enables strict determinism. Use separate
output directories/source packages for comparisons; old configuration/source
resume guards remain intact. No running process is hot-patched by these files.

## Mac CPU diagnostic, 7 November 2026

Synthetic native-sized tensors; two CPU threads, PyTorch 2.8.0, one warmup and
three timed repetitions. No training, dataset samples or accelerator execution.

| Component | Legacy median | Updated median | Scope |
| --- | ---: | ---: | --- |
| Disparity loss + backward | 75.18 ms | 21.55 ms | One microbatch: 4 × 1 × 250 × 400, 16 predictions; **3.49×** faster for this component |
| Read/decode one spike block | 25.91 ms | 0.52 ms | Synthetic native-format bytes; old FP32 versus new uint8 host representation |
| Prepare a cached batch | — | 7.00 ms | Eight samples, flips and two stacked microbatches |

The CPU profiler recorded **19 scalar reads and 32 `nonzero` calls** in the old
loss/backward diagnostic, versus **zero of each** in the new diagnostic. The new
path instead performs one packed host read at the pre-optimizer boundary. CPU
profiling cannot measure the NPU cost of these host reads or predict utilization.
The decoder comparison also benefits from retaining compact uint8 rather than
converting to host FP32. The accelerator still performs the required FP32 cast.

These are component timings, not a full-training speedup. CPU tests compare the
loss and gradients in FP32/FP64 and three paired AdamW updates with accumulation
and clipping. The masked sum/count is mathematically the same objective as the
masked mean, but reduction order can produce small floating-point differences;
bitwise trajectory identity is not promised.

Reproduce from this repository root:

```bash
python projects/snn-depth-estimation/experiments/b-attention-mechanism-v1/diagnose_cpu.py \
  --output projects/snn-depth-estimation/experiments/b-attention-mechanism-v1/outputs/cpu-speed-diagnostic/result.json
python projects/snn-depth-estimation/experiments/b-attention-mechanism-v1/test_speed_runtime.py -v
```

The JSON includes timings, operator counts and CPU profiler tables. Its output
folder is ignored by Git. Synthetic input timings exclude real EXR parsing and
remote filesystem latency; the diagnostic does not measure model-forward cost.

## Short diagnostic on the actual NPU host

Deploy/freeze the updated code through the site's existing working launch path.
The local launcher is still CUDA-oriented; these optimizations do not add a new
Huawei deployment layer. Do not interpret unchanged server code as testing this
update. First run the existing three-step preflight with the new source.

For a disposable short worker run, enable:

```bash
export B_ATTENTION_PROFILE_CPU=1
```

Use a fresh output directory and the existing worker's `--stop-at 200` option
(absolute optimizer-update count). If the site's wrapper owns the stop budget,
pass it there too: the checked-in legacy `run_on_gpu.py` still hardcodes 30k and
is not a short-benchmark launcher. Do not launch it expecting a 200-step stop.

The profiler waits 5 updates, warms up for 2, captures 5, then disables itself.
It produces `cpu-trace.json` and `cpu-profile.txt` beside `train.jsonl`. Marked
regions cover input waiting, host-to-device copies, model forward, loss,
backward, metric transfer, optimizer and end-of-step synchronization. This is a
**host trace**: it locates CPU work and blocking calls but does not replace the
Huawei profiler for device kernels/fallbacks. Profiling overhead affects the
captured updates; exclude the first 20 updates from throughput comparisons.

For speed-mode-on/off comparison, retain the same input mode, source, model
variant, seed, batch, precision and server load; change only speed mode. Compare
median/p90 completed-update time over updates 21–200. Do not combine these
benchmark runs with the scientific endpoint table. Return the short logs and
trace before increasing training duration.

## Reading the remaining waits

| Evidence | Meaning / next action |
| --- | --- |
| High `consumer_wait_seconds` | Inputs are not ready: try RAM preload; inspect real disk/EXR cost and host RAM pressure. |
| High `phase_wait_seconds` | Shared validation lock is holding this worker; check other jobs or stale request files before changing model code. |
| Low input wait, expensive `host_to_device` | Transfer remains synchronous. Test backend-supported pinned memory/async transfer on the actual NPU stack; it is not enabled blindly here. |
| Many short forward/backward operations with host gaps | Python dispatch, recurrent updates, signed-code loops or checkpoint recomputation may dominate. Use device tracing before attempting compilation/fusion. |
| Time inside `metrics_to_host` / `end_step_sync` | May be waiting for already queued accelerator work, not doing CPU computation. Inspect preceding device kernels. |
| Large pauses every 500 updates | Check `performance.jsonl` checkpoint durations and filesystem throughput. |

`seconds` retains synchronized training-step duration. `update_wall_seconds`
adds input and phase wait, but excludes the subsequent JSON write,
checkpoint/evaluation. Measure external wall time for end-to-end throughput;
`performance.jsonl` separately records checkpoint time. One explicit end-of-step
synchronization remains for reliable completed-step timing. The patch removes
many interior waits, not every synchronization or every CPU operation.

## RTX 3070 Laptop check, 7 November 2026

A short remote Windows/CUDA training-loop benchmark measured:

| Loop | Average GPU utilization | Mean update time |
| --- | ---: | ---: |
| Original loss/metrics | 86.05% | 1.1034 s |
| Speed runtime | 88.53% | 1.0862 s |

Full-resolution 50-bin input, hybrid B, 16 refinements, FP32, microbatch 1 with
accumulation 2; eight warmup and 30 measured updates per mode. PyTorch 2.0.1+cu118.
Inputs were cached synthetic uint8 tensors in both modes; no dataset I/O,
validation or checkpoint saving. The laptop was idle before the benchmark.
GPU-wide utilization was sampled every 500 ms (65 and 64 measured samples).
Peak allocated memory was approximately 3.69 GiB. The small 1.6% throughput
change is from one sequential comparison, not a replicated speed claim.

Reproduce with `benchmark_cuda.py --output <new-output-folder>` in the existing
CUDA environment. This standalone harness has no Slurm dependency; it imports
the current B model and speed runtime and does not save model weights. It uses
synthetic inputs and does not replace the native-data NPU preflight. Raw local
records are in ignored `outputs/3070-util-20261107/`.

The Mac's 3.49× loss-component speedup did not translate into a large NVIDIA
training speedup. GPU utilization is already high in this cached-input setting.
No Huawei throughput has been measured. Its backend/kernel behavior, dispatch,
real loading and transfers still require the short on-NPU trace; microbatch 4
and the full NPU input pipeline were not reproduced here.
