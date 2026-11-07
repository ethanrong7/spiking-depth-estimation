# Reducing input-side CPU work on the B runs

The new input pipeline is CPU/backend-neutral and leaves the existing accelerator
calls unchanged. It can feed the current CUDA worker or a Huawei compatibility
layer that already runs that worker. It does not implement or validate the NPU
backend itself. No running job is changed automatically.

## Enable for the next launch

Set these variables in the environment that actually launches the worker:

```bash
export B_ATTENTION_INPUT_MODE=preload
export B_ATTENTION_PRELOAD_GIB=12
export B_ATTENTION_INPUT_WORKERS=2
export B_ATTENTION_PREFETCH_DEPTH=2
```

Then use the existing launch command. Both B wrappers read these settings and
forward them to the worker. Equivalent flags are `--input-mode preload
--preload-gib 12 --input-workers 2 --prefetch-depth 2`. Deploy the updated files
and freeze a new source package first. Existing processes cannot acquire the new
pipeline without a restart. Existing source/config identity checks still apply;
do not bypass them to hot-patch an old checkpoint or running frozen package.

For smaller host memory, use `B_ATTENTION_INPUT_MODE=fast`: faster bit decoding
and ahead-of-time compact batch preparation, without keeping the corpus in RAM.
Use `legacy` to restore the original loader. The default is now `fast`; full
corpus preloading remains explicit, so no extra ten-GiB cache is allocated by default.

## What changes

- The old bit reader constructs several large integer index arrays every time a
  spike file is decoded. The new path reads the required bytes and uses
  little-endian `numpy.unpackbits`, preserving the native first 50 frames exactly.
- `preload` decodes each Train/Dev sample once at startup. Subsequent updates read
  cached host tensors instead of re-reading spike files and EXR depth.
- Spikes remain uint8 on the host; depth-derived flow remains FP32 and masks bool.
  The full 960-Train/60-Dev cache is approximately **9.97 GiB per training process**.
  The explicit budget covers cached tensors, not all process/loader memory.
  Allow extra RAM for batch buffers, the model and libraries. Concurrent runs each
  own their cache; there is no shared-memory cache across processes.
- Bounded producer threads prepare two four-sample microbatches in advance,
  including deterministic flips, stacking and valid-pixel counts. The consumer
  receives batches in the original order even if producers finish out of order.
- Binary spikes transfer as uint8 and are cast to FP32 on the accelerator **before**
  native `2*spikes-1` preprocessing. Spike bytes transferred per optimizer update
  drop from 320 MB to 80 MB; target/mask traffic is unchanged. These are tensor-size
  calculations, not measured NPU transfer timings.
- The precomputed CPU mask counts avoid a device synchronization for each
  microbatch's valid-pixel denominator. Loss weighting remains unchanged.
- The three-step disposable preflight uses fast decoding without loading the full
  corpus. Real production, including a new resume process, builds its own RAM cache.

No model, optimizer, batch size, sample ordering, augmentation seed, refinement
count, evaluation split or learning-rate schedule changes. Cache entries are
immutable; flips and stacked batches use separate storage. No device operations
occur in the producer threads. Pinning, device streams and asynchronous DMA have
not been introduced because the current NPU compatibility layer is unknown.

## Read the timings before claiming a utilization fix

The worker writes `train/B/input-pipeline.json` with the actual mode, cache bytes,
preload duration and producer settings. Existing `train.jsonl` rows retain
`consumer_wait_seconds` and `seconds`, and add `input_prepare_seconds` and
`host_batch_bytes`. `decode_seconds` retains total producer preparation time for
compatibility; under preload it no longer means disk decoding.

- High consumer wait means the trainer is waiting for prepared inputs. Compare
  matched windows after cache warmup, excluding validation and checkpoint pauses.
- Producer preparation time can overlap device work. Do not add it to step time
  or interpret it alone as device idle time.
- With negligible consumer wait, more preload workers will not solve poor
  utilization. The recurrent model, many small launches/synchronizations, backend
  operator fallbacks or checkpoint/evaluation overhead may dominate instead.
- The default [speed runtime](SPEED.md) now consolidates metric transfers and
  reduces checkpoint frequency. Synchronous transfers and recurrent dispatch
  remain possible bottlenecks; backend throughput must still be measured.

Measure full wall time per fixed number of optimizer updates as well as device
utilization. No Huawei utilization or throughput improvement has been measured
from this Mac. The NPU preflight must verify uint8 transfers and FP32 casting on
that actual stack before production.

## CPU checks

From repository root, in an environment with the root model dependencies:

```bash
python projects/snn-depth-estimation/experiments/b-attention-mechanism-v1/test_input_pipeline.py -v
python projects/snn-depth-estimation/experiments/b-attention-mechanism-v1/test_controls.py -v
```

These use tiny synthetic samples, not full-dataset caching or training. They
check bit-order decoding, target/mask formulas, exact batches and weighting
against the old seeded loader, unchanged RNG, cache immutability and decode count,
budget enforcement, ordered resume, and launch/worker option forwarding.
