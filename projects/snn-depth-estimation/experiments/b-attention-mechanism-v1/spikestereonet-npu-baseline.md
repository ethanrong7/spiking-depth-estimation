# SpikeStereoNet versus hybrid B on NPU: matched full 30k training

Updated 8 November 2026. Status: **planned; neither new run has a result**.
Work from `main`, independently of `nmrf-experiments`. This document specifies
training; editing it does not submit jobs.

**This revision supersedes the earlier 7.5k baseline screen. Train BOTH Original
non-I SpikeStereoNet and hybrid attention B from initialization to 30,000
optimizer updates, using the historical NVIDIA training hyperparameters.**
There is no performance-based early-stop or promotion gate. Intermediate rankings
do not change either run's budget. Total: two seed-1234 runs, 60,000 updates.

## 1. Question and comparison

Does hybrid B's pre-correlation feature exchange improve the final disparity
estimate over Original SpikeStereoNet when both are trained on NPU with the
same NVIDIA-recipe hyperparameters and a fixed execution policy?

| New run | Architecture | Seed | Updates | Primary result |
| --- | --- | ---: | ---: | --- |
| `original-non-i` | Native CNN, native correlation pyramid, multiscale ALIF updater and disparity head | 1234 | 30,000 | TBD |
| `hybrid-b` | The same native model plus historical placement-B signed-T4 epipolar descriptor exchange before correlation | 1234 | 30,000 | TBD |

Start both fresh. Do not resume the old 7.5k NPU B checkpoint: it used a different
microbatch/distributed arrangement and changed execution settings during training.
Resume only checkpoints produced under this new, fixed protocol after interruption.
Conventional B and crossed-gradient variants are outside these two runs. This
comparison tests the **whole B addition**, not whether spikes outperform conventional
attention or which mechanism causes any improvement.

## 2. Historical evidence and what is being reproduced

| Backend / historical run | Update | Mean-scene EPE, px ↓ | Pooled bad3, % ↓ |
| --- | ---: | ---: | ---: |
| NVIDIA Original non-I | 30,000 | 2.691127 | 10.712 |
| NVIDIA hybrid B | 30,000 | 2.034117 | 9.626333 |
| NVIDIA conventional B | 30,000 | 2.768814 | 10.935683 |
| NPU conventional B, old `speed-a7` | 5,000 | 3.1642131477594377 | 12.88370049893856 |
| NPU hybrid B, old `speed-a7` | 5,000 | 3.1039837976296742 | 13.334167237083117 |
| NPU conventional B, old `speed-a7` | 7,500 | 2.438361665606499 | 12.519350358843804 |
| NPU hybrid B, old `speed-a7` | 7,500 | 2.4370980968077975 | 12.738550477723281 |

The historical NVIDIA B improvement over Original was about 24.4% at 30k, for
one seed. Old NPU hybrid and conventional B were nearly tied at 7.5k; there was
no Original NPU baseline. Neither observation predicts the new 30k outcome.
Old NPU results are context, not members of the new matched pair.

The historical NVIDIA recipe is available in this clone:
[Original worker](../v0-normalized-bridge-v1/worker.py),
[placement worker](../spiking-attention-placement-v1/worker.py),
[native builder/forward/loss](../spiking-transformer-stereo/scripts/run_experiment.py),
and [sample/flip stream](../shared-gpu-30k-v1/common.py).
Read those functions rather than current launcher defaults: some launchers still
stop at 7.5k, and the newer mechanism worker can override execution policy.

Old NPU aggregates above came from the supplied `speed-a7` endpoint CSV and raw
`seed-1234/{linear-identity,signed-surrogate}/train/B/evaluations.jsonl`.
Its frozen source identity was
`d67b3721cacd4287c92d293053c3683566b5d88754b28df639de72c293434ad6`.
The external export includes `source-speed-a5/source.tar.gz` with `worker_npu.py`.
These artifacts are not assumed present in a fresh clone. All settings needed
to specify this new comparison are below; historical scene comparisons need the
original raw logs and must not be reconstructed from aggregate means.

## 3. Exact model identities and initialization

Build both with `build_model('original', 1234)` in the linked native driver.
Native arguments: `hidden_dims=[128,128,128]`, `context_norm='batch'`,
`shared_backbone=False`, `n_snn_layers=3`, `n_downsample=2`, `corr_levels=4`,
`corr_radius=4`, `corr_implementation='reg'`, `mixed_precision=False`,
`feature_encoder='cnn'`, `context_encoder='cnn'`,
`matched_initialization_seed=1234`, `slow_fast_gru=False`.

Original has no placement module or `fnet` hook. For hybrid B use
[`spiking_attention_placement.install(model, 'B')`](../../core/spiking_attention_placement.py),
whose branch seed is **2017**. Preserve this exact `DescriptorAttention`:

- 256-dimensional descriptors project to width 16, followed by LayerNorm and
  separate learned Q/K/V projections, including their existing biases.
- Each projection is encoded by `signed_rate(current, steps=4, leak=0.9)`:
  threshold ±1, signed events −1/0/+1, differentiable soft reset, average four
  steps, original surrogate derivative and full gradients through recurrence.
- Bidirectional full-row attention uses `softmax(Q K^T / 4) V`. Both directions
  read the original unmodified feature pair. Preserve padded descriptors,
  full-row support, and continuous softmax; add no causal/positive-disparity mask.
- A shared 16→256 output projection, initialized to zero (weight and bias),
  adds the residual to each descriptor map before native correlation.
- Preserve non-reentrant activation checkpointing during training. No added
  FFN, temperature tuning, tanh replacement or alternate spike primitive.

Do not substitute the separate `EpipolarSpikingDescriptor` class: its width,
heads and FFN differ. Neither arm includes A cost filtering, v0, extra `+I`,
encoder conversion, guided correction or NMRF. Freeze BatchNorm using the native
`freeze_bn()` in both runs. Preserve native ALIF dynamics and reset all neuron
state between microbatches/evaluation batches using the driver.

Use the same native initial tensors for both runs and the historical saved
seed-1234 initialization when available. Load only native keys into Original;
load the full initialization into B. Do not load trained weights/optimizer state.
The historical full B initialization hash was
`b37335857fd34f9f5aaa1cf203f906304f7edbd965447ff7dad3953172d47bd4`;
the shared fnet/cnet/context_zqr_convs hash was
`2085df6ea20073c3a9804d79f6baee94e8e9bfe41facda47e3f292fd3cc9394d`.
Use the linked worker's sorted-name/tensor-byte hashing procedure. Compare all
native parameters between the new arms, not just the shared subset. If archived
initial tensors are unavailable, generate them once on CPU with the pinned
builder and check the historical hashes; matching seeds across framework
versions alone do not establish matching weights. Record any mismatch explicitly.

## 4. Shared NVIDIA training hyperparameters — apply to BOTH runs

| Setting | Required value |
| --- | --- |
| Model/train seed | 1234; B branch seed 2017 |
| Training endpoint | **30,000 optimizer updates per model**, 240,000 presentations per model |
| Scheduler horizon | **30,000 from update zero**, never shortened or restarted |
| Effective batch | **8** |
| Per-model device arrangement | **One NPU, physical microbatch 4, two accumulation microbatches per update**, matching the historical one-GPU arrangement |
| Accumulation order | First four samples, then next four; reset neuron state after each; optimizer step once after both |
| Optimizer | AdamW, LR/max LR 2e−4, betas (0.9, 0.999), epsilon 1e−8, weight decay 1e−5, amsgrad=False |
| Parameter groups | All trainable parameters in the same group, including B; no special branch LR or no-decay group |
| Schedule | OneCycleLR, total_steps=30000, pct_start=0.01, anneal_strategy='linear', cycle_momentum=False, div_factor=25, final_div_factor=10000 |
| LR endpoints | Initial 8e−6; peak 2e−4; configured minimum 8e−10; preserve the scheduler's actual step semantics |
| Update order | zero_grad(set_to_none=True), two weighted backwards, global norm clipping 1.0, optimizer.step(), scheduler.step() |
| Loss weighting | Native per-microbatch disparity loss × that microbatch's valid-pixel count / valid-pixel count across all eight samples; accumulate both gradients, with **no extra division by two** |
| Sequence loss | Sum over 16 predictions of `0.9**(15-i) * mean(abs(pred_i-target)[mask])`; no extra auxiliary loss |
| Train / evaluation refinements | 16 / 32 |
| Input | All 50 sensor bins as channels, native 250×400 resolution, no crops; FP32 before native `2*spikes-1` normalization |
| Train / Dev | 960 frames / 120 scenes; 60 frames / 30 scenes |
| Sampling | One CPU torch.randperm(960) with generator seed 1234, cycled; do not reshuffle each epoch |
| Augmentation | Historical H/V flips by absolute sample position; preserve the horizontal target sign reversal |
| Evaluation loader | Batch 2, no augmentation, generator seed 2225 (=1234+991) |
| Validation updates | 0, 250, 500, 1000, 1500, 2500, then every 2500 from 5000 through **30000** |
| Primary endpoint | Fixed **30,000-update** mean-scene EPE; no best-checkpoint substitution |
| Test/confirmation split | Not accessed for this comparison |

For zero-based update `u`, global sample position `p=8*u+j` addresses
`order[p % 960]`, j=0…7. Flips use SHA256 of `recipe-hv-v1:1234:{p}`;
H is byte 0 <128, V is byte 1 <128. Reuse the linked implementation. Preloading
must preserve tensors, sample order and flips exactly for both arms.

Use [`configs/synthetic-split.json`](../../../../configs/synthetic-split.json),
SHA256 `bad4369b6fd8b0cdf871a20b1727c2d9b59733a16bd33f75904227bdfef4510a`.
Provide the external read-only dataset root through configuration. The signed
target is `-(289.740625 * 0.08) / depth`. Preserve finite positive depth and
absolute disparity <512 dataset validity, then the driver's finite-target/<700
loss mask. Preserve padding to a multiple of 32, unpadding and output sign.

Do not reuse the old four-NPU × physical-batch-2 topology simply because its
effective batch was also eight. This plan intentionally matches microbatch size,
accumulation order and loss weighting as well as effective batch. If physical
batch four does not fit, report the measured memory failure before revising the
shared protocol; do not silently shrink either model's microbatch.

## 5. What matches NVIDIA and what cannot be identical on NPU

Training hyperparameters above must match. Device kernels and arithmetic cannot
be made bitwise identical merely by copying these settings. Both new arms use
one common fixed NPU execution policy from initialization to update 30,000.

| Item | Historical NVIDIA | New NPU pair |
| --- | --- | --- |
| Framework | PyTorch 2.7.1+cu128 | Record installed torch/torch_npu/CANN/driver versions; old port used Python 3.11.14, torch 2.9.0+cpu, torch_npu 2.9.0.post1+gitdc51c2d |
| Precision | FP32 tensors, AMP off; cuDNN convolution TF32 allowed, matmul TF32 off | FP32, AMP off, convolution/matmul HF32 off; TF32 and HF32 are not identical modes |
| Determinism | cuDNN deterministic=True, benchmark=False; this is not itself global strict deterministic execution | Keep previously requested strict determinism disabled, fixed for both runs; no cuDNN on NPU |
| AdamW implementation | foreach/fused omitted (backend defaults) | foreach=False, fused=False for the validated portable implementation; identical AdamW hyperparameters, different kernel dispatch |
| Loss implementation | Native boolean-indexed masked mean | Use native loss expression and the original two-microbatch weighting for this reproduction; do not silently enable the newer fixed-shape reduction override |
| Input / diagnostics | CPU FIFO prefetch; frequent logs/checkpoints | Preload allowed with identical decoded tensors/order; diagnostics every 100 and checkpoints every 500; these do not alter the training budget |

For the NPU policy keep JIT compilation disabled, strict deterministic algorithms
disabled and `HCCL_DETERMINISTIC=false`; the per-model run is single-device and
needs no rank gradient collective. Do not enable AMP, HF32, fused optimizers,
change batch size or switch policies mid-run to increase utilization. Log actual
resolved flags instead of assuming the newer worker's defaults match this table.
This is **NVIDIA-hyperparameter-matched training on NPU**, not a claim of identical
CUDA arithmetic. The within-NPU pair is the primary comparison.

Use one assigned Ascend device per model (the old study used Ascend910B2C).
If two devices are allocated, run the models concurrently on separate devices;
otherwise run sequentially on one. Do not invent site launch commands or device
allocations. Preload budget 12 GiB per process, two preparation workers and
prefetch depth two are permitted. Preserve native forward/backward calculations;
reduced logging and preload must not replace the model, loss or sample stream.

## 6. Handoff to the executing agent

The checked-in workers are CUDA-oriented and B-specific; `original-non-i` is
not an existing B variant argument. The external validated NPU archive can be
used if available. Otherwise port the linked worker, using the NPU-port guidance
in [b-attention-verification.md](b-attention-verification.md). The protocol here
supersedes its staged budgets for **these two new runs only**.

1. Add/select Original and historical hybrid-B modes with separate output paths.
   Preserve their model definitions and shared initial native weights.
2. Use one NPU, physical batch four and two accumulation backwards; remove the
   old distributed rank weighting/gradient SUM path for these single-device runs.
3. Adapt CUDA device/RNG/synchronization calls to NPU; restore optimizer tensors
   to the device and protect training RNG around validation. Omit only B-specific
   Q/K/V diagnostics for Original, not finite checks or clipping.
4. Set **both stop budget and scheduler horizon to 30000 in every launcher
   layer**. Disable the old 5k/7.5k tie, promotion and practical-gap stopping rules.
5. Pin source/configuration before training. Verify one disposable native-input
   update and resume for the new path. Check the zero-initialized B residual makes
   its initial output match Original within recorded numerical tolerance. Reuse
   existing port checks for unchanged paths; do not treat smoke outputs as results.

The historical CUDA/NPU discrepancy also warrants a small same-weight/same-batch
comparison of predictions, loss, gradients and one optimizer step when CUDA is
available. Record convolution precision differences. This is a port-fidelity
check, not an extra full training run or a performance-based screen.

Train BOTH arms to completion even if one is worse at 5k, 7.5k or later. A
nonfinite loss, invalid data or hardware failure warrants diagnosis/recovery,
not declaring that model an architectural loser. Preserve model, optimizer,
scheduler, RNG and sample position on resume; never reset the LR schedule.
Save named 7.5k and 30k checkpoints plus recovery checkpoints every 500 updates
and after evaluation. At 30k mark `COMPLETED`, updates=30000,
samples_seen=240000, schedule_total=30000, and retain the full Dev endpoint.

## 7. Output and analysis contract

Write both new runs within this experiment's ignored outputs directory:

```text
outputs/original-hybrid-b-npu-30000-v1/
  source-registry.json
  config.json
  seed-1234/
    original-non-i/
      metadata.json
      train.jsonl
      evaluations.jsonl
      status.json
      result.json
      last.pt
      step-7500.pt
      step-30000.pt
    hybrid-b/                  # identical file layout
  summaries/
    endpoints.csv
    scene-deltas.csv
    interpretation.md
```

Use environment/configured storage for large checkpoints if needed, retaining
links in the run folder. Do not commit raw logs, datasets or weights. Record exact
source revision, dirty changes, resolved configuration, initial-weight identity,
allocated device and environment. Keep old NPU/CUDA evidence separately labeled.

- `train.jsonl`: update, samples_seen, ordered sample IDs/flips, loss, gradient
  norm, LR used/next, step time, input wait and device memory. Log microbatch=4,
  accumulation=2 and device_count=1 in metadata.
- `evaluations.jsonl`: update, each frame's scene/frame/valid count/EPE/bad3,
  scene means, overall mean-scene EPE and pooled bad3. Mean-scene EPE averages
  frames within each scene, then scenes equally; pooled bad3 counts valid pixels
  with absolute error >3 px across all Dev frames. State whether bad3 is stored
  as fraction or percent and convert once when making tables.
- `endpoints.csv`: model, seed, backend, update, schedule_total, physical_batch,
  accumulation, effective_batch, mean_scene_epe, bad3_percent, source_identity,
  execution_policy. Primary rows are the new two models at 30k.
- `scene-deltas.csv`: update, scene, Original EPE, hybrid-B EPE,
  Original-minus-B EPE (positive favors B), plus paired bad3 if available.
- `interpretation.md`: 30k absolute/relative EPE change, scene wins, median scene
  difference, bad3, full learning curves and gap concentration in difficult
  scenes. Keep all scenes; do not drop failures or select the best interim step.

## 8. Interpretation after BOTH 30k endpoints

| Observation | What it supports / next step |
| --- | --- |
| Hybrid B improves Original on mean EPE with broad scene support | The B addition reproduces a useful benefit on NPU at this seed; replicate the pair at another seed before a robust superiority claim. |
| Hybrid B improves mean EPE but a few scenes dominate, or bad3 worsens | Report the distribution/metric tradeoff; test whether the gain repeats rather than describing universal improvement. |
| The models are close | No meaningful advantage established under this recipe; inspect uncertainty and historical reproducibility. |
| Hybrid B is worse | The historical improvement did not reproduce in this matched pair; investigate the discrepancy before investing in attribution of a presumed gain. |

The old 0.05 px band may be reported descriptively, not used to stop either run
or as a significance test. Two models at one seed do not identify the mechanism.
Even a clear B gain would not isolate spiking: a separately trained, matched
conventional B control is needed for that claim. No additional controls or seeds
are automatically included in this two-run protocol.
