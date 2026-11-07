# B attention verification — experiment and Huawei NPU handoff

Updated 7 November 2026. Read this before implementing or launching this study.

**Execution update, 7 November:** use the default [speed runtime](SPEED.md) for all
four cells. It consolidates host reads, avoids dynamic loss indexing and uses
500-update recovery checkpoints. The objective, paired data and 30k LR horizon
are unchanged. Read that page before launching or interpreting timing logs.

**Objective:** determine why placement B's hybrid signed-spiking feature exchange
learned a better stereo estimator than the matched conventional attention block.
Separate the forward representation from the local learning rule, then replicate
that result. This is a mechanism experiment, not a new attention architecture.

**Current status:** the four encoder variants and their CPU tests exist. The
training worker and launch template still target NVIDIA CUDA/cccl3. No Huawei NPU
port or NPU training has been validated. This handoff specifies the intended NPU
execution; it does not claim the existing commands already support it.

## 1. Results that motivate this experiment

These are completed **historical NVIDIA runs**, seed 1234, exactly 30,000 optimizer
updates, evaluated on the same Native Synthetic Dev set: 60 frames in 30 scenes.
EPE is mean-scene disparity error in native pixels; bad3 is the pooled percentage
of valid pixels with absolute disparity error greater than 3 pixels. Lower is better.

| Model | EPE, px | bad3, % | Role |
| --- | ---: | ---: | --- |
| Original SpikeStereoNet, no added placement | 2.691127 | 10.712 | Architectural context |
| B, conventional linear Q/K/V | 2.768814 | 10.935683 | Conventional reference; job 2902 |
| B, hybrid signed T4 Q/K/V | **2.034117** | **9.626333** | Hybrid reference |

Hybrid B lowered EPE by **0.734697 px / 26.53%** relative to conventional B, and
bad3 by **1.309350 percentage points**. This is a large observed one-seed gap,
not a demonstrated statistically significant or cross-dataset advantage.
Conventional B did not improve the Original baseline at this fixed endpoint.

Both B variants retain the same native CNN, recurrent ALIF updater, correlation
pyramid, feature projections, parameter count (9,312 added parameters), attention
support and training settings. B exchanges information bidirectionally across the
opposite epipolar row **before** correlation. Q/K/V width is 16, attention logits
are divided by 4, and the output is projected back to 256 channels and added as a
residual. Both directions use the original feature pair. The conventional encoder
returns linear projected values; it does not use tanh. The hybrid encoder applies
signed T4 coding to Q, K and V before the same continuous softmax and value mixing.

The learning curves matter: conventional B's lowest recorded Dev EPE was 2.582825
at 15k; hybrid B's was 1.763581 at 17.5k. Both worsened by 30k. Keep fixed 30k as the
primary endpoint, and retain the full curves rather than silently selecting each
run's best checkpoint.

### Embedded historical curve summary

The following compact values were copied from the completed B evaluation records
and verified against them on 7 November 2026. They are included here so a new
computer can plot the historical comparison without the old repository or logs.
Values are rounded to six decimals. Raw predictions, checkpoints and scene-level
historical records are not included; do not infer unreported uncertainty or
historical scene breakdowns from this summary. All new NPU analysis must use the
new runs' logs.

| Updates | Conventional EPE, px | Hybrid EPE, px | Conventional bad3, % | Hybrid bad3, % |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 44.376375 | 44.376375 | 100.000000 | 100.000000 |
| 250 | 15.607711 | 10.794515 | 83.362931 | 65.146748 |
| 500 | 8.585500 | 7.918224 | 48.652466 | 46.365749 |
| 1000 | 5.002392 | 5.016571 | 24.602816 | 23.243349 |
| 1500 | 4.541310 | 4.729780 | 18.800166 | 18.787883 |
| 2500 | 4.270697 | 4.132856 | 16.452350 | 17.932716 |
| 5000 | 3.441170 | 2.651039 | 14.381683 | 12.870150 |
| 7500 | 2.921007 | 2.346435 | 12.832750 | 12.012433 |
| 10000 | 2.832781 | 2.177690 | 10.969100 | 10.373983 |
| 12500 | 2.633178 | 2.203002 | 12.076300 | 11.256683 |
| 15000 | 2.582825 | 2.034606 | 11.406416 | 10.018333 |
| 17500 | 2.804930 | 1.763581 | 11.073666 | 9.592483 |
| 20000 | 2.718351 | 2.124071 | 11.145266 | 9.725316 |
| 22500 | 2.706026 | 2.040907 | 11.118850 | 10.029150 |
| 25000 | 2.719396 | 2.078313 | 11.002883 | 9.881266 |
| 27500 | 2.629380 | 1.910924 | 10.977400 | 9.412350 |
| 30000 | 2.768814 | 2.034117 | 10.935683 | 9.626333 |

Provenance labels only (no files need to be retrieved): conventional B was job
2902, source identity `17f577a9bf16f6cface49e9134be085f0aac4301958c97212b5b5c9783430c0f`;
hybrid B used placement source identity
`606dcfe39ee26b7eecb4f52768376db76a80c5a4243929795cec2d233ba82f56`.
These identities describe the old experiments, not dependencies of the NPU study.
The new mechanism cells have **no reported NPU EPE results yet**.

## 2. What we want to find out

The existing substitution changes forward values and backward derivatives together.
We need the following factorial comparison, with all other calculations fixed:

| Variant ID | Forward Q/K/V values | Local encoder derivative | Question |
| --- | --- | --- | --- |
| `linear-identity` | Unbounded linear projections | Identity | How does conventional B perform? |
| `signed-surrogate` | Original signed T4 code | Original surrogate through recurrence/reset | How does hybrid B perform? |
| `linear-surrogate` | Unbounded linear projections | Original signed T4 surrogate derivative | Does the learning rule help without forward spike codes? |
| `signed-identity` | Original signed T4 code | Identity / straight-through | Does the forward encoding help without the original surrogate? |

The signed code averages four events in {-1, 0, +1}; its output has nine levels
between -1 and +1. Every internal step receives the same projected input. This is
not four new sensor observations and is not a fully spike-only attention block.
The original four-step leak (.9), threshold (1) and differentiable soft reset
remain fixed. No tanh, extra normalization, temperature tuning or new FFN is added.

The crossed backward rules are deliberately prescribed surrogate derivatives.
They change the local derivative at Q/K/V encoding, not the entire downstream
loss gradient. Evaluate the surrogate using each current run's projected inputs.
Do not borrow weights, codes or gradients from a trained reference model.

### Staged NPU run plan — minimum-resource investigation

**Do not launch eight 30k runs initially. Start with a 5k screen, spend more only
on an informative contrast, and preserve resumable checkpoints.** This plan
supersedes the earlier full-matrix-first execution plan.

Historical B already showed a useful gap at 5k: conventional EPE 3.441170 versus
hybrid 2.651039, a 23.0% reduction. At 2.5k the reduction was only 3.2%; at 7.5k it
was 19.7%. This motivates 5k as the initial screening budget, without assuming
that the NPU runs will reproduce the historical trajectory.

| Stage | Fresh runs / continuation | Budget | Decision |
| --- | --- | ---: | --- |
| 0. Port validation | Tiny CPU/NPU checks and disposable recovery preflight | Engineering checks only | Validate all four implementations before scientific training. |
| 1. Establish the NPU signal | `linear-identity` and `signed-surrogate`, seed 1234 | 2 × 5k = **10k total updates** | Is there a useful hybrid advantage on the new setup? |
| 2. Identify the contributing component | `linear-surrogate` and `signed-identity`, seed 1234 | 2 × 5k = **10k additional** | Compare the complete four-cell factorial at the identical 5k checkpoint. |
| 3. Replicate the deciding contrast | The two cells needed for the leading explanation, seed 2345 | Usually 2 × 5k = **10k additional** | Does that specific forward-rule or backward-rule effect repeat? |
| 4. Confirm persistence | Resume the informative cells to a common later checkpoint, potentially 30k | Conditional; record before continuing | Does the early mechanism also explain later/final performance? |

The first factorial answer costs **20k optimizer updates in total**, versus 240k
for eight complete 30k runs. A two-cell second-seed replication brings this to
30k total. These are update budgets, not guaranteed wall-clock or energy savings;
measure throughput on the actual NPU. Budget disposable preflight separately.

#### Decisions between stages

- After stage 1, inspect fixed-5k EPE, bad3, paired scene differences and the
  planned learning-curve points. Use the existing 0.05 px practical band to flag
  small differences, not to claim significance or equivalence. A clear advantage
  supports stage 2; an inconsistent trajectory weakens that decision.
- If no useful gap appears, inspect port fidelity. If the implementation checks
  pass but the trajectory remains inconclusive, extend **only those two reference
  runs** to 7.5k (5k additional updates total). Do not fund crossed training merely
  to explain a gain that has not reproduced.
- If stage 1 uses 7.5k as the deciding checkpoint, train both crossed cells to
  7.5k before comparing them. Never compare a crossed 5k result with a 7.5k reference.
- If all four 5k results are ambiguous, extend those same runs to 7.5k rather
  than immediately starting another full seed. Record the extension and preserve
  the initial 5k outcome; this is an adaptive exploratory study.
- Choose the seed-2345 pair from the question: `linear-surrogate` versus
  `linear-identity` tests the surrogate benefit with continuous forward values;
  `signed-identity` versus `linear-identity` tests signed forward encoding with
  an identity derivative. Other paired comparisons may be appropriate; record the
  selection before launching the second seed.
- A two-cell replication supports only its specific contrast. To claim a replicated
  factorial interaction, repeat **all four** cells at seed 2345. To claim a crossed
  cell matches hybrid B at the second seed, also run that seed's hybrid reference.
  Do not infer unrun cells from the first seed.
- Continue to longer training only when the earlier result justifies that cost.
  Every later comparison must use the same endpoint, and must report early and
  late outcomes even if rankings change. No automatic best-checkpoint selection.

Use [configs/matrix-npu.json](configs/matrix-npu.json) as the **catalog of eight
possible fresh cells**, not an instruction to launch all eight. Its `updates=30000`
and [configs/protocol-npu.json](configs/protocol-npu.json) describe the retained
full training horizon. The execution agent must add an explicit stage/run budget
selection before launching; the current files do not implement this staged queue.
Historical CUDA endpoints remain context, never reused NPU controls. The older
`configs/matrix.json` is a CUDA plan and must not drive this study.

#### Pause the run; do not compress the learning-rate schedule

Keep `OneCycleLR(total_steps=30000)` from the first optimizer update. For the initial
screen, **pause at optimizer update 5000**. Do not change `total_steps` to 5000,
accelerate the decay, reset the optimizer, or restart the schedule when extending.
Retain the full Dev set, batch contract, data order and evaluation points. This
tests the early portion of the original recipe rather than a new short recipe.

The current lower-level worker has a `--stop-at` argument, but `run_on_gpu.py`
hard-codes 30000 and asserts a 30k endpoint. The NPU agent must implement a real
stage-budget option and forward it through every wrapper. Keep the stage budget
separate from immutable model/optimizer configuration, so increasing the pause
point does not invalidate an otherwise identical resume. At 5k/7.5k:

- Finish the scheduled evaluation and atomically save model, optimizer, scheduler,
  RNG, sample order and evaluation counters.
- Retain a named checkpoint for that screened update in addition to `last.pt`.
- Mark the run `SCREEN_COMPLETED` or equivalent, with its actual update, rather
  than `COMPLETED_30000`. The wrapper must accept a valid screened endpoint without
  looking for `step-30000.pt` or reporting failure.
- Resume from that complete checkpoint, preserving the 30k scheduler position.
  Full Dev evaluation is required at every selected comparison endpoint.

### Fixed training contract

- Native Synthetic manifest: 960 Train frames, 60 Dev frames; no Test access.
  Manifest SHA256:
  `bad4369b6fd8b0cdf871a20b1727c2d9b59733a16bd33f75904227bdfef4510a`.
- All 50 sensor bins, native resolution, matched cyclic sample order and HV flips.
- Full schedule horizon: 30,000 updates / 240,000 sample presentations. Initial
  per-run pause: 5,000 updates / 40,000 presentations. Physical batch 4,
  accumulation 2, effective batch 8; keep valid-count microbatch loss weighting.
- 16 train / 32 Dev refinement iterations; preserve disparity sign, masks, padding,
  full valid evaluation support, and scene averaging.
- AdamW: peak LR 2e-4, weight decay 1e-5, epsilon 1e-8; gradient clipping norm 1.
  Linear OneCycle, total_steps=30000, pct_start=.01, cycle_momentum=False.
- FP32 first. Do not enable BF16/FP16, fused optimizer substitutions, compiled
  execution or mixed precision for one cell only.
- Train/model seed 1234 or 2345; branch seed = seed+783 (2017 or 3128). Seed sample
  order and flips with the train seed; Dev loader uses seed+991. Within each seed,
  every cell begins from identical CPU-initialized tensors and presentation order.
- Evaluate updates 0, 250, 500, 1000, 1500, 2500, then 5000 through 30000 every 2500.
  Complete each selected stage budget before its decision; do not stop a cell
  midway merely because an intermediate score is worse. Later stages are conditional.

If NPU memory requires a different physical batch, preserve effective batch and
valid-pixel weighting and apply the same change to every cell. Record the changed
recipe and verify accumulation behavior; do not call the run an exact historical
reproduction. Preserve model/optimizer/scheduler and CPU/device RNG state on resume.

## 3. Hardware and NPU porting instructions

**Target hardware: Huawei Ascend NPUs, not NVIDIA GPUs.** Exact NPU model, available
memory, device count, host, scheduler and software versions have not been supplied.
Record those facts from the actual environment; do not assume Ascend 910B, 910C or
any particular capacity. Initially use one NPU per run. Independent cells may run
on separate available NPUs; distributed training is not needed for this study.

TorchNPU is the PyTorch adapter for Ascend. Select an explicitly compatible set of
PyTorch, `torch_npu`, CANN, Python, firmware and driver versions using the official
[Ascend README](https://github.com/Ascend/pytorch/blob/master/README.md) and
[compatibility matrix](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md).
Record the selected versions instead of installing an arbitrary latest release.
Operator support must be checked on that selected stack.

### Mathematics and data logic to preserve

| Component | Porting expectation |
| --- | --- |
| `core/attention_encoder_controls.py` | Pure tensor operations and custom PyTorch autograd; preserve the equations and all four forward/backward definitions. No CUDA kernel is implemented here. Test actual NPU backward execution. |
| `core/spiking_attention_placement.py::DescriptorAttention` | Keep projections, LayerNorm, matmul, softmax, scale, full-row support and residual unchanged. Verify NPU checkpoint recomputation and gradients. |
| `core/epipolar_spiking_stereo.py::signed_rate` | Preserve signed thresholds, leak, averaging and soft-reset gradients. Do not replace it with a different native spike primitive for convenience. |
| Native loading and phase-2 deterministic prefetch | CPU decoding, sample indices, flips and JSON records can retain their logic. Verify native depth decoding and data paths on the NPU host. |
| Loss, metrics and schedule | Keep formulas, units, masks, denominators and update counts unchanged. |
| Native ALIF neuron equations | Keep the existing `backend='torch'` path and state resets. Test SpikingJelly's operations and scripted reset on the installed stack; do not switch to a CuPy backend. |

These are portability expectations from source inspection, not an assertion that
every operator is already supported on every Ascend device.

### Code that requires adaptation or explicit verification

| Location / current CUDA dependency | Required NPU work |
| --- | --- |
| `../conventional-attention-placement-v1/worker.py`: `.cuda()` and CUDA availability checks | Introduce a device/backend abstraction; move model, batches and restored optimizer tensors with `.to(device)`. Validate an assigned NPU rather than CUDA device count. |
| CUDA seeding, `get_rng_state_all`, `set_rng_state_all`, and `torch.random.fork_rng(devices=[0])` | Save/restore the selected NPU backend's RNG state alongside Python/NumPy/CPU/loader RNG. Make the evaluation RNG guard explicitly NPU-aware; the existing default fork targets CUDA. Verify resume and evaluation do not alter the intended training RNG stream. |
| CUDA synchronization, cache operations, memory counters and OOM handling | Use supported NPU equivalents through the adapter. Check their availability on the pinned version and record unsupported telemetry as null, not zero. Preserve atomic checkpoints on failure. |
| Device determinism and TF32 controls | B now disables PyTorch deterministic algorithms and cuDNN determinism; benchmarking stays off and precision is unchanged. Preserve this relaxed policy across all four NPU cells, verify the compatibility layer honors it, and record any native NPU settings. Seeds/order/flips remain fixed. See README's device-determinism section for source/resume implications. |
| `core/spikestereonet.py`: `torch.cuda.amp.autocast` wrappers | Keep mixed precision disabled. Use a device-neutral no-op context for the FP32 study, or a supported backend context if later registered. Do not silently lower precision. |
| `core/corr.py` and `core/utils/utils.py::bilinear_sampler` | The registered `reg` path uses PyTorch correlation, pooling and `grid_sample`, not a custom correlation CUDA extension. Check forward AND backward support, including height-one correlation rows, `align_corners=True`, zero padding, fractional/OOV coordinates and pyramids. Audit CPU-created scalar tensors for device mixing. |
| `core/ALIF.py`, transposed convolutions and scripted reset | Check gradients through recurrence, state reset and `torch.jit.script` behavior. If scripting must be replaced by eager tensor code, preserve equations and verify values/gradients. |
| Non-reentrant `torch.utils.checkpoint` and `_CrossedCode.backward` | Test recomputation, saved tensors, finite gradients and identical forward outputs when only the backward rule changes. Run FP64 reference checks on CPU; do not assume NPU FP64 support. |
| `run_on_gpu.py`, `slurm/run.sbatch` and their old allocation assumptions | Implement an NPU launcher using the actual site allocation mechanism. The old CUDA template also refers to an allocation guard absent from this standalone checkout; it is not an NPU dependency and must not be used as a launcher. |
| Historical initialization hash assertions | Framework versions can change initial tensors. Prefer an identical saved CPU initialization per seed for all four NPU cells. Record its identity. Keep historical checks when the original environment is reproduced; otherwise add a distinct NPU identity check, not a blanket deletion of validation. |
| `freeze_source.py` | Default mode packages this standalone checkout, including current source changes, into a new source capsule. No external archive is needed. Use this mode after the NPU port. The optional `--reference` mode is only for legacy CUDA work and is not part of this study. |
| Legacy manifest path and registry assumptions | The old runner points to a missing `spiking-transformer-stereo/configs/manifest.json`. Use the included root `configs/synthetic-split.json` via an explicit argument. Read source identity from the newly generated capsule registry; never require old campaign registrations or private deployment files. |

This is a **fresh study using the model source included in this standalone
repository**. Do not seek an old source capsule or silently claim byte-identical
historical reproduction. Preserve one fixed model source across all eight cells
and record it in the new registry. Use a separate NPU worker/adapter so the old
CUDA implementation remains readable. The portable `run.py` loop has different
sampling/loss-accumulation details; do not substitute it for the specified matched
protocol without declaring and applying that change to every cell.

### Minimum port validation before full training

1. Run `test_controls.py` on CPU: diagonal implementation parity, crossed values
   and local gradients, module initialization, checkpoint recomputation, seed
   streams and wrapper argument forwarding. Seven CPU tests passed during code
   preparation; the standalone `.venv` lacked SpikingJelly, so those checks used
   the existing research test environment. This is not an NPU validation result.
2. Parameterize the tiny tensor tests for NPU FP32 and compare against CPU using
   identical tensors/weights and stated tolerances. Add the native correlation
   sampling and recurrent-state checks above. Report measured numerical deltas.
3. On one native Train batch, verify finite predictions/loss, model backward and
   an optimizer step for all four cells. Q/K/V gradients can be zero at the initial
   zero output projection; verify them after that output projection has learned.
4. Run the disposable pause-at-2/resume-to-3 check on NPU, including optimizer,
   scheduler, RNG and sample order. Compare with an uninterrupted three-step
   preflight at the same initialization. Discard these weights before production.
5. Record the port diff, environment and measured memory. Start fresh production
   only when the reference and crossed cells execute under one fixed protocol.

This document requests preparation and defines the intended hardware; actual job
submission still requires the user's execution instruction and real site details.

## 4. Expected interpretation of the results

Let L = linear forward, S = signed forward, I = identity derivative and G = the
original surrogate derivative. Compare the same fixed stage endpoint within the
same seed/backend: initially 5k, or 7.5k after a documented extension. Apply the
same contrasts again at 30k if final confirmation is funded. Treat 0.05 px as a
descriptive practical band, not a statistical equivalence test.

**The screen identifies an early-learning mechanism.** Even a replicated 5k effect
does not yet establish the cause of the historical final-30k advantage. A selected
two-cell second-seed result is a targeted replication, not a replicated full matrix.

| Observed outcome, replicated where possible | Supported interpretation | Next experiment |
| --- | --- | --- |
| L,G improves over L,I and approaches S,G | The surrogate learning rule can help without forward spike codes. | Separate gradient shape from overall gradient scale; test transfer of the useful rule. |
| S,I improves over L,I and approaches S,G | The signed forward representation can help without its original surrogate. | Separate continuous bounding, static quantization and T4 encoding with a common backward rule. |
| Both crossed cells improve | More than one ingredient or route may contribute. | Quantify both effects and their interaction; do not assign the entire gain to one property. |
| Only S,G performs well | The original encoding/derivative pairing appears important. | Replicate the interaction, then locate the effect in Q/K versus V. |
| Neither diagonal advantage nor crossed gains reproduce | The original gap may be seed-sensitive, backend-sensitive or affected by a port mismatch. | Check port fidelity and scene-level results; use another paired seed if justified. |
| A crossed cell beats S,G | A simpler or different training/encoding combination may be preferable. | Repeat it and develop the method around the actual successful property. |

Report E(L,I)-E(S,I), E(L,G)-E(S,G), E(L,I)-E(L,G), E(S,I)-E(S,G), and the
interaction [E(L,I)-E(S,I)]-[E(L,G)-E(S,G)]. Positive individual contrasts favor
the named forward or gradient change. Include per-scene differences and seed
variation. Scenes are not substitutes for independent training seeds.

A result does not by itself show better raw correspondence ranking, exploitation
of sensor-time dynamics, lower energy or a uniquely spiking advantage over all
continuous alternatives. Use the identified ingredient to design the next test,
not the label “spiking” as the explanation.

## 5. Where to write logs and how to make them graph-ready

**Keep this study's outputs inside this experiment folder, under ignored
`outputs/`.** Do not mix NPU runs with historical CUDA logs. Use a unique study ID
for each source/protocol revision. From the standalone repository root:

```bash
export B_ATTENTION_DIR="$PWD/projects/snn-depth-estimation/experiments/b-attention-mechanism-v1"
export B_ATTENTION_OUTPUT_ROOT="$B_ATTENTION_DIR/outputs/npu-<study-id>"
mkdir -p "$B_ATTENTION_OUTPUT_ROOT"
```

Replace `<study-id>` with a descriptive unique identifier. The future NPU runner
must accept this path as an argument/environment setting. These variables define
the output contract; the NPU launcher that consumes them has not been implemented.
When compute uses a remote checkout, preserve this relative tree there and sync
compact logs back to the same experiment folder. Keep large checkpoints remote if
needed and record their location; do not copy datasets into this directory.

```text
b-attention-mechanism-v1/
  b-attention-verification.md
  outputs/                              # ignored by Git and source packaging
    npu-<study-id>/
      study.json                        # cell catalog, selected stages, protocol/source
      stage-decisions.jsonl             # outcomes, selected contrasts and next budgets
      port-validation.json              # environment, checks, measured differences
      seed-1234/
        linear-identity/
          launch.json
          stdout.log
          stderr.log
          preflight/B/                  # discarded checks, excluded from graphs
          train/B/
            metadata.json
            status.json
            train.jsonl
            evaluations.jsonl
            recoveries.jsonl
            checkpoint.json
            result.json
            last.pt
            step-5000.pt                # retain screened checkpoints
            step-7500.pt
            step-30000.pt
        signed-surrogate/...
        linear-surrogate/...
        signed-identity/...
      seed-2345/...
      summaries/
        curves.csv                      # derived after reading committed JSONL
        endpoints.csv
        scene-deltas.csv
        factorial-contrasts.csv
      figures/                          # later plots; not required for execution
```

The existing worker already emits most `train/B` records. The NPU port must retain
those fields and add the backend identity to metadata; study/port records, explicit
stdout/stderr capture and CSV exports are additional handoff requirements, not
files already produced by the current implementation.

### Record contract

| File | Required content |
| --- | --- |
| `study.json` / `metadata.json` | Study/run ID, arm B, variant, forward/backward rules, seed and branch seed, manifest identity, Git revision plus exact source identity/dirty patch, full optimizer/schedule/batch/iteration config; NPU model/memory/device, driver/firmware/CANN/torch/torch_npu/Python versions, precision flags, initialization identity, start time and deployment job ID. |
| `train.jsonl` | One row per committed optimizer update: update, samples_seen, loss, gradient_norm, placement_gradients, lr_used, lr_next, indices, augmentation_hv, seconds and measured peak memory. Preserve original names. NPU synchronization must make timing meaningful. |
| `evaluations.jsonl` | update, samples_seen, mean_scene_epe, scene_epe mapping, pooled_epe, bad1/bad3/bad5 and per_frame rows containing scene/frame IDs, EPE, bad rates and valid_pixels. Record no missing measurement as zero. |
| `checkpoint.json` / `recoveries.jsonl` | Last complete update, scheduler counter, completed evaluation steps and each restored update. Resume must reconcile uncommitted rows so graphs have one committed record per update. |
| `status.json` / `result.json` | Running/paused/failed/completed state, latest update/time, completion counters and finite-state check. A valid screened evaluation/checkpoint establishes only its stage endpoint. Only a valid 30k evaluation plus completed checkpoint establishes the full-schedule endpoint. |

Keep EPE in **pixels**, bad rates in **percent**, learning rate as its raw numeric
value, durations in **seconds**, memory in **MiB**, and timestamps in ISO-8601 UTC.
Use JSON numbers with full available precision. Store undefined metrics as `null`
with an explanatory status; do not emit NaN/Infinity or invent an endpoint.

### Future graph exports

Use a long-form `curves.csv` with columns:

```text
study_id,backend,seed,variant,update,samples_seen,metric,value,unit
```

`endpoints.csv` should contain one row per completed seed/variant/stage, including
`stage`, `endpoint_kind` (`screen` or `full_schedule`), `mean_scene_epe`, `bad3`,
the actual `update` (5000, 7500 or 30000), `schedule_total=30000`, and source/protocol
identity. Never label a screened endpoint as a completed 30k experiment.
`scene-deltas.csv` should identify the seed, scene, both compared variants and both
EPE values before their difference. Derive paired factorial contrasts within seed.

Plot EPE and bad3 versus optimizer update, and training loss/LR separately. Plot
individual seeds as well as any aggregate. Do not merge CUDA and NPU curves into
one seed average, include preflight data, interpolate missing endpoints, or replace
fixed-stage results with best-Dev values. Keep 5k/7.5k screening panels separate
from 30k confirmation panels. Record the reason and chosen comparison before each
extension or second-seed launch in `stage-decisions.jsonl`. Preserve JSONL as the source of truth and
regenerate CSV/figures from it.

## Fresh-clone inputs and repository map

All paths below are relative to this repository, not another checkout. The
repository provides the architecture, control implementations, unit tests, split
manifest, run matrix and training specification. **The NPU backend remains work
for the receiving agent**; pulling the repository does not make the CUDA worker
NPU-compatible.

| Included file | Role |
| --- | --- |
| [Encoder controls](../../core/attention_encoder_controls.py) | Four variants, signed local Jacobian and B installation. |
| [B attention](../../core/spiking_attention_placement.py) | Unchanged attention computation and original code path. |
| [Neuron encoder](../../core/epipolar_spiking_stereo.py) | Original `signed_rate` and surrogate threshold. |
| [Tests](test_controls.py) | CPU wiring, derivative, initialization and wrapper checks. |
| [Worker adapter](worker.py), [settings](settings.py) | Variant/seed routing; adapt for the NPU worker. |
| [Training loop](../conventional-attention-placement-v1/worker.py) | Reference training/evaluation/checkpoint semantics; CUDA calls need porting. |
| [Data/order contract](../spiking-attention-placement-phase2/protocol.py), [common utilities](../shared-gpu-30k-v1/common.py) | Deterministic sampling/flips, logging and recovery helpers. |
| [Native driver helpers](../spiking-transformer-stereo/scripts/run_experiment.py) | `build_model`, `ManifestDataset`, `forward`, `disparity_loss`, `metric_values`, `reset_model`. Reuse these helpers; its campaign-specific `main()` is not the new launcher. |
| [Split manifest](../../../../configs/synthetic-split.json) | Exact relative Train/Dev filenames. Use only `train` and `dev`; ignore `confirmation` and `overfit`. |
| [NPU matrix](configs/matrix-npu.json), [NPU protocol](configs/protocol-npu.json) | Eight fresh cells and fixed settings, with unknown hardware fields explicit. |
| [Dependencies](../../../../requirements.txt), [source packager](freeze_source.py) | Environment dependencies and self-contained source snapshot. |

External inputs that cannot come from source control:

- **Native Synthetic data files.** Set `SPIKE_DATA_ROOT` to the dataset root; never
  assume a local personal path. The manifest supplies relative filenames. Example:
  `Train/scene_0000/spike_l/0001_spike_l.dat`, corresponding `spike_r` file, and
  `Train/scene_0000/depth/0001_depth.exr`. Dev uses held-out Train scenes starting
  with `scene_0120`. Verify every required Train/Dev file exists before launching.
  All 50 binary spike bins are loaded; raw `.dat` uses 250×400 geometry and EXR
  stores depth. The included loader uses signed flow `-(289.740625*0.08)/depth`,
  finite positive depth, and `abs(flow)<512` validity. Preserve that convention.
  If data are unavailable, request their location/access; do not substitute RGB
  or another event dataset and call it the same experiment.
- **An assigned Ascend environment.** Obtain NPU model/count/memory and site
  allocation instructions, plus compatible CANN/TorchNPU packages. Do not invent
  account, queue, container or device settings.

Install the chosen backend's compatible `torch`/`torchvision`/`torch_npu` stack
first. Resolve the remaining root requirements using constraints that retain
those versions; do not let an unconstrained requirements install replace the NPU
build. For CPU-only checks use a CPU PyTorch environment plus the listed model
dependencies, including `spikingjelly==0.0.0.0.14`. No old virtual environment is
required. From the root of a fresh checkout:

```bash
python projects/snn-depth-estimation/experiments/b-attention-mechanism-v1/test_controls.py -v
```

After implementing and testing the NPU port, package the checkout (including the
port files) without an old archive:

```bash
export B_ATTENTION_DIR="$PWD/projects/snn-depth-estimation/experiments/b-attention-mechanism-v1"
export B_ATTENTION_SOURCE="$B_ATTENTION_DIR/outputs/source-npu-<revision-id>"
python "$B_ATTENTION_DIR/freeze_source.py" --output "$B_ATTENTION_SOURCE"
```

The output directory must be new. It contains `source.tar.gz` and
`source-registry.json`; stage both on the NPU host, unpack there, and supply the
dataset/output paths to the future NPU launcher. Logs/checkpoints under `outputs/`
are excluded. The fresh-source registry explicitly records that this is not a
historically equivalent CUDA reproduction.

## Agent completion criteria

The next execution agent should deliver a tested NPU backend/launcher consuming the
included NPU cell catalog with explicit stage selection and resumable pause budgets,
the exact packaged source/configuration, port-validation records,
and the output tree above. Report any unsupported operation or recipe change
explicitly. Do not claim NPU readiness from CPU tests alone. Once runs finish,
provide the paired EPE/bad3 table, learning curves, scene deltas and factorial
interpretation, including negative or inconsistent outcomes. Distinguish an early
mechanism screen from later confirmation; completion does not require running
every catalog cell to 30k.

## Optional input-side optimization

[INPUT_PIPELINE.md](INPUT_PIPELINE.md) documents the RAM cache, compact uint8
spike transfers and ordered CPU batch preparation. These changes preserve the
scientific input stream and remain opt-in. Record the chosen mode per run and
verify the target NPU's uint8 transfer/cast path in preflight. Compare consumer
wait and wall time before concluding that input loading explains low utilization.
