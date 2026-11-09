# Hybrid B ablation — 911 — NPU

Prepared 9 November 2026. **Status: executable NPU batch implemented; CPU checks completed, NPU validation runs automatically on the server. No jobs launched locally.**
Work on `main` in the standalone spiking-depth-estimation repository. The user-specified
`911` is a study label, not an accelerator model, random seed or training budget.

## Objective and scope

Determine whether Hybrid B's advantage over conventional cross-attention comes
from its **forward signed-T4 Q/K/V encoding**, its **surrogate training derivative**,
or their interaction. All models consume the same 50-bin spike-camera input.
“Continuous” describes internal Q/K/V values, not a different sensor modality.

Use the eight available NPU slots for four variants × two paired seeds, with one
independent model per device. Train every admitted run to **30,000 optimizer
updates**, using the completed Original/Hybrid NPU recipe below. No score-based
5k/7.5k gate applies. Intermediate curves are diagnostic, not endpoint selection.
NVIDIA confirmation is a later decision, not part of this batch.

This protocol supersedes the staged budgets in `b-attention-verification.md` for
this named batch only. Do not modify or restart older studies. Do not add Original,
A-site controls, positional routing, temperature tuning, bounded-continuous or
static-quantization arms to this batch.

## Existing evidence and reference implementation

Completed Ascend910B2C study `original-hybrid-b-npu-30000-v1`, seed 1234:

| Model | Update | Mean-scene EPE (px) ↓ | Pooled bad3 (%) ↓ |
| --- | ---: | ---: | ---: |
| Original non-I | 30,000 | 2.364822164674600 | 10.688950480024020 |
| Hybrid B | 30,000 | 1.938087996343772 | 10.302883766591549 |

Hybrid B reduced EPE by 18.045%, winning 18/30 Dev scenes. That comparison establishes
a benefit for the B package at this seed, not a spiking-specific advantage.

The completed NPU worker was `worker_baseline_npu.py`. Its single-device training
loop is now included as [`worker_ablation_npu.py`](worker_ablation_npu.py), with
seed/variant routing added. The original NPU backend and input-preparation helpers
are included unchanged. [`npu-source-provenance.json`](npu-source-provenance.json)
records the imported source identity and per-file hashes. The archived model,
data, loss and protocol dependencies matched this checkout when imported.

This handoff is self-contained: the server does not need the Research monorepo,
old source archive or historical result folders to run it. The historical numbers
above are references, not results of this new batch.

## Execute on the NPU server

Use the working Ascend Python environment from the completed run. Its dependencies
must include this repository's Python requirements and the matching `torch_npu` /
CANN installation; do not replace the vendor PyTorch installation with a generic
CPU/CUDA wheel. Run from the repository root, in a persistent terminal/session.
The dataset root must contain the files named in `configs/synthetic-split.json`.

```bash
python projects/snn-depth-estimation/experiments/b-attention-mechanism-v1/run_ablation_on_npu.py \
  --data-root /path/to/synthetic-dataset \
  --devices 0 1 2 3 4 5 6 7 \
  --execute
```

Replace the dataset path and device IDs with the server's actual allocation. The
command creates a frozen source registry and two CPU initial states, runs tiny
on-device encoder/checkpoint checks, then three native-shape preflight updates
with pause/resume for each cell. **All eight preflights must pass before any
production training starts.** It then runs eight fresh cells to 30k and exports
CSV summaries automatically. No code editing or historical-file recovery is
required. The native training worker remains FP32, with strict determinism off.

- Omit `--execute` to prepare the source/configuration/initializations only.
- Add `--preflight-only --execute` to stop after successful NPU validation.
- Add `--resume --execute` to continue an interrupted batch using the same source,
  output path, device mapping and hyperparameters. Completed cells are skipped;
  unfinished cells restore their own checkpoint. No trained weights cross cells.
- Optional `--output-root /path/to/output` relocates the whole batch. Default is
  `projects/snn-depth-estimation/experiments/b-attention-mechanism-v1/outputs/hybrid-b-ablation-911-NPU`.
- Fewer device IDs queue the same eight cells sequentially per device. Batch size,
  seeds and step budget are unchanged. With eight concurrent runs, allow at least
  96 GiB for preload budgets **plus** model/process/OS overhead. Use fewer devices
  if host memory is insufficient; do not silently change the data recipe.
- Inspect each cell's `stdout.log`, `stderr.log`, `status.json` and `train.jsonl`.
  Failed validation prevents production admission. Preserve failure evidence;
  fix the reported issue and use a new output root if code/configuration changes.
- A deliberately created `pause.request` must be removed deliberately before
  resuming that cell. The runner never cancels unrelated jobs.

Regenerate partial or final tables while/after training:

```bash
python projects/snn-depth-estimation/experiments/b-attention-mechanism-v1/summarize_ablation_npu.py \
  --output-root projects/snn-depth-estimation/experiments/b-attention-mechanism-v1/outputs/hybrid-b-ablation-911-NPU
```

The summarizer checks frame metrics, presentation/flip/LR streams and final
completion. It labels missing cells as partial and never substitutes best-Dev
checkpoints. Exact hardware behavior remains unverified until the server's NPU
checks pass; local CPU tests cannot establish NPU utilization or numerical parity.

## Four-way matrix and device allocation

Letters are reporting labels; use the exact variant strings for code and paths.

| Label | Variant | Forward encoder | Local backward rule | Seed 1234 slot | Seed 2345 slot |
| --- | --- | --- | --- | ---: | ---: |
| C | `linear-identity` | Unbounded linear Q/K/V | Identity | 1 | 5 |
| H | `signed-surrogate` | Signed-T4 Q/K/V | Original surrogate through T4/reset | 2 | 6 |
| G | `linear-surrogate` | Unbounded linear Q/K/V | Signed-T4 encoder's surrogate Jacobian at the current projected input | 3 | 7 |
| S | `signed-identity` | Signed-T4 Q/K/V | Identity / straight-through for the complete encoder | 4 | 8 |

Slots are logical, not assumed physical IDs. Map them to the eight allocated
Ascend devices after inspection. Each process sees one device and world_size=1.
No DDP, HCCL gradient reduction or tensor parallelism. Do not use eight NPUs to
train one model. A device shortage changes concurrency, not batch size or budget.

**Execution choice: eight fresh runs.** This implementation does not reuse the
historical H/1234 endpoint. Re-running it keeps all cells on the same current
source/environment and removes any dependency on old server artifacts. The
runner records this decision in `reuse-decision.json` before production. This
repeat is the seed-1234 matrix cell, not a third independent seed.

## Architecture and encoder semantics

All four models use native Original non-I SpikeStereoNet plus the SAME historical
placement-B `DescriptorAttention` from
[`core/spiking_attention_placement.py`](../../core/spiking_attention_placement.py).

- Native CNN feature and context encoders; native correlation pyramid, multiscale
  ALIF recurrent updater and disparity head. No additional +I or v0 conversion.
- Native arguments: hidden_dims=[128,128,128], context_norm=batch,
  shared_backbone=False, n_snn_layers=3, n_downsample=2, corr_levels=4,
  corr_radius=4, corr_implementation=reg, mixed_precision=False,
  feature_encoder=cnn, context_encoder=cnn, slow_fast_gru=False.
  Set matched_initialization_seed to the run seed.
- B projects descriptors 256→16, applies LayerNorm, then three learned biased
  16→16 Q/K/V projections. One attention head; score scale 1/4 (=1/sqrt(16)).
- Both directions use the original left/right feature pair and full opposite-row
  support, including the native padded descriptors. No new disparity/causal mask.
- Continuous softmax and value mixing; shared biased 16→256 output projection and
  residual addition. Output projection weight/bias start at zero. No extra FFN.
- 9,312 added parameters; native + B total 12,157,558. The four variants have
  identical parameter shapes, keys, initialization and counts within each seed.
- Keep non-reentrant activation checkpointing during training, native freeze_bn(),
  16 training refinements / 32 evaluation refinements and native state resets.

For a scalar projected current I, signed_rate starts v=0, then repeats four times:

```text
u = 0.9 * v + I
s = H(u - 1) - H(-u - 1)       # event in {-1, 0, +1}; H(x)=1 when x>=0
v = u - s                     # differentiable soft reset
code = sum(s over four steps) / 4
```

The same current is used in all four internal steps. Codes have nine possible
levels from -1 to +1 in increments of 0.25. Weights remain FP32. This is not a
pure binary model or four separate sensor observations.

**Use the existing exact controls** in
[`core/attention_encoder_controls.py`](../../core/attention_encoder_controls.py):
`controlled_code`, `signed_value_and_derivative`, and `install`.

- C uses f(I)=I, derivative 1. H uses the original signed_rate and its ordinary
  surrogate autograd path. These diagonal variants must reproduce the original
  conventional/hybrid operators, not approximations.
- G outputs I unchanged. Its backward uses the local derivative of the entire
  four-step signed encoder, including leak, soft reset and averaging, evaluated
  at the same I via a shadow recurrence. It is NOT just one threshold derivative.
- S outputs the exact signed_rate code, but the local encoder derivative is 1.
  It is NOT identity gradients through each internal threshold separately.
- The threshold surrogate is H'(x)=1/(1+abs(x))^2. Preserve the existing analytic
  Jacobian implementation and first-order semantics. Do not replace it with a
  detached spike, tanh derivative, sigmoid or a backend spike primitive.
- Only the local Q/K/V encoder VJP changes. Projections, LayerNorm, attention
  products, softmax, residual and native ALIF updater retain normal autograd.

## Exact shared training configuration

These values come from the completed NPU Hybrid B metadata, not the old 7.5k
four-NPU mechanism screen or current CUDA launch defaults.

| Setting | Required value |
| --- | --- |
| Model/train/order/augmentation seeds | 1234 and 2345, paired across all four variants |
| B branch seed | seed+783: 2017 / 3128 |
| Dev loader generator seed | seed+991: 2225 / 3336 |
| Endpoint | 30,000 optimizer updates; 240,000 presentations per run |
| Device count / world size | 1 / 1 per model |
| Physical batch / accumulation / effective batch | 4 / 2 / 8 |
| Optimizer | AdamW; max_lr=2e-4, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-5, amsgrad=False, foreach=False, fused=False |
| Parameter groups | One group containing all trainable parameters; no separate branch LR or decay exemptions |
| Scheduler | OneCycleLR(total_steps=30000, max_lr=2e-4, pct_start=0.01, anneal_strategy='linear', cycle_momentum=False, div_factor=25, final_div_factor=10000) |
| LR endpoints | Initial 8e-6, peak 2e-4, configured minimum 8e-10; preserve original scheduler step ordering |
| Clipping | Global gradient norm 1.0 after both microbatch backwards |
| Update order | zero_grad(set_to_none=True); accumulate two weighted backwards; clip; optimizer.step(); scheduler.step() |
| Loss | Native boolean-indexed masked sequence L1; sum_i 0.9^(15-i) × mean absolute valid error for 16 predictions |
| Accumulation weighting | Each microbatch loss × its valid count / valid count across all eight samples; no extra division by 2 |
| Reduction | No inter-rank gradient collective; preserve completed worker's finite/valid handling |
| Input | All 50 bins as channels, native 250×400; no crops; uint8 host cache, FP32 device input before native 2*spikes-1 normalization |
| Train / Dev | 960 frames from 120 scenes / 60 frames from 30 scenes |
| Sampling | One seeded CPU randperm(960), cycled; no per-epoch reshuffle |
| Augmentation | Historical deterministic-by-position H/V flip selection; horizontal flip also negates signed target flow |
| Train / Dev iterations | 16 / 32 |
| Dev loader | Batch 2, no augmentation, same 60 frames and native masks |
| Evaluations | 0,250,500,1000,1500,2500,5000,7500,10000,12500,15000,17500,20000,22500,25000,27500,30000 |
| Primary endpoint | Final update 30000, not the best Dev checkpoint |
| Loss implementation name | native_boolean_indexed_masked_mean; no speed-runtime masked-sum replacement |
| Precision | FP32; AMP off; convolution HF32 off; matmul HF32 off; JIT compilation off |
| Execution | strict deterministic algorithms=False; cudnn_deterministic=False; cudnn_benchmark=False; HCCL_DETERMINISTIC=false |
| Input preparation | preload; 12 GiB budget per process (12884901888 bytes); 2 workers; prefetch depth 2 |
| Logging / diagnostic / checkpoint cadence | Status every 25 updates; detailed gradients every 100; recovery checkpoint every 500 and after evaluation |
| Named checkpoints | Preserve 7500 and 30000; optional intermediate snapshots must not change primary selection |
| CPU profiling | Off in production; record ordinary step/input-wait timing |
| Test/confirmation | Do not access |

The source manifest is [`configs/synthetic-split.json`](../../../../configs/synthetic-split.json),
SHA256 `bad4369b6fd8b0cdf871a20b1727c2d9b59733a16bd33f75904227bdfef4510a`.
Use a configured read-only dataset root. Signed disparity target is
`-(289.740625*0.08)/depth`. Preserve finite positive depth, abs(disparity)<512
loader validity and the driver's finite-target/abs(target)<700 loss mask.
Keep native padding to a multiple of 32 and unpadding/output sign.

For zero-based update u and item j=0…7, p=8*u+j and index=order[p % 960].
Flip flags come from SHA256(`recipe-hv-v1:{seed}:{p}`): H=byte0<128,
V=byte1<128. Use the existing seed-aware protocol; remove any remaining hard-coded
1234/2017/2225 routing from the new worker. Fix the sample stream across variants
within each seed, including after resume.

The reference environment was Python 3.11.14, torch 2.9.0+cpu,
torch_npu 2.9.0.post1+gitdc51c2d, Ascend910B2C, CANN 8.5.1. Prefer the same
working environment and record actual driver/container versions. Do not silently
mix NPU types or precision policies within a paired seed. Device arithmetic need
not match CUDA bitwise; this is a within-NPU study.

## Initialization and minimum validation

Generate or load one full CPU native+B initialization per seed and share it across
all four cells. Construct the native model using the run seed, then B using
seed+783 in an isolated CPU RNG context. Save the initial tensors once. Do not
initialize seed 2345 by loading seed 1234 or restore trained reference weights.

Historical seed-1234 full B state hash:
`b37335857fd34f9f5aaa1cf203f906304f7edbd965447ff7dad3953172d47bd4`.
Native-only hash: `8ceac15b09ce188f9ee52ad5a8eca8e68727cbbb2887a012874d7246dc6495ca`.
Use the established sorted-state-name/tensor-byte procedure. Record seed-2345's
own identity; do not require it to equal seed 1234. The launcher records whether
fresh seed-1234 tensors match the historical reference. Different CPU/PyTorch
implementations can initialize different bytes from the same seed; this is
reported, not silently called a historical replication. Because all eight runs
are fresh, they may use the new initialization, but **all four cells within a
seed must share exact tensors**, and each worker must independently reconstruct
that state. No historical endpoint is mixed into the new comparison.

Run existing [`test_controls.py`](test_controls.py) and relevant input/resume tests.
Adapt device tests without weakening their scientific assertions:

1. At identical currents, C/G forward values are identical and H/S forward values
   are identical; H matches original signed_rate exactly. Their VJPs match the
   intended local Jacobians with recorded dtype-appropriate tolerance.
2. Test B with an opened output projection on tiny tensors so Q/K/V gradients
   are exercised. At untouched zero-output initialization, zero Q/K/V parameter
   gradients are expected and are not a broken-control signal.
3. Verify checkpointed/non-checkpointed outputs and gradients, including both
   crossed custom-autograd functions, on NPU FP32. Keep FP64 reference checks on
   CPU if the accelerator does not support FP64. Do not use finite-difference
   gradcheck as the correctness oracle for intentional surrogate derivatives.
4. Perform a disposable native-shape three-update port/resume check for each
   variant, with a pause after update 2. Verify restored state, sample/flip/LR
   continuation and finite loss/gradients. With determinism off, numerical equality
   of resumed versus separate uninterrupted trajectories is tolerance-based;
   serialized state restoration and sample/LR addressing remain exact.

Preflights use separate folders and never seed production optimizer state. Pin
one validated source/configuration for production. A smoke failure is grounds
to fix the implementation before training, not to drop an unfavorable arm.

## Analysis, decisions and scope of claims

Report all four EPE values separately for each seed. Let E_C,E_H,E_G,E_S be their
fixed-30k mean-scene EPEs. Positive contrasts below mean an improvement:

| Contrast | Formula | Meaning |
| --- | --- | --- |
| Hybrid package | E_C−E_H | Hybrid advantage over ordinary attention |
| Forward encoding with identity derivative | E_C−E_S | Signed forward versus linear, fixed identity backward |
| Forward encoding with surrogate derivative | E_G−E_H | Signed forward versus linear, fixed surrogate rule |
| Surrogate with linear forward | E_C−E_G | Derivative effect with ordinary continuous values |
| Surrogate with signed forward | E_S−E_H | Derivative effect with signed values |
| Interaction | E_S+E_G−E_H−E_C | Positive: larger signed-forward benefit under the surrogate than under identity |

These are effects of the specified training interventions, not unique additive
percentages of a neuron's biological mechanisms. The “same surrogate” means the
same derivative function evaluated at each model's current values; trajectories
and weights can diverge after learning begins.

Use 0.05 px as a prospective descriptive practical band. A contrast >=0.05 px in
both seeds is a promising replicated screen; a reversal or one small effect is
inconclusive for a robust advantage. A gap inside ±0.05 is not proof of equivalence.
Report bad3, median paired scene difference, scene wins and concentration of mean
gains alongside EPE. Do not hide bad3 regressions. Two seeds are preliminary, not
a population significance claim; scenes are not extra training seeds.

| Observed pattern | Interpretation / next decision |
| --- | --- |
| H does not beat C consistently | Hybrid superiority has not reproduced here; do not expand a mechanism story assuming it has. |
| G is within the practical band of H and both beat C, while S does not | Surrogate training is a leading sufficient explanation under this test; confirm the relevant contrast before a claim. |
| S approaches H and both beat C, while G does not | Forward encoding is the leading explanation; next test a bounded continuous control. |
| G and S each beat C | Both interventions can help; inspect their interaction and whether H adds anything. |
| Only H beats C materially | The paired forward/backward design matters; neither ingredient alone is sufficient in these controls. |
| G or S beats H | Prefer the observed better combination; revise the method rather than defend H. |
| Seeds disagree | Add the deciding third seed or inspect optimization sensitivity before NVIDIA confirmation. |

Forward-encoding effects still combine bounding, quantization, integration and
reset. Do not call them uniquely caused by discrete spikes. No latency/energy
advantage follows from these accuracy runs. Do not tune temperature or learning
rate mid-study. Report a change in recipe as a separate experiment.

## Outputs for later graphing

Default study output root, relative to this experiment directory:

```text
outputs/hybrid-b-ablation-911-NPU/
  config.json
  matrix.json
  source-registry.json
  reuse-decision.json
  initializations/seed-1234.pt
  initializations/seed-2345.pt
  initializations/metadata.json
  preflight/seed-1234/{variant}/...
  seed-1234/{linear-identity,signed-surrogate,linear-surrogate,signed-identity}/
    metadata.json
    train.jsonl
    evaluations.jsonl
    status.json
    checkpoint.json
    result.json
    recoveries.jsonl             # when resumed
    last.pt
    step-7500.pt
    step-30000.pt
  seed-2345/{same four variants}/...
  summaries/
    endpoints.csv
    curves.csv
    scene-deltas.csv
    factorial-contrasts.csv
    completion-audit.json
    interpretation.md
```

Use configured storage for large checkpoints if necessary, with a manifest of
locations. Retain raw JSONL beside the compact graph-ready tables. Never overwrite
old studies or commit checkpoints/raw logs. This runner creates eight fresh
cells; all endpoint records therefore have reuse=False.

- Metadata: seed, variant, forward/backward rules, source revision and source
  identity, initialization identity, full resolved optimizer/scheduler/loss/input
  configuration, dtype/flags, device mapping and environment, job identity.
- Train: update, samples_seen, ordered indices, flips, loss, pre-clip gradient norm,
  LR used/next, step/input-wait time, device memory, physical batch and accumulation.
- Evaluations: update, all 60 frame identities/valid counts/EPE/bad3, 30 scene means,
  mean-scene EPE and pooled bad3. Average frames within scenes, then scenes equally;
  bad3 is valid pixels with error >3px pooled over Dev, reported as percent.
- Endpoints: study, backend, seed, variant, forward_rule, backward_rule, update,
  mean_scene_epe, bad3_percent, source_identity, reuse flag.
- Curves: study, backend, seed, variant, update, samples_seen, metric, value, unit.
- Scene deltas: seed, update, scene, compared variants, both EPEs and signed gap.
- Factorial contrasts: one row per seed/update/contrast with the above sign
  convention. Two-seed aggregate is separate and retains both individual effects.

At completion verify 30,000 contiguous train rows, the expected sample/LR streams,
17 full-Dev evaluations, fixed endpoint, finite model/optimizer state and intact
resume metadata. Recompute endpoint metrics from frame records. Retain all valid
runs and every scene; no best-checkpoint substitution or backend pooling.

## Copyable prompt for the NPU server's Codex

```text
Run the implemented batch described in:
projects/snn-depth-estimation/experiments/b-attention-mechanism-v1/hybrid-b-ablation-911-NPU.md

Read repository instructions and this protocol. The implementation is already
included: run_ablation_on_npu.py, worker_ablation_npu.py, npu_backend.py,
input_pipeline_npu.py, validate_ablation_npu.py and summarize_ablation_npu.py.
No old Research checkout or source archive is needed. Preserve unrelated changes.
Use the existing working Ascend environment and configured read-only dataset root.

I authorize this four-variant/two-seed batch on my eight assigned NPUs. Inspect
allocated device IDs and host memory, then execute from the repository root:

python projects/snn-depth-estimation/experiments/b-attention-mechanism-v1/run_ablation_on_npu.py --data-root ACTUAL_DATASET_ROOT --devices ACTUAL_ASSIGNED_DEVICE_IDS --execute

Replace the placeholders with verified server values. Use a persistent job/session
according to the server's established practice. The launcher handles initialization,
NPU checks, pause/resume preflights, source recording, eight fresh training runs
and graph-ready tables. All preflights must pass before production. Each cell is
one NPU, physical batch 4, accumulation 2, FP32, native masked loss, 30,000 updates,
strict deterministic execution off. Do not turn on the older speed-loss override,
add a 7.5k gate, reuse old endpoints or change the registered recipe.

If interrupted, inspect the failure/status logs and resume using the same command
plus --resume; do not restart completed cells or mix source/configurations. If a
backend validation fails, report its exact log and diagnose before training; do
not bypass the assertion. Do not commit/push, launch NVIDIA jobs or add experiments.

Report source/configuration identity, seed/variant/device mapping, validation
status, job/session details and output paths. On completion, report all eight
30k results and the per-seed factorial contrasts, including bad3 and scene-level
variation. Distinguish forward encoding, surrogate training and interaction;
do not claim discrete spikes alone explain any forward-encoding advantage.
```
