# Original SpikeStereoNet on NPU: matched 7.5k baseline screen

Created 8 November 2026. Status: **planned; no baseline result yet**.
This document specifies one new run and the decisions it should inform. It does
not submit training. Work from `main`, independently of `nmrf-experiments`.

## 1. Question and scope

Does adding pre-correlation feature exchange improve Original SpikeStereoNet
under the NPU protocol, regardless of whether that exchange uses signed spikes
or conventional continuous Q/K/V?

Run **Original non-I SpikeStereoNet**, seed **1234**, to **7,500 optimizer updates**.
Keep the **30,000-update OneCycle horizon**. Evaluate at the same checkpoints as
the completed B references. This costs one 7.5k training run, not another pair of
30k runs. Do not start the crossed-gradient controls, other seeds or longer
training automatically.

This is an exploratory screen against existing references. Those references
changed input preparation, determinism and loss implementation during training.
The new Original run uses the final execution policy from initialization. Thus,
a difference is evidence worth following up, but is not a perfectly isolated
architecture comparison. Confirmation must train the selected models with one
unchanged execution policy from the start.

## 2. Existing evidence, included for a self-contained handoff

The supplied NPU records contain two substantive runs, both seed 1234, evaluated
on 60 Dev frames from 30 scenes. The other two factorial variants only completed
three-update preflights; they have no scientific training endpoint.

| Model | Update | Mean-scene EPE, px ↓ | Pooled bad3, % ↓ |
| --- | ---: | ---: | ---: |
| B conventional: linear-identity | 5,000 | 3.1642131477594377 | 12.88370049893856 |
| B hybrid: signed-surrogate | 5,000 | 3.1039837976296742 | 13.334167237083117 |
| B conventional: linear-identity | 7,500 | 2.438361665606499 | 12.519350358843804 |
| B hybrid: signed-surrogate | 7,500 | 2.4370980968077975 | 12.738550477723281 |
| Original non-I, new run | 5,000 | TBD | TBD |
| Original non-I, new run | 7,500 | TBD | TBD |

At 7.5k the hybrid advantage over conventional B is only 0.00126357 px (0.0518%);
hybrid wins 16/30 scenes and has slightly worse bad3. The staged investigation
therefore stopped before crossed-control training. It did **not** establish
whether either B model beats Original on NPU.

Historical NVIDIA context, not substitutes for NPU baselines: Original at 30k
was 2.691127 EPE; conventional B was 2.768814; hybrid B was 2.034117. At 7.5k,
conventional B was 2.921007 and hybrid B 2.346435. Compare identical update
counts; never compare the new 7.5k Original with historical 30k B as an ablation.

Provenance of the NPU table: exported study `speed-a7`, endpoint CSV plus raw
`seed-1234/{linear-identity,signed-surrogate}/train/B/evaluations.jsonl`.
Frozen NPU source identity:
`d67b3721cacd4287c92d293053c3683566b5d88754b28df639de72c293434ad6`.
The export also contains `source-speed-a5/source.tar.gz`, its registry, endpoint
and restoration audits, and stage decisions. These are external run artifacts,
not files assumed to exist in a fresh clone. The aggregate comparison above can
be read without them; paired scene analysis requires importing their raw logs.

## 3. Model definition: remove B and train Original from initialization

Use the native model construction in
[`run_experiment.py`](../spiking-transformer-stereo/scripts/run_experiment.py),
`build_model('original', 1234)`, with its normal forward/reset path. Preserve the
CNN feature/context encoders, native correlation pyramid, multiscale ALIF
recurrent updater and disparity head.

- No B feature-exchange module and no hook on `fnet`.
- No A cost filter, local-attention replacement, v0 updater, additional `+I`
  pathway, guided correction, encoder conversion or NMRF component.
- Native arguments: `hidden_dims=[128,128,128]`, `context_norm='batch'`,
  `shared_backbone=False`, `n_snn_layers=3`, `n_downsample=2`, `corr_levels=4`,
  `corr_radius=4`, `corr_implementation='reg'`, `mixed_precision=False`,
  `feature_encoder='cnn'`, `context_encoder='cnn'`,
  `matched_initialization_seed=1234`, `slow_fast_gru=False`.
- Retain native neuron dynamics/thresholds and reset behavior from the pinned
  source. Do not substitute a different neuron or detach its reset gradient.
- Call the same `freeze_bn()` during training. Reset neuron state between
  microbatches and evaluation batches, as the reference worker does.

This must be independently trained Original, not a trained B checkpoint with
its branch disabled. Use the same **initial backbone weights** as the B runs.
Prefer the NPU study's saved seed-1234 initialization and load the native keys;
exclude only `placement.*`. Require all native keys/shapes to match and reject
other missing/unexpected keys. Do not import trained optimizer state or weights.
If the saved initialization is unavailable, construct it using the pinned
seeded builder and verify it before training; a matching seed alone is not proof
of matching weights across framework versions.

The recorded complete B initialization hash is
`b37335857fd34f9f5aaa1cf203f906304f7edbd965447ff7dad3953172d47bd4`.
Original has no placement parameters, so its complete hash must differ. The
historical shared fnet/cnet/context_zqr_convs hash is
`2085df6ea20073c3a9804d79f6baee94e8e9bfe41facda47e3f292fd3cc9394d`.
Use the reference worker's hashing procedure and compare all remaining native
parameters against the saved initialization when available.

## 4. Hyperparameters and data contract

| Setting | Required value |
| --- | --- |
| Run identity | `original-non-i`, seed 1234 |
| Training endpoint | 7,500 updates; 60,000 presentations |
| Scheduler horizon | 30,000, unchanged when pausing/resuming |
| Optimizer | AdamW; betas (0.9, 0.999); epsilon 1e−8; weight decay 1e−5 |
| AdamW implementation | `foreach=False`, `fused=False`; no LR scaling by worker count |
| Peak LR | 2e−4 |
| Schedule | OneCycleLR; `pct_start=0.01`, `anneal_strategy='linear'`, `cycle_momentum=False`, `div_factor=25`, `final_div_factor=10000` |
| Gradient clipping | Global norm 1.0, after summing all rank gradients |
| Effective batch | 8 |
| Distributed execution | 4 NPUs, physical batch 2 per rank, accumulation 1 |
| Loss reduction | Each rank contributes its valid-count-weighted share of the global batch; SUM rank gradients, not an additional average |
| Input | All 50 spike bins as channels; native 250×400 resolution; no crops |
| Input conversion | Host uint8; cast to FP32 on device before native `2*spikes-1` normalization |
| Train / Dev | 960 frames from 120 scenes / 60 frames from 30 scenes |
| Sampling | Seeded single permutation of the 960 training entries, cycled |
| Augmentation | Existing deterministic H/V flips by seed and absolute presentation index; preserve flow sign handling |
| Seeds | Model/sampling/augmentation 1234; evaluation-loader generator 2225 (=1234+991) |
| Refinements | 16 training; 32 evaluation |
| Loss | Native sequence disparity L1: sum of `0.9**(15-i)` times masked mean absolute error for prediction i=0…15; same valid-count rank weighting |
| Validity | Native depth decoder and dataset mask: finite positive depth, absolute converted disparity <512; retain the driver's finite-target and <700 loss mask |
| Disparity conversion | Native signed target `-(289.740625 * 0.08) / depth`; retain padding/unpadding and output sign |
| Dev checkpoints | 0, 250, 500, 1,000, 1,500, 2,500, 5,000, 7,500 |
| Primary endpoint | Fixed 7,500-update mean-scene EPE; retain 5k as an intermediate result |
| Evaluation input | Native unaugmented Dev images, same 60 frame identities and valid masks; batch 2 as in the reference evaluator |
| Test/confirmation split | Do not access for this screen |

Use the included [`configs/synthetic-split.json`](../../../../configs/synthetic-split.json).
Expected manifest SHA256:
`bad4369b6fd8b0cdf871a20b1727c2d9b59733a16bd33f75904227bdfef4510a`.
Supply the actual dataset root via configuration. Dataset files are external and
must remain read-only. Reuse the NPU reference's rank partitioning of the global
sample/flip stream; shuffling independently per rank changes the experiment.

## 5. NPU execution policy: fixed from the first update

Use the existing working Ascend port, on four available allocated Ascend910B2C
NPUs. The reference environment recorded Python 3.11.14, PyTorch 2.9.0+cpu and
`torch_npu=2.9.0.post1+gitdc51c2d`; the `+cpu` suffix does not imply CPU execution
when torch_npu is driving the model. Record the actual CANN/driver/container
versions; they are not specified in the supplied compact configuration.

Keep FP32 throughout, convolution/matmul HF32 disabled, mixed precision disabled,
JIT compilation disabled, strict deterministic algorithms disabled and
`HCCL_DETERMINISTIC=false`. Preserve the reference's backend flag setters and log
the resolved values. Do not turn on AMP, graph compilation or larger batches to
raise utilization during this accuracy screen.

Use `preload`, 12 GiB cache budget per rank, two preparation workers and prefetch
depth two, matching the final NPU reference configuration. Retain the validated
speed-runtime masked sum/count implementation and batched finite checks.
Checkpoint every 500 updates and after validation; save named 5k and 7.5k
checkpoints. Status cadence 25; detailed gradient diagnostics every 100 updates.
There is no placement module in Original: omit placement-gradient diagnostics
instead of failing on absent Q/K/V. Preserve global clipping and finite checks.
Default CPU profiling is off. Store per-update timings and accelerator telemetry;
utilization is diagnostic, not a model-selection criterion.

The old references changed input loading, then determinism at update 1500, then
speed runtime at 2723 (conventional) / 2712 (hybrid). Do not recreate these
mid-run migrations in the new baseline. Keep their history visible when
interpreting the exploratory comparison.

## 6. Execution handoff and minimum checks

The checked-in `worker.py`/`run_on_gpu.py` are B/CUDA-oriented. Do not launch them
unchanged for Original or assume `--variant original-non-i` already exists.
The supplied NPU source archive contains `worker_npu.py`; it is not assumed to
be present on `main`. Use that validated port as the implementation reference,
record its provenance, and make the smallest Original-mode adaptation:

1. Construct the native model without calling the B installer; use a separate
   run identity/output directory, not one of the four B encoder IDs.
2. Remove only B-specific initialization/full-model-hash and Q/K/V diagnostic
   assumptions. Verify the native starting weights as described above.
3. Preserve rank loss weighting, gradient SUM before clipping, data addressing,
   metric definitions, optimizer and scheduler; add a real 7.5k stop budget to
   every launcher layer. Do not use a wrapper that forces 30k completion.
4. Freeze the resulting source/configuration once and retain its revision. Do
   not silently use another branch's worker or change policies while training.

Before the 7.5k run, perform one disposable native-input forward/backward and
recovery check for the new Original mode. Verify the initial Original forward
matches B at initialization when B's residual output projection is zero, within
recorded numerical tolerance. Reuse the completed port tests for unchanged paths.

Separately, the historical CUDA/NPU discrepancy merits a small same-weight,
same-batch comparison of predictions, loss, summed gradients and one optimizer
update, with convolution precision settings explicitly matched. Reuse existing
checks where they cover this. A material unexplained discrepancy should be
investigated before spending on the full screen. This is a fidelity check, not
an EPE result or a reason to require another 30k CUDA run.

At 7.5k evaluate all Dev frames, save model/optimizer/scheduler/RNG/order, and mark
`SCREEN_COMPLETED` with actual update 7500 and schedule horizon 30000. Keep the
checkpoint resumable, but do not continue automatically.

## 7. Output contract and analysis

Write under this experiment's ignored folder, for example:

```text
outputs/original-npu-baseline-7500-v1/
  source-registry.json
  config.json
  seed-1234/original-non-i/
    metadata.json
    train.jsonl
    evaluations.jsonl
    status.json
    result.json
    last.pt
    step-5000.pt
    step-7500.pt
  references/                  # imported B evaluation logs + provenance, if available
  summaries/
    endpoints.csv
    scene-deltas.csv
    interpretation.md
```

Use environment/configured storage if the site needs checkpoints elsewhere;
keep compact summaries and links in this folder. Do not commit raw logs or
checkpoints. Record source commit, local changes, resolved configuration,
initial-weight identity, device mapping, framework versions and execution policy.

- `train.jsonl`: update, samples_seen, global sample IDs/flips, loss, gradient
  norm, LR used/next, completed-step time, input wait, memory and rank count.
- `evaluations.jsonl`: update, per-frame scene/frame/valid count/EPE/bad3,
  per-scene EPE, mean-scene EPE and pooled bad3. Average frames within scenes,
  then scenes equally; bad3 is valid pixels with absolute error >3 px pooled
  across Dev. Retain every scene, including large failures.
- `endpoints.csv`: model, seed, backend, update, schedule_total,
  effective_batch, mean_scene_epe, bad3, source_identity, execution_policy.
- `scene-deltas.csv`: update, scene, Original EPE, conventional-B EPE,
  hybrid-B EPE and Original-minus-B differences (positive favors B).
- `interpretation.md`: fixed-endpoint table, absolute/relative changes, scene
  wins and median paired differences, bad3 agreement/disagreement, and the
  mixed-history comparison limitation. Plot curves later from these records.

The embedded B table is sufficient for mean comparisons. If raw B frame/scene
records are unavailable, mark scene comparisons unavailable; do not reconstruct
them from aggregate metrics. Imported B records must retain their old identities,
not be relabeled as new runs under this baseline's execution policy.

## 8. Decision after the screen

Use 0.05 px as a **descriptive practical band**, not a significance test. Consider
bad3, scene wins and the full curve as well as the fixed 7.5k mean. One seed does
not establish equivalence or robust superiority.

| Observation at matched 7.5k | Interpretation and proposed next step |
| --- | --- |
| Both B references clearly improve Original | Feature exchange has a useful lead; replicate Original and the informative B models with another seed and one fixed policy. A spiking-specific claim still needs conventional B. |
| Both are close to Original | No useful incremental benefit established at this checkpoint; investigate historical reproducibility before extending training. |
| Both are worse than Original | Added feature exchange hurts in this screen; do not fund spike/gradient attribution as though a gain had reproduced. |
| Small, metric-dependent or scene-concentrated differences | Treat as inconclusive; specify the next resolving comparison before spending more. |

A later 30k confirmation is conditional and must extend the compared models to
the same endpoint. Do not add the no-attention v0/current-I package to this
baseline: that would answer a different question. No new Original EPE, spiking
mechanism attribution or publication claim is asserted by this plan.
