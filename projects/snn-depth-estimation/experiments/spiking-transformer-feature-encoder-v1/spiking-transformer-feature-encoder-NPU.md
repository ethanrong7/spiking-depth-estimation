# Spiking Transformer feature encoders — eight-NPU experiment

Prepared 10 November 2026. Branch: `hybrid-b-experiments`.

**Status: experiment specification and implementation handoff. The four new encoder adapters and launcher are not implemented by this document. No new training has been launched.** The server-agent prompt at the end requests implementation followed by NPU validation and execution.

## 1. Goal and decision

Determine whether replacing SpikeStereoNet's left/right matching-feature CNN with a spiking Transformer improves stereo disparity estimation, and how three spiking encoder families compare with a conventional Transformer. Use SpikeStereoNet Original non-I as the surrounding stereo system.

Run four architectures at two paired seeds: **eight independent single-NPU jobs**, not one eight-device distributed job. Train each admitted run to **30,000 optimizer updates**. Primary metric: final Dev mean-scene EPE; secondary: pooled bad3, scene-level changes and measured computation cost.

This is an architecture screen. Encoders differ in hierarchy, attention, normalization and neuron dynamics. Beating Swin establishes an encoder-package result, not that binary spikes alone caused it. A promising encoder can later receive a matched non-spiking ablation and transfer to a second stereo system. Do not add those runs to this batch.

The existing Original result is historical context, not an additional production arm: NPU seed 1234 at 30k had EPE **2.3648221647 px**, bad3 **10.6889504800%**. It is not a two-seed paired control for this new batch. The old Hybrid B module is not part of this experiment.

## 2. Experiment matrix and sources

| Variant ID | Encoder configuration | What it tests | Seed 1234 slot | Seed 2345 slot |
| --- | --- | --- | ---: | ---: |
| `spikformer` | Spikformer, 8 blocks, width 384, 8 heads, MLP expansion 4 | Spike Q/K/V self-attention without softmax | 0 | 1 |
| `qkformer` | QKFormer-10-384; widths 96/192/384; depths 1/2/7; 6 heads; MLP expansion 4 | Hierarchical Q-K gating followed by the upstream final-stage spiking self-attention | 2 | 3 |
| `meta-spikeformer` | Meta-SpikeFormer-8-384; widths 96/192/384/480; upstream convolutional stages plus 6+2 Transformer blocks; 8 heads; MLP expansion 4 | Spiking convolution/Transformer backbone with dense-prediction precedent | 4 | 5 |
| `swin` | Swin-T: widths 96/192/384/768; depths 2/2/6/2; heads 3/6/12/24; patch size 4; window size 7; MLP expansion 4 | Conventional learned Q/K/V, scaled softmax attention, LayerNorm and GELU | 6 | 7 |

Slots are logical positions in the eight actually allocated NPU IDs. Do not assume these IDs are available until checking the server allocation.

Author implementations and papers:

- Spikformer, ICLR 2023: <https://arxiv.org/abs/2209.15425>; <https://github.com/ZK-Zhou/spikformer>, `imagenet/model.py`.
- QKFormer, NeurIPS 2024: <https://arxiv.org/abs/2403.16552>; <https://github.com/zhouchenlin2096/QKFormer>, `imagenet/qkformer.py`.
- Meta-SpikeFormer / Spike-driven Transformer V2, ICLR 2024: <https://arxiv.org/abs/2404.03663>; <https://github.com/BICLab/Spike-Driven-Transformer-V2>, `classification/models.py`.
- Swin Transformer, ICCV 2021: <https://arxiv.org/abs/2103.14030>; <https://github.com/microsoft/Swin-Transformer>, `models/swin_transformer.py`.

These are specified stereo adaptations of the full backbone block sequences, not reproductions of their ImageNet experiments. Remove classification heads and global spatial pooling. Preserve defining attention equations, residual placement and upstream neuron settings. Do not approximate every architecture with the same home-made attention block. Retain each upstream license and record the exact imported commit in `UPSTREAM.md` before training; source URLs alone are not version pins.

Widths put these in a broadly comparable compact-backbone regime, **not exact parameter or FLOP matching**. Measure encoder, adapter and total trainable parameters separately. Do not silently resize models after seeing Dev rankings. No pretrained weights, classification distillation, auxiliary reconstruction loss, stochastic depth or dropout in this first common-recipe batch; set all dropout/drop-path probabilities to zero in all four adapters. Keep Swin relative-position bias and normal attention scaling, and each SNN's upstream attention scaling.

## 3. Stereo integration and temporal input

### Scope

Build native Original with `shared_backbone=False`, then replace only `model.fnet`. Keep the context CNN, correlation pyramid, multiscale ALIF updater, context projections and disparity head unchanged in architecture. Train the whole model end to end; do not freeze the downstream solver. Do not install placement A/B, +I, normalized-v0 or guided refinement.

The adapter must support native `fnet([left, right])` and return two dense continuous maps with **256 channels on the native quarter-resolution grid**. The same encoder weights process both views; no encoder cross-view attention is added.

### Explicit input choice: T5, ten sensor bins per step

This study uses all 50 chronological sensor bins, grouped into **five consecutive chunks of ten bins**. For a raw binary tensor `[B,50,H,W]`, reshape without summing or reordering into `[T=5,B,C=10,H,W]`.

- This is five neuron steps over different sensor chunks, not five repeated copies of a static 50-channel tensor.
- Within each chunk, bin identity remains encoded by channel position. It is not a 50-neuron-step experiment; make that limitation explicit in reports.
- SpikeStereoNet currently maps raw inputs to `2*x-1` before calling `fnet`. Recover raw binary inputs inside the new adapter with `(x+1)/2`; do not normalize twice. Keep the context CNN's existing signed input unchanged.
- SNN states persist across the five chunks, with full temporal gradients. Reset all encoder states before/after independent forwards, including evaluation and different microbatches. Left and right states must not leak into each other; concatenating views on the batch axis is allowed.
- Swin processes the same five chunks with shared weights, treating time as a batch dimension for its spatial backbone. Ordered temporal fusion below gives it access to all bins. It has no hidden-state recurrence; this batch does not compare against a temporally self-attending ANN. Do not describe it as such.
- This T5 grouping is a new experiment-specific adapter configuration. It does not modify existing T4 static-input or T50 encoder studies.

### Common dense readout

Use the same readout design for all four backbones; input channel counts differ by backbone:

1. Collect a quarter-resolution feature, an eighth-resolution feature and the final deep feature for each of the five chunks.
2. For Spikformer, use the stem outputs after its second and third pooling stages, plus the final Transformer output. For QKFormer use stage 1, stage 2 and stage 3 outputs. For Meta-SpikeFormer use the completed quarter/eighth convolutional stages and final Transformer stage. For Swin use stage 1, stage 2 and stage 4 outputs.
3. Project each tap to 64 channels using a learned 1x1 convolution. Bilinearly resize the latter two to the quarter grid (`align_corners=False`), then concatenate: 192 channels per chunk. Do not add separate nonlinearities or normalization at these lateral projections.
4. Concatenate the five chunk features in chronological order: 960 channels. Apply `Conv1x1(960,256) -> GroupNorm(32,256) -> GELU -> Conv3x3(256,256,padding=1)`. The last layer is linear, with no final threshold, softmax or descriptor L2 normalization. Use this same continuous readout in every arm.
5. Preserve native solver padding/unpadding and metric masks. Internally pad encoder inputs on the bottom/right to a multiple of 32 if necessary; pad raw spikes with zero. Crop the final quarter-grid maps to the native solver's exact expected grid. Do not stretch the input image to fit a classifier or resize ground truth. Handle Swin window padding/masks at each stage.

Map shapes must be tested against the existing native encoder on the actual padded input shape. Preserve spatial tap coordinates. A continuous readout and the existing CNN context/RSNN solver mean this is not an entirely spike-only network.

### Neurons and normalization

Use each pinned author's neuron configuration, including threshold, leak/tau, charging rule, reset rule, detach-reset choice and surrogate derivative. Export every effective setting in model metadata. Do not substitute the old signed-T4 Hybrid B encoder. Defaults visible in the candidate author implementations include LIF tau 2 and detached reset, with some attention thresholds differing from ordinary neuron thresholds; preserve those distinctions.

**BatchNorm must train inside the new feature encoder.** The native `freeze_bn()` visits every BatchNorm2d module, including any replacement encoder. In this experiment, preserve native normalization behavior outside `fnet`, while restoring the new encoder's normalization layers to training mode during optimization. All modules enter evaluation mode for Dev. Do not leave new BN running statistics at initialization. Keep upstream BN epsilon/momentum and record them; common-readout GroupNorm has epsilon 1e-5.

Use the same microbatch across all eight arms because it affects training BN. For checkpointed segments, recomputation must neither advance persistent neuron state nor update BN statistics twice. Prefer ordinary forward/backward for the initial implementation; introduce activation checkpointing only if required for memory and after checking these semantics.

## 4. Shared training hyperparameters

This reuses the completed NPU data/optimizer/evaluation recipe where applicable. Encoder architecture, input adapter and new-encoder BN behavior are intentional changes; microbatch is reduced to accommodate temporal backbones.

| Parameter | Value |
| --- | --- |
| Seeds | 1234 and 2345 |
| Endpoint | 30,000 optimizer updates; 240,000 stereo-pair presentations per run |
| Dataset manifest | `configs/synthetic-split.json` in this repository |
| Train / Dev | 960 training frames from 120 scenes; 60 Dev frames from 30 scenes |
| Input | All 50 bins; native 250x400; no image crops |
| Physical batch / gradient accumulation | **2 / 4**, effective stereo-pair batch 8 |
| Memory fallback | If any arm cannot fit batch 2, set **1 / 8 for all eight arms before production** and record it. No architecture-specific microbatch mixture. |
| Optimizer | AdamW, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-5, amsgrad=False, foreach=False, fused=False |
| Parameter groups | One group; all trainable parameters, no encoder-specific LR or weight-decay exemptions |
| Schedule | OneCycleLR: total_steps=30000, max_lr=2e-4, pct_start=0.01, anneal_strategy=linear, cycle_momentum=False, div_factor=25, final_div_factor=10000 |
| LR | Initial 8e-6; peak 2e-4; configured minimum 8e-10; use existing scheduler step ordering |
| Gradient clipping | Global norm 1.0 after accumulation |
| Train / evaluation refinements | 16 / 32 |
| Loss | Native masked sequence L1, sum over 16 predictions with weights 0.9^(15-i) |
| Accumulation | Weight each microbatch loss by its valid-pixel count divided by total valid pixels in the effective batch; no additional division by accumulation count |
| Sampling | Existing protocol: one seeded randperm(960), cycled; do not add epoch reshuffling |
| Augmentation | Existing paired H/V flips and signed-target correction; identical presentation/augmentation stream across arms at each seed |
| Native model | hidden_dims=[128,128,128], n_snn_layers=3, n_downsample=2, corr_levels=4, corr_radius=4, corr_implementation=reg, context_norm=batch, slow_fast_gru=False |
| Precision | FP32; AMP off; convolution/matmul HF32 off; same across all arms |
| Device determinism | Strict deterministic algorithms off; seeded RNGs remain enabled; preserve current NPU backend settings |
| Distribution | world_size=1; one independent job per NPU; no HCCL gradient reduction |
| Evaluation | Updates 0, 1000, 2500, then every 2500 through 30000; full Dev, native masks, no augmentation; Dev batch 2 |
| Checkpoints | Recovery every 1000 updates and after evaluation; retain 10000, 20000 and 30000 |
| Runtime logging | Train summary every 25 updates; no per-step host copies of attention matrices or firing-rate tensors |
| Data pipeline | Existing uint8 RAM preload, 2 preparation workers, prefetch depth 2; budget up to 12 GiB/process after checking total host RAM |
| Test split | Do not use for architecture selection or this initial screen |

Use the existing target conversion, finite/positive-depth handling, disparity validity and native loss masks. These must be imported from the checked-in dataset/training code rather than re-created from memory.

### Seed and initialization rules

For each seed, construct native Original once and save its initial state. Reuse exactly the same initial tensors outside `fnet` across all four architectures. Initialize each replacement encoder and readout freshly under `encoder_seed=seed+783` (2017 or 3128); use no trained checkpoint. Different tensor shapes mean identical seeds do not imply identical encoder weights.

Use the seed for Python, NumPy, torch, NPU RNG, training order and augmentation, with separate generators where supported. Dev loader generator seed is `seed+991` (2225 or 3336). Avoid consuming data-order RNG from model construction. Pairing controls data exposure and shared solver initialization; strict determinism remains off, so bitwise rerun equivalence is not promised.

## 5. NPU implementation and admission

Existing local references, all present in a fresh checkout:

- Native model: `projects/snn-depth-estimation/core/spikestereonet.py`.
- Builder/dataset/evaluation: `projects/snn-depth-estimation/experiments/spiking-transformer-stereo/scripts/run_experiment.py`.
- Current NPU loop/backend: `projects/snn-depth-estimation/experiments/b-attention-mechanism-v1/worker_ablation_npu.py` and `npu_backend.py`.
- Current preload: `input_pipeline_npu.py` in that experiment directory; shared input helpers under `experiments/shared-gpu-30k-v1/`.
- Presentation stream: `projects/snn-depth-estimation/experiments/spiking-attention-placement-phase2/protocol.py`.

The B worker is a reference, **not this batch's executable**: it hard-codes B installation, eight B initialization layouts, a particular physical batch and run checks. Create a new encoder-specific launcher/worker rather than passing new variant strings to the old command. Preserve old experiments and their defaults.

Ordinary convolution, linear layers, normalization, pooling, tensor reshaping, softmax and matrix products should use PyTorch operations through the installed `torch_npu` backend. Their on-device support and performance must be checked. Author CuPy/CUDA multi-step neuron implementations need a torch/NPU equivalent that preserves forward dynamics and surrogate gradients. Disable optional Swin fused CUDA window kernels and use its normal tensor path. Do not globally monkey-patch CUDA to mean NPU or silently fall back to CPU operations.

Before production, perform one focused validation pass per architecture:

- Check native-size forward/backward, output grid, reset isolation and nonzero finite encoder/solver gradients.
- Compare the neuron port and representative attention block outputs/gradients against the pinned torch reference on small tensors. Include the intended reset derivative.
- Run a short learning/throughput preflight, nominally 200 optimizer updates, and a checkpoint-resume check. Record loss, firing activity, peak NPU memory, input-wait and steady-state step time. Use separate preflight output directories; production starts fresh at update zero.
- Check all four architectures' normalization modes and ensure model.eval() actually freezes new BN during Dev. If checkpointing is used, include its forward/gradient/state/statistics check here.

This preflight is for correctness and feasibility, not Dev-based ranking or a hyperparameter search. A broken port must be fixed before launching it; do not replace it with a simplified model bearing the same name. No extra pilot-seed sweep is included. If a configuration needs a substantive numerical or optimization change, record the revised protocol before fresh production runs, rather than changing it midway.

Check actual NPU allocation, installed PyTorch/torch_npu/CANN versions and host RAM. Reuse the working vendor environment; do not install a generic CUDA PyTorch wheel over it. With eight preload processes, verify the combined memory budget; a shared read-only cache or smaller preload budgets may be used without changing sample order or numerical input. Record the choice. Performance is not established until measured on this server.

## 6. Outputs and interpretation

Default output location, relative to repository root:

```text
projects/snn-depth-estimation/experiments/spiking-transformer-feature-encoder-v1/outputs/
  encoder-screen-npu-v1/
    config.json
    source-registry.json
    environment.json
    preflight/<variant>/
    seed-1234/<variant>/
    seed-2345/<variant>/
    summaries/
    figures/
```

Each production cell writes `metadata.json`, `status.json`, `train.jsonl`, `eval.jsonl`, `stdout.log`, `stderr.log`, a final `result.json`, and a `checkpoints/` directory. Keep raw outputs and weights ignored by Git. Allow an explicit `--output-root` override; no personal paths or dependency on the Research monorepo, Downloads or historical cluster storage.

Record source commit plus effective configuration, imported upstream commits/licenses, device identity, seed details, actual batch/accumulation, precision, neuron parameters, normalization policy, parameter counts and preprocessing. Record any source patch used for execution. No preflight result is a final endpoint.

Export graph-ready CSVs:

- `curves.csv`: study, backend, seed, variant, update, samples_seen, metric, value, unit. Include mean_scene_epe, pooled_bad3_percent, train_loss, learning_rate, step_seconds, input_wait_seconds and peak_memory_bytes at their measured cadence.
- `endpoints.csv`: seed, variant, update, mean_scene_epe, pooled_bad3_percent, encoder_params, readout_params, total_params, train_seconds, inference_ms_per_pair, peak_memory_bytes.
- `scenes.csv`: seed, variant, update, scene_id, frame_count, mean_scene_epe, valid_pixels, bad3_pixels.
- `paired-deltas.csv`: seed, variant, reference=swin, update, scene_id, epe_delta, bad3_percentage_point_delta.

Use the native metric aggregation: mean-scene EPE, pooled valid-pixel bad3. Report scene wins and seed-specific results; do not hide reversals in a two-seed mean. Use device synchronization around measured inference timing, equal native resolution, batch one, 32 refinements and identical warmup/measurement counts. Report measured latency, not sparsity-based estimates of NPU energy savings. If not measured, leave latency blank rather than inventing it.

| Outcome at the fixed 30k endpoint | Next decision |
| --- | --- |
| An SNN encoder beats Swin in both seeds, with useful EPE magnitude and reasonable bad3/cost | Candidate for an additional paired seed and transfer; then a matched non-spiking control |
| SNN and Swin both look better than historical Original, but SNN does not beat Swin | Motivation for Transformer features, without established spiking-specific superiority; historical baseline needs a current paired check before a paper claim |
| Rankings reverse between seeds | Report instability; prioritize replication over detailed mechanism attribution |
| All replacement encoders are poor | Examine training/Dev curves, firing activity, readout resolution and data limits before more full runs |

Use an average relative EPE improvement of approximately 3% or more over Swin, positive in both seeds, as a pragmatic follow-up signal, not a significance test. Also inspect scene concentration and bad3. Every valid run continues to 30k regardless of interim rank. No automatic 50k/80k extension or best-checkpoint replacement of the primary endpoint.

## 7. Copy-paste prompt for the NPU server agent

```text
Work in the standalone spiking-depth-estimation checkout on branch
hybrid-b-experiments. Read AGENTS.md and:
projects/snn-depth-estimation/experiments/spiking-transformer-feature-encoder-v1/spiking-transformer-feature-encoder-NPU.md

Implement and run that eight-NPU experiment: Spikformer, QKFormer,
Meta-SpikeFormer and conventional Swin, each with seeds 1234 and 2345,
using SpikeStereoNet Original non-I as the surrounding stereo architecture.
This document is a specification; the new encoder adapters and launcher still
need to be created. The old Hybrid B launcher does not execute this experiment.

Preserve unrelated work. Inspect the current checkout and fetch/fast-forward
the requested branch only if safe; never discard local changes. Import author
implementations with pinned commits and licenses. Implement the specified full
block sequences, T5 x 10-bin input adapter, common dense readout, fresh per-seed
initialization and normalization/state rules. Replace only fnet. No Hybrid B,
+I, v0 or cost-filter branch. All four systems train end to end from scratch.

Use the checked-in native model, dataset, loss, evaluation, NPU backend and
preloading helpers listed in the document. Write new encoder-specific model
adapters, worker, launcher and CSV summarizer. Do not alter old run defaults.
The launcher must support preparation, preflight-only, execute and explicit
resume modes, accept data-root/output-root/devices arguments, and reject
accidental reuse of another run's output directory.

Keep all 50 sensor bins. Use the documented five chronological 10-bin chunks,
not repeated static inputs. Swin gets the same chunks and ordered fusion.
Preserve the native quarter-resolution, 256-channel descriptor interface.
Preserve author neuron equations/surrogate derivatives when replacing CuPy
with NPU-compatible torch operations. Keep native BN behavior outside fnet,
but let the new encoder BN learn. Ensure reset and checkpoint recomputation
cannot leak state or update BN statistics twice.

Find the configured dataset root and eight allocated NPU IDs on this machine;
ask only if required resource details cannot be found. Do not guess paths or
interrupt unrelated jobs. Use the existing working torch_npu/CANN environment.
Validate the implementation and run the bounded preflights described in the
specification. Profile memory and input waiting. Use physical batch 2 and
accumulation 4, or batch 1 and accumulation 8 for ALL arms if memory requires it.
Maintain effective batch 8, FP32 and the fixed 30k schedule. Do not choose a
weaker encoder to make a preflight pass.

Once code and preflights pass, record an exact experiment-code Git commit and
resolved configurations before production. Commit only this experiment's
implementation/configuration/tests/docs, leaving unrelated changes untouched.
Run eight fresh jobs concurrently, one per allocated NPU, to 30,000 updates.
The source running each job must remain fixed. Resume failures from their own
checkpoints and preserve failure logs; do not silently restart or change the
architecture/hyperparameters. Report a substantive blocker rather than
launching an invalid arm. No extra architectures, tuning sweeps or longer runs.

Write graph-ready logs, checkpoints, per-scene evaluations and summaries under
the document's outputs/encoder-screen-npu-v1 location (or a recorded explicit
override). Preserve fixed 30k endpoints and all learning curves, separating
seeds. Report EPE, pooled bad3, scene wins against Swin, parameters, memory and
measured timing. Compare historical Original only as labelled context. Do not
claim an encoder win proves that spikes alone are responsible. Report the
implementation commit, effective settings, preflight results, launch commands,
output paths and final completion status.
```
