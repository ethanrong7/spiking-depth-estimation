# Spiking Transformer stereo: working evidence

Updated **5 November 2026**. Latest read-only cluster check: **13:04 Shanghai / 05:04 UTC**.


| Paper          | Year | Input   | Camera | Architecture                                 |
| -------------- | ---- | ------- | ------ | -------------------------------------------- |
| SpikeStereoNet | 2026 | Spiking | Stereo | Feature map + Standard cost volume + RSNN    |
| RAFT-Stereo    | 2021 | RGB     | Stereo | Feature map + Standard cost volume + ConvGRU |
| STTR           | 2021 | RGB     | Stereo | Feature map + Transformer + Optimisations    |
| SSDEFormer     | 2022 | Spiking | Stereo | Feature map + Transformer                    |
|                |      |         |        |                                              |


## 1. North star

**Develop a useful, distinctive method for stereo depth from spike-camera input, and identify what actually produces its gains.** Separate learned content routing, the signed forward code, its surrogate training gradient, and the current pathway into the recurrent decoder. Improving an existing architecture can be a worthwhile contribution; a better explanation alone does not establish novelty or publication readiness.

**Current evidence:** A's spiking configuration has lower mean EPE than its corresponding tanh control under both routing designs in **two completed seeds**. Content-dependent routing is not a consistent winner: spiking positional wins seed 2345. The strongest current-reference model remains retrained **no-attention v0, 1.851657 px**, measured at seed 1234. Guided v0 beats Original but loses to its own no-attention parent. B remains the strongest A/B/C placement at seed 1234, **2.034117 px**; these placement rankings have not been replicated with the new A-control seed.

**Working mechanism, not an established cause:** cost preparation and normalized current may help the recurrent decoder use evidence without improving the strongest raw match. The CPU interventions below identify strong dependence on cost offsets in trained B. They do not show that offsets explain its gain after independent retraining. The forward-code versus surrogate-gradient question remains open.

**Common 30k protocol:** native Synthetic, all 50 input bins, 250 × 400; 960 Train frames from 120 scenes, 60 Dev frames from 30 scenes; effective batch 8 (4 × accumulation 2), HV flips, AdamW peak LR 2e−4, fixed 30k OneCycle, 16 training/32 evaluation refinements. Each endpoint uses 240,000 presentations. EPE is mean-scene disparity error in native pixels; bad3 is percent over 3 px; lower is better. These are Dev results, not untouched-test results.

**Naming:** guided v0 was previously called “C”; it is distinct from recurrent placement C. Older feedforward B and pyramid-free A are separate experiments. “Continuous” in the current A factorial means **tanh Q/K/V**, not every possible conventional Transformer.

## 2. Completed experiments and what they tell us

### Architecture comparison · seed 1234

All primary rows are seed 1234, fixed 30k. Controls share endpoints, rather than constituting additional independent replications.

| Model                      | Intervention / matched parent                                                            | EPE ↓ / bad3 ↓         | Interpretation                                                                                                                               |
| -------------------------- | ---------------------------------------------------------------------------------------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Original non-I             | Native CNN, correlation pyramid, ALIF updater                                            | **2.691127 / 10.712%** | Shared reference.                                                                                                                            |
| Full normalized v0         | Normalized multiscale LIF attention/current/FFN updater                                  | **1.990950 / 9.610%**  | Package gain; does not isolate attention.                                                                                                    |
| No-attention v0, retrained | Remove/freeze v0 attention; retain normalized current, FFN and recurrence                | **1.851657 / 9.393%**  | Best reference; most package benefit survives without attention.                                                                             |
| Guided v0                  | Spiking Q/K over 36 signed candidates, gated correction; parent=no-attention v0          | **1.92164 / 9.277%**   | −0.76948 versus Original, **+0.06999 versus parent**. Wins 17/30 parent scenes; scene 0123 dominates loss. Positional follow-up gate failed. |
| Placement A / SC           | Five-candidate disparity-axis cost filter before native pyramid                          | **2.159759 / 10.423%** | −0.531368 versus Original; 15/30 scene wins. Four scenes supply 89% of net gain.                                                             |
| Placement B                | Bidirectional epipolar feature exchange before correlation                               | **2.034117 / 9.626%**  | −0.657010 versus Original; 23/30 wins; beats A in 21/30 scenes. Four scenes supply 68% of gain.                                              |
| Placement C, recurrent     | Source-token spiking Transformer replaces multiscale ALIF updater                        | **2.827199 / 11.254%** | +0.136072 versus Original; 11/30 wins. Scale structure/capacity also change.                                                                 |
| A / SP                     | Retrained spiking, positional Q/K; content-bearing V                                     | **2.481007 / 10.781%** | SC better by 0.321248 at this seed.                                                                                                          |
| A / CC                     | Retrained `tanh` Q/K/V; content routing, support/projections/norm/residual matched to SC | **2.778887 / 10.575%** | SC better by 0.619128; 17/30 wins, but median gap only 0.0068.                                                                               |
| A / CP                     | Retrained continuous, positional Q/K; content-bearing V                                  | **2.868223 / 10.459%** | CC better by 0.089337 EPE; CP has lower bad3.                                                                                                |
| Compact S²M² v2b           | Separate compact spiking matching/refinement system                                      | **4.703018 / 35.664%** | Loses to Original; does not evaluate the published system.                                                                                   |

[Verified A/B/C endpoints and diagnostics](file:///Users/ethanrong/Documents/Git/Research/projects/snn-depth-estimation/experiments/spiking-attention-placement-v1/analysis/final-results.md) · [v0 paired endpoints](file:///Users/ethanrong/Documents/Git/Research/projects/snn-depth-estimation/experiments/v0-normalized-bridge-v1/results/paired-30000.json) · [Guided interpretation](file:///Users/ethanrong/Documents/Git/Research/projects/snn-depth-estimation/experiments/spiking-attention-placement-phase2/results/guided-30k-interpretation.md).

A/B replay matches all 60 official frames. B improves its own raw candidate rank/argmax slightly, yet remains worse than Original on these correspondence measures. A's rank improves but top-1 worsens. **Decoder-compatible cost shaping** is therefore a hypothesis worth testing; better final depth does not establish more accurate raw match selection. C briefly beats Original at refinement 8, then accumulates an in-view deficit by refinement 32.

### Experiment A · two completed 30k seeds

All four rows use Original's CNN, native correlation pyramid and ALIF updater, with the same added five-candidate cost-filter architecture. They differ in coding and the input to Q/K. Each seed has identical exposure and the same 60 Dev frame identities/valid counts across its four arms.

| A control | Seed 1234 EPE ↓ | Seed 2345 EPE ↓ | Two-seed mean EPE ↓ | Seed 2345 bad3 ↓ |
|---|---:|---:|---:|---:|
| SC · spiking content | 2.159759 | 2.193481 | 2.176620 | 10.455% |
| SP · spiking positional | 2.481007 | 2.074089 | 2.277548 | 10.025% |
| CC · tanh content | 2.778887 | 2.299441 | 2.539164 | 10.223% |
| CP · tanh positional | 2.868223 | 2.632070 | 2.750147 | 11.236% |

The two-seed mean weights the seeds equally; it is descriptive, not a completed three-seed significance/equivalence result. **SC wins seed 1234; SP wins seed 2345.**

**What SP means:** Q/K use learned projections of normalized candidate right-x position, with the correlation input zeroed. V still receives correlation and position. Its weights can be learned and nonuniform, but do not change with the sample's matching costs. SC uses matching cost and position for Q/K as well as V. Both mix the same five neighbors. SP is therefore a content-independent routing control, not a removal of stereo information.

| Contrast; positive favors the named effect | Seed 1234 | Seed 2345 | What follows |
|---|---:|---:|---|
| Signed coding advantage with content routing: CC − SC | +0.619128 px | +0.105960 px | Lower mean EPE repeats, but the advantage shrinks substantially. |
| Signed coding advantage with positional routing: CP − SP | +0.387217 px | +0.557981 px | Lower mean EPE repeats under positional routing too. |
| Content-routing advantage with signed coding: SP − SC | +0.321248 px | **−0.119392 px** | Direction reverses; content routing is not a consistent improvement. |
| Content-routing advantage with tanh: CP − CC | +0.089337 px | +0.332629 px | Content beats position in these two continuous runs. |
| Coding × routing interaction: (SP − SC) − (CP − CC) | +0.231911 px | **−0.452021 px** | No stable interaction direction across these seeds. |

**Strongest supported claim:** the signed-spiking training configuration achieves lower mean EPE than the tested tanh configuration in both seeds and under both routing designs. This comparison bundles the forward transfer function, internal T4 dynamics and backward derivative; it does not identify discrete spikes as the cause or establish superiority over conventional attention generally.

**Scene and metric qualifications:** seed-1 SC beats CC in 17/30 scenes; scene 0134 contributes about 92% of the net mean gap. Removing it descriptively leaves 0.0521 px, but it stays in the primary mean. In seed 2, SC beats CC in only **12/30 scenes**, with a median CC−SC difference of **−0.01415 px** and slightly worse bad3 (**10.455% vs 10.223%**). Its lower mean EPE comes from larger improvements in fewer scenes. SP beats SC in **20/30 scenes** in seed 2. There is no admitted seed-2345 Original endpoint here, so **A's improvement over Original has not yet been replicated across two seeds**.

[Saved second-seed endpoints](file:///Users/ethanrong/Documents/Git/Research/storage/experiment-runs/snn-depth-estimation/spiking-transformer-stereo/operations/seed2345-endpoints-20261105.json) · [Fresh endpoint/status audit](file:///Users/ethanrong/Documents/Git/Research/projects/snn-depth-estimation/experiments/spiking-attention-placement-phase2/results/live-status-20261105-obsidian.json) · [Registered factorial and inference rules](file:///Users/ethanrong/Documents/Git/Research/projects/snn-depth-estimation/experiments/spiking-attention-placement-phase2/STUDY.md).

### Frozen interventions and historical retraining

**A/B coding:** signed-rate T4 followed by continuous softmax. A constant projected current is encoded for four internal neuron steps; these are not four sensor observations. The models retain analog operations, and replacing internal codes does not remove any of the 50 sensor bins.

**Frozen probes measure checkpoint reliance; retraining measures whether the component is needed after learning.** At A's 30k checkpoint, branch-off/uniform/`tanh` gives **4.8436/2.8693/2.1716**, versus 2.1598 baseline. At B's earlier **23,665** checkpoint these give **2.7386/2.4156/2.1306**, versus 2.0631. They are not B's final-checkpoint ablations. Full-v0 attention-off raises 1.9909→2.2931, despite the better retrained no-attention result. Guided gate-off raises 1.9216→1.9937; score shuffle changes little and does not establish correspondence-specific benefit.

Historical three-seed, batch-2/no-flip studies found full v0 **2.1015**, no-attention **2.1685**, minimal-current LIF **2.1810**, versus Original **3.3530**. Independent normalized-current rescue also helped Original in two 5k seeds. However, a batch×flip factorial changed rankings (B8/HV at 10k: Original **2.4431**, current-I **2.1839**, v0 **2.7719**). These motivate current-pathway controls under today's recipe; they are not extra current-protocol replications. Earlier architecture screens, loss repair, smooth-signaling and diagnostic details remain in the archive.

**Historical conventional-attention comparison:** a separately trained non-SNN updater used continuous Q/K/V, softmax and sigmoid hidden states, bypassing the LIF neurons. At the earlier 10k, batch-2/no-flip protocol, its three-seed mean was **2.1989 px**, versus **2.1015 px** for the LIF updater; LIF won all three paired seeds. This compares updater operators, changing neuron dynamics and attention normalization together. It is **not** an A-site conventional-attention control under the current 30k recipe. [Conventional-attention report](file:///Users/ethanrong/Documents/Git/Research/projects/snn-depth-estimation/experiments/spiking-transformer-stereo/campaigns/analog-attention-v1/results/final-report.md).

### CPU mechanism investigation · completed 5 November

These small interventions explain dependence within trained models; they do not measure the benefit of independently trained replacements. A reuses its earlier completed two-case replay. B uses new 30k-checkpoint replays on **0120/0001 and 0123/0001**, selected explanatory cases rather than a representative dataset sample.

| Test / architecture | What was changed | Finding | Interpretation limit |
|---|---|---|---|
| Trained A, two native cases | Match unfiltered cost rows to the filter output's mean/std, keeping decoder fixed | Recovers **73.7% / 75.9%** of bypass EPE damage. Learned profile still gives better EPE/bad3; fixed five-neighbor smoothing fails. | Calibration and learned shape both matter in these fitted cases; this is not a causal percentage of the training gain. |
| Trained B, two native cases | Match pre-exchange cost rows to post-exchange row mean/std | Recovers **80.8% / 93.4%** of bypass EPE damage. | Uses statistics of the computed trained output; not an independent fast replacement. |
| Trained B, same cases | Restore only row mean; separately restore only row std | **Mean shift: 79.0% / 89.2% recovery. Scale only: 5.2% / 4.3%.** | A common row shift preserves candidate ordering and score differences, yet changes decoder behavior substantially. The shift can still encode image-dependent information. |
| Trained B, same cases | Uniform opposite-row mixing; separately signed codes → tanh | Uniform EPE **0.9015 / 3.1781**, tanh **0.9082 / 2.9591**, trained baseline **0.9021 / 2.9291**. | Learned mixing helps the harder case; exact forward spike values have small effects on these two frames. This is not a new full-Dev 30k ablation. |
| Normalized-current v0 equations | Tiny synthetic current gain/offset change before LIF | GroupNorm nearly removes that common gain/offset before membrane evolution. | Explains an operating property; does not isolate the cause of trained EPE gains. Historical no-attention/minimal-cell retraining is stronger evidence for the current-path package. |
| Guided v0, cached full-Dev probes and tiny equation check | Gate off, content shuffle; examine bounded Q/K scores | Guided v0 remains worse than its retrained parent. Switching guidance off scarcely changes its large scene-0123 failure. Four-bit heads have logits in [0,2], so correction is soft, not a sharp 36-way selector. | The branch is used; reliable content-specific selection and incremental mean-EPE benefit are unproven. The bound does not show that sharper attention would improve EPE. |

**Most useful new lead:** learned cost preparation may partly act through the numerical bias presented to the recurrent decoder, rather than better ranking alone. A's cached full-Dev analysis finds raw argmax error worsens in all 30 scenes after filtering, while removing the filter harms the fitted decoder in all 30. A/B preserve the native solver, which need not be invariant to cost offsets. Test a trainable calibration control before treating this explanation as a new method or combining all modules.

**Scope:** B used 115 seconds of CPU model computation including an initial corrected metric-mask error; 18 saved predictions were independently checked, weights stayed unchanged, and no GPU was used. CPU baseline EPE differed from historical GPU values by −0.00113/−0.01452 px, so all new effects are within-CPU contrasts. [Full CPU findings and exact tables](file:///Users/ethanrong/Documents/Git/Research/projects/snn-depth-estimation/experiments/success-mechanism-cpu-audit-v1/REPORT.md) · [Earlier A CPU controls](file:///Users/ethanrong/Documents/Git/Research/projects/snn-depth-estimation/experiments/trained-cost-filter-cpu-controls-v1/results/cpu-within-backend-20261104/REPORT.md).

### Forward spikes versus surrogate gradients · CPU validation complete

The deciding experiment independently controls the **forward values** and **local training derivative**, with A's content routing and all other training settings fixed:

| Forward values | Backward derivative | Evidence status |
|---|---|---|
| Signed T4 code | Original surrogate through T4/soft reset | Existing SC training results |
| Continuous tanh | Ordinary tanh derivative | Existing CC training results |
| **Continuous tanh** | **Original signed recurrence's surrogate derivative** | CPU control validated; **not trained** |
| **Signed T4 code** | **Tanh-shaped derivative** | CPU control validated; **not trained** |

CPU checks used A's trained filter weights with three tiny synthetic cost rows. All four combinations passed in FP32/FP64, including checkpoint recomputation. Changing only the derivative left output values **bitwise identical** while changing Q/K/V gradient magnitude and direction. Existing signed/signed and tanh/tanh outputs and gradients matched their controls. No optimizer steps, native-frame inference or EPE measurements were performed in this check.

At current 4, the signed/tanh forward values are **1 / 0.999329**, while their original derivatives are **0.062651 / 0.001341**—about a 47× difference. This makes a gradient explanation plausible; it does not establish that larger gradients improve stereo learning.

**Decision after matched training:** if continuous values with the signed derivative retain the gain, discrete forward values are unnecessary for that recovered benefit. If signed values with tanh-shaped gradients retain it, the forward coding function contributes. If only the original combination wins, investigate interaction; if both crossed controls work, both may offer sufficient routes. A forward-code benefit would still need a smooth range/dead-zone-matched control before attributing it specifically to discreteness. An inference-only gradient swap cannot answer the training question.

[CPU validation and exact training comparison](file:///Users/ethanrong/Documents/Git/Research/projects/snn-depth-estimation/experiments/spiking-attention-cost-landscape-v1/FORWARD_BACKWARD_CPU_RESULT.md).

## 3. Current status and outstanding experiments

**Fresh read-only audit: 5 November, 13:04 Shanghai. No jobs are running or queued for this account.** The old 2869–2872 chain and partial-step counters below the previous note are superseded by completed replacement jobs **2885–2888**.

| Study / seed | Baseline and change | Completed evidence | Remaining work / status |
|---|---|---|---|
| A/B/C placements / 1234 | Original plus cost filtering / feature exchange / updater replacement | All three 30k endpoints | One-seed placement comparison complete. |
| Phase 2 / 1234 | A coding × routing controls; shared Original reference | All five 30k endpoints, including reused Original/SC | Complete for this seed. |
| Phase 2 / 2345 | Same A factorial, new paired seed | **SC/SP/CC/CP all complete at 30k**, jobs **2885/2886/2887/2888**, all exit 0:0 | Original has no run directory/admitted endpoint in this audit. |
| Phase 2 / 3456 | Registered third seed of Original plus four A controls | No run directories observed | Not started in the audited phase-2 tree; not queued. |
| Whole registered replication matrix | 13 new training tasks, plus reused seed-1234 Original/SC | **7/13 new tasks complete** | Six without completed endpoints: Original2345 and all five seed3456 arms. |
| Forward/backward factorial | Existing SC/CC plus two crossed derivative/forward controls | Scalar and full-filter CPU checks complete | New training controls are proposed, not submitted. |
| Calibration / B uniform controls | Same decoder; learned row-offset or normalization replacement, or uniform descriptor exchange | Frozen CPU evidence only | Independent retraining remains proposed. |

[Current cluster/status receipt](file:///Users/ethanrong/Documents/Git/Research/projects/snn-depth-estimation/experiments/spiking-attention-placement-phase2/results/live-status-20261105-obsidian.json). The registered phase-2 practical difference is **0.05 px**; its formal classification requires the specified three seeds and scene-level intervals. Two-seed directional agreement does not satisfy that rule. All valid controls follow the fixed 30k endpoint regardless of interim ranking. No new jobs were submitted for this update; proposed experiments are not a running queue. Stopped E4, pyramid-free A and old v0 replication remain stopped. Encoder conversion is separate: [[Converting SpikeStereoNet into full SNN]].

## 4. Paper direction and deciding questions

**There is no percentage-improvement threshold that separates CCF-B from CCF-A.** A targeted improvement to an existing architecture can support a CCF-A paper when its contribution is distinctive, useful and convincingly evaluated. Conversely, an ordinary bias/normalization fix that only improves one inherited baseline may remain limited, even with a large EPE gain. Explaining an engineering change more scientifically does not by itself create novelty.

The present evidence supports a mechanism investigation, not an established CCF-A contribution. Two promising questions are **how matching-cost calibration interacts with recurrent updates**, and **whether the signed training derivative helps learn useful representations without requiring discrete inference codes**. Neither requires abandoning the native dataset to run the next deciding experiment.

| Question before writing a strong paper | Next deciding evidence | What it would establish |
|---|---|---|
| Does the code advantage survive replication and fair controls? | Complete registered seed/baseline matrix; report EPE, bad3, scene tails and seed variability. Give continuous alternatives comparable tuning. | Scope and consistency of the advantage; no universal SNN claim. |
| Is the benefit in forward values, the backward rule, or their combination? | Train the two crossed controls from matched initial states, initially at the two existing seeds. Keep the native diagonal implementations. | Separates the interventions under this recipe; CPU validation alone cannot decide the winner. |
| Is cost calibration learnable without the full attention block? | Compare a small learned row-offset adapter and plain centered/standardized costs with the matched original model and B. Controls must use raw inputs, with no trained-output Y statistics. | Whether the frozen CPU shortcut is a sufficient deployable replacement after learning. |
| Does a new method improve the strongest relevant parent? | Test incremental value against retrained no-attention v0; preserve current/decoder interfaces. A minimal-current retrain under the current recipe explains v0 separately. | Avoids attributing inherited gains to added guidance or attention. |
| Is there a distinctive contribution beyond known attention/normalization/confidence methods? | Identify a reproducible failure and a specific solution that beats strong simple alternatives; check closest prior art once the successful mechanism is concrete. | A possible CCF-A methods/empirical contribution. A simpler baseline improvement alone may justify a narrower story. |
| Does the claim generalize, and at what cost? | An independently evaluated cohort or relevant transfer setting; matched latency/memory/energy only if claimed. For sensor robustness, connect cost changes to actual acquisition changes. | Evidence proportional to the claimed contribution, rather than artificial perturbations alone. |

**Immediate priorities:** finish interpreting the registered replication; use the crossed forward/backward experiment to answer the spiking-specific question, and a trained calibration control to test the simpler explanation. These are plans, not new launch authorization. Avoid stacking A+B+guidance+I before identifying which component adds value.

**Venue framing:** a focused CCF-B submission could center on a narrower validated improvement and careful controls, subject to venue fit. A credible CCF-A direction would offer a consequential, distinctive method or finding beyond one configuration, with appropriate comparisons and evidence. A universal theory, new sensor, entirely new architecture or arbitrary number of datasets is not a prerequisite; acceptance is not established by this notebook.

**Closest-work context:** [FoundationStereo](https://arxiv.org/html/2501.09898v1) already uses disparity-axis attention; [Selective-Stereo](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Selective-Stereo_Adaptive_Frequency_Information_Selection_for_Stereo_Matching_CVPR_2024_paper.html) is a precedent for a targeted updater improvement in existing stereo backbones. Placement alone is not our novelty. [Verified Middlebury-paper novelty comparison](file:///Users/ethanrong/Documents/Git/Research/projects/snn-depth-estimation/literature-search-20261104-middlebury-novelty/papers.md).

All 40 native Test scenes were used historically and cannot now be called untouched confirmation. Existing data still supports the controlled mechanism study. New scenes or an external benchmark would strengthen independent generalization evidence. No field-wide SOTA, sensor-time attention or energy advantage has been established.

[CPU-informed direction report](file:///Users/ethanrong/Documents/Git/Research/projects/snn-depth-estimation/experiments/success-mechanism-cpu-audit-v1/REPORT.md) · [Earlier roadmap (its 4 November status is historical)](file:///Users/ethanrong/Documents/Git/Research/projects/snn-depth-estimation/experiments/spiking-attention-paper-direction/README.md) · [Backup before this update](file:///Users/ethanrong/Documents/Git/Research/storage/experiment-runs/snn-depth-estimation/spiking-transformer-stereo/research-notes/Spiking%20Transformer%20-%20before-second-seed-refresh-20261105-130709.md) · [Full older experiment notebook](file:///Users/ethanrong/Documents/Git/Research/storage/experiment-runs/snn-depth-estimation/spiking-transformer-stereo/research-notes/Spiking%20Transformer%20-%20before-master-refresh-2026-11-04-200603.md).


What about transformers for spiking cameras but monocular, could we see their failure modes and perhaps understand how to implement it for stereo vision? 