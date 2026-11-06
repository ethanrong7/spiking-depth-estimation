# Spiking depth estimation

Standalone source for native spike-camera stereo depth, starting with SpikeStereoNet and the existing spiking-attention / Transformer implementations.

## Contents

- `projects/snn-depth-estimation/core/`: all current model, data, loss and utility code.
- `projects/snn-depth-estimation/vendor/`: required baseline source with licenses and provenance.
- `projects/snn-depth-estimation/experiments/`: reusable experiment implementations, training drivers and probes. Their original paths are preserved because scripts import one another by path.
- `projects/snn-depth-estimation/tests/`: imported implementation tests.
- `run.py`, `runtime.py`, `configs/`: fresh portable execution entry point and Native Synthetic split/configuration.
- `Spiking Transformer.md`: byte-for-byte snapshot of the Obsidian master sheet. Its links retain the original targets; linked research documents are not included.

Papers, previous results, run records, dashboards, trained weights, source capsules, datasets, Git history and old cluster configuration files are excluded. The original Research repository is unchanged. Upstream license/provenance documents are retained for the copied code.

## Setup

Use Python 3.10 or newer. Install a suitable PyTorch build for your machine, then:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py models
python run.py smoke --model spikestereonet
```

The smoke command uses random 50-bin tensors on CPU and checks output shape/finite values. It does not measure accuracy. No GPU is selected automatically.

The portable runner supports SpikeStereoNet; normalized v0, no-attention v0 and guided v0; attention placements A/B/C; epipolar and SSDE-inspired models; S2M2 spiking/v2/continuous; coordinate Transformer/E3; direct attention; and temporal spiking correspondence. Other implementations remain available in `core/` and the experiment code. Placement A changes the correlation class within its Python process: use separate processes for separate models.

## Inputs and inference

Point `SPIKE_DATA_ROOT` to an existing Native Synthetic directory containing `Train/` and `Test/`. Data remain outside this repository; paths can refer to the existing dataset without copying it.

For inference supply native packed `.dat` files (50 bins, 250 × 400) or NumPy `[50,H,W]` arrays, and matching model weights:

```bash
python run.py infer --model spikestereonet --device cpu \
  --checkpoint /path/to/weights.pt \
  --left /path/to/left.dat --right /path/to/right.dat \
  --output outputs/disparity.npy
```

Output is disparity `x_left - x_right`, in native pixels. Internal losses use the original negative signed-flow convention. Raw depth follows `Z = f B / disparity` where valid; calibration must match the camera. New checkpoints contain a `model` state dictionary. A raw state dictionary or `state_dict` wrapper is also accepted, including a uniform DataParallel prefix. Checkpoint loading is strict and uses PyTorch's tensor-safe loader; old full training-state pickles may require a separate trusted conversion. No pretrained weights are included.

## Fresh training and evaluation

Run these on an authorized CUDA resource. These commands do not submit scheduler jobs.

```bash
export SPIKE_DATA_ROOT=/path/to/Synthetic
python run.py train --device cuda --model spikestereonet \
  --config configs/train-native.json --output runs/original-seed1234
python run.py evaluate --device cuda --model spikestereonet \
  --checkpoint runs/original-seed1234/weights.pt \
  --output outputs/original-dev.json
```

Choose a different `--model` for a new experiment. The default config uses 30,000 optimizer steps, batch 4 with accumulation 2, 16 training refinements and a fixed OneCycle schedule. Evaluation defaults to 32 refinements and Dev only. The retained split lists input filenames, not prior measurements. Training creates a new output directory and saves inference weights periodically. It does not resume interrupted optimizer state or select a best checkpoint automatically.

This fresh harness reuses the current models and losses; it is not an exact replay of the historical training order, augmentation seeds, frozen source versions or run protocol. New results need their own measurements. Commit the chosen code/configuration before a real cluster run to give it an exact revision.

## Retained experiment code

These files preserve additional architectures, controls and probes. Older campaign entry points can depend on frozen source capsules, registrations, old configurations or checkpoints that were deliberately excluded. They are retained as implementation source, not as ready-to-resume old jobs. Use the fresh runner for the listed model families; adapt another experiment's configuration explicitly before running it. No old cluster controller is enabled by the new entry point.

## Verification

Verified: all copied source files match their originals; the master note matches byte-for-byte; all Python files parse; all 16 listed models pass a 64 × 128 CPU forward check; three integration tests pass for recurrent/direct gradients, state reset and packed/NumPy input decoding. Checks used the existing local PyTorch 2.8 CPU environment; a clean dependency installation and GPU training/evaluation have not been run.

Run the new integration checks with:

```bash
python -m unittest discover -s tests -v
```
 The full historical test suite includes tests against excluded campaign artifacts and is not a standalone acceptance suite.

The repository is initialized locally with no commit or remote. Configure GitHub and make the first commit when ready. See `UPSTREAM.md` and each vendor license before redistribution.
