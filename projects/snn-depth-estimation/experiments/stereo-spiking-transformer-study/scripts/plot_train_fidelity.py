"""Render all fixed native QA views from verified saved files; no inference."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
SCENES = ('0087', '0009', '0349', '0204')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    sources = {}

    def load(member):
        path = args.raw_root/member
        expected = inventory['files'][member]
        if path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
            raise ValueError('Saved visual differs from copied inventory: '+member)
        sources[member] = expected['sha256']
        with np.load(path, allow_pickle=False) as packed:
            return {key: packed[key] for key in packed.files}

    outputs = []
    for center in (1, 4, 8):
        rgb_fig, rgb_axes = plt.subplots(4, 7, figsize=(18, 8))
        crop_fig, crop_axes = plt.subplots(8, 12, figsize=(21, 12))
        for row, scene in enumerate(SCENES):
            prefix = 'TRAIN/C/'+scene
            for view_index, view in enumerate(('l', 'r')):
                for step_index, step in enumerate((0, 25, 50)):
                    data = load(f'{prefix}/{view}/visual-{center}-{step}.npz')
                    assert bool(data['sensor_output_available']) == (step != 0)
                    rgb_ax = rgb_axes[row, view_index*3+step_index]
                    rgb_ax.imshow(data['rgb_thumbnail'], interpolation='nearest')
                    if row == 0:
                        rgb_ax.set_title(f'{view.upper()} {step}/50', fontsize=10)
                    axes = crop_axes[row*2+view_index, step_index*4:step_index*4+4]
                    axes[0].imshow(data['gray_crop'], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
                    for channel, ax in enumerate(axes[1:]):
                        if step == 0:
                            ax.text(.5, .5, 'No new\noutput', ha='center', va='center', fontsize=7)
                            ax.set_facecolor('#eeeeee')
                        elif channel == 0:
                            ax.imshow(data['spike_crop'], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
                        else:
                            # Fixed theoretical crossing scale, retaining counts above1.
                            ax.imshow(np.log1p(data['event_crop'][channel-1]), cmap='magma',
                                      vmin=0, vmax=np.log1p(37), interpolation='nearest')
                    if row == 0 and view_index == 0:
                        for title, ax in zip(('Gray', 'Spike', 'ON log1p', 'OFF log1p'), axes):
                            ax.set_title(f'{step}/50 {title}', fontsize=8)
                crop_axes[row*2+view_index, 0].set_ylabel(scene+' '+view.upper(), fontsize=9)
            gt = load(f'{prefix}/disparity-visual-{center}.npz')
            rgb_axes[row, 6].imshow(gt['native_crop'], cmap='viridis', vmin=0, vmax=256, interpolation='nearest')
            if row == 0:
                rgb_axes[row, 6].set_title('GT crop: 0–256px', fontsize=10)
            rgb_axes[row, 0].set_ylabel(scene+(' historical' if row < 2 else ' new'), fontsize=9)
        for axes in (rgb_axes, crop_axes):
            for ax in axes.flat:
                ax.set_xticks([])
                ax.set_yticks([])
        rgb_fig.suptitle(f'Fixed native Train panel · interval {center} · RGB thumbnails + endpoint GT crop', fontsize=13)
        crop_fig.suptitle(f'Interval {center} · native crop rows222:318, cols416:544 · inherited EMA clamp\n'
                         'Source-frame time; no physical calibration · step0 has no new sensor output', fontsize=12)
        for fig, kind in ((rgb_fig, 'rgb'), (crop_fig, 'crops')):
            fig.tight_layout(rect=(0, 0, 1, .95))
            path = STUDY/'results'/f'train-fidelity-native-{kind}-{center}-v1.png'
            fig.savefig(path, dpi=130)
            plt.close(fig)
            outputs.append(dict(path=str(path.relative_to(STUDY)), sha256=sha(path)))
    receipt = dict(scope='All72 predeclared Train views; saved-output rendering only; no inference or Test',
        inventory_sha256=sha(args.inventory), source_members_sha256=sources,
        renderer_sha256=sha(Path(__file__)), outputs=outputs,
        event_display='log1p(count), fixed37 crossing-scale maximum; no threshold/activity tuning')
    (STUDY/'results/train-fidelity-native-visuals-v1.sources.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(plates=len(outputs), source_members=len(sources), all_visual_cases=72)))


if __name__ == '__main__':
    main()
