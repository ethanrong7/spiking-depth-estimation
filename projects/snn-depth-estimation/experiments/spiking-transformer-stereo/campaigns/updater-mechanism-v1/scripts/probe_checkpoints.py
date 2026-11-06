"""Frozen final-checkpoint development interventions; no training or Test access."""
import argparse
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import socket
import sys

EXPERIMENT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(EXPERIMENT / 'scripts'))
import probe_refinements as probe
import run_experiment as runner
import torch
from torch.utils.data import DataLoader


@contextmanager
def reset_each_cell_call(model):
    """Reset Q/K/V/output voltages before each cell call; retain hidden spikes.

    This removes internal membrane carry, not all recurrence: hidden outputs,
    disparity, and cross-scale inputs still feed subsequent refinements.
    """
    handles = [cell.register_forward_pre_hook(
        lambda module, inputs: module.reset_state(clear_diagnostics=False))
        for cell in model.update_block.cells]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'data-root', 'source-registry', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--checkpoint', action='append', required=True, metavar='SEED=LAST_PT')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('scientific execution requires Slurm CUDA')
    if torch.cuda.device_count() != 1 or '5090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('requires one allocated RTX5090')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    manifest = json.loads(args.manifest.read_text())
    digest = probe.file_sha256(args.manifest)
    if digest != 'bad4369b6fd8b0cdf871a20b1727c2d9b59733a16bd33f75904227bdfef4510a':
        raise ValueError('manifest differs from frozen five-method cohort')
    rows = manifest['dev']
    if len(rows) != 60:
        raise ValueError('requires frozen 60-frame dev set')
    source = json.loads(args.source_registry.read_text())
    for relative, expected in source['source_files'].items():
        if probe.file_sha256(runner.PROJECT.parents[1] / relative) != expected:
            raise ValueError('source differs from registry: ' + relative)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(command=sys.argv, source=source, manifest_sha256=digest,
                    script_sha256=probe.file_sha256(__file__), hostname=socket.gethostname(),
                    torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                    slurm={k: v for k, v in os.environ.items() if k.startswith('SLURM_')},
                    split='dev', cohort=rows, checkpoints={}, conditions={}, paired_scene_errors={})
    loader = DataLoader(runner.ManifestDataset(args.data_root, rows), batch_size=2, shuffle=False, num_workers=0)
    def save():
        (args.output / 'probe.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    save()
    seen = set()
    for entry in args.checkpoint:
        label, path = entry.split('=', 1)
        if label not in ('1234', '2345', '3456') or label in seen or Path(path).name != 'last.pt':
            raise ValueError('require unique paired seed labels and final last.pt checkpoints')
        seen.add(label)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        config = checkpoint['config']
        if (config['model'] != 'spiking' or config['seed'] != int(label)
                or not config['normalize_residual_current'] or checkpoint['updates'] != 10000):
            raise ValueError('requires registered final10k full normalized LIF checkpoint')
        if checkpoint['manifest_sha256'] != digest:
            raise ValueError('checkpoint manifest differs')
        for relative, expected in checkpoint['source']['source_files'].items():
            if '/core/' in relative and source['source_files'].get(relative) != expected:
                raise ValueError('probe core differs from training: ' + relative)
        model = runner.build_model('spiking', int(label), True).cuda().eval()
        model.load_state_dict(checkpoint['model'], strict=True)
        metadata['checkpoints'][label] = dict(path=str(path), sha256=probe.file_sha256(path),
            update=checkpoint['updates'], config=config, training_source=checkpoint['source'])
        conditions = [('full', {}, 32), ('center_masked', {'center_only': True}, 32),
                      ('branch_disabled', {'disable_attention': True}, 32), ('membrane_reset', {}, 32),
                      ('full16', {}, 16), ('full8', {}, 8)]
        for name, flags, iterations in conditions:
            before = [(c.spatial_mixing, c.attention_enabled, len(c._forward_pre_hooks)) for c in model.update_block.cells]
            context = reset_each_cell_call(model) if name == 'membrane_reset' else model.update_block.attention_intervention(**flags) if flags else nullcontext()
            with context:
                result = probe.evaluate_condition(model, loader, iterations, args.output / label / name, True)
            after = [(c.spatial_mixing, c.attention_enabled, len(c._forward_pre_hooks)) for c in model.update_block.cells]
            if before != after:
                raise RuntimeError('intervention cleanup failed')
            if name == 'full':
                history = json.loads((Path(path).parent / 'history.json').read_text())
                expected = next(row for row in history if row['update'] == 10000)
                metric_error = abs(result['mean_scene_epe'] - expected['mean_scene_epe'])
                scene_error = max(abs(result['scene_epe'][scene] - value) for scene, value in expected['scene_epe'].items())
                metadata['checkpoints'][label]['reproduction'] = dict(mean_epe_abs_difference=metric_error, max_scene_epe_abs_difference=scene_error, tolerance_px=1e-5)
                save()
                if max(metric_error, scene_error) > 1e-5:
                    raise RuntimeError('full32 differs from recorded final metrics; stop interpretation')
            metadata['conditions'].setdefault(label, {})[name] = result
            full = metadata['conditions'][label]['full']['scene_epe']
            metadata['paired_scene_errors'].setdefault(label, {})[name] = {
                scene: dict(full_epe=full[scene], intervention_epe=value, delta_epe=value-full[scene])
                for scene, value in result['scene_epe'].items()}
            if name != 'full':
                import numpy as np
                differences = []
                for frame in result['per_frame']:
                    filename = frame['scene'] + '-' + frame['frame'] + '.npz'
                    with np.load(args.output / label / 'full' / 'predictions' / filename) as base, np.load(args.output / label / name / 'predictions' / filename) as changed:
                        delta = np.abs(changed['prediction'] - base['prediction'])
                        differences.append(dict(scene=frame['scene'], frame=frame['frame'], mean_abs_prediction_delta=float(delta.mean()), max_abs_prediction_delta=float(delta.max())))
                result['prediction_delta_from_full32'] = differences
            save()
            print(label, name, result['mean_scene_epe'], flush=True)
        del model, checkpoint
        torch.cuda.empty_cache()
    if seen != {'1234', '2345', '3456'}:
        raise ValueError('campaign requires all three paired seeds')
    metadata['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
