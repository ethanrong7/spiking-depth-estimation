"""Prepare, validate and run the eight single-NPU B encoder ablations.

No accelerator work without --execute. Each device runs one cell at a time;
all preflights must pass before any production cell starts. Re-run with --resume
after interruption. Historical endpoints are not silently imported.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[3]
PROJECT = EXP.parents[1]
STUDY = 'hybrid-b-ablation-911-NPU'
SEEDS = (1234, 2345)
VARIANTS = ('linear-identity', 'signed-surrogate', 'linear-surrogate', 'signed-identity')
HISTORICAL_INITIAL = 'b37335857fd34f9f5aaa1cf203f906304f7edbd965447ff7dad3953172d47bd4'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def cells(devices):
    return [dict(seed=seed, variant=variant, device=devices[i % len(devices)])
            for i, (seed, variant) in enumerate((s, v) for s in SEEDS for v in VARIANTS)]


def environment(device, seed):
    # Do not invent server scratch paths or modify the caller's CANN environment.
    return {**os.environ, 'ASCEND_RT_VISIBLE_DEVICES': str(device),
            'PYTHONHASHSEED': str(seed), 'HCCL_DETERMINISTIC': 'false',
            'OMP_NUM_THREADS': '4', 'PYTHONDONTWRITEBYTECODE': '1',
            'WORLD_SIZE': '1', 'RANK': '0', 'LOCAL_RANK': '0'}


def verify_source(output):
    registry = read_json(output/'source-registry.json')
    for name, expected in registry['source_files'].items():
        path = ROOT/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen source changed: {name}; use a new study output for changed code')
    return registry['source_identity_sha256']


def prepare(args):
    output = args.output_root
    requested = dict(study=STUDY, cells=cells(args.devices),
                     data_root=str(args.data_root), manifest=str(args.manifest),
                     updates=30000, schedule_total=30000, physical_batch=4,
                     accumulation=2, device_count_per_model=1)
    if (output/'config.json').exists():
        if read_json(output/'config.json') != requested:
            raise RuntimeError('Existing batch configuration differs; preserve it and use a new output root')
        verify_source(output)
        return
    if output.exists():
        raise RuntimeError(f'Output already exists without a complete configuration: {output}')
    from freeze_source import freeze_checkout
    freeze_checkout(output)
    sys.path[:0] = [str(PROJECT), str(EXP.parent/'shared-gpu-30k-v1')]
    from settings import load
    import torch
    torch.set_num_threads(4)
    from core.attention_encoder_controls import install
    contract = load('ablation_initial_contract', EXP.parent/'spiking-attention-placement-phase2/protocol.py')
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != contract.MANIFEST_SHA:
        raise RuntimeError('Dataset manifest does not match the completed NPU protocol')
    driver_path = EXP.parent/'spiking-transformer-stereo/scripts/run_experiment.py'
    sys.path.insert(0, str(driver_path.parent))
    driver = load('ablation_initial_driver', driver_path)
    initialization_records = []
    for seed in SEEDS:
        model = install(driver.build_model('original', seed), 'B',
                        variant='signed-surrogate', seed=seed)
        state = model.state_dict()
        identity = contract.tensor_hash(state)
        # This is an all-fresh within-backend batch. Seeded initialization can
        # differ across PyTorch/CPU implementations; never claim historical
        # bitwise replication merely because the seed number matches.
        historical_match = identity == HISTORICAL_INITIAL if seed == 1234 else None
        initialization_records.append(dict(seed=seed, sha256=identity,
            historical_reference_sha256=HISTORICAL_INITIAL if seed == 1234 else None,
            historical_reference_matches=historical_match, torch=torch.__version__))
        path = output/'initializations'/f'seed-{seed}.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(seed=seed, model=state, sha256=identity), path)
        del model, state
    write_json(output/'initializations'/'metadata.json', initialization_records)
    if initialization_records[0]['historical_reference_matches'] is False:
        print('NOTICE: fresh seed-1234 initialization differs from historical NPU bytes. '
              'All new cells share the new state; no historical endpoint is reused.', flush=True)
    write_json(output/'reuse-decision.json', dict(reuse=False,
               reason='Eight fresh paired runs; no external result/source archive required.'))
    write_json(output/'matrix.json', requested['cells'])
    write_json(output/'config.json', requested)


def worker_command(args, cell, *, preflight=False, resume=False, pause=False):
    root = args.output_root
    if preflight:
        root = root/'preflight'
    root = root/f"seed-{cell['seed']}"/cell['variant']
    cmd = [sys.executable, str(EXP/'worker_ablation_npu.py'),
           '--seed', str(cell['seed']), '--variant', cell['variant'],
           '--root', str(root), '--data-root', str(args.data_root),
           '--manifest', str(args.manifest), '--source-registry',
           str(args.output_root/'source-registry.json'), '--initialization',
           str(args.output_root/'initializations'/f"seed-{cell['seed']}.pt"),
           '--stop-at', '3' if preflight else '30000', '--input-mode', 'preload',
           '--preload-gib', '12', '--input-workers', '2', '--prefetch-depth', '2']
    if preflight:
        cmd.append('--preflight')
    if pause:
        cmd += ['--pause-at', '2']
    if resume:
        cmd.append('--resume')
    return root, cmd


def run_worker(args, cell, *, preflight=False, resume=False, pause=False):
    root, command = worker_command(args, cell, preflight=preflight, resume=resume, pause=pause)
    root.mkdir(parents=True, exist_ok=True)
    if (root/'last.pt').exists() != resume:
        raise RuntimeError(f'Resume flag does not match checkpoint at {root}')
    if (root/'pause.request').exists():
        raise RuntimeError(f'Remove the pause.request deliberately before resuming {root}')
    record = dict(**cell, command=command, preflight=preflight, resume=resume,
                  time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(root/'launch.json', record)
    with (root/'launch-history.jsonl').open('a') as handle:
        handle.write(json.dumps(record)+'\n')
    print(f"START {'preflight' if preflight else 'production'} {cell}", flush=True)
    with (root/'stdout.log').open('ab') as out, (root/'stderr.log').open('ab') as err:
        result = subprocess.run(command, cwd=ROOT, env=environment(cell['device'], cell['seed']),
                                stdout=out, stderr=err)
    expected = 75 if pause else 0
    if result.returncode != expected:
        raise RuntimeError(f'Worker exited {result.returncode}, expected {expected}: {root}/stderr.log')
    if pause:
        if read_json(root/'checkpoint.json')['update'] != 2:
            raise RuntimeError(f'Pause did not save update 2: {root}')
    else:
        saved = read_json(root/'result.json')
        if saved['updates'] != (3 if preflight else 30000):
            raise RuntimeError(f'Incomplete worker result: {root}')
    print(f"DONE {'preflight' if preflight else 'production'} {cell}", flush=True)


def preflight_cell(args, cell):
    root, _ = worker_command(args, cell, preflight=True)
    if (root/'result.json').exists():
        # A successful resumed preflight is reusable only under the frozen source.
        result = read_json(root/'result.json')
        if result['status'] == 'PREFLIGHT_COMPLETED' and (root/'restore-audit-2.json').exists():
            return
        raise RuntimeError(f'Incomplete preflight evidence: {root}')
    probe = root/'encoder-validation.json'
    root.mkdir(parents=True, exist_ok=True)
    if not probe.exists():
        command = [sys.executable, str(EXP/'validate_ablation_npu.py'), '--output', str(probe)]
        with (root/'encoder-validation.log').open('ab') as handle:
            subprocess.run(command, cwd=ROOT, env=environment(cell['device'], cell['seed']),
                           stdout=handle, stderr=handle, check=True)
    if read_json(probe)['status'] != 'passed':
        raise RuntimeError(f'Encoder validation failed: {probe}')
    if not (root/'last.pt').exists():
        run_worker(args, cell, preflight=True, pause=True)
    else:
        update = read_json(root/'checkpoint.json')['update']
        if update < 2:
            run_worker(args, cell, preflight=True, pause=True, resume=True)
        elif update not in (2, 3):
            raise RuntimeError(f'Unexpected preflight checkpoint; inspect {root}')
    run_worker(args, cell, preflight=True, resume=True)
    if not (root/'restore-audit-2.json').exists():
        raise RuntimeError(f'Missing resume audit: {root}')


def on_devices(args, function):
    """Sequential queue per assigned device; parallel independent devices."""
    matrix = cells(args.devices)
    def queue(device):
        for cell in matrix:
            if cell['device'] == device:
                function(args, cell)
    with ThreadPoolExecutor(max_workers=len(args.devices)) as pool:
        futures = [pool.submit(queue, device) for device in args.devices]
        errors = []
        for future in futures:
            try:
                future.result()
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError('\n'.join(errors))


def validate_preflight_matrix(args, identity):
    """Admit a uniform backend and paired initial states, not merely exit code 0."""
    records = []
    for cell in cells(args.devices):
        root, _ = worker_command(args, cell, preflight=True)
        result = read_json(root/'result.json')
        metadata = read_json(root/'metadata.json')
        config = metadata['config']
        if (result['status'] != 'PREFLIGHT_COMPLETED' or result['updates'] != 3
                or result['source_identity'] != identity or not result['finite_model_optimizer']):
            raise RuntimeError(f'Invalid preflight completion: {root}')
        if (config['seed'], config['variant']) != (cell['seed'], cell['variant']):
            raise RuntimeError(f'Misrouted preflight: {root}')
        if read_json(root/'restore-audit-2.json')['status'] != 'passed':
            raise RuntimeError(f'Resume validation failed: {root}')
        probe = read_json(root/'encoder-validation.json')
        if probe['status'] != 'passed':
            raise RuntimeError(f'Encoder validation failed: {root}')
        backend = metadata['rank_environments'][0]
        signature = {key: backend[key] for key in ('python', 'torch', 'torch_npu', 'npu_name', 'precision')}
        records.append(dict(**cell, backend=signature, initial_state=metadata['initial_state_sha256']))
    if len({json.dumps(r['backend'], sort_keys=True) for r in records}) != 1:
        raise RuntimeError('Preflights used different NPU models/software/precision policies')
    for seed in SEEDS:
        if len({r['initial_state'] for r in records if r['seed'] == seed}) != 1:
            raise RuntimeError(f'Initial weights differ within seed {seed}')
    return records


def production_cell(args, cell):
    root, _ = worker_command(args, cell)
    if (root/'result.json').exists():
        if not args.resume:
            raise RuntimeError(f'Existing result: {root}; use --resume for a batch continuation')
        result = read_json(root/'result.json')
        if result['status'] == 'COMPLETED' and result['updates'] == 30000:
            return
        raise RuntimeError(f'Invalid completion record: {root}')
    run_worker(args, cell, resume=args.resume and (root/'last.pt').exists())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT/'configs/synthetic-split.json')
    parser.add_argument('--output-root', type=Path, default=EXP/'outputs'/STUDY)
    parser.add_argument('--devices', type=int, nargs='+', required=True,
                        help='Allocated physical NPU IDs; one worker per ID, typically eight IDs')
    parser.add_argument('--execute', action='store_true', help='Run NPU validation and training')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if len(set(args.devices)) != len(args.devices) or not 1 <= len(args.devices) <= 8 or min(args.devices) < 0:
        parser.error('Specify 1–8 unique nonnegative allocated device IDs')
    for name in ('data_root', 'manifest', 'output_root'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if not args.data_root.is_dir() or not args.manifest.is_file():
        parser.error('Dataset root/manifest must exist')
    return args


def main(argv=None):
    args = parse_args(argv)
    # Linux file lock prevents duplicate orchestrators launching onto the same devices.
    import fcntl
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    with (args.output_root.parent/(args.output_root.name+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(args)
        print(json.dumps(dict(output=str(args.output_root), cells=cells(args.devices)), indent=2), flush=True)
        if not args.execute:
            print('Prepared only. Add --execute for NPU preflights and the full batch.')
            return 0
        identity = verify_source(args.output_root)
        on_devices(args, preflight_cell)
        validation = validate_preflight_matrix(args, identity)
        write_json(args.output_root/'port-validation.json', dict(status='passed', source_identity=identity,
                   cells=validation, checks='encoder VJP/checkpoint and native update-2 resume per cell'))
        if args.preflight_only:
            return 0
        verify_source(args.output_root)
        try:
            on_devices(args, production_cell)
        finally:
            from summarize_ablation_npu import summarize
            summarize(args.output_root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
