"""CPU-only Slurm task wrapper: train, preserve endpoint, then parity replay."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=(1234, 2345, 3456), required=True)
    p.add_argument('--arm', choices=('SP', 'CC', 'CP', 'SC', 'Original'), required=True)
    p.add_argument('--reuse-receipt', type=Path, help='Replay only; never train a historical reference')
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    source = Path(config['source']); raw = Path(config['raw_root'])
    project = source / 'projects/snn-depth-estimation'
    exp = project / 'experiments/spiking-attention-placement-phase2'
    python = config['python']
    guard = [python, '-u', str(exp / 'slurm/allocation_guard.py')]
    env = os.environ.copy()
    env.update(STEREO_SOURCE=str(source), STEREO_RUN_ROOT=str(raw),
               STEREO_ENV=str(Path(python).parents[1]), STEREO_DATA=config['data_root'])
    folder = raw / 'train' / f'seed-{a.seed}' / a.arm
    folder.mkdir(parents=True, exist_ok=True)
    def status(phase, **values):
        path = folder / 'task-replay-status.json'; temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(phase=phase, time_utc=datetime.now(timezone.utc).isoformat(),
                    seed=a.seed, arm=a.arm, job=os.environ['SLURM_JOB_ID'],
                    task_kind='HISTORICAL_REFERENCE_REPLAY' if a.reuse_receipt else 'TRAINING_AND_REPLAY',
                    training_performed=not bool(a.reuse_receipt), **values), indent=2) + '\n')
        temporary.replace(path)
    if a.reuse_receipt:
        receipt = json.loads(a.reuse_receipt.read_text())
        assert receipt['status'] == 'VERIFIED_REUSE' and receipt['current_source_identity'] == config['source_identity']
        assert (receipt['seed'], receipt['arm']) == (a.seed, a.arm)
        endpoint, = [row for row in receipt['endpoints'] if row['update'] == 30000]
        assert endpoint['checkpoint_verified']
        evaluations = Path(receipt['reference']) / 'evaluations.jsonl'
        status('HISTORICAL_REFERENCE_REPLAY_PENDING', reuse_receipt=str(a.reuse_receipt))
    else:
        command = guard + [python, '-u', str(exp / 'run_on_gpu.py'), '--seeds', str(a.seed), '--arms', a.arm, '--stage', 'all']
        status('TRAINING', command=command)
        training = subprocess.run(command, env=env)
        if training.returncode:
            status('TRAINING_FAILED', exit_code=training.returncode)
            return training.returncode
        endpoint = json.loads((folder / 'endpoint-30000.json').read_text())
        assert endpoint['source_identity'] == config['source_identity']
        assert endpoint['update'] == 30000
        evaluations = folder / 'evaluations.jsonl'
        status('TRAINING_COMPLETED_REPLAY_PENDING', endpoint=str(folder / 'endpoint-30000.json'))
    replay = guard + [python, '-u', str(exp / 'replay.py'), '--source-root', str(project),
                     '--data-root', config['data_root'], '--manifest', str(project / 'experiments/spiking-transformer-stereo/configs/manifest.json'),
                     '--checkpoint', endpoint['checkpoint'], '--evaluations', str(evaluations),
                     '--raw-root', str(raw / 'diagnostics'), '--seed', str(a.seed), '--arm', a.arm]
    if a.reuse_receipt:
        replay.extend(['--reuse-receipt', str(a.reuse_receipt)])
    replay_env = env.copy()
    replay_env['STEREO_RUN_ROOT'] = str(raw / 'diagnostics' / 'admission' / os.environ['SLURM_JOB_ID'])
    try:
        with (folder / 'replay.out').open('a') as log:
            result = subprocess.run(replay, env=replay_env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        if result.returncode:
            status('HISTORICAL_REFERENCE_REPLAY_FAILED' if a.reuse_receipt else 'TRAINING_COMPLETED_REPLAY_FAILED', exit_code=result.returncode, command=replay)
            return 87
        receipt = raw / 'diagnostics' / f'seed-{a.seed}' / a.arm / 'replay-30000.json'
        row = json.loads(receipt.read_text())
        assert row['parity']['passed'] and row['update'] == 30000
        assert row['checkpoint_sha256'] == endpoint['checkpoint_sha256']
        status('HISTORICAL_REFERENCE_REPLAY_COMPLETED' if a.reuse_receipt else 'TRAINING_AND_REPLAY_COMPLETED', replay=str(receipt), parity=row['parity'])
        return 0
    except Exception as error:
        status('HISTORICAL_REFERENCE_REPLAY_FAILED' if a.reuse_receipt else 'TRAINING_COMPLETED_REPLAY_FAILED', error=repr(error), command=replay)
        return 87


if __name__ == '__main__':
    raise SystemExit(main())
