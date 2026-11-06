"""Run read-only A/B diagnostics under an existing own-worker phase lock."""
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone


def output(*args):
    return subprocess.check_output(args, text=True, timeout=20).strip()


def main(config_path):
    config = json.loads(Path(config_path).read_text())
    root = Path(config['output_root']); root.mkdir(parents=True, exist_ok=True)
    job, uuid = str(config['job']), config['gpu_uuid']
    assert job == os.environ['SLURM_JOB_ID'] == '2805'
    assert socket.gethostname().split('.')[0] == 'cccl3' and output('id', '-un') == 'rongliang'
    assert uuid == 'GPU-91144565-173b-4884-43c0-1cbff4ebdb05'
    allocation = output('scontrol', 'show', 'job', '-do', job)
    assert 'UserId=rongliang(1004)' in allocation.split() and 'GRES=gpu:1(IDX:1)' in allocation.split()
    assert 'job_'+job+'/' in Path('/proc/self/cgroup').read_text()
    driver, = [p.read_text() for p in Path('/proc/driver/nvidia/gpus').glob('*/information') if uuid in p.read_text()]
    assert re.search(r'Device Minor:\s*1\s', driver)
    phase = Path(config['phase_root']); assert phase.exists()
    assert hashlib.sha256(Path(config['evaluator']).read_bytes()).hexdigest() == config['evaluator_sha256']
    status_path = Path(config['training_status'])
    training = json.loads(status_path.read_text()); worker = Path('/proc') / str(training['pid'])
    assert 'job_'+job+'/' in (worker/'cgroup').read_text() and b'worker.py' in (worker/'cmdline').read_bytes()
    apps = list(csv.reader(output('nvidia-smi', '--id='+uuid, '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits').splitlines(), skipinitialspace=True))
    assert {int(row[0]) for row in apps} == {training['pid']}, 'Unexpected GPU co-tenant'
    receipt = dict(started_utc=datetime.now(timezone.utc).isoformat(), job=job, gpu_uuid=uuid,
                   worker_before=training, allocation=allocation, optimizer_updates_by_diagnostic=0,
                   evaluator_sha256=config['evaluator_sha256'], resident_apps_before=apps)
    receipt_path = root / 'overlap-2805-receipt.json'
    def save(state, **values):
        receipt.update(state=state, updated_utc=datetime.now(timezone.utc).isoformat(), **values)
        temporary = receipt_path.with_suffix('.tmp'); temporary.write_text(json.dumps(receipt, indent=2)+'\n'); temporary.replace(receipt_path)
    def memory():
        data = output('nvidia-smi', '--id='+uuid, '--query-gpu=uuid,memory.free,memory.used', '--format=csv,noheader,nounits')
        gpu, free, used = next(csv.reader([data], skipinitialspace=True)); assert gpu == uuid
        return dict(free_mib=int(free), used_mib=int(used))
    request = phase / f'validation-frozen-ablation-{os.getpid()}.request'
    save('WAITING_EXCLUSIVE_PHASE_LOCK', memory_before=memory())
    request.touch()
    try:
        with (phase/'gpu-phase.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            deadline = time.monotonic()+30
            while memory()['free_mib'] < 20480:
                assert time.monotonic() < deadline, 'CC did not release enough cached GPU memory'
                time.sleep(.2)
            save('LOCKED_HEADROOM_VERIFIED', memory_after_cache_release=memory(),
                 cc_update_at_lock=json.loads(status_path.read_text())['update'])
            env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = uuid
            for arm in ('A', 'B'):
                item = config['arms'][arm]
                command = [sys.executable, '-u', config['evaluator'], '--source-root', config['source_root'],
                           '--probes', config['probes'],
                           '--checkpoint', item['checkpoint'], '--evaluations', item['evaluations'],
                           '--data-root', config['data_root'], '--manifest', config['manifest'],
                           '--output-root', str(root), '--arm', arm, '--seed', '1234']
                assert memory()['free_mib'] >= 20480
                save('EVALUATING_'+arm, command=command, memory_before_model=memory())
                with (root/f'{arm}.out').open('a') as log:
                    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900)
                assert result.returncode == 0, (arm, result.returncode)
            save('EVALUATIONS_COMPLETED_RELEASE_PENDING', memory_after=memory())
    except Exception as error:
        save('FAILED_LOCK_RELEASED', error=repr(error))
        raise
    finally:
        request.unlink(missing_ok=True)
    deadline = time.monotonic()+90
    while json.loads(status_path.read_text())['update'] <= receipt['cc_update_at_lock']:
        assert time.monotonic() < deadline, 'CC progress not yet observed after release'
        time.sleep(1)
    save('COMPLETED_CC_ADVANCING', worker_after=json.loads(status_path.read_text()))


if __name__ == '__main__':
    main(sys.argv[1])
