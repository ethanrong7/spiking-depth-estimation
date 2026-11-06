#!/usr/bin/env python3
"""Continue an unchanged frozen trainer inside one finite GPU allocation."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import threading


def run_budget(command, output, seconds, *, clock=time.time, spawn=subprocess.Popen, health_interval=0):
    if not 600 <= seconds <= 30 * 86400:
        raise ValueError('continuation must have a finite budget of at most 30 days')
    end = clock() + seconds
    if os.environ.get('SLURM_JOB_END_TIME'):
        end = min(end, float(os.environ['SLURM_JOB_END_TIME']))
    stopping = [False]
    health_failed = [False]
    child = [None]

    def stop(*_):
        stopping[0] = True
        if child[0] is not None and child[0].poll() is None:
            child[0].send_signal(signal.SIGUSR1)

    for number in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(number, stop)
    health_done = threading.Event()
    if health_interval:
        from first_hour_health import check_health
        def monitor():
            last_step = 0
            records = []
            for check in range(1, 5):
                if health_done.wait(health_interval):
                    return
                try:
                    record = check_health(output, last_step)
                except Exception as error:
                    record = dict(step=last_step, passed=False, errors=[str(error)])
                record['minute'] = check * 15
                records.append(record)
                temporary = output / 'health.tmp'
                temporary.write_text(json.dumps({'checks': records, 'first_hour_passed': check == 4 and record['passed']}, indent=2))
                temporary.replace(output / 'health.json')
                print(json.dumps({'event': 'health_check', **record}), flush=True)
                if not record['passed']:
                    health_failed[0] = True
                    stop()
                    return
                last_step = record['step']
        threading.Thread(target=monitor, daemon=True).start()
    while not stopping[0]:
        remaining = int(end - clock())
        if remaining < 600 or (output / 'STOP').exists():
            return 0
        environment = os.environ.copy()
        environment.pop('SPIKESTEREO_SEGMENT_DEADLINE', None)
        # The frozen trainer verifies and restores its full state on every restart.
        child[0] = spawn(command + ['--segment-seconds', str(min(18000, remaining))],
                         env=environment)
        result = child[0].wait()
        if health_failed[0]:
            return 2
        child[0] = None
        if result:
            return result
        state = json.loads((output / 'status.json').read_text())
        if state['status'] != 'PAUSED_BUDGET':
            return 0
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--budget-seconds', type=int, required=True)
    parser.add_argument('--allow-fresh', action='store_true')
    parser.add_argument('--health-interval', type=int, default=0)
    args = parser.parse_args()
    project = Path(os.environ['SPIKESTEREO_PROJECT_ROOT'])
    output = Path(os.environ['SPIKESTEREO_OUTPUT'])
    if not args.allow_fresh and not (output / 'latest.pth').is_file():
        raise RuntimeError('continuation requires an existing full-state checkpoint')
    command = [sys.executable, '-u', str(project / 'experiments/sew-t50-training/scripts/train.py'),
        '--config', os.environ['SPIKESTEREO_CONFIG'], '--data-root', os.environ['SPIKESTEREO_DATA_ROOT'],
        '--output', str(output), '--provenance', os.environ['SPIKESTEREO_PROVENANCE']]
    raise SystemExit(run_budget(command, output, args.budget_seconds, health_interval=args.health_interval))


if __name__ == '__main__':
    main()
