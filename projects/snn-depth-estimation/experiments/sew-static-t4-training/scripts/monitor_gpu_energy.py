"""Read-only GPU board-power telemetry for existing Slurm jobs (no CUDA work)."""
import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
import subprocess
import time


def command(args):
    return subprocess.check_output(args, text=True, timeout=15).strip()


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--max-hours', type=float, default=24 * 22)
    args = parser.parse_args()
    started = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    runs = json.loads(args.registry.read_text())['runs']
    for run in runs:
        out = Path(run['output']) / ('gpu-energy-' + started)
        out.mkdir(exist_ok=False)
        run.update(out=out, energy_j=0., observed_seconds=0., previous=None,
                   first_step=None, last_step=None, samples=0, skipped=0)
        run['file'] = (out / 'samples.csv').open('w', buffering=1)
        run['writer'] = csv.writer(run['file'])
        run['writer'].writerow(['utc', 'monotonic_s', 'job_id', 'gpu_uuid',
                               'trainer_pid', 'step', 'status', 'power_w',
                               'utilization_pct', 'integrated_energy_j'])
        atomic_json(out / 'metadata.json', {
            'variant': run['variant'], 'job_id': run['job_id'],
            'run_output': run['output'], 'start_utc': started,
            'method': 'nvidia-smi power.draw, trapezoidal integration',
            'scope': 'gross GPU board energy; includes validation and I/O waits; excludes CPU/RAM/PSU',
            'interval_seconds': args.interval, 'historical_energy_available': False,
            'attribution': 'trainer PID cgroup Slurm job; exclusive compute-process GPU',
            'registry': str(args.registry), 'script': str(Path(__file__).resolve()),
        })
    deadline = time.monotonic() + args.max_hours * 3600
    active = {str(r['job_id']) for r in runs}
    next_scheduler = 0.
    while active and time.monotonic() < deadline:
        tick = time.monotonic()
        try:
            if tick >= next_scheduler:
                rows = command(['squeue', '-h', '-j', ','.join(active), '-o', '%i %T'])
                active = {row.split()[0] for row in rows.splitlines()
                          if row.split()[1] in ('RUNNING', 'COMPLETING')}
                next_scheduler = tick + 30
            gpu_rows = csv.reader(command([
                'nvidia-smi', '--query-gpu=uuid,power.draw,utilization.gpu',
                '--format=csv,noheader,nounits']).splitlines())
            powers = {r[0].strip(): (float(r[1]), float(r[2])) for r in gpu_rows}
            processes = list(csv.reader(command([
                'nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                '--format=csv,noheader,nounits']).splitlines()))
            owners = {}
            for pid, gpu in processes:
                try:
                    group = Path('/proc') / pid.strip() / 'cgroup'
                    owners[pid.strip()] = (gpu.strip(), group.read_text())
                except (FileNotFoundError, PermissionError):
                    pass
            for run in runs:
                if str(run['job_id']) not in active:
                    continue
                matches = [(pid, gpu) for pid, (gpu, group) in owners.items()
                           if '/job_' + str(run['job_id']) + '/' in group]
                if len(matches) != 1:
                    run['previous'] = None
                    run['skipped'] += 1
                    continue
                pid, gpu = matches[0]
                if sum(row[1].strip() == gpu for row in processes) != 1:
                    run['previous'] = None
                    run['skipped'] += 1
                    continue
                watts, utilization = powers[gpu]
                if not math.isfinite(watts) or watts < 0:
                    raise ValueError('Invalid GPU power')
                status = json.loads((Path(run['output']) / 'status.json').read_text())
                step = int(status['step'])
                now = time.monotonic()
                previous = run['previous']
                if previous and previous[2] == gpu and now - previous[0] <= args.interval * 3:
                    elapsed = now - previous[0]
                    run['energy_j'] += (watts + previous[1]) * .5 * elapsed
                    run['observed_seconds'] += elapsed
                run['previous'] = (now, watts, gpu)
                if run['first_step'] is None:
                    run['first_step'] = step
                run['last_step'] = step
                run['samples'] += 1
                run['writer'].writerow([
                    dt.datetime.now(dt.timezone.utc).isoformat(), now,
                    run['job_id'], gpu, pid, step, status.get('status'), watts,
                    utilization, run['energy_j']])
                updates = step - run['first_step']
                atomic_json(run['out'] / 'summary.json', {
                    'start_utc': started, 'last_sample_step': step,
                    'first_sample_step': run['first_step'], 'updates_elapsed': updates,
                    'samples': run['samples'], 'skipped_samples': run['skipped'],
                    'observed_seconds': run['observed_seconds'],
                    'estimated_gpu_energy_j': run['energy_j'],
                    'estimated_gpu_energy_kwh': run['energy_j'] / 3_600_000,
                    'average_gpu_power_w': run['energy_j'] / run['observed_seconds']
                        if run['observed_seconds'] else None,
                    'gross_j_per_elapsed_update': run['energy_j'] / updates if updates else None,
                    'last_sample_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                    'coverage_note': 'Intervals with missing attribution are omitted; J/update is valid only with continuous coverage.',
                })
        except Exception as error:
            print(dt.datetime.now(dt.timezone.utc).isoformat(), repr(error), flush=True)
            for run in runs:
                run['previous'] = None
        time.sleep(max(.1, args.interval - (time.monotonic() - tick)))
    for run in runs:
        atomic_json(run['out'] / 'monitor-ended.json', {
            'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'reason': 'jobs no longer active' if not active else 'finite monitor deadline',
        })
        run['file'].close()


if __name__ == '__main__':
    main()
