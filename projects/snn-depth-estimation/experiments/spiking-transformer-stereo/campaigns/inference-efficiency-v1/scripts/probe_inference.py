"""Slurm-only paired FP32 dense-CUDA full-model inference probe; no accuracy scoring."""
import argparse
import ctypes
import gc
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(EXPERIMENT / 'scripts'))
import probe_refinements as probe
import run_experiment as runner
import numpy as np
import torch


class EnergyCounter:
    """Use existing system NVML directly; absence/support failures are recorded."""
    def __init__(self, uuid):
        self.lib = None
        self.reason = None
        try:
            self.lib = ctypes.CDLL('libnvidia-ml.so.1')
            self.check(self.lib.nvmlInit_v2())
            self.handle = ctypes.c_void_p()
            self.check(self.lib.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(self.handle)))
            self.read()
        except (OSError, AttributeError, RuntimeError) as error:
            self.reason = str(error)
            self.lib = None

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError('NVML return code ' + str(code))

    def read(self):
        result = ctypes.c_ulonglong()
        self.check(self.lib.nvmlDeviceGetTotalEnergyConsumption(self.handle, ctypes.byref(result)))
        return result.value


def canonical_gpu_uuid(value):
    """Torch may return a bare UUID; NVML and nvidia-smi require GPU-prefix."""
    value = value.decode() if isinstance(value, bytes) else str(value)
    return value if value.startswith(('GPU-', 'MIG-')) else 'GPU-' + value


def percentile(values):
    return {name: float(np.percentile(values, q)) for name, q in [('p10', 10), ('median', 50), ('p90', 90), ('p95', 95)]} | {'mean': float(np.mean(values)), 'min': min(values), 'max': max(values), 'count': len(values)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['source-registry', 'manifest', 'data-root', 'lif-checkpoint', 'ann-checkpoint', 'protocol', 'output']:
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires Slurm CUDA; never execute on Mac or login')
    if torch.cuda.device_count() != 1 or '5090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Requires one RTX5090 allocation')
    protocol = json.loads(a.protocol.read_text())
    if protocol['refinements'] != 32 or protocol['batch_size'] != 2:
        raise ValueError('Frozen primary input/refinement budget differs')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(8)
    source = json.loads(a.source_registry.read_text())
    for relative, digest in source['source_files'].items():
        if probe.file_sha256(runner.PROJECT.parents[1] / relative) != digest:
            raise ValueError('Source registry mismatch: ' + relative)
    manifest_digest = probe.file_sha256(a.manifest)
    if manifest_digest != protocol['manifest_sha256']:
        raise ValueError('Frozen manifest mismatch')
    rows = json.loads(a.manifest.read_text())['dev'][:2]
    # Only spike files are decoded. Ground truth and Test files are never read.
    inputs = []
    for side in ('spike_l', 'spike_r'):
        decoded = [runner.get_block_spikes(str(a.data_root / row[side])).astype(np.float32) for row in rows]
        value = torch.from_numpy(np.stack(decoded)).cuda()
        if tuple(value.shape) != (2, 50, 250, 400):
            raise ValueError('Unexpected native input shape')
        inputs.append(value)
    torch.cuda.synchronize()
    gpu = torch.cuda.get_device_properties(0)
    uuid = canonical_gpu_uuid(gpu.uuid)
    energy = EnergyCounter(uuid)
    a.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(status='running', command=sys.argv, protocol=protocol, protocol_sha256=probe.file_sha256(a.protocol),
        script_sha256=probe.file_sha256(__file__), source=source, manifest_sha256=manifest_digest,
        development_frames=rows, input_shape=list(inputs[0].shape), input_sha256={row[side]: probe.file_sha256(a.data_root / row[side]) for row in rows for side in ('spike_l', 'spike_r')},
        hostname=socket.gethostname(), torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        gpu=dict(name=gpu.name, uuid=uuid, total_memory=gpu.total_memory, capability=list(torch.cuda.get_device_capability())),
        slurm={k: v for k, v in os.environ.items() if k.startswith('SLURM_')},
        precision=dict(parameter_dtype='float32', input_dtype='float32', autocast=False, matmul_tf32=False, cudnn_tf32=True, cudnn_benchmark=False, cudnn_deterministic=True),
        energy_counter=dict(available=energy.lib is not None, unavailable_reason=energy.reason, scope='device-wide NVML cumulative millijoules; includes board/background power, no idle subtraction'),
        checkpoints={}, visits=[], limitations=['One seed, two fixed development frames, one GPU; dense CUDA implementation only.', 'Per-sample latency is batch latency divided by 2, not independent batch1 response latency.', 'No accuracy scoring or neuromorphic-energy claim.'])
    def save():
        (a.output / 'probe.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    checkpoints = {}
    for name, path, model_name in [('lif', a.lif_checkpoint, 'spiking'), ('ann', a.ann_checkpoint, 'analog')]:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        c = checkpoint['config']
        if path.name != 'last.pt' or checkpoint['updates'] != 10000 or c['seed'] != 1234 or c['model'] != model_name or not c['normalize_residual_current']:
            raise ValueError('Expected normalized final10k seed1234 ' + name)
        if checkpoint['manifest_sha256'] != manifest_digest:
            raise ValueError('Checkpoint manifest mismatch')
        for relative, digest in checkpoint['source']['source_files'].items():
            if '/core/' in relative and source['source_files'].get(relative) != digest:
                raise ValueError('Exact checkpoint core mismatch: ' + relative)
        checkpoints[name] = checkpoint
        metadata['checkpoints'][name] = dict(path=str(path), sha256=probe.file_sha256(path), updates=checkpoint['updates'], config=c, training_source=checkpoint['source'])
    save()
    started = time.monotonic()
    with torch.inference_mode():
        for block in range(3):
            # ABBA / BAAB / ABBA gives each method equal positions within every block.
            order = ['lif', 'ann', 'ann', 'lif'] if block % 2 == 0 else ['ann', 'lif', 'lif', 'ann']
            for position, name in enumerate(order):
                if time.monotonic() - started > 360:
                    raise RuntimeError('Six-minute measured-work guard exceeded')
                # Other users/processes would invalidate isolation: fail instead of fabricating a comparison.
                processes = subprocess.check_output(['nvidia-smi', '-i', uuid, '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).splitlines()
                if any(pid.strip().isdigit() and int(pid) != os.getpid() for pid in processes):
                    raise RuntimeError('Other compute process on measured GPU: ' + str(processes))
                model = runner.build_model('spiking' if name == 'lif' else 'analog', 1234, True).cuda().eval()
                model.load_state_dict(checkpoints[name]['model'], strict=True)
                for _ in range(5):
                    prediction = runner.forward(model, *inputs, 32)
                    del prediction
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                baseline = dict(allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved())
                latencies, walls = [], []
                for repetition in range(5):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    wall = time.perf_counter()
                    begin.record()
                    prediction = runner.forward(model, *inputs, 32)
                    end.record()
                    end.synchronize()
                    walls.append((time.perf_counter() - wall) * 1000)
                    latencies.append(begin.elapsed_time(end))
                    if not torch.isfinite(prediction).all():
                        raise FloatingPointError('Nonfinite inference output')
                    del prediction
                peak = dict(allocated_bytes=torch.cuda.max_memory_allocated(), reserved_bytes=torch.cuda.max_memory_reserved())
                measured_energy = dict(available=False, reason=energy.reason or 'Counter block too short')
                if energy.lib is not None:
                    torch.cuda.synchronize()
                    before = energy.read()
                    begin_wall = time.perf_counter()
                    count = 0
                    while time.perf_counter() - begin_wall < 5 and count < 100:
                        prediction = runner.forward(model, *inputs, 32)
                        del prediction
                        torch.cuda.synchronize()
                        count += 1
                    after = energy.read()
                    seconds = time.perf_counter() - begin_wall
                    if seconds >= 5 and after > before:
                        measured_energy = dict(available=True, millijoules=after-before, seconds=seconds, batches=count, samples=2*count, joules_per_sample=(after-before)/1000/(2*count))
                    else:
                        measured_energy = dict(available=False, reason='Counter did not advance or capped block shorter than5s', seconds=seconds, batches=count)
                metadata['visits'].append(dict(block=block, position=position, model=name, parameters=sum(v.numel() for v in model.parameters()),
                    baseline_memory=baseline, peak_memory=peak, batch_cuda_ms=latencies, amortized_sample_cuda_ms=[x/2 for x in latencies], batch_host_wall_ms=walls, energy=measured_energy))
                runner.reset_model(model)
                del model
                gc.collect()
                torch.cuda.empty_cache()
                save()
                print(block, position, name, percentile(latencies), flush=True)
    metadata['summary'] = {}
    for name in ('lif', 'ann'):
        visits = [v for v in metadata['visits'] if v['model'] == name]
        metadata['summary'][name] = dict(batch_cuda_ms=percentile([t for v in visits for t in v['batch_cuda_ms']]),
            amortized_sample_cuda_ms=percentile([t for v in visits for t in v['amortized_sample_cuda_ms']]),
            peak_allocated_bytes=max(v['peak_memory']['allocated_bytes'] for v in visits), peak_reserved_bytes=max(v['peak_memory']['reserved_bytes'] for v in visits))
    metadata['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
