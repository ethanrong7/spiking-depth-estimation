"""Two-case CPU diagnostic of the admitted trained B model; no training."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '2'
import importlib.util
import argparse
import json
from pathlib import Path
import signal
import sys
import time
from types import MethodType
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BASE = REPO / 'storage/experiment-runs/snn-depth-estimation'
RAW = BASE / 'success-mechanism-cpu-audit-v1'
STEREO = BASE / 'spiking-transformer-stereo'
CONDITIONS = ('baseline', 'bypass', 'raw_match_Y_moments', 'Y_match_X_moments', 'uniform', 'tanh')
RESULT_NAME = 'B.json'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(result):
    (HERE / 'results').mkdir(exist_ok=True)
    (HERE / 'results' / RESULT_NAME).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


def main():
    global RESULT_NAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--panel', choices=('main','split'), default='main')
    args = parser.parse_args()
    conditions = CONDITIONS if args.panel == 'main' else ('baseline','shift_only','scale_only')
    RESULT_NAME = 'B.json' if args.panel == 'main' else 'B-calibration-split.json'
    sys.path.insert(0, str(HERE.parent / 'trained-cost-filter-cpu-controls-v1'))
    old_runner = load('previous_cpu_runner', HERE.parent / 'trained-cost-filter-cpu-controls-v1/run.py')
    from controls import fit_control
    source = BASE / 'trained-cost-filter-cpu-controls-v1/source'
    project = source / 'projects/snn-depth-estimation'
    registry = json.loads((source / 'source-registry.json').read_text())
    for rel, sha in registry['source_files'].items():
        assert old_runner.digest(source / rel) == sha, rel
    checkpoint = STEREO / 'spiking-attention-placement-v1/train/B/step-30000.pt'
    expected = '126caf6fe5c334bf6c143d91db9e4046f6a48cfe328380f9701b576cef391b84'
    assert old_runner.digest(checkpoint) == expected
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    sys.path[:0] = [str(project), str(project / 'core'), str(project / 'experiments/spiking-transformer-stereo/scripts')]
    driver = load('B_cpu_driver', project / 'experiments/spiking-transformer-stereo/scripts/run_experiment.py')
    probes = load('B_cpu_metrics', HERE.parent / 'spiking-attention-placement-v1/probes.py')
    from core.spiking_attention_placement import install
    from core.corr import CorrBlock1D
    from core.epipolar_spiking_stereo import signed_rate
    import cv2
    cv2.setNumThreads(1)
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert ck['updates'] == 30000 and ck['samples_seen'] == 240000
    assert ck['source_identity'] == registry['source_identity_sha256']
    model = install(driver.build_model('original', 1234), 'B')
    model.load_state_dict(ck['model'], strict=True)
    del ck
    model.eval()
    original_state = old_runner.state_digest(model)
    manifest = json.loads((project / 'experiments/spiking-transformer-stereo/configs/manifest.json').read_text())
    cohort = [next(r for r in manifest['dev'] if r['scene'] == s and r['frame'] == '0001') for s in ('scene_0120', 'scene_0123')]
    dataset = driver.ManifestDataset(REPO / 'data/Synthetic', cohort)
    RAW.mkdir(parents=True, exist_ok=True)
    output = RAW / args.run_id
    output.mkdir(exist_ok=False)
    result = dict(status='running', checkpoint_sha256=expected, source_identity=registry['source_identity_sha256'],
                  evaluator_sha256=old_runner.digest(__file__), torch=torch.__version__, cpu_threads=2,
                  updates=30000, seed=1234, iterations=32, gpu_used=False, training_steps=0,
                  scope='two selected cases; within-CPU intervention, not benchmark or historical GPU parity', rows=[])
    native_corr = CorrBlock1D.corr
    native_compute = model.placement.compute
    native_codes = model.placement.codes.forward
    raw_pair = []
    stats = []
    coding_stats = []
    condition = 'baseline'

    def capture(module, args):
        raw_pair[:] = args

    def corr(left, right):
        y = native_corr(left, right)
        x = native_corr(*raw_pair)
        mode = condition if condition in CONDITIONS[:4] else 'baseline'
        out, diagnostic = fit_control(x, y, mode)
        if condition in ('shift_only','scale_only'):
            xd, yd = x.double(), y.double()
            mx, my = xd.mean(-1,keepdim=True), yd.mean(-1,keepdim=True)
            if condition == 'shift_only':
                out = (xd+my-mx).to(x.dtype)
            else:
                sx, sy = xd.std(-1,keepdim=True,unbiased=False), yd.std(-1,keepdim=True,unbiased=False)
                out = torch.where(sx>1e-8,mx+sy/sx.clamp_min(1e-8)*(xd-mx),xd).to(x.dtype)
        diagnostic['mean_raw_row_std'] = x.std(-1, unbiased=False).mean().item()
        diagnostic['mean_learned_row_std'] = y.std(-1, unbiased=False).mean().item()
        diagnostic['raw_mean'] = x.mean().item()
        diagnostic['learned_mean'] = y.mean().item()
        stats.append(diagnostic)
        return out

    def codes(self, x):
        z = self.norm(x)
        currents = [layer(z) for layer in (self.q, self.k, self.v)]
        coded = [signed_rate(c, 4) for c in currents]
        if condition == 'baseline':
            coding_stats.append({name: dict(zero_fraction=(s == 0).float().mean().item(),
                saturated_fraction=(s.abs() == 1).float().mean().item(),
                current_abs_mean=c.abs().mean().item(),
                signed_tanh_MAE=(s-c.tanh()).abs().mean().item())
                for name, c, s in zip(('q','k','v'), currents, coded)})
        return tuple(c.tanh() for c in currents) if condition == 'tanh' else tuple(coded)

    def uniform(self, left, right):
        l, r = left.permute(0,2,3,1), right.permute(0,2,3,1)
        _, _, vl = self.codes(self.input(l))
        _, _, vr = self.codes(self.input(r))
        a = l + self.output(vr.mean(-2, keepdim=True).expand_as(vl))
        b = r + self.output(vl.mean(-2, keepdim=True).expand_as(vr))
        return a.permute(0,3,1,2).contiguous(), b.permute(0,3,1,2).contiguous()

    hook = model.placement.register_forward_pre_hook(capture)
    CorrBlock1D.corr = staticmethod(corr)
    model.placement.codes.forward = MethodType(codes, model.placement.codes)
    def timeout(*args):
        raise TimeoutError('300-second CPU forward budget')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(200 if args.panel == 'split' else 300)
    started = time.monotonic()
    try:
        with torch.no_grad():
            for index, label in enumerate(cohort):
                left, right, target, valid, _ = dataset[index]
                left, right, target, valid = [v[None] for v in (left, right, target, valid)]
                for condition in conditions:
                    stats.clear()
                    coding_stats.clear()
                    model.placement.compute = MethodType(uniform, model.placement) if condition == 'uniform' else native_compute
                    prediction = driver.forward(model, left, right, 32)
                    driver.reset_model(model)
                    metrics = probes.frame_metrics(prediction, target, valid)[0]
                    error = (prediction-target).abs().squeeze(1)[valid.bool()]
                    row = dict(scene=label['scene'], frame=label['frame'], condition=condition,
                        metrics=metrics, capped5=error.clamp(max=5).mean().item(),
                        excess5=(error-5).clamp(min=0).mean().item(), cost=stats[-1], coding=list(coding_stats))
                    assert abs(row['capped5']+row['excess5']-metrics['full']['epe']) < 1e-5
                    if condition == 'baseline':
                        repeat = driver.forward(model, left, right, 32)
                        driver.reset_model(model)
                        row['exact_baseline_repeat'] = torch.equal(prediction, repeat)
                        assert row['exact_baseline_repeat']
                    torch.save(prediction.cpu(), output / f"{label['scene']}-{condition}.pt")
                    result['rows'].append(row)
                    save(result)
                    print(json.dumps({k:row[k] for k in ('scene','condition','capped5','excess5')} | {'epe':metrics['full']['epe'],'bad3':metrics['full']['bad3']}), flush=True)
        result['status'] = 'complete'
    except Exception as exc:
        result['status'] = 'failed'
        result['error'] = repr(exc)
        raise
    finally:
        signal.alarm(0)
        hook.remove()
        CorrBlock1D.corr = staticmethod(native_corr)
        model.placement.compute = native_compute
        model.placement.codes.forward = native_codes
        result['elapsed_seconds'] = time.monotonic()-started
        result['weights_unchanged'] = old_runner.state_digest(model) == original_state
        result['cuda_initialized'] = torch.cuda.is_initialized()
        save(result)


if __name__ == '__main__':
    main()
