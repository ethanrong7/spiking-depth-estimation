"""External, parity-gated seed1234 endpoint lookup diagnostic. Never submits jobs.

Requires an existing authorized single 4090/5090 Slurm allocation. Importing this
module and running its synthetic tests do not load data/models or require CUDA.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch

EPSILON = .25
STABILIZER = 1e-8
ANCHORS = (-2., 0., 2.)
PREFIXES = (1, 2, 4, 8, 16, 32)
PROBE_SHAS = {'da0fa6ab64de0649af1e7afc9c586e79fc0ccc83740824431713e1a8eb766098',
              '45876d358b3420e328ad4f5937764b2fa637109438e27a7a67d12d46b7c4e979'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def state_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def backend_flags():
    return dict(autocast=torch.is_autocast_enabled(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic)


def moments(value, support):
    x = value.detach().double()[support]
    n = x.numel()
    total, squares, absolute = x.sum().item(), x.square().sum().item(), x.abs().sum().item()
    return dict(count=n, sum=total, sum_sq=squares, abs_sum=absolute,
                mean=total/n if n else None, rms=(squares/n)**.5 if n else None)


def error_decomposition_tau5(probes, prediction, target, valid):
    """Exact error split on frozen full/inview/oov masks, prefix32 only.

    Start with the same float32 absolute errors used by frame_metrics; sum the
    two terms in float64. The official EPE remains the frozen float32 reducer's
    result, whose tiny reduction-rounding difference is checked separately.
    """
    prediction, target, valid = [v.detach().cpu() for v in (prediction, target, valid)]
    reference = probes.frame_metrics(prediction, target, valid)
    error = (prediction[:, 0] - target[:, 0]).abs().double()
    x = torch.arange(target.shape[-1])[None, None, :]
    inside = (x + target[:, 0] >= 0) & (x + target[:, 0] <= target.shape[-1]-1)
    result = []
    for b in range(target.shape[0]):
        strata = {}
        for name, mask in [('full', valid[b]), ('inview', valid[b] & inside[b]), ('oov', valid[b] & ~inside[b])]:
            values = error[b]
            stats = dict(absolute_error=moments(values, mask),
                capped_error=moments(values.clamp_max(5), mask),
                excess_error=moments((values-5).clamp_min(0), mask),
                over5_fraction=moments((values > 5).double(), mask))
            count = stats['absolute_error']['count']
            assert count == reference[b][name]['valid_pixels']
            residual = stats['absolute_error']['sum'] - stats['capped_error']['sum'] - stats['excess_error']['sum']
            assert abs(residual) <= 1e-12 * max(1, stats['absolute_error']['sum'])
            rounding = abs(stats['absolute_error']['mean']-reference[b][name]['epe']) if count else None
            assert rounding is None or rounding <= max(1e-5, abs(reference[b][name]['epe'])*1e-6)
            strata[name] = dict(threshold_px=5, statistics=stats, identity_sum_residual=residual,
                official_epe=reference[b][name]['epe'],
                official_epe_reduction_rounding_difference=rounding)
        result.append(strata)
    return result


def candidate_pyramid(candidate, levels):
    result = [candidate.bool()]
    for _ in range(levels - 1):
        result.append(result[-1].unfold(0, 2, 2).all(-1))
    return result


def native_neighbors(x, candidate):
    """ALL interpolation contributors native; at an integer floor=ceil."""
    lo, hi = x.floor().long(), x.ceil().long()
    inside = (lo >= 0) & (hi < candidate.numel())
    return inside & candidate[lo.clamp(0, candidate.numel()-1)] & candidate[hi.clamp(0, candidate.numel()-1)]


def anchor_coords(coords, geometry, offset):
    result = coords.detach().clone()
    result[:, 0] = geometry['right_cells'] + offset
    return result


def profile_records(block, coords, output, lower_coords, lower_output, upper_coords,
                    upper_output, geometry, labels, regime, offset):
    """Per-frame sufficient statistics, never pixel-based significance tests."""
    b, _, h, w = output.shape
    k = 2*block.radius + 1
    expected = (b, block.num_levels*k, h, w)
    assert all(t.shape == expected and torch.isfinite(t).all()
               for t in (output, lower_output, upper_output))
    # Aggregate diagnostic tensors on CPU, avoiding many tiny synchronizations
    # for each moment/stratum while keeping the actual prediction on its device.
    c, lo, hi = [v.detach().cpu().double().reshape(b, block.num_levels, k, h, w)
                 for v in (output, lower_output, upper_output)]
    coords, lower_coords, upper_coords = [v.detach().cpu() for v in (coords, lower_coords, upper_coords)]
    geometry = {name: value.detach().cpu() for name, value in geometry.items()}
    a = (coords[:, :1].double() - lower_coords[:, :1].double())[:, None]
    z = (upper_coords[:, :1].double() - coords[:, :1].double())[:, None]
    nonzero = (a > 0) & (z > 0)
    sa, sz = [torch.where(nonzero, t, torch.ones_like(t)) for t in (a, z)]
    slope = ((sa/sz)*(hi-c) + (sz/sa)*(c-lo))/(sa+sz)
    curve = 2*((hi-c)/sz - (c-lo)/sa)/(sa+sz)
    candidates = candidate_pyramid(geometry['candidate_mask'], block.num_levels)
    delta = torch.arange(-block.radius, block.radius+1, device=coords.device).view(1, k, 1, 1)
    records = []
    for level in range(block.num_levels):
        scale = 2**level
        x, xl, xu = [(v[:, :1]/scale + delta) for v in (coords, lower_coords, upper_coords)]
        width = block.corr_pyramid[level].shape[-1]
        assert candidates[level].numel() == width
        tensor = (xl >= 0) & (xu <= width-1)
        unpadded = tensor & native_neighbors(x, candidates[level]) & native_neighbors(xl, candidates[level]) & native_neighbors(xu, candidates[level])
        rms = c[:, level].square().mean(1).sqrt()
        denominator = rms.clamp_min(STABILIZER)[:, None]
        frac = (x - x.floor()).double()
        knot = (xl.floor() != xu.floor()).double()
        for stratum, left in [('native_valid', geometry['support']), ('inview', geometry['inview']),
                               ('oov', geometry['support'] & ~geometry['inview'])]:
            if regime == 'gt_anchor' and stratum != 'inview':
                continue
            for support_type, right in [('returned', torch.ones_like(tensor)),
                                         ('tensor_inbounds', tensor), ('unpadded', unpadded)]:
                mask = left[:, None] & right
                fdmask = mask & nonzero[:, 0]
                whole = mask.all(1)
                fdwhole = fdmask.all(1)
                normalized_support = fdwhole[:, None].expand_as(fdmask)
                fields = {
                    'lookup': (c[:, level], mask),
                    'slope': (slope[:, level], fdmask),
                    'finite_difference_curvature': (curve[:, level], fdmask),
                    'normalized_slope': (slope[:, level]/denominator, normalized_support),
                    'normalized_finite_difference_curvature': (curve[:, level]/denominator, normalized_support),
                    'fractional_part': (frac, mask),
                    'stencil_knot_crossing': (knot, fdmask),
                    'profile_rms': (rms, whole),
                    'profile_std': (c[:, level].std(1, unbiased=False), whole),
                    'profile_range': (c[:, level].amax(1)-c[:, level].amin(1), whole),
                    'tiny_profile_scale': ((rms <= STABILIZER).double(), whole),
                    'zero_fd_step': ((~nonzero[:, 0, 0]).double(), whole),
                }
                for low, high in ((0., .25), (.25, .5), (.5, .75), (.75, 1.)):
                    fields[f'fractional_bin_{low:g}_{high:g}'] = (((frac >= low) & (frac < high)).double(), mask)
                for index in range(b):
                    records.append(dict(**labels[index], regime=regime, anchor_offset=offset,
                        level=level, left_stratum=stratum, support_type=support_type,
                        left_site_count=int(left[index].sum()), profile_count=int(whole[index].sum()),
                        finite_difference_profile_count=int(fdwhole[index].sum()),
                        statistics={name: moments(value[index], valid[index]) for name, (value, valid) in fields.items()}))
    return records


def reduce_records(records):
    """Equal-frame, then equal-scene means of each per-frame moment mean.

    Keep raw per-frame sums/sumsq in the payload for other declared reductions.
    No pooling of pixels or repeated prefixes as independent observations.
    """
    groups = defaultdict(list)
    keys = ('prefix', 'iteration', 'regime', 'anchor_offset', 'level', 'left_stratum', 'support_type')
    for row in records:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, rows in groups.items():
        assert len({(r['scene'], r['frame']) for r in rows}) == len(rows)
        summary = {}
        for name in rows[0]['statistics']:
            by_scene = defaultdict(list)
            for row in rows:
                metric = row['statistics'][name]
                if metric['count']:
                    by_scene[row['scene']].append(metric['mean'])
            scenes = {s: sum(v)/len(v) for s, v in by_scene.items()}
            summary[name] = dict(per_scene=scenes, mean_scene=sum(scenes.values())/len(scenes) if scenes else None,
                defined_frames=sum(len(v) for v in by_scene.values()), defined_scenes=len(scenes))
        result.append(dict(zip(keys, key), statistics=summary))
    return result


class LookupObserver:
    def __init__(self, probes, target, valid, padding, labels):
        self.probes, self.target, self.valid = probes, target, valid
        self.padding, self.labels = padding, labels
        self.prefix, self.iteration = None, 0
        self.records, self.traces = [], []
        self.geometry = None

    def collect(self, block, call, coords, center, geometry, regime, offset, iteration):
        lower, upper = coords.detach().clone(), coords.detach().clone()
        lower[:, 0] -= EPSILON
        upper[:, 0] += EPSILON
        lo, hi = call(block, lower), call(block, upper)
        labels = [dict(label, prefix=self.prefix, iteration=iteration) for label in self.labels]
        self.records.extend(profile_records(block, coords, center, lower, lo, upper, hi,
                                             geometry, labels, regime, offset))
        self.traces.append(dict(prefix=self.prefix, iteration=iteration, regime=regime, anchor_offset=offset,
                                coords=coords.detach().cpu().clone(), lower=lower.detach().cpu(), upper=upper.detach().cpu()))

    @contextmanager
    def installed(self, native):
        original = native.CorrBlock1D
        call = original.__call__
        observer = self

        class Observed(original):
            def __call__(self, coords):
                output = call(self, coords)
                observer.iteration += 1
                if observer.prefix == 32:
                    if observer.geometry is None:
                        observer.geometry = observer.probes.fine_geometry(observer.target, observer.valid,
                            observer.padding, coords.shape[-2:], stride=4)
                        identity = hashlib.sha256()
                        for key in ('support', 'inview', 'right_cells', 'candidate_mask'):
                            identity.update(observer.geometry[key].cpu().contiguous().numpy().tobytes())
                        observer.labels = [dict(label, batch_geometry_sha256=identity.hexdigest()) for label in observer.labels]
                    geometry = observer.geometry
                    observer.collect(self, call, coords, output, geometry, 'actual_trajectory', None, observer.iteration)
                    if observer.iteration == 32:
                        for offset in ANCHORS:
                            anchor = anchor_coords(coords, geometry, offset)
                            observer.collect(self, call, anchor, call(self, anchor), geometry, 'gt_anchor', offset, None)
                return output

        native.CorrBlock1D = Observed
        try:
            yield
        finally:
            native.CorrBlock1D = original


@contextmanager
def capture_predictions(driver, predictions, observer=None):
    original = driver.forward

    def forward(model, left, right, iters, *args, **kwargs):
        if observer:
            observer.prefix, observer.iteration = iters, 0
        prediction = original(model, left, right, iters, *args, **kwargs)
        predictions[iters] = prediction.detach().cpu().clone()
        if observer:
            assert observer.iteration == iters, (iters, observer.iteration)
        return prediction

    driver.forward = forward
    try:
        yield
    finally:
        driver.forward = original


def official_parity(probes, frames, official):
    result = probes.assert_parity(frames, official)
    keyed = {(r['scene'], r['frame']): r for r in official}
    differences = [abs(row['iterations']['32']['full'][metric] - keyed[(row['scene'], row['frame'])][metric])
                   for row in frames for metric in ('bad1', 'bad3', 'bad5')]
    assert max(differences) <= 1e-5
    result['max_frame_bad_difference'] = max(differences)
    return result


def admit(checkpoint, registry, receipt, arm, checkpoint_sha):
    """Mirror frozen phase2 replay lineage admission, bounded to seed1234/30k."""
    config = checkpoint['config']
    assert config['seed'] == 1234 and checkpoint['updates'] == 30000
    assert checkpoint['samples_seen'] == 240000
    if receipt:
        assert arm == 'SC' and receipt['status'] == 'VERIFIED_REUSE'
        assert (receipt['seed'], receipt['arm'], receipt['branch_seed']) == (1234, 'SC', 2017)
        assert receipt['current_source_identity'] == registry['source_identity_sha256']
        assert receipt['stream_audit']['actual_indices_hv_lr_verified']
        assert receipt['common_initial_state_sha256'] == receipt['native_initial_state_sha256']
        assert config['arm'] == receipt['historical_arm'] == 'A'
        endpoint = [r for r in receipt['endpoints'] if r['update'] == 30000]
        assert len(endpoint) == 1 and endpoint[0]['checkpoint_sha256'] == checkpoint_sha
        assert endpoint[0]['archived_source_identity'] == checkpoint['source_identity'] == receipt['archived_source_identity']
        return endpoint[0]['evaluation']
    assert config['arm'] == arm and config['branch_seed'] == 2017
    assert checkpoint['source_identity'] == registry['source_identity_sha256']
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'checkpoint', 'evaluations', 'data-root', 'manifest', 'output-root', 'probes'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--arm', choices=('SC', 'SP', 'CC', 'CP'), required=True)
    parser.add_argument('--reuse-receipt', type=Path)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert any(name in torch.cuda.get_device_name() for name in ('4090', '5090'))
    output = args.output_root/f'seed-1234/{args.arm}/update-30000'
    output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(args.source_root), str(args.source_root/'core'),
                   str(args.source_root/'experiments/spiking-transformer-stereo/scripts')]
    registry_path = args.source_root.parents[1]/'source-registry.json'
    registry = json.loads(registry_path.read_text())
    for rel, expected in registry['source_files'].items():
        assert digest(args.source_root.parents[1]/rel) == expected, rel
    assert digest(args.probes) in PROBE_SHAS, 'use admitted external placement probes'
    probes = load_module('lookup_frozen_probes', args.probes)
    assert digest(args.manifest) == probes.MANIFEST_SHA
    driver = load_module('lookup_frozen_driver', args.source_root/'experiments/spiking-transformer-stereo/scripts/run_experiment.py')
    from core.spiking_attention_phase2 import install
    import core.spikestereonet as native
    from core.utils.utils import InputPadder
    import cv2
    torch.set_num_threads(6)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    checkpoint_sha = digest(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    reuse = json.loads(args.reuse_receipt.read_text()) if args.reuse_receipt else None
    admitted_evaluation = admit(checkpoint, registry, reuse, args.arm, checkpoint_sha)
    official = [json.loads(line) for line in args.evaluations.read_text().splitlines() if line.strip()]
    official = [r for r in official if r['update'] == 30000]
    assert len(official) == 1 and official[0]['samples_seen'] == 240000
    if admitted_evaluation:
        assert official[0] == admitted_evaluation
    model = install(driver.build_model('original', 1234), args.arm, seed=2017)
    model.load_state_dict(checkpoint['model'], strict=True)
    assert all(torch.equal(t.cpu(), checkpoint['model'][n]) for n, t in model.state_dict().items())
    checkpoint_source_identity = checkpoint['source_identity']
    del checkpoint
    model = model.cuda().eval()
    assert all(p.dtype == torch.float32 for p in model.parameters())
    expected_backend = dict(autocast=False, matmul_tf32=False, cudnn_tf32=True,
                            cudnn_benchmark=False, cudnn_deterministic=True)
    assert backend_flags() == expected_backend
    initial_state = state_digest(model)
    manifest = json.loads(args.manifest.read_text())
    assert len(manifest['dev']) == 60
    dataset = driver.ManifestDataset(args.data_root, manifest['dev'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=0,
        generator=torch.Generator().manual_seed(2225))
    provenance = dict(arm=args.arm, seed=1234, update=30000, training_steps=0,
        source_identity=registry['source_identity_sha256'], checkpoint_sha256=checkpoint_sha,
        checkpoint_source_identity=checkpoint_source_identity,
        checkpoint=str(args.checkpoint), source_root=str(args.source_root),
        source_git_commit=registry['git_commit'], source_dirty_status=registry['dirty_status'],
        source_registry_sha256=digest(registry_path), manifest_sha256=digest(args.manifest),
        evaluations_sha256=digest(args.evaluations), probes_sha256=digest(args.probes),
        evaluator_sha256=digest(__file__), reuse_receipt_sha256=digest(args.reuse_receipt) if reuse else None,
        slurm_job_id=os.environ['SLURM_JOB_ID'], gpu=torch.cuda.get_device_name(),
        gpu_uuid=str(getattr(torch.cuda.get_device_properties(0), 'uuid', 'unavailable')),
        torch=torch.__version__, cuda=torch.version.cuda, epsilon=EPSILON, stabilizer=STABILIZER,
        anchor_offsets=ANCHORS, prefixes=PREFIXES, trajectory_prefix=32,
        error_decomposition=dict(threshold_px=5, prefix=32, strata=['full', 'inview', 'oov'],
            reduction='same float32 absolute errors, float64 component sums; equal-frame then equal-scene',
            over5_fraction_units='fraction, not percent'),
        precision=backend_flags(), model_dtype=str(next(model.parameters()).dtype))
    write_json(output/'provenance.json', provenance)
    baseline, observed, records, tau5_records = [], [], [], []
    try:
        with torch.no_grad():
            for batch_index, (left, right, target, valid, indices) in enumerate(loader):
                assert backend_flags() == expected_backend
                left, right, target, valid = [v.cuda() for v in (left, right, target, valid)]
                labels = [dict(scene=dataset.rows[i]['scene'], frame=dataset.rows[i]['frame'],
                    arm=args.arm, seed=1234, checkpoint_sha256=checkpoint_sha,
                    source_identity=registry['source_identity_sha256']) for i in indices.tolist()]
                predictions = {}
                with capture_predictions(driver, predictions):
                    base_rows = probes.replay_batch(model, driver, left, right, target, valid, 'A')
                for b, label in enumerate(labels):
                    base_rows[b].update(label)
                baseline.extend(base_rows)
                batch_keys = {(r['scene'], r['frame']) for r in labels}
                batch_reference = [r for r in official[0]['per_frame'] if (r['scene'], r['frame']) in batch_keys]
                official_parity(probes, base_rows, batch_reference)
                assert state_digest(model) == initial_state, 'baseline model state changed'
                observer = LookupObserver(probes, target, valid, InputPadder(left.shape, divis_by=32)._pad, labels)
                actual_predictions = {}
                with observer.installed(native), capture_predictions(driver, actual_predictions, observer):
                    observed_rows = probes.replay_batch(model, driver, left, right, target, valid, 'A')
                for b, label in enumerate(labels):
                    observed_rows[b].update(label)
                observed.extend(observed_rows)
                assert set(predictions) == set(actual_predictions) == set(PREFIXES)
                assert all(torch.equal(predictions[p], actual_predictions[p]) for p in PREFIXES), 'instrumented prediction differs'
                official_parity(probes, observed_rows, batch_reference)
                decompositions = error_decomposition_tau5(probes, predictions[32], target, valid)
                for b, decomposition in enumerate(decompositions):
                    observed_rows[b]['error_decomposition_tau5'] = decomposition
                    for stratum, entry in decomposition.items():
                        tau5_records.append(dict(**labels[b], prefix=32, iteration=32,
                            regime='error_decomposition_tau5', anchor_offset=None, level=None,
                            left_stratum=stratum, support_type='official_mask', statistics=entry['statistics']))
                records.extend(observer.records)
                torch.save(dict(labels=labels, geometry={k: v.detach().cpu() for k, v in observer.geometry.items()},
                    traces=observer.traces), output/f'traces-batch-{batch_index:02d}.pt')
                assert state_digest(model) == initial_state, 'model state changed'
        # Reuse the admitted official reducer and replay parity; no new EPE cohort.
        result = dict(provenance=provenance, baseline_parity=official_parity(probes, baseline, official[0]['per_frame']),
            instrumented_parity=official_parity(probes, observed, official[0]['per_frame']),
            exact_prediction_parity_all_prefixes=True, model_state_unchanged=True,
            state_sha256=initial_state, checkpoint_unchanged=digest(args.checkpoint) == checkpoint_sha,
            baseline_frames=baseline, instrumented_frames=observed, official_summary=probes.summarize(observed),
            per_frame_lookup=records, lookup_summary=reduce_records(records),
            error_decomposition_tau5_summary=reduce_records(tau5_records),
            interpretation='observational model-wise actual trajectories; GT-anchor support pairs geometrically, no mediation claim')
        assert result['checkpoint_unchanged']
        assert all(digest(args.source_root.parents[1]/rel) == expected for rel, expected in registry['source_files'].items())
        write_json(output/'lookup.json', result)
    except Exception as error:
        write_json(output/'failure.json', dict(error=repr(error), provenance=provenance,
            baseline_frames=baseline, instrumented_frames=observed, completed_batches=len(baseline)//2))
        raise
    finally:
        driver.reset_model(model)
    print(output/'lookup.json')


if __name__ == '__main__':
    main()
