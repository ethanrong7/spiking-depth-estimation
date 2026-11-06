"""Data boundaries for the prospective stereo study; no implicit temporal pooling.

All tensors use positive left-reference disparity d=x_left-x_right. A sample
returns left/right [50,C,H,W], target/valid [1,H,W]; DataLoader adds B. Events
retain ON/OFF count channels. Legacy centered data require explicit engineering
access and never qualify as the causal paired benchmark.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

T = 50
LEGACY_MANIFEST_SHA256 = 'eb52ec22890423efc2d5455904c3155d6aaa8c8a96199eb8fcd371305d62534d'


def safe_path(root, relative):
    name = Path(relative)
    if name.is_absolute() or '..' in name.parts:
        raise ValueError('Data asset must be relative to its configured root')
    base = Path(root).resolve()
    result = (base / name).resolve()
    if not result.is_relative_to(base):
        raise ValueError('Data asset escapes configured root')
    return result


def read_pfm(path, shape):
    """Decode PFM bottom-up storage; refuse implicit disparity scale changes."""
    with Path(path).open('rb') as stream:
        if stream.readline().strip() != b'Pf':
            raise ValueError('Single-channel disparity PFM required')
        width, height = map(int, stream.readline().split())
        scale = float(stream.readline().strip())
        if (height, width) != tuple(shape) or abs(scale) != 1:
            raise ValueError('Unexpected PFM shape/scale')
        array = np.frombuffer(stream.read(), dtype='<f4' if scale < 0 else '>f4')
        if array.size != height * width:
            raise ValueError('PFM payload length mismatch')
    return np.flipud(array.reshape(height, width)).astype(np.float32, copy=True)


def read_packed_spikes(path, shape):
    count = T * int(shape[0]) * int(shape[1])
    payload = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    if count % 8 or payload.size != count // 8:
        raise ValueError('Expected exactly 50 complete LSB-first spike planes')
    return np.unpackbits(payload, bitorder='little').reshape(T, 1, *shape)


def _tensor_sample(left, right, disparity, row, crop=None):
    shape = disparity.shape
    if left.shape[-2:] != shape or right.shape != left.shape or left.shape[0] != T:
        raise ValueError('Stereo/target geometry or temporal count mismatch')
    valid = np.isfinite(disparity) & (disparity > 0)
    target = np.where(valid, disparity, 0).astype(np.float32)
    tensors = dict(left=torch.from_numpy(left.astype(np.float32, copy=True)),
                   right=torch.from_numpy(right.astype(np.float32, copy=True)),
                   target=torch.from_numpy(target[None]),
                   valid=torch.from_numpy(valid[None].copy()))
    if crop is not None:
        y, x, h, w = map(int, crop)
        if min(y, x) < 0 or min(h, w) <= 0 or y + h > shape[0] or x + w > shape[1]:
            raise ValueError('Crop outside image')
        tensors = {key: value[..., y:y+h, x:x+w].contiguous() for key, value in tensors.items()}
    return {**tensors, 'sample_id': row['sample_id'], 'scene_id': row.get('scene_id', row.get('source_id')),
            'metadata': {**row, 'crop': crop, 'disparity_sign': 'x_left_minus_x_right'}}


class LegacyNativeEngineeringDataset:
    """Audited FT3D centered 960/60 input for engineering only; never Test.

    Explicit crop changes spatial support for an engineering memory diagnostic;
    it does not resize pixels or disparity. No event observations are fabricated.
    """
    def __init__(self, source_root, derived_root, *, purpose, split='train', crop=None,
                 manifest_sha256=LEGACY_MANIFEST_SHA256):
        if purpose != 'engineering' or split not in ('train', 'dev'):
            raise ValueError('Legacy native data supports explicit engineering Train/Dev only')
        self.source_root, self.derived_root = Path(source_root), Path(derived_root)
        raw = (self.derived_root / 'manifest.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest_sha256:
            raise ValueError('Legacy manifest identity changed')
        manifest = json.loads(raw)
        if manifest['protocol_id'] != 'flyingthings3d-centered-native-v1':
            raise ValueError('Unrecognized centered dataset')
        self.rows, self.crop = manifest[split], crop
        for row in self.rows:
            if (row['spike_shape'] != [50, 540, 960] or row['split'] != split
                    or row['half_frame_future_lookahead'] is not True
                    or row['vertical_flip'] is not False):
                raise ValueError('Legacy timing/geometry contract mismatch')
            if row['global_indices'] != list(range(50*row['center_index']-24, 50*row['center_index']+26)):
                raise ValueError('Legacy centered chronological mapping changed')
            for key in ('spike_l', 'spike_r', 'disparity_l'):
                if 'TEST' in Path(row[key]).parts:
                    raise ValueError('Legacy engineering access is TRAIN-only')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        left, right = [read_packed_spikes(safe_path(self.derived_root, row['spike_' + view]), (540, 960))
                       for view in ('l', 'r')]
        disparity = read_pfm(safe_path(self.source_root, row['disparity_l']), (540, 960))
        return _tensor_sample(left, right, disparity,
                              {**row, 'evidence_scope': 'engineering_centered_noncausal',
                               'time_unit': 'source_frame_interval', 'physical_fps': None}, self.crop)


def validate_paired_manifest(manifest, *, allow_engineering_pilot=False):
    """Check metadata only, including Test identities without opening assets."""
    pilot = manifest.get('schema') == 'stereo-paired-engineering-pilot-v1'
    if pilot and (not allow_engineering_pilot or manifest.get('evidence_scope') != 'engineering_only'
                  or manifest.get('test') != []):
        raise ValueError('Pilot requires explicit engineering access and no fabricated Test')
    if (manifest.get('schema') != 'stereo-paired-v1' and not pilot) or manifest.get('T') != T:
        raise ValueError('Expected paired schema with all 50 sensor-time steps')
    if manifest.get('disparity') != 'positive_left_pixels' or manifest.get('rectified') is not True:
        raise ValueError('Rectified positive left-reference disparity required')
    if manifest.get('time_unit') not in ('seconds', 'source_frame_interval'):
        raise ValueError('Physical or explicitly normalized time required')
    if manifest.get('time_unit') == 'seconds' and not manifest.get('physical_time_source'):
        raise ValueError('Seconds require physical timing provenance')
    if not manifest.get('source_identity') or not manifest.get('simulators') or not manifest.get('license'):
        raise ValueError('Source, simulator provenance and license required')
    for modality in ('spike', 'event'):
        simulator = manifest['simulators'].get(modality, {})
        if not all(simulator.get(k) for k in ('implementation', 'version', 'configuration')):
            raise ValueError('Both simulators require pinned implementation and configuration')
    owners, samples = {}, set()
    for split in (('train', 'val') if pilot else ('train', 'val', 'test')):
        rows = manifest.get(split, [])
        if not rows:
            raise ValueError('Train, Val and sealed Test must each have scene identities')
        for row in rows:
            scene = row['scene_id']
            if scene in owners and owners[scene] != split:
                raise ValueError('Scene leakage between splits')
            owners[scene] = split
            if row['sample_id'] in samples:
                raise ValueError('Repeated sample identity')
            samples.add(row['sample_id'])
            edges = np.asarray(row['bin_edges'], dtype=np.float64)
            if (edges.shape != (51,) or not np.isfinite(edges).all() or not np.all(np.diff(edges) > 0)
                    or not np.isclose(edges[-1], row['target_time'], rtol=0, atol=1e-9)):
                raise ValueError('Exactly 50 causal bins ending at target required')
            if (not np.isfinite(row.get('source_support_end', float('inf')))
                    or row['source_support_end'] > row['target_time'] + 1e-9):
                raise ValueError('Generation used future source context')
            if row.get('event_channels') != ['ON', 'OFF']:
                raise ValueError('Keep event polarity in distinct ON/OFF channels')
            for key in ('spike', 'event', 'disparity'):
                name = Path(row[key])
                if name.is_absolute() or '..' in name.parts:
                    raise ValueError('Manifest data paths must be relative')
    return True


class PairedStereoDataset:
    """Read immutable paired NPZ representations and positive disparity NPY.

    Spike NPZ: left/right [50,1,H,W] binary; event NPZ: left/right
    [50,2,H,W] nonnegative counts, ordered ON/OFF. Event counts may exceed one.
    This loader does not assign scenes, simulate sensors, rescale, or normalize.
    """
    def __init__(self, data_root, manifest_path, modality, split, *, test_release=None, purpose='study', sample_ids=None):
        if modality not in ('spike', 'event') or split not in ('train', 'val', 'test'):
            raise ValueError('Unknown modality/split')
        self.root, self.modality = Path(data_root), modality
        self.manifest = json.loads(Path(manifest_path).read_text())
        validate_paired_manifest(self.manifest, allow_engineering_pilot=purpose == 'engineering')
        if self.manifest.get('schema') == 'stereo-paired-engineering-pilot-v1' and split == 'test':
            raise ValueError('Engineering pilot has no Test set')
        if split == 'test':
            release = test_release or {}
            raw_digest = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
            if (release.get('manifest_sha256') != raw_digest or not release.get('frozen_registration_sha256')
                    or release.get('test_access_authorized') is not True):
                raise ValueError('Test remains sealed until configuration and registration freeze')
        self.rows = self.manifest[split]
        if sample_ids is not None:
            if not isinstance(sample_ids, list) or not sample_ids or len(set(sample_ids)) != len(sample_ids):
                raise ValueError('Selected sample IDs must be a nonempty unique ordered list')
            indexed = {row['sample_id']: row for row in self.rows}
            if any(sample_id not in indexed for sample_id in sample_ids):
                raise ValueError('Selected sample must belong to the requested split')
            self.rows = [indexed[sample_id] for sample_id in sample_ids]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(safe_path(self.root, row[self.modality]), allow_pickle=False) as packed:
            left, right = packed['left'], packed['right']
        expected_c = 1 if self.modality == 'spike' else 2
        for array in (left, right):
            if array.ndim != 4 or array.shape[:2] != (T, expected_c) or not np.isfinite(array).all():
                raise ValueError('Invalid temporal/channel layout or nonfinite observations')
            if np.any(array < 0) or np.any(array != np.floor(array)):
                raise ValueError('Inputs must be nonnegative integral counts')
            if self.modality == 'spike' and np.any(array > 1):
                raise ValueError('Intensity spikes must be binary')
        disparity = np.load(safe_path(self.root, row['disparity']), allow_pickle=False)
        if disparity.ndim != 2:
            raise ValueError('Disparity must be H,W')
        return _tensor_sample(left, right, disparity, row)
