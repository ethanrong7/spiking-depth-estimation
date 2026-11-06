"""Bind the fixed Train-only fidelity panel to two immutable source roots.

Metadata only: no network, source-array decoding, simulator, model or job.
The output is not a GPU-preparation registration or a training dataset.
verify_source_bytes is the read-only adapter for a future cluster generator.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from prepare_new_train_acquisition import validate_manifest

INDEX_SHA = '77d56a04b84374a7d121ddf812a519502f886ac151414aacf429a1531a8c6e74'
SCENES = ('TRAIN/C/0087', 'TRAIN/C/0009', 'TRAIN/C/0349', 'TRAIN/C/0204')
CENTERS = (1, 4, 8)
ALIASES = ('historical_native', 'new_train_attempt002')


def require(value, message):
    if not value:
        raise ValueError(message)


def read_record(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def relative_path(value):
    require(isinstance(value, str) and value, 'Nonempty source member required')
    path = PurePosixPath(value)
    require(not path.is_absolute() and '..' not in path.parts and 'TEST' not in path.parts
            and str(path) == value, 'Bounded canonical Train source path required')
    return path


def select_panel(index, roots, records):
    """Pure metadata selection; records are attributed native acquisition rows."""
    require(index.get('test') == [] and index.get('test_values_opened') is False,
            'Test must remain sealed')
    require(set(roots) == set(ALIASES) and set(records) == set(ALIASES), 'Two explicit source aliases required')
    root_paths = {key: PurePosixPath(value) for key, value in roots.items()}
    require(all(path.is_absolute() and '..' not in path.parts for path in root_paths.values()), 'Absolute configured roots required')
    a, b = root_paths.values()
    require(a != b and a not in b.parents and b not in a.parents, 'Source roots must be distinct and nonoverlapping')
    sequences = index['sequences']
    require(len({row['source_id'] for row in sequences}) == len(sequences), 'Duplicate source scene')
    require(all(re.fullmatch(r'TRAIN/C/[0-9]{4}', row['source_id'])
                and row['split'] in ('train', 'val') and row['original_split'] == 'TRAIN' for row in sequences),
            'Only original TRAIN metadata may enter this adapter')
    lookup = {row['source_id']: row for row in sequences}
    selected = []
    for number, sid in enumerate(SCENES):
        require(sid in lookup, 'Fixed fidelity scene missing: ' + sid)
        row = lookup[sid]
        alias = ALIASES[0 if number < 2 else 1]
        require(row['split'] == 'train' and row['root_alias'] == alias
                and row['historical_exposure'] is (number < 2), 'Fixed Train scene/root/exposure identity changed')
        require(len(row['frames']) == 10 and row['reference_centers'] == list(range(1, 9)), 'Ten native frames/eight Train centers required')
        files = []
        for position, frame in enumerate(row['frames']):
            stem = f'{position+6:04d}'
            require(frame['index'] == position and frame['source_frame'] == stem, 'Native source frame order changed')
            roles = [('rgb_l', 'rgb', 'left', '.png'), ('rgb_r', 'rgb', 'right', '.png')]
            if position in CENTERS:
                roles.append(('disparity_l', 'gt', 'left', '.pfm'))
            for field, role, view, suffix in roles:
                name = frame.get(field)
                relative_path(name)
                expected = f'{"frames_cleanpass" if role == "rgb" else "disparity"}/{sid}/{view}/{stem}{suffix}'
                require(name == expected, 'Source frame/view/path identity changed')
                require(name in records[alias], 'Acquired source member missing: ' + name)
                receipt = records[alias][name]
                require(receipt.get('role') == role and receipt.get('source_id') == sid
                        and type(receipt.get('size')) is int and receipt['size'] > 0
                        and re.fullmatch(r'[0-9a-f]{64}', receipt.get('sha256', '')), 'Attributed source receipt required')
                files.append(dict(path=name, role=role, view=view, frame_index=position,
                                  size=receipt['size'], sha256=receipt['sha256']))
        selected.append(dict(source_id=sid, split='train', root_alias=alias,
                             historical_exposure=row['historical_exposure'], centers=list(CENTERS),
                             frames=copy.deepcopy(row['frames']), files=files))
    require(sum(len(row['files']) for row in selected) == 92, 'Exactly80 RGB and12 target disparity sources required')
    return dict(schema='stereo-study-train-fidelity-source-panel-v1',
                scope='fixed_Train_only_source_metadata', registered_for_generation=False,
                source_roots={key: str(value) for key, value in root_paths.items()}, sequences=selected,
                test=[], test_values_opened=False, source_arrays_opened=False,
                sensor_generation_performed=False, training_compatible=False,
                source_steps_per_view=450, window_bins=50, target_count=12)


def source_file(panel, sequence, record):
    """Resolve one declared member without fallback or crossing source roots."""
    require(sequence['source_id'] in SCENES and sequence['root_alias'] in ALIASES, 'Unregistered source identity')
    require(record in sequence['files'], 'Unregistered source member')
    root = Path(panel['source_roots'][sequence['root_alias']]).resolve()
    path = root.joinpath(*relative_path(record['path']).parts).resolve()
    require(path.is_relative_to(root), 'Source member escapes its configured root')
    return path


def verify_source_bytes(panel):
    """Hash-only readback before future decoding; never opens source arrays."""
    require(panel['schema'] == 'stereo-study-train-fidelity-source-panel-v1'
            and panel['test'] == [] and panel['test_values_opened'] is False, 'Train-only source panel required')
    count = size = 0
    for sequence in panel['sequences']:
        for record in sequence['files']:
            path = source_file(panel, sequence, record)
            require(path.is_file() and path.stat().st_size == record['size'], 'Missing/changed source size: ' + record['path'])
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1048576), b''):
                    digest.update(block)
            require(digest.hexdigest() == record['sha256'], 'Changed source bytes: ' + record['path'])
            count += 1
            size += record['size']
    require(count == 92, 'Incomplete fixed source panel')
    return dict(status='PASS_FIXED92_SOURCE_BYTE_READBACK', files=count, bytes=size,
                source_arrays_decoded=False, Test_values_opened=False, GPU_used=False)


def accepted_source_metadata(source_state, historical_state, deployment_path):
    """Shared accepted acquisition bindings; metadata only, no payload read."""
    source_state, historical_state = Path(source_state), Path(historical_state)
    index, index_sha = read_record(source_state/'train-val-multiroot-source-index-v1.json')
    require(index_sha == INDEX_SHA, 'Frozen multi-root source index changed')
    old_index, old_index_sha = read_record(historical_state/'selected-source-index.json')
    old, old_sha = read_record(historical_state/'source-download-receipt-v4.json')
    old_audit, old_audit_sha = read_record(historical_state/'source-acquisition-audit-v4.json')
    manifest, manifest_sha = read_record(source_state/'acquisition-recovery-approved-v1.json')
    new, new_sha = read_record(source_state/'payload-attempt002-completed-v1.json')
    audit, audit_sha = read_record(source_state/'payload-attempt002-independent-audit-v1.json')
    deployment, deployment_sha = read_record(deployment_path)
    historical, acquired = (index['root_aliases'][key] for key in ALIASES)
    require(old_index_sha == historical['source_index_sha256'] == old['manifest_sha256'] == old_audit['manifest_sha256']
            and old_sha == old_audit['download_receipt_sha256'] and old['status'] == 'COMPLETE'
            and old_audit['status'] == 'PASS' and len(old['completed']) == 4020, 'Historical native acquisition evidence changed')
    require({row['source_id'] for row in old_index['sequences']} ==
            {row['source_id'] for row in index['sequences'] if row['root_alias'] == ALIASES[0]}, 'Historical source cohort changed')
    validate_manifest(manifest, execution=True)
    require(manifest_sha == acquired['acquisition_manifest_sha256'] == new['manifest_sha256'] == audit['manifest_sha256']
            and new_sha == acquired['terminal_receipt_sha256'] == audit['terminal_receipt_sha256']
            and audit_sha == acquired['independent_rehash_receipt_sha256']
            and deployment_sha == manifest['input_receipts_sha256']['deployment'], 'New acquisition/deployment bindings changed')
    require(new['status'] == 'COMPLETE840_NATIVE_TRAIN_SOURCE_FILES' and new['readback_all840_PASS'] is True
            and audit['status'] == 'PASS_EXACT840_SOURCE_FILES_REHASHED' and audit['files'] == 840
            and set(new['completed']) == {row['path'] for row in manifest['members']}
            and new['returned_bytes'] == audit['payload_bytes'] == manifest['budget']['disk_payload_bytes'] == 938899825
            and all(new[key] is False for key in ('test_access', 'pixel_arrays_decoded', 'model_inference', 'GPU_used')),
            'Complete native new-Train source acceptance required')
    require(acquired['relative_path'] == manifest['output_root_relative'], 'New acquisition root binding changed')
    relative_path(acquired['relative_path'])
    roots = {ALIASES[0]: deployment[historical['deployment_key']],
             ALIASES[1]: str(PurePosixPath(deployment[acquired['deployment_base_key']])/acquired['relative_path'])}
    binding = dict(source_index=index_sha, historical_index=old_index_sha,
        historical_download=old_sha, historical_audit=old_audit_sha, acquisition_manifest=manifest_sha,
        new_download=new_sha, independent_new_audit=audit_sha, deployment=deployment_sha)
    return index, roots, {ALIASES[0]: old['completed'], ALIASES[1]: new['completed']}, binding


def build_from_receipts(source_state, historical_state, deployment_path):
    index, roots, records, binding = accepted_source_metadata(source_state, historical_state, deployment_path)
    panel = select_panel(index, roots, records)
    panel['input_sha256'] = binding
    panel['verification'] = 'Expected native source bytes from accepted acquisition receipts; no new source-byte reads yet'
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-state', type=Path, required=True)
    parser.add_argument('--historical-state', type=Path, required=True)
    parser.add_argument('--cluster-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[5]
    require(any(args.output.resolve().is_relative_to(repository/key) for key in ('storage', 'scratch')),
            'Fresh ignored source-panel metadata output required')
    panel = build_from_receipts(args.source_state, args.historical_state, args.cluster_config)
    with args.output.open('x') as stream:
        json.dump(panel, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='FIXED_TRAIN_SOURCE_PANEL_BOUND', output=str(args.output),
        source_files=92, target_count=12, source_arrays_opened=False, GPU_preparation_registered=False)))


if __name__ == '__main__':
    main()
