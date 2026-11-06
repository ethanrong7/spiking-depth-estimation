"""Bounded metadata-only audit of already selected, pinned RGB HDF5 shards.

No array indexing/decoding is permitted. Uses only object shape, dtype and
creation properties. Raw HTTP ranges, source pointers and partial failures are
preserved outside Git. Does not acquire source pixels or release sealed Test.
"""
import argparse
import contextlib
import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import time


def selected_shards(proposal):
    shards = {}
    for scene in proposal['scenes']:
        if scene['split'] != 'test':
            continue
        for frame in scene['frames']:
            source = frame['rgb_hdf5']
            key = (source['repository'], source['revision'], source['path'])
            item = {k: source[k] for k in ('repository','revision','path','size','lfs_sha256','left_dataset','right_dataset')}
            if (not re.fullmatch(r'[0-9a-f]{40}', item['revision'])
                    or not re.fullmatch(r'[0-9a-f]{64}', item['lfs_sha256'])
                    or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', item['repository'])
                    or not re.fullmatch(r'individual_quantities/test/[0-9]+_[0-9]+\.h5', item['path'])
                    or item['left_dataset'] != 'frame_cleanpass_left'
                    or item['right_dataset'] != 'frame_cleanpass_right'):
                raise ValueError('Exact preselected public RGB shard identities required')
            if key not in shards:
                shards[key] = dict(item, selected_indices=[], scene_ids=[])
            current = shards[key]
            if any(current[k] != v for k,v in item.items()):
                raise ValueError('Inconsistent pinned shard identity')
            current['selected_indices'].append(source['file_index'])
            current['scene_ids'].append(scene['scene_id'])
    for item in shards.values():
        item['selected_indices'] = sorted(set(item['selected_indices']))
        item['scene_ids'] = sorted(set(item['scene_ids']))
    if len(shards) != 12:
        raise ValueError('Existing prospective cohort must retain exactly twelve selected shards')
    return [shards[k] for k in sorted(shards)]


class MetadataRanges(io.RawIOBase):
    def __init__(self, source, session, *, max_requests=64, max_bytes=131072, max_seconds=120):
        self.size = source['size']
        self.url = 'https://huggingface.co/datasets/{repository}/resolve/{revision}/{path}'.format(**source)
        self.session = session
        self.pos = self.total = 0
        self.calls = []
        self.started = time.monotonic()
        self.max_requests, self.max_bytes, self.max_seconds = max_requests, max_bytes, max_seconds

    def readable(self): return True
    def seekable(self): return True
    def writable(self): return False
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        if whence not in (0,1,2): raise ValueError('Unknown seek origin')
        target = offset + (0 if whence == 0 else self.pos if whence == 1 else self.size)
        if not 0 <= target <= self.size: raise ValueError('Out-of-file metadata seek')
        self.pos = target
        return target

    def read(self, n=-1):
        if n < 0: raise RuntimeError('Unbounded file read prohibited')
        n = min(n, self.size-self.pos)
        if not n: return b''
        if (len(self.calls) >= self.max_requests or self.total+n > self.max_bytes
                or time.monotonic()-self.started > self.max_seconds):
            raise RuntimeError('Metadata request/byte/time bound reached')
        start, end = self.pos, self.pos+n-1
        with self.session.get(self.url, headers={'Range':f'bytes={start}-{end}'},
                              stream=True, timeout=(10,20)) as response:
            if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                raise RuntimeError('Pinned finite HTTP range not honored')
            data = response.raw.read(n+1)
            if len(data) != n: raise RuntimeError('Short or oversized metadata range')
            self.url = response.url
        self.calls.append(dict(offset=start,bytes=n,sha256=hashlib.sha256(data).hexdigest()))
        self.total += n; self.pos += n
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer)); buffer[:len(data)] = data
        return len(data)


@contextlib.contextmanager
def forbid_dataset_values(h5py):
    original = h5py.Dataset.__getitem__
    def prohibited(*args, **kwargs):
        raise RuntimeError('Test dataset value access prohibited during metadata audit')
    h5py.Dataset.__getitem__ = prohibited
    try: yield
    finally: h5py.Dataset.__getitem__ = original


def inspect_rgb_metadata(handle, source, h5py):
    objects = []
    with forbid_dataset_values(h5py):
        for name in (source['left_dataset'], source['right_dataset']):
            dataset = handle[name]
            if not isinstance(dataset, h5py.Dataset): raise ValueError('RGB dataset missing')
            row = dict(name=name,shape=list(dataset.shape),dtype=str(dataset.dtype),
                       chunks=list(dataset.chunks) if dataset.chunks else None,
                       compression=dataset.compression,compression_options=dataset.compression_opts)
            if (row['shape'][1:] != [540,960,3] or len(row['shape']) != 4
                    or row['dtype'] != 'uint8' or row['chunks'] != [1,540,960,3]
                    or row['compression'] != 'gzip'
                    or any(type(i) is not int or not 0 <= i < row['shape'][0] for i in source['selected_indices'])):
                raise ValueError('Selected RGB shape/dtype/chunks/index bounds differ from proposal')
            objects.append(row)
    if objects[0]['shape'] != objects[1]['shape']:
        raise ValueError('Stereo RGB shape mismatch')
    return objects


def verify_pointer(source, session):
    url = 'https://huggingface.co/datasets/{repository}/raw/{revision}/{path}'.format(**source)
    with session.get(url, stream=True, timeout=(10,20)) as response:
        response.raise_for_status(); raw = response.raw.read(4097)
    if len(raw) > 4096: raise ValueError('Expected small pinned LFS pointer')
    lines = raw.decode().splitlines()
    if (f'oid sha256:{source["lfs_sha256"]}' not in lines or f'size {source["size"]}' not in lines
            or 'version https://git-lfs.github.com/spec/v1' not in lines):
        raise ValueError('Provider LFS identity changed')
    return dict(text=raw.decode(),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    # This is metadata work only; no model/CUDA libraries are imported.
    import h5py
    import requests
    raw = args.proposal.read_bytes(); sources = selected_shards(json.loads(raw))
    args.output.mkdir(parents=True,exist_ok=False)
    report = dict(schema='stereo-selected-rgb-metadata-audit-v1',status='RUNNING',
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        proposal_sha256=hashlib.sha256(raw).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dataset_values_accessed=False,test_pixels_accessed=False,model_constructed=False,
        policy='Only two preselected RGB dataset object headers/properties; Dataset.__getitem__ forbidden; no values/labels/inference',
        bounds=dict(shards=12,requests_per_shard=64,range_bytes_per_shard=131072,seconds_per_shard=120,total_seconds=600),
        h5py=h5py.__version__,requests_version=requests.__version__,shards=[])
    def save(): (args.output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    began = time.monotonic(); save()
    for source in sources:
        item = dict(source=source,status='INITIALIZING'); report['shards'].append(item)
        remote = None
        try:
            if time.monotonic()-began > 600: raise RuntimeError('Total metadata audit time bound reached')
            with requests.Session() as session:
                item['pointer'] = verify_pointer(source,session)
                remote = MetadataRanges(source,session,max_seconds=min(120,600-(time.monotonic()-began)))
                with h5py.File(remote,'r') as handle:
                    item['objects'] = inspect_rgb_metadata(handle,source,h5py)
            item['status'] = 'PASS_SELECTED_RGB_METADATA_ONLY'
        except Exception as error:
            item.update(status='FAILED_OR_INCOMPLETE_METADATA_ONLY',error=repr(error))
        finally:
            if remote is not None:
                item.update(requests=remote.calls,metadata_range_bytes=remote.total)
            save()
        if item['status'] != 'PASS_SELECTED_RGB_METADATA_ONLY': break
    report.update(status='PASS_ALL12_SELECTED_RGB_METADATA_ONLY' if len(report['shards']) == 12
                  and all(x['status']=='PASS_SELECTED_RGB_METADATA_ONLY' for x in report['shards']) else 'INCOMPLETE_OR_FAILED',
                  metadata_range_bytes=sum(x.get('metadata_range_bytes',0) for x in report['shards']),
                  range_requests=sum(len(x.get('requests',[])) for x in report['shards']),elapsed_seconds=time.monotonic()-began)
    save(); print(json.dumps({k:v for k,v in report.items() if k not in ('shards','policy')},indent=2))
    if report['status'] != 'PASS_ALL12_SELECTED_RGB_METADATA_ONLY': raise SystemExit(1)


if __name__ == '__main__': main()
