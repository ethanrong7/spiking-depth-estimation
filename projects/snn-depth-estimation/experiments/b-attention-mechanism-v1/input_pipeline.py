"""CPU-only compact input cache and ordered batch preparation for CUDA/NPU.

No accelerator calls or random sampling occur in producer threads. Spike bytes
stay uint8 until the consumer casts them on its device, before model execution.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'shared-gpu-30k-v1'))
from common import flags


def read_spike_bytes(path, shape=(50, 250, 400)):
    """Decode exactly the historical first 50 frames, little-endian bit order."""
    count = int(np.prod(shape))
    with Path(path).open('rb') as stream:
        packed = np.frombuffer(stream.read((count + 7) // 8), dtype=np.uint8)
    if packed.size * 8 < count:
        raise ValueError(f'Truncated spike input: {path}')
    return torch.from_numpy(np.unpackbits(packed, bitorder='little', count=count).reshape(shape))


class NativeBytesDataset(Dataset):
    """Identical native targets/masks; compact binary spikes, no large bit indices."""
    def __init__(self, root, rows):
        self.root, self.rows = Path(root), rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from core.utils.reader import read_depth
        row = self.rows[index]
        depth = read_depth(str(self.root / row['depth'])).astype(np.float32)
        valid = np.isfinite(depth) & (depth > 0)
        target = np.zeros_like(depth)
        target[valid] = -(289.740625 * .08) / depth[valid]
        valid &= np.abs(target) < 512
        left, right = [read_spike_bytes(self.root / row[key]) for key in ('spike_l', 'spike_r')]
        if left.shape[-2:] != depth.shape:
            raise ValueError('Spike/depth geometry mismatch')
        return left, right, torch.from_numpy(target[None]), torch.from_numpy(valid), index


class RamDataset(Dataset):
    """Fully preload immutable samples within an explicit total tensor budget.

    __getitem__ returns read-only-by-contract cached tensors. Consumers must not
    mutate them. The prepared loader below flips out of place and stacks into
    owned batch storage. Normal DataLoader collation also makes owned storage.
    """
    def __init__(self, dataset, budget_bytes):
        self.rows = dataset.rows
        self.samples, self.bytes = [], 0
        started = time.monotonic()
        for index in range(len(dataset)):
            item = dataset[index]
            size = sum(v.numel() * v.element_size() for v in item[:4])
            # Native shapes are fixed. Fail before loading the entire dataset.
            if index == 0 and size * len(dataset) > budget_bytes:
                raise MemoryError(f'RAM cache needs {size * len(dataset) / 2**30:.2f} GiB; '
                                  f'budget is {budget_bytes / 2**30:.2f} GiB')
            if self.bytes + size > budget_bytes:
                raise MemoryError('RAM cache tensor budget exceeded')
            self.samples.append(item)
            self.bytes += size
        self.preload_seconds = time.monotonic() - started

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def prepare_batch(dataset, order, step, seed):
    started = time.monotonic()
    indices = [order[(step * 8 + j) % len(order)] for j in range(8)]
    hv = [list(flags(step * 8 + j, seed)) for j in range(8)]
    items = []
    for index, (horizontal, vertical) in zip(indices, hv):
        item = list(dataset[index])
        if horizontal:
            item[:4] = [x.flip(-1) for x in item[:4]]
            item[2] = -item[2]
        if vertical:
            item[:4] = [x.flip(-2) for x in item[:4]]
        items.append(item)
    microbatches = []
    for offset in (0, 4):
        tensors = tuple(torch.stack([item[j] for item in items[offset:offset+4]]) for j in range(4))
        microbatches.append(dict(tensors=tensors, valid_count=int(tensors[3].sum())))
    seconds = time.monotonic() - started
    return dict(step=step, indices=indices, flags=hv, microbatches=microbatches,
                valid_count=sum(m['valid_count'] for m in microbatches),
                host_batch_bytes=sum(t.numel()*t.element_size() for m in microbatches for t in m['tensors']),
                decode_seconds=seconds, input_prepare_seconds=seconds)


class PreparedPrefetch:
    """Bounded futures, strictly ordered consumption and absolute-step recovery."""
    def __init__(self, dataset, order, start, stop, seed, workers=2, depth=2):
        if workers < 1 or depth < 1:
            raise ValueError('workers and depth must be positive')
        if not order and start < stop:
            raise ValueError('Empty sample order')
        self.dataset, self.order, self.next, self.stop = dataset, tuple(order), start, stop
        self.seed, self.depth = seed, depth
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='stereo-prepare')
        self.pending = deque()
        self._fill()

    def _fill(self):
        while len(self.pending) < self.depth and self.next < self.stop:
            step = self.next
            self.next += 1
            self.pending.append((step, self.pool.submit(prepare_batch, self.dataset, self.order, step, self.seed)))

    def get(self, step):
        if not self.pending or self.pending[0][0] != step:
            raise RuntimeError('Prefetch order mismatch')
        _, future = self.pending.popleft()
        result = future.result()
        self._fill()
        return result

    def close(self):
        for _, future in self.pending:
            future.cancel()
        self.pool.shutdown(wait=True)


class InputPipeline:
    def __init__(self, mode='preload', budget_gib=12., workers=2, depth=2, seed=1234):
        if mode not in ('fast', 'preload') or budget_gib <= 0 or workers < 1 or depth < 1:
            raise ValueError('Invalid input pipeline settings')
        self.mode, self.budget = mode, int(budget_gib * 2**30)
        self.workers, self.depth, self.seed = workers, depth, seed
        self.metadata = dict(mode=mode, budget_bytes=self.budget, workers=workers,
                             prefetch_depth=depth, spike_host_dtype='uint8', cached_bytes=0,
                             preload_seconds=0., cache_scope='per_process_train_and_dev')

    def datasets(self, root, train_rows, dev_rows, preflight=False):
        train, dev = NativeBytesDataset(root, train_rows), NativeBytesDataset(root, dev_rows)
        # Disposable 3-step probes should not decode/preload the complete corpus.
        if self.mode == 'preload' and not preflight:
            print('Preloading compact Train/Dev tensors into host RAM...', flush=True)
            # Combined reservation before loading Train when shapes match native data.
            estimate = (len(train_rows) + len(dev_rows)) * (2*50*250*400 + 5*250*400)
            if estimate > self.budget:
                raise MemoryError(f'Combined native cache needs {estimate/2**30:.2f} GiB; '
                                  f'budget is {self.budget/2**30:.2f} GiB')
            train = RamDataset(train, self.budget)
            dev = RamDataset(dev, self.budget - train.bytes)
            self.metadata.update(cached_bytes=train.bytes+dev.bytes,
                                 preload_seconds=train.preload_seconds+dev.preload_seconds)
            print(f'Preloaded {(train.bytes+dev.bytes)/2**30:.2f} GiB in '
                  f'{self.metadata["preload_seconds"]:.1f}s', flush=True)
        self.metadata.update(preflight=preflight, train_frames=len(train), dev_frames=len(dev))
        return train, dev

    def prefetch(self, dataset, order, start, stop):
        return PreparedPrefetch(dataset, order, start, stop, self.seed, self.workers, self.depth)
