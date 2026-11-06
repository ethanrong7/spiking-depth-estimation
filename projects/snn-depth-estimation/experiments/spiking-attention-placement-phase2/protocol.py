"""Phase-2 presentation contract and auditable source/endpoint records."""
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'shared-gpu-30k-v1'))
from common import Prefetch as BasePrefetch, flags

SEEDS = {1234: 2017, 2345: 3128, 3456: 4239}
ARMS = ('SC', 'SP', 'CC', 'CP', 'Original')
MANIFEST_SHA = 'bad4369b6fd8b0cdf871a20b1727c2d9b59733a16bd33f75904227bdfef4510a'
EVALS = [0, 250, 500, 1000, 1500, 2500] + list(range(5000, 30001, 2500))


def presentation(order, step, seed):
    return ([order[(step * 8 + j) % len(order)] for j in range(8)],
            [list(flags(step * 8 + j, seed)) for j in range(8)])


def expected_lr(step):
    """LR used by update step+1 under the frozen linear 30k OneCycle."""
    initial, maximum, minimum = 2e-4 / 25, 2e-4, 2e-4 / 25 / 1e4
    if step <= 299:
        return initial + (maximum - initial) * step / 299
    return maximum + (minimum - maximum) * (step - 299) / (29999 - 299)


def validate_row(row, order, seed):
    step = row['update'] - 1
    indices, hv = presentation(order, step, seed)
    assert row['indices'] == indices and row['augmentation_hv'] == hv, ('stream', row['update'])
    assert row['samples_seen'] == row['update'] * 8
    assert math.isclose(row['lr_used'], expected_lr(step), rel_tol=1e-10, abs_tol=1e-14), ('lr', row['update'])
    assert math.isclose(row['lr_next'], expected_lr(step + 1), rel_tol=1e-10, abs_tol=1e-14)


class Prefetch(BasePrefetch):
    def __init__(self, dataset, order, start, stop, seed):
        self.seed = seed
        super().__init__(dataset, order, start, stop)

    def _fill(self):
        while len(self.pending) < 2 and self.next < self.stop:
            step = self.next
            self.next += 1
            self.pending.append((step, self.pool.submit(self._decode, step)))

    def _decode(self, step):
        import time
        started = time.monotonic()
        indices, hv = presentation(self.order, step, self.seed)
        items = []
        for index, (h, v) in zip(indices, hv):
            item = list(self.dataset[index])
            if h:
                item[:4] = [x.flip(-1) for x in item[:4]]
                item[2] = -item[2]
            if v:
                item[:4] = [x.flip(-2) for x in item[:4]]
            items.append(item)
        return dict(step=step, indices=indices, flags=hv, items=items, decode_seconds=time.monotonic() - started)


def audit_history(path, order, seed, completed):
    count = 0
    if Path(path).exists():
        with Path(path).open() as handle:
            for line in handle:
                row = json.loads(line)
                count += 1
                assert row['update'] == count
                validate_row(row, order, seed)
    assert count == completed, ('history incomplete', count, completed)
    return dict(updates=count, presentations=count * 8, actual_indices_hv_lr_verified=True)


def tensor_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
