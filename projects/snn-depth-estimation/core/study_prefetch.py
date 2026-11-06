"""Bounded ordered CPU decode/pinning and optional one-batch CUDA prefetch.

Only deterministic, read-only datasets are supported. Queue positions are
speculative: checkpoints commit presentations only after an optimizer update.
Rebuilding at that committed position discards all speculative buffers. Workers
never draw from Python/NumPy/Torch RNG or perform model/CUDA computation.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import time

import torch


def prefetch_options(training):
    values = dict(workers=0, batches=2, pin_memory=True, device_prefetch=True)
    values.update(training.get('prefetch', {}))
    if set(values) != {'workers', 'batches', 'pin_memory', 'device_prefetch'}:
        raise ValueError('Unknown prefetch option')
    if type(values['workers']) is not int or not 0 <= values['workers'] <= 4:
        raise ValueError('Prefetch workers must be0..4')
    if type(values['batches']) is not int or not 1 <= values['batches'] <= 4:
        raise ValueError('Bounded prefetch batches must be1..4')
    if any(type(values[key]) is not bool for key in ('pin_memory', 'device_prefetch')):
        raise ValueError('Pin/device prefetch options must be boolean')
    return values


class OrderedBatchStream:
    """Keep at most `batches` CPU futures and one next batch on the GPU.

    Decode threads return results in sampler order, including exceptions. A
    completed next CPU batch may be copied on a separate CUDA stream while the
    current graph runs. No wait for an unready future is introduced by staging.
    CUDA input storage is recorded on the consumer stream before use; pinned
    source storage remains referenced until its copy event completes.
    """
    def __init__(self, dataset, *, seed, position, microbatch, end_position,
                 device, options=None, indices_fn=None, collate_fn=None):
        # Lazy import avoids a module cycle with the training entry points.
        from core.study_training import collate_samples, sample_indices
        self.dataset, self.seed = dataset, seed
        self.position, self.scheduled = position, position
        self.microbatch, self.end = microbatch, end_position
        if position < 0 or microbatch < 1 or end_position < position or (end_position-position) % microbatch:
            raise ValueError('Stream boundaries must contain complete microbatches')
        self.device = torch.device(device)
        self.options = prefetch_options({'prefetch': options or {}})
        self.indices_fn, self.collate_fn = indices_fn or sample_indices, collate_fn or collate_samples
        self.pool = ThreadPoolExecutor(max_workers=self.options['workers'], thread_name_prefix='stereo-decode') if self.options['workers'] else None
        self.pin = self.pool is not None and self.device.type == 'cuda' and self.options['pin_memory']
        self.copy_stream = torch.cuda.Stream(device=self.device) if self.pin and self.options['device_prefetch'] else None
        self.queue, self.host_inflight = deque(), deque()
        self.staged = None
        self.closed = False
        self._fill()

    def _prepare(self, position):
        tick = time.perf_counter()
        indices = self.indices_fn(self.seed, position, self.microbatch, len(self.dataset))
        batch = self.collate_fn([self.dataset[index] for index in indices])
        if self.pin:
            batch = {key: value.pin_memory() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        return batch, indices, time.perf_counter()-tick

    def _fill(self):
        if self.pool is not None:
            while len(self.queue) < self.options['batches'] and self.scheduled < self.end:
                position = self.scheduled
                self.queue.append((position, self.pool.submit(self._prepare, position)))
                self.scheduled += self.microbatch

    def _host(self, position):
        tick = time.perf_counter()
        if self.pool is None:
            batch, indices, prepare = self._prepare(position)
        else:
            queued, future = self.queue.popleft()
            if queued != position:
                raise RuntimeError('Prefetch sampler order changed')
            batch, indices, prepare = future.result()
            self._fill()
        return batch, indices, {'cpu_prepare_seconds': prepare, 'host_wait_seconds': time.perf_counter()-tick}

    def _copy(self, prepared):
        batch, indices, timing = prepared
        tick = time.perf_counter()
        begin = end = None
        if self.device.type == 'cuda':
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        def transfer():
            if begin is not None:
                begin.record()
            result = {key: value.to(self.device, non_blocking=self.pin) if isinstance(value, torch.Tensor) else value
                      for key, value in batch.items()}
            if end is not None:
                end.record()
            return result
        if self.copy_stream is not None:
            with torch.cuda.stream(self.copy_stream):
                result = transfer()
            self.host_inflight.append((end, batch))
        else:
            result = transfer()
        timing['h2d_submit_seconds'] = time.perf_counter()-tick
        return result, indices, timing, begin, end

    def take(self, position):
        if self.closed or position != self.position or position >= self.end:
            raise RuntimeError('Batch stream must be consumed at its exact uncommitted position')
        if self.staged is None:
            result = self._copy(self._host(position))
        else:
            staged_position, result = self.staged
            if staged_position != position:
                raise RuntimeError('Staged device batch position changed')
            self.staged = None
        batch, indices, timing, begin, end = result
        if self.copy_stream is not None:
            consumer = torch.cuda.current_stream(self.device)
            consumer.wait_event(end)
            for value in batch.values():
                if isinstance(value, torch.Tensor):
                    value.record_stream(consumer)
        while self.host_inflight and self.host_inflight[0][0].query():
            self.host_inflight.popleft()
        self.position += self.microbatch
        return batch, indices, timing, (begin, end)

    def stage_next(self):
        """Enqueue one ready next H2D copy, without waiting for CPU decoding."""
        if self.copy_stream is None or self.staged is not None or self.position >= self.end:
            return False
        if not self.queue or not self.queue[0][1].done():
            return False
        # A failed speculative read belongs to the next registered sample
        # position. Let the current valid optimizer update commit; take() will
        # surface the failure only when that position is actually consumed.
        if self.queue[0][1].exception() is not None:
            return False
        self.staged = (self.position, self._copy(self._host(self.position)))
        return True

    def close(self):
        if self.closed:
            return
        self.closed = True
        for _, future in self.queue:
            future.cancel()
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
        if self.copy_stream is not None:
            self.copy_stream.synchronize()
        self.queue.clear()
        self.host_inflight.clear()
        self.staged = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
