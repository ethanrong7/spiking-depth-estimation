"""Resumable shuffled sampling and full-state checkpoints for the T50 study."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def segment_deadline(seconds, now, inherited=None, slurm_end=None):
    """Allocation time is independent of the persistent optimizer schedule."""
    if not 600 <= seconds <= 18000:
        raise ValueError('allocation budget must be between 10 minutes and five hours')
    deadline = float(inherited) if inherited is not None else now + seconds
    if slurm_end is not None:
        deadline = min(deadline, float(slurm_end))
    return deadline


class ShuffledBatchSampler(Sampler):
    """Track committed optimizer batches, never the loader's prefetch cursor."""
    def __init__(self, size, seed, consumed=0, batch_size=2, effective_batch=8):
        self.size, self.seed = int(size), int(seed)
        self.batch_size, self.effective_batch = batch_size, effective_batch
        self.usable = self.size // effective_batch * effective_batch
        if not self.usable or consumed < 0 or consumed % effective_batch:
            raise ValueError("invalid committed sample position")
        self.consumed = consumed

    def __iter__(self):
        cursor = self.consumed
        while True:
            epoch, offset = divmod(cursor, self.usable)
            order = np.random.default_rng(self.seed + epoch).permutation(self.size)
            for start in range(offset, self.usable, self.batch_size):
                yield order[start:start + self.batch_size].tolist()
                cursor += self.batch_size

    def commit_update(self):
        self.consumed += self.effective_batch

    def state_dict(self):
        return dict(size=self.size, seed=self.seed, consumed=self.consumed,
                    usable=self.usable, batch_size=self.batch_size,
                    effective_batch=self.effective_batch)

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in expected.keys() - {'consumed'}:
            if expected[key] != state[key]:
                raise ValueError(f"sampler changed: {key}")
        if state['consumed'] < 0 or state['consumed'] % self.effective_batch:
            raise ValueError("checkpoint splits an optimizer batch")
        self.consumed = state['consumed']


def atomic_save(payload, destination):
    destination = Path(destination)
    temporary = destination.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    descriptor = os.open(str(destination.parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_state(path, *, model, optimizer, scheduler, sampler, step, config,
               binding, rng, best, resume_verified):
    atomic_save(dict(format='sew-t50-training-v1', model=model.state_dict(),
                     optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     sampler=sampler.state_dict(), step=step, config=config,
                     binding=binding, rng=rng, best=best,
                     next_lr=[group['lr'] for group in optimizer.param_groups],
                     resume_verified=resume_verified), path)


def restore_state(path, *, model, optimizer, scheduler, sampler, config, binding):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['format'] != 'sew-t50-training-v1':
        raise ValueError("not a full-training checkpoint")
    if saved['config'] != config or saved['binding'] != binding:
        raise ValueError("source/configuration/data identity changed")
    model.load_state_dict(saved['model'])
    # Construct scheduler before loading optimizer; its constructor changes LR.
    optimizer.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    sampler.load_state_dict(saved['sampler'])
    if scheduler.total_steps != config['schedule_num_steps'] + 100:
        raise ValueError("schedule horizon changed")
    if scheduler.last_epoch != saved['step']:
        raise ValueError("scheduler clock differs from completed updates")
    if sampler.consumed != saved['step'] * config['effective_batch_size']:
        raise ValueError("sampler clock differs from completed updates")
    if [g['lr'] for g in optimizer.param_groups] != saved['next_lr']:
        raise ValueError("learning rate changed during checkpoint restoration")
    if scheduler.get_last_lr() != saved['next_lr']:
        raise ValueError("optimizer and scheduler learning rates disagree")
    for state in optimizer.state.values():
        if 'step' in state and int(state['step']) != saved['step']:
            raise ValueError("Adam clock differs from completed updates")
    return saved
