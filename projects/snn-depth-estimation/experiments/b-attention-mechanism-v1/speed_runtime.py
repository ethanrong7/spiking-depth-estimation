"""Less host synchronization; unchanged disparity objective and gradient clipping."""
import math
import os
from contextlib import nullcontext

import torch


class SpeedRuntime:
    checkpoint_every = 500
    diagnostic_every = 100
    status_every = 25

    def __init__(self):
        setting = os.environ.get('B_ATTENTION_PROFILE_CPU', '0')
        if setting not in ('0', '1'):
            raise ValueError('B_ATTENTION_PROFILE_CPU must be 0 or 1')
        self.profile_cpu = setting == '1'
        self.profiler = None
        self.profile_steps = 0

    def start_profile(self, outdir):
        if not self.profile_cpu:
            return
        def save(profile):
            profile.export_chrome_trace(str(outdir/'cpu-trace.json'))
            (outdir/'cpu-profile.txt').write_text(profile.key_averages().table(
                sort_by='self_cpu_time_total', row_limit=60)+'\n')
        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(wait=5, warmup=2, active=5, repeat=1),
            on_trace_ready=save)
        self.profiler.start()

    def span(self, name):
        return torch.profiler.record_function(name) if self.profiler else nullcontext()

    def profile_step(self):
        if self.profiler:
            self.profiler.step()
            self.profile_steps += 1
            if self.profile_steps >= 12:
                self.stop_profile()

    def stop_profile(self):
        if self.profiler:
            self.profiler.stop()
            self.profiler = None

    @property
    def config(self):
        return dict(version=1, loss='fixed_shape_masked_sum',
                    checkpoint_every=self.checkpoint_every,
                    diagnostic_every=self.diagnostic_every,
                    status_every=self.status_every,
                    profile_cpu=self.profile_cpu,
                    host_metric_reads_per_update=1)

    def loss(self, preds, target, valid):
        # Boolean indexing allocates a dynamically sized result for each of the
        # 16 predictions. Fixed-shape where/sum avoids those nonzero operations.
        mask = valid[:, None] & torch.isfinite(target) & (target.abs() < 700)
        count = mask.sum()
        target = torch.where(mask, target, 0)
        gamma = .9 ** (15 / max(1, len(preds) - 1))
        loss = sum(gamma ** (len(preds) - i - 1) *
                   torch.where(mask, (p - target).abs(), 0).sum() / count.clamp_min(1)
                   for i, p in enumerate(preds))
        # Keep checks on device until the single pre-optimizer metric transfer.
        finite = torch.stack([torch.isfinite(p).all() for p in preds]).all()
        return loss, finite & (count > 0) & torch.isfinite(loss)

    def summarize(self, losses, checks, norm, branch_grads):
        names, values = list(branch_grads), list(branch_grads.values())
        packed = torch.stack([sum(losses), norm.detach(),
                              torch.stack(checks).all().to(norm.dtype), *values])
        # One host transfer instead of a scalar read for every tensor.
        total, norm_value, okay, *gradient_values = packed.detach().cpu().tolist()
        if not okay or not math.isfinite(total) or not math.isfinite(norm_value) or norm_value <= 0:
            raise FloatingPointError('Nonfinite predictions/loss/gradient or empty mask/zero gradient')
        if not all(math.isfinite(v) for v in gradient_values):
            raise FloatingPointError('Nonfinite attention gradients')
        return total, norm_value, dict(zip(names, gradient_values))
