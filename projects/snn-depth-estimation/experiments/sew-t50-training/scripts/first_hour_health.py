"""Read scalar telemetry only; no CUDA work or parameter changes."""
import json
import math
import time


def check_health(output, previous_step, now=None):
    now = time.time() if now is None else now
    status = json.loads((output / 'status.json').read_text())
    rows = []
    with (output / 'events.jsonl').open() as stream:
        for line in stream:
            if line.endswith('\n'):
                row = json.loads(line)
                if row['event'] == 'update':
                    rows.append(row)
    recent = rows[-100:]
    errors = []
    step = status.get('step', 0)
    if step <= previous_step:
        errors.append('optimizer update counter did not advance')
    if not recent or now - recent[-1]['time'] > 900:
        errors.append('no recent optimizer update')
    for row in recent:
        if not all(math.isfinite(row[k]) for k in ('loss', 'train_epe', 'preclip_norm', 'lr_used')):
            errors.append('nonfinite telemetry')
        if not row['replay_exact'] or row['preclip_norm'] <= 0:
            errors.append('replay/gradient check failed')
        if not all(row['gradient_groups'].get(k, 0) > 0
                   for k in ('fnet', 'cnet', 'update_block', 'context_zqr_convs')):
            errors.append('missing component gradient')
    if not (output / 'latest.pth').exists():
        errors.append('recovery checkpoint absent')
    return dict(time=now, step=step, passed=not errors, errors=sorted(set(errors)),
                latest_loss=recent[-1]['loss'] if recent else None,
                checked_updates=len(recent))
