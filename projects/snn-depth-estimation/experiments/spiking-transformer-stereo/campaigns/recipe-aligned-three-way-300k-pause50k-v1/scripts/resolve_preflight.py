"""Reduce exactly four released480-update engineering workers; never release training."""
import argparse
from pathlib import Path
from stage import read,write,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();paths=sorted(a.root.glob('wave*/rank*/result.json'))
    if len(paths)!=4:raise RuntimeError('exactly four engineering worker results required')
    results=[read(path) for path in paths]
    if sorted((r['wave'],r['rank'],r['arm']) for r in results)!=[(1,0,'original'),(1,1,'original_I'),(2,0,'spiking_normalized'),(2,1,'original')]:raise RuntimeError('worker identities differ')
    if len({r['source_identity_sha256'] for r in results})!=1 or len({r['registration_sha256'] for r in results})!=1:raise RuntimeError('mixed source/registration')
    for r in results:
        if r['status']!='PASS' or r['engineering_optimizer_updates']!=480 or r['scientific_optimizer_updates']!=0 or r['optimizer_step_min_max']!=[480,480] or r['scheduler_epoch']!=480 or r['scheduler_total_steps']!=300100:raise RuntimeError('worker budget/clock or health failure')
        if any(r[k]!='PASS' for k in ('named_parameter_gate','resume_gate','learning_gate','throughput_gate')) or not r['final_model_optimizer_finite']:raise RuntimeError('worker subgate failed')
    if len({r['initialization']['common_state_sha256'] for r in results})!=1:raise RuntimeError('common seed1234 initialization differs')
    write(a.root/'status.json',dict(status='PASS',source_identity_sha256=results[0]['source_identity_sha256'],registration_sha256=results[0]['registration_sha256'],
        engineering_optimizer_updates=1920,scientific_optimizer_updates=0,concurrent_gpus=2,waves=2,
        results={str(path.relative_to(a.root)):r for path,r in zip(paths,results)},result_sha256={str(path.relative_to(a.root)):sha(path) for path in paths},root_review_required_before_scientific_release=True))


if __name__=='__main__':main()
