"""Reduce the four fixed concurrent engineering workers; no training release is created."""
import argparse
from pathlib import Path
from fresh import read,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    paths=sorted(a.root.glob('wave*/rank*/result.json'))
    if len(paths)!=4:raise RuntimeError('expected exactly four predeclared engineering worker results')
    results=[read(path) for path in paths]
    if sorted((r['wave'],r['rank'],r['arm']) for r in results)!=[(1,0,'original'),(1,1,'original_I'),(2,0,'spiking_normalized'),(2,1,'original')]:raise RuntimeError('health worker identities differ')
    if len({r['source_identity_sha256'] for r in results})!=1:raise RuntimeError('mixed health sources')
    if any(r['status']!='PASS' or r['scientific_optimizer_updates']!=0 or r['engineering_optimizer_updates']!=260 for r in results):raise RuntimeError('native health gate did not pass')
    if any(r['zero_step_native_backward_passes']!=9 or r['parity']['status']!='PASS' or not r['all128_input_identity_order_hv_lr_validity_exact'] or not r['final_rng_exact'] for r in results):raise RuntimeError('native numerical/stream identity gate differs')
    proof=[r['initialization']['common_state_sha256'] for r in results]
    if len(set(proof))!=1:raise RuntimeError('seed1234 shared encoder initialization differs')
    write(a.root/'status.json',dict(status='PASS',source_identity_sha256=results[0]['source_identity_sha256'],registration_sha256=results[0]['registration_sha256'],
        native_zero_optimizer_backward_passes=36,scientific_optimizer_updates=0,engineering_optimizer_updates=sum(r['engineering_optimizer_updates'] for r in results),
        total_measured_updates=4*256,physical_batch=4,effective_batch=8,concurrent_worker_count=2,two_waves=True,
        results={str(path.relative_to(a.root)):r for path,r in zip(paths,results)},root_training_review_required=True))


if __name__=='__main__':main()
