"""Reconstruct baseline-repeat scene EPE from saved additive scalar receipts; never decode GT/maps."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def close(a,b):
    if not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Scalar reconstruction mismatch')


def reconstruct(frames):
    groups={}
    for row in frames:
        if row['valid_pixels']<=0:raise ValueError('Nonempty fixed GT support required')
        close(row['absolute_error_sum']/row['valid_pixels'],row['epe'])
        groups.setdefault((row['scene_id'],row['condition']),[]).append(row)
    scenes={}
    for (scene,condition),rows in groups.items():
        scenes.setdefault(scene,{})[condition]=sum(r['absolute_error_sum'] for r in rows)/sum(r['valid_pixels'] for r in rows)
    for values in scenes.values():
        if set(values)!=set(('first','second','repeat')):raise ValueError('Incomplete per-scene call panel')
    means={name:sum(r[name] for r in scenes.values())/len(scenes) for name in ('first','second','repeat')}
    return dict(equal_scene_epe=means,paired_scene_epe_changes=[dict(scene_id=scene,**values,
        second_minus_first=values['second']-values['first'],repeat_minus_second=values['repeat']-values['second'])
        for scene,values in sorted(scenes.items())],equal_scene_second_minus_first=means['second']-means['first'],
        equal_scene_repeat_minus_second=means['repeat']-means['second'])


def analyze(root,inventory_path):
    inventory=json.loads(inventory_path.read_text()); files=inventory['files']
    if (inventory['exact_file_set_verified'] is not True or inventory['remote_before_after_equal'] is not True
            or set(files)!={p.name for p in root.iterdir() if p.is_file()} or len(files)!=21):
        raise ValueError('Full copied file set and before/after receipt required')
    for name,record in files.items():
        p=root/name
        if p.is_symlink() or p.stat().st_size!=record['bytes'] or sha(p)!=record['sha256']:
            raise ValueError('Copied raw member changed')
    r=json.loads((root/'result.json').read_text())
    if (r['status']!='COMPLETED_BASELINE_REPEAT_DIAGNOSTIC' or len(r['samples'])!=20
            or len(r['frame_metrics'])!=60 or r['test_access'] is not False or r['prior_forward_calls']!=0
            or r['weights_buffers_unchanged'] is not True or r['replace_pilot_metrics'] is not False):
        raise ValueError('Complete unchanged baseline diagnostic required')
    if r['slurm_job_id']!=inventory['job_id'] or inventory['sacct'].split('|')[1]!='COMPLETED':
        raise ValueError('Completed exact allocation receipt required')
    frame_by={}
    for row in r['frame_metrics']:
        key=(row['sample_id'],row['condition'])
        if key in frame_by:raise ValueError('Duplicate frame/call')
        frame_by[key]=row
    ids=r['config']['sample_ids']
    for ordinal,sample in enumerate(r['samples']):
        if (sample['ordinal']!=ordinal or sample['sample_id']!=ids[ordinal] or sample['process_cold']!=(ordinal==0)
                or sample['artifact_sha256']!=files[sample['artifact']]['sha256']
                or len(sample['calls'])!=3
                or not all(c['input_unchanged'] and c['weights_buffers_unchanged'] for c in sample['calls'])):
            raise ValueError('Exact ordered repeat/input/state/artifact binding required')
        metrics=[frame_by[(sample['sample_id'],name)] for name in ('first','second','repeat')]
        if len({m['valid_pixels'] for m in metrics})!=1:raise ValueError('GT support changed across calls')
        for name in ('primary_first_vs_second','primary_second_vs_repeat'):
            for values in sample[name].values():
                if not 0<=values['changed_pixels']<=values['pixels'] or not 0<=values['mean_absolute_difference']<=values['maximum_absolute_difference']:
                    raise ValueError('Invalid pointwise scalar receipt')
    summary=reconstruct(r['frame_metrics'])
    for name,value in summary['equal_scene_epe'].items():close(value,r['equal_scene_epe'][name])
    for key in ('equal_scene_second_minus_first','equal_scene_repeat_minus_second'):close(summary[key],r[key])
    for computed,saved in zip(summary['paired_scene_epe_changes'],r['paired_scene_epe_changes']):
        if computed['scene_id']!=saved['scene_id']:raise ValueError('Scene order mismatch')
        for key in ('first','second','repeat','second_minus_first','repeat_minus_second'):close(computed[key],saved[key])
    first_id=ids[0];first={name:frame_by[(first_id,name)]['epe'] for name in ('first','second','repeat')}
    first.update(second_minus_first=first['second']-first['first'],repeat_minus_second=first['repeat']-first['second'])
    samples=r['samples']
    return dict(case=r['case'],job_id=r['slurm_job_id'],allocated_seconds=int(inventory['sacct'].split('|')[3]),
        source_identity_sha256=r['source_identity_sha256'],source_registry_sha256=r['source_registry_sha256'],
        config_sha256=r['config_sha256'],checkpoint_sha256=r['checkpoint']['checkpoint_sha256'],
        gpu=r['gpu'],allocated_gpu_uuid=r['allocated_gpu_uuid'],process_id=r['process_id'],
        samples=20,scenes=10,primary_predictions=60,**summary,
        process_cold_first_target=dict(sample_id=first_id,epe=first,pointwise=samples[0]['primary_first_vs_second'],
            full_returned_max_difference=samples[0]['comparisons']['first_vs_second']['maximum_absolute_difference']),
        subsequent19_targets=reconstruct([m for m in r['frame_metrics'] if m['ordinal']>0]),
        first_second_changed_targets=sum(s['primary_first_vs_second']['all_pixels']['changed_pixels']>0 for s in samples),
        second_repeat_changed_targets=sum(s['primary_second_vs_repeat']['all_pixels']['changed_pixels']>0 for s in samples),
        second_repeat_all_returned_exact_targets=sum(s['comparisons']['second_vs_repeat']['exact_equal'] for s in samples),
        post_rng_matches_targets=sum(s['post_rng_matches_between_calls'] for s in samples),
        max_primary_first_second_difference=max(s['primary_first_vs_second']['all_pixels']['maximum_absolute_difference'] for s in samples),
        max_primary_second_repeat_difference=max(s['primary_second_vs_repeat']['all_pixels']['maximum_absolute_difference'] for s in samples),
        result_sha256=files['result.json']['sha256'],inventory_sha256=sha(inventory_path),raw_root=str(root),
        verified_raw_files=len(files),verified_raw_bytes=inventory['total_bytes'],
        pointwise_scope='Saved scalar receipts; NPZ primary maps/GT archived and hashed, not decoded by this analyzer')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,nargs=2,required=True);p.add_argument('--inventory',type=Path,nargs=2,required=True)
    p.add_argument('--readback-root',type=Path,required=True);p.add_argument('--readback-inventory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=[analyze(root,inv) for root,inv in zip(a.raw,a.inventory)]
    if [r['case'] for r in rows]!=['original-I-spike','original-I-event']:raise ValueError('Fixed two baseline cases required')
    if rows[0]['job_id']==rows[1]['job_id']:raise ValueError('Fresh independent allocation/process per modality required')
    raw=[json.loads((root/'result.json').read_text()) for root in a.raw]
    if raw[0]['config']['sample_ids']!=raw[1]['config']['sample_ids']:raise ValueError('Common20 ordered IDs required')
    proof_inventory=json.loads(a.readback_inventory.read_text())
    proof_path=a.readback_root/'result.json'
    proof=json.loads(proof_path.read_text())
    record=proof_inventory['files']['result.json']
    if (set(proof_inventory['files'])!={'result.json'} or proof_path.is_symlink()
            or proof_path.stat().st_size!=record['bytes'] or sha(proof_path)!=record['sha256']
            or proof_inventory['remote_before_after_equal'] is not True
            or proof_inventory['exact_file_set_verified'] is not True
            or proof_inventory['sacct'].split('|')[1]!='COMPLETED'
            or proof['status']!='PASS_ALL120_STORED_PREDICTION_MAPS_AND40_GT_SUPPORTS'
            or proof['prediction_maps']!=120 or proof['GT_supports']!=40 or proof['pointwise_comparisons']!=80
            or proof['model_loaded'] is not False or proof['test_access'] is not False
            or [v['source_report_sha256'] for v in proof['cases']]!=[v['result_sha256'] for v in rows]
            or not all(v['remote_before_after_hashes_equal'] for v in proof['cases'])):
        raise ValueError('Exact independent registered Slurm saved-map readback required')
    readback=dict(job_id=proof['slurm_job_id'],allocated_seconds=int(proof_inventory['sacct'].split('|')[3]),
        status=proof['status'],prediction_maps=120,GT_supports=40,pointwise_comparisons=80,
        source_identity_sha256=proof['source_identity_sha256'],source_registry_sha256=proof['source_registry_sha256'],
        config_sha256=proof['config_sha256'],result_sha256=record['sha256'],inventory_sha256=sha(a.readback_inventory),
        raw_root=str(a.readback_root),allocated_gpu_uuid=proof['allocated_gpu_uuid'])
    result=dict(schema='stereo-study-baseline-repeat-analysis-v1',scope='Two fresh processes, one per modality, one process-cold target each; descriptive numerical reproducibility, no causal mechanism/selection/pilot replacement',
        analysis='Mac analyzer: saved additive scalar arithmetic only. Separate registered Slurm readback independently verifies archived primary/GT pixels; no new inference',
        independent_saved_map_readback=readback,
        analyzer_sha256=sha(Path(__file__)),cases=rows,allocated_seconds=sum(r['allocated_seconds'] for r in rows)+readback['allocated_seconds'],
        raw_files=sum(r['verified_raw_files'] for r in rows)+1,raw_bytes=sum(r['verified_raw_bytes'] for r in rows)+record['bytes'],test_access=False)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fields=['case','job_id','first','second','repeat','second_minus_first','repeat_minus_second']
    with a.output.with_suffix('.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow(dict(case=row['case'],job_id=row['job_id'],**row['equal_scene_epe'],
            second_minus_first=row['equal_scene_second_minus_first'],repeat_minus_second=row['equal_scene_repeat_minus_second']))
    print(json.dumps(dict(output=str(a.output),sha256=sha(a.output),cases=[dict(case=r['case'],**r['equal_scene_epe'],
        delta=r['equal_scene_second_minus_first'],changed_targets=r['first_second_changed_targets']) for r in rows])))

if __name__=='__main__':main()
