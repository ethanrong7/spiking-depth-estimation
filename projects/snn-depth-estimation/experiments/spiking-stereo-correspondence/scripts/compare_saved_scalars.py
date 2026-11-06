"""Read saved scalars only: matched P2 endpoint versus immutable historical arms.

No torch/model/checkpoint/data-array imports or execution. Pending observations
stay missing; only a terminal, artifact-verified P2 can populate primary results.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

PROJECT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).resolve().parents[1]
BASELINE=PROJECT/'experiments/spiking-transformer-stereo/campaigns/recipe-aligned-three-way-v1'
MANIFEST=PROJECT/'experiments/spiking-transformer-stereo/configs/manifest.json'
MANIFEST_SHA='bad4369b6fd8b0cdf871a20b1727c2d9b59733a16bd33f75904227bdfef4510a'
BASE_ID='cb43497fdc02bf612794dd48d516ee71e5658f10969e47ed6f4683591f3e8ec5'
HISTORICAL_ID='83e3aadbe4f2618dd76c8bb3021b903e11e6a11a9ffbdf074aff9f1869e557f6'
P2_ID='31ed4dcc25a0980c96b5df2b57fe090cbb229d378386f83e3b4badd618adcec3'
P2_RUN='main-seed1234-u2500-p2-v1'
GRID=(0,1000,2500)
ENDPOINT=2500
ARMS={'original-seed1234-10k-v1':'Original ALIF', 'original_i-seed1234-10k-v1':'ALIF + current I'}
METRICS=('mean_scene_epe','pooled_epe','bad1','bad3','bad5')
BASELINE_PINS={
  "configs/registration.json": "0c0fc90d4e365353f5d41e83454ea80c86031b0eaf015b5f231881801e9db003",
  "configs/original-seed1234-10k-v1-physical4.json": "cd21187946ae32c349b4cfdbad46aec541e72f25cf8e8272bad0fe0d0ba5d317",
  "configs/original_i-seed1234-10k-v1-physical4.json": "2c41a978f87a5b1f0e9e34f9e5b9fda3d8dde654b1bf7e9e66fc954818dc452d",
  "results/audited/dev-trajectories.csv": "d762ba2d240ca3116a3b188ac11ed15b23b0afebde3d167456b53e04a82e4970",
  "results/audited/dev-scenes.csv": "3d41bf65b103f9232520ee4cbfdce65b68d8ae1e1b2cdbb69e4e4eb84d1d46fb",
  "results/audited/dev-frames.csv": "a03fc12ccfa2c287bddbb5759f64f43f6bd7181c92a52c15c92f926b6094262f",
  "results/audited/summary.json": "8d95ed430db2da48f8b6ca66b20a6b2e8fc53a0cdfe5cc9f1262578f0d25cbc4",
  "results/audited/final-report.md": "1ddfe70171cfcbdfc190852f132b6b3d865aa3e7e967360cb9d11df262184f7c"
}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):return json.loads(Path(path).read_text())


def read_jsonl(path):
    rows=[]
    for line in Path(path).read_text().splitlines():
        if line.strip():rows.append(json.loads(line))
    return rows


def csv_rows(path):
    with Path(path).open(newline='') as stream:return list(csv.DictReader(stream))


def numeric(row, names):return {key:float(row[key]) for key in names}


def close(actual,expected,label,tolerance=1e-9):
    if not math.isfinite(actual) or not math.isfinite(expected) or abs(actual-expected)>tolerance:
        raise ValueError(f'{label} mismatch: {actual} versus {expected}')


def frame_map(rows):
    result={}
    for row in rows:
        key=(str(row['scene']),str(row['frame']))
        if key in result:raise ValueError('duplicate frame '+str(key))
        result[key]=row
    return result


def scene_means(rows):
    return {scene:statistics.mean(float(r['epe']) for r in rows if r['scene']==scene)
        for scene in sorted({r['scene'] for r in rows})}


def derived_metrics(scenes):
    values=list(scenes.values())
    return dict(median_scene_epe=statistics.median(values),worst_five_scene_epe=statistics.mean(sorted(values,reverse=True)[:5]))


def load_baselines():
    provenance={name:dict(sha256=digest(BASELINE/name),size=(BASELINE/name).stat().st_size) for name in BASELINE_PINS}
    for name,expected in BASELINE_PINS.items():
        if provenance[name]['sha256']!=expected:raise ValueError('pinned historical artifact changed: '+name)
    if digest(MANIFEST)!=MANIFEST_SHA:raise ValueError('historical manifest changed')
    manifest=read_json(MANIFEST);registration=read_json(BASELINE/'configs/registration.json')
    audit=read_json(BASELINE/'results/audited/summary.json')
    if (audit['status']!='ALL_THREE_AUDITED_PASS' or audit['source_identity_sha256']!=HISTORICAL_ID
            or registration['base_source_identity_sha256']!=BASE_ID or registration['manifest_sha256']!=MANIFEST_SHA):
        raise ValueError('historical source/manifest audit identity mismatch')
    if len(manifest['train'])!=960 or len(manifest['dev'])!=60:raise ValueError('native Train960/Dev60 required')
    cohort={(r['scene'],r['frame']) for r in manifest['dev']}
    trajectories=csv_rows(BASELINE/'results/audited/dev-trajectories.csv')
    frames=csv_rows(BASELINE/'results/audited/dev-frames.csv')
    scenes=csv_rows(BASELINE/'results/audited/dev-scenes.csv')
    curves={};references={}
    for arm,label in ARMS.items():
        curves[arm]={}
        for update in GRID:
            totals=[r for r in trajectories if r['run_id']==arm and int(r['update'])==update]
            group=[dict(r,valid_pixels=int(r['valid_pixels']),**numeric(r,('epe','bad1','bad3','bad5')))
                for r in frames if r['run_id']==arm and int(r['update'])==update]
            if len(totals)!=1 or set(frame_map(group))!=cohort:raise ValueError('historical cohort/visit mismatch')
            per_scene=scene_means(group)
            supplied={r['scene']:float(r['epe']) for r in scenes if r['run_id']==arm and int(r['update'])==update}
            if set(supplied)!=set(per_scene):raise ValueError('historical scene set differs')
            for scene,value in supplied.items():close(per_scene[scene],value,'historical per-scene EPE')
            metrics=numeric(totals[0],METRICS)
            close(statistics.mean(per_scene.values()),metrics['mean_scene_epe'],'historical mean-scene EPE')
            curves[arm][update]=dict(update=update,presentations=update*8,**metrics,**derived_metrics(per_scene),
                scene_epe=supplied,per_frame=group,valid_pixels=sum(r['valid_pixels'] for r in group),split='dev')
            references[update]=references.get(update,frame_map(group))
            if {key:int(row['valid_pixels']) for key,row in references[update].items()}!={key:int(row['valid_pixels']) for key,row in frame_map(group).items()}:
                raise ValueError('historical arms use unequal valid-pixel counts')
    return curves,references,dict(historical_source_identity=HISTORICAL_ID,base_source_identity=BASE_ID,
        manifest_sha256=MANIFEST_SHA,files=provenance,all_frames_full_native_support=all(int(r['valid_pixels'])==100000 for group in references.values() for r in group.values()),cohort_frames=60,cohort_scenes=len({r['scene'] for r in manifest['dev']}),
        mask_binding='Exact immutable ManifestDataset/reader lineage; per-frame valid counts verified. Counts alone do not prove pixel-mask identity.')


def validate_metadata(metadata):
    config=metadata.get('config',{})
    expected=dict(stage='main',run_id=P2_RUN,model='correspondence',seed=1234,updates=2500,effective_batch=8,time_steps=50)
    if any(config.get(key)!=value for key,value in expected.items()):raise ValueError('only exact P2 main2500 is eligible; engineering/P1 excluded')
    if (metadata.get('manifest_sha256')!=MANIFEST_SHA or metadata.get('base_source',{}).get('source_identity_sha256')!=BASE_ID
            or metadata.get('source',{}).get('source_identity_sha256')!=P2_ID):
        raise ValueError('P2 source/manifest lineage mismatch')


def validate_new_visit(row,references,require_frames=True):
    update=int(row['update'])
    if update not in GRID:raise ValueError('unscheduled P2 visit')
    if row.get('split')!='dev' or int(row.get('scientific_training_updates',-1))!=update:
        raise ValueError('diagnostic/engineering visit is ineligible')
    if not require_frames:return
    supplied=frame_map(row.get('per_frame',[]));expected=references[update]
    if set(supplied)!=set(expected):raise ValueError('new/historical frame cohort differs')
    for key,record in supplied.items():
        if int(record['valid_pixels'])!=int(expected[key]['valid_pixels']):raise ValueError('new/historical per-frame valid count differs')
    scenes=scene_means(list(supplied.values()))
    if set(scenes)!=set(row['scene_epe']):raise ValueError('new per-scene set differs')
    for scene,value in scenes.items():close(value,float(row['scene_epe'][scene]),'new scene EPE')
    close(statistics.mean(scenes.values()),float(row['mean_scene_epe']),'new mean-scene EPE')
    count=sum(int(r['valid_pixels']) for r in supplied.values())
    if count!=int(row['valid_pixels']):raise ValueError('new valid count aggregate differs')
    close(sum(float(r['error_sum']) for r in supplied.values())/count,float(row['pooled_epe']),'new pooled EPE')
    for key in ('bad1','bad3','bad5'):
        close(sum(float(r[key])*int(r['valid_pixels']) for r in supplied.values())/count,float(row[key]),'new '+key,tolerance=1e-6)


def verify_receipt(root,path):
    receipt=read_json(path)
    jobs=read_json(STUDY/'results/jobs.json')
    job=next((r for r in jobs if str(r.get('job_id'))=='2598'),{})
    if job.get('state')!='COMPLETED' or job.get('exit_code')!='0:0' or job.get('source_identity')!=P2_ID:
        raise ValueError('terminal successful P2 scheduler receipt required')
    if int(receipt.get('allocated_gpu_seconds',-1))!=int(job.get('allocated_seconds',-2)):
        raise ValueError('artifact/scheduler accounting differs')
    if (str(receipt.get('job_id'))!='2598' or receipt.get('source_identity')!=P2_ID or not receipt.get('remote_before_after_equal')
            or not (receipt.get('local_exact_transferred_set_sha_size_equal') or receipt.get('local_exact_set_sha_size_equal'))):
        raise ValueError('exact terminal P2 artifact verification required')
    records=receipt.get('transferred_files',receipt.get('files',[]))
    by_name={r['path']:r for r in records}
    for name in ('metadata.json','result.json','validation.jsonl','training.jsonl','sample-order.json'):
        if name not in by_name:raise ValueError('verified required scalar file absent: '+name)
    for name,record in by_name.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or Path(name).suffix not in ('.json','.jsonl'):
            raise ValueError('only safe scalar artifact members are accepted')
        actual=root/name
        if actual.stat().st_size!=int(record['size']) or digest(actual)!=record['sha256']:
            raise ValueError('verified scalar artifact changed: '+name)
    return receipt


def load_new(root,references,receipt_path=None):
    if root is None:return {},False,dict(status='PENDING',reason='No saved P2 scalars supplied')
    root=Path(root)
    metadata_path=root/'metadata.json'
    if not metadata_path.exists():metadata_path=root/'metadata-summary.json'
    metadata=read_json(metadata_path);validate_metadata(metadata)
    rows=read_jsonl(root/'validation.jsonl') if (root/'validation.jsonl').exists() else read_json(root/'validation-summary.json').get('rows',[])
    visits={}
    for row in rows:
        update=int(row['update'])
        if update in visits:raise ValueError('duplicate P2 validation visit')
        validate_new_visit(row,references,require_frames=bool(row.get('per_frame')))
        visits[update]=row
    complete=False;result=read_json(root/'result.json') if (root/'result.json').exists() else None
    proof=dict(status='PARTIAL_SAVED_SCALARS',source_identity=P2_ID,metadata_sha256=digest(metadata_path),
        observed_visits=sorted(visits),final_result_verified=False,config_sha256=metadata.get('config_sha256'),
        git_commit=metadata['source'].get('git_commit'),source_archive_sha256=metadata['source'].get('archive_sha256'),
        loader_driver_sha256=metadata['base_source'].get('source_files',{}).get('projects/snn-depth-estimation/experiments/spiking-transformer-stereo/scripts/run_experiment.py'),
        observation_file_sha256=digest(root/'validation.jsonl') if (root/'validation.jsonl').exists() else digest(root/'validation-summary.json'),
        per_frame_cohort_validated_updates=[int(r['update']) for r in visits.values() if r.get('per_frame')],
        partial_scope='Saved main Dev scalars only; primary/secondary/AUC/ratios remain pending until terminal artifact review')
    if result and receipt_path:
        receipt=verify_receipt(root,receipt_path)
        checks=dict(status='COMPLETE_MAIN',updates=2500,presentations=20000,scientific_training_updates=2500,
            optimizer_steps=2500,scheduler_steps=2500,parameters_finite=True,optimizer_state_finite=True)
        if any(result.get(k)!=v for k,v in checks.items()):raise ValueError('P2 terminal result does not pass exact endpoint/health contract')
        if set(visits)!=set(GRID):raise ValueError('completed P2 must preserve all registered visits')
        for row in visits.values():validate_new_visit(row,references,require_frames=True)
        training=read_jsonl(root/'training.jsonl')
        if [r['update'] for r in training]!=list(range(1,2501)) or [r['presentations'] for r in training]!=list(range(8,20001,8)):
            raise ValueError('P2 update/presentation log is incomplete')
        if any(r.get('stage')!='main' or r.get('scientific_training_updates')!=r['update'] for r in training):
            raise ValueError('non-main update in scientific curve')
        order=read_json(root/'sample-order.json')
        if sorted(order.get('order',[]))!=list(range(960)) or order.get('augmentation')!='historical recipe-hv-v1':
            raise ValueError('P2 sample-order or augmentation contract differs')
        for step,row in enumerate(training):
            expected_indices=[order['order'][(step*8+i)%960] for i in range(8)]
            expected_flags=[]
            for i in range(8):
                key=hashlib.sha256(f'recipe-hv-v1:1234:{step*8+i}'.encode()).digest()
                expected_flags.append([key[0]<128,key[1]<128])
            if row.get('sample_indices')!=expected_indices or row.get('augmentation_flags')!=expected_flags:
                raise ValueError('P2 presentation order/augmentation log differs')
        if result['final']!=visits[2500]:raise ValueError('terminal final row differs from saved validation')
        best=min((visits[u] for u in GRID if u>0),key=lambda r:float(r['mean_scene_epe']))
        close(float(result['best']),float(best['mean_scene_epe']),'secondary best trained EPE')
        if result['best_update']!=best['update']:raise ValueError('secondary best trained update differs')
        complete=True;proof.update(status='COMPLETE_ARTIFACT_VERIFIED',final_result_verified=True,
            receipt_sha256=digest(receipt_path),result_sha256=digest(root/'result.json'),allocated_gpu_seconds=receipt.get('allocated_gpu_seconds'))
    return visits,complete,proof


def common_auc(visits):
    if any(update not in visits for update in GRID):return None
    return sum((b-a)*(float(visits[a]['mean_scene_epe'])+float(visits[b]['mean_scene_epe']))/2 for a,b in zip(GRID,GRID[1:]))/ENDPOINT


def build_comparison(baselines,new_visits,complete):
    methods=[];curves=[];scenes=[]
    all_methods={arm:(label,baselines[arm],True) for arm,label in ARMS.items()}
    all_methods[P2_RUN]=('Spiking correspondence Transformer v1',new_visits,complete)
    for identity,(label,visits,eligible) in all_methods.items():
        final=visits.get(ENDPOINT) if eligible else None
        trained=[row for update,row in visits.items() if update>0 and update in GRID] if eligible else []
        best=min(trained,key=lambda row:float(row['mean_scene_epe'])) if trained else None
        row=dict(method=label,identity=identity,seed=1234,comparison_endpoint_updates=ENDPOINT,updates=ENDPOINT if eligible else None,presentations=20000 if eligible else None,
            status='MEASURED_HISTORICAL' if identity in ARMS else ('MEASURED_TERMINAL_AWAITING_SCIENTIFIC_REVIEW' if eligible else 'PENDING'),
            **{key:float(final[key]) if final else None for key in METRICS},
            **(derived_metrics(final['scene_epe']) if final else dict(median_scene_epe=None,worst_five_scene_epe=None)),
            best_trained_scheduled_update=int(best['update']) if best else None,
            best_trained_scheduled_epe=float(best['mean_scene_epe']) if best else None,
            normalized_common_grid_auc_including_zero=common_auc(visits) if eligible else None)
        methods.append(row)
        for update in GRID:
            observed=visits.get(update)
            curves.append(dict(method=label,identity=identity,update=update,presentations=update*8,
                status='MEASURED' if observed else 'PENDING',**{key:float(observed[key]) if observed else None for key in METRICS}))
    ratios={}
    for arm in ARMS:
        baseline=baselines[arm][ENDPOINT]
        actual=new_visits.get(ENDPOINT) if complete else None
        ratios[arm]=dict(new_over_reference_epe=float(actual['mean_scene_epe'])/baseline['mean_scene_epe'] if actual else None,
            new_minus_reference_epe=float(actual['mean_scene_epe'])-baseline['mean_scene_epe'] if actual else None)
        for scene,reference in baseline['scene_epe'].items():
            new=float(actual['scene_epe'][scene]) if actual else None
            scenes.append(dict(reference_identity=arm,scene=scene,reference_epe=reference,new_epe=new,new_minus_reference_epe=new-reference if new is not None else None))
    return dict(status='COMPLETE_SAVED_SCALAR_COMPARISON_AWAITING_ASTRA_REVIEW' if complete else 'PENDING_FINAL_2500',
        endpoint=2500,presentations=20000,common_grid=list(GRID),methods=methods,curves=curves,paired_scene_differences=scenes,
        endpoint_ratios=ratios,practical_10k_target_applicable=False,
        interpretation='One seed, whole-system comparison at matched2500 exposure. No 10k comparability/convergence/attention-functionality claim. P1 and disposable engineering Dev2 are excluded.')


def csv_text(rows):
    if not rows:return ''
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows);return stream.getvalue()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--p2-run-root',type=Path)
    parser.add_argument('--artifact-verification',type=Path)
    parser.add_argument('--output-prefix',type=Path,default=STUDY/'results/p2-saved-scalar-comparison-v1')
    args=parser.parse_args()
    baselines,references,pins=load_baselines()
    visits,complete,proof=load_new(args.p2_run_root,references,args.artifact_verification)
    report=build_comparison(baselines,visits,complete)
    report.update(generated_utc=datetime.now(timezone.utc).isoformat(),provenance=dict(baselines=pins,new=proof,
        analysis_script_sha256=digest(__file__),no_model_or_checkpoint_execution=True))
    prefix=args.output_prefix;prefix.parent.mkdir(parents=True,exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for label,key in [('methods','methods'),('curves','curves'),('scenes','paired_scene_differences')]:
        prefix.with_name(prefix.name+'-'+label).with_suffix('.csv').write_text(csv_text(report[key]))
    print(json.dumps(dict(status=report['status'],output=str(prefix.with_suffix('.json')),observed_new_visits=sorted(visits))))


if __name__=='__main__':main()
