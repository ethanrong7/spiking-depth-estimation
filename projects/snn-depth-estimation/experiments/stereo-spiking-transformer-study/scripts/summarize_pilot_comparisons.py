"""Analyze compact completed pilot receipts only; no torch/data/model imports.

Writes a fresh .json and10-row .csv after all present receipts pass validation.
No checkpoint arrays, dataset files, Val curves or remote paths are opened.
Missing cells remain null; an existing partial/invalid receipt is rejected.
All intervals describe paired scene resampling for one selected training seed.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
TRAIN_SOURCE = 'e84cbdafeeb332ac33d6ae509168cd418bae1958bcc8d47b4fe2cc4c7349149a'
TRAIN_REGISTRY = '0ebf2bbd421a483707e547e1e5c269b1141beab3d0849596009279c90116bfcb'
MANIFEST = 'be80defb27c948b9766a6bb46381b56ce9dd1ee7062782f53dc6be2b518b99c5'
TRAIN_IDS = 'f15c7cefdb20f47fde5fab67f93a0cec5f7ccb977f618329decce200904057c9'
VAL_IDS = 'f4d15f5ed39ebf3a9f4accc0edb053163f63ec1fc9856d7da46d3650230b83ed'
DRAW_COUNT, SEED = 2000, 9471
MODEL_NAMES = ('original-I', 'A', 'B', 'continuous-A', 'continuous-B')
CONDITIONS = ('correct', 'shuffled', 'masked')
SCHEDULER_FIELDS = ('run','job_id','state','stop_at_update','config_sha256',
                    'source_identity','source_identity_sha256','git_commit','manifest_sha256','allocated_seconds')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def captured_json(path):
    """Parse and hash the same captured bytes, even if a watcher replaces it."""
    raw=path.read_bytes()
    return json.loads(raw),hashlib.sha256(raw).hexdigest(),len(raw)


def scheduler_receipts_used(jobs, details):
    result=[]
    for detail in details:
        wanted=((detail['run'],detail['completion']['job_id']),
                ('functionality-'+detail['run'],detail['diagnostic_job_id']))
        for run,job_id in wanted:
            matching=[row for row in jobs if row.get('run') == run and str(row.get('job_id')) == str(job_id)]
            require(len(matching) == 1 and matching[0].get('state') == 'COMPLETED', 'Used scheduler receipt must be unique and completed')
            result.append({key:matching[0][key] for key in SCHEDULER_FIELDS if key in matching[0]})
    return result


def require(value, message):
    if not value: raise ValueError(message)


def numeric(value, *, positive=False, unit=False):
    require(type(value) in (float, int) and math.isfinite(value), 'Finite numeric metric required')
    require(value > 0 if positive else value >= 0, 'Metric must be positive' if positive else 'Metric must be nonnegative')
    if unit: require(value <= 1, 'Metric fraction outside[0,1]')
    return float(value)


def near(a, b): return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-10)


def samples(draws=DRAW_COUNT, seed=SEED, scenes=10):
    return np.random.default_rng(seed).integers(0, scenes, size=(draws, scenes))


def vector(values):
    result = np.asarray(values, dtype=np.float64)
    require(result.ndim == 1 and len(result) > 0 and np.isfinite(result).all() and (result >= 0).all(), 'Invalid nonnegative finite scene vector')
    return result


def paired_ratio_draws(model, control, indices):
    a, b = vector(model), vector(control)
    require(a.shape == b.shape and indices.ndim == 2 and indices.shape[1] == len(a)
            and (indices >= 0).all() and (indices < len(a)).all(), 'Paired scene vector/index mismatch')
    denominator = b[indices].mean(1)
    require((denominator > 0).all() and b.mean() > 0, 'Ratio denominator must remain positive in every draw')
    return float(a.mean()/b.mean()), a[indices].mean(1)/denominator


def joint_interaction_draws(model_spike, control_spike, model_event, control_event, indices):
    rs, spike = paired_ratio_draws(model_spike, control_spike, indices)
    re, event = paired_ratio_draws(model_event, control_event, indices)
    require(rs > 0 and re > 0 and (spike > 0).all() and (event > 0).all(), 'Log interaction requires positive arm means in every draw')
    return float(math.log(rs)-math.log(re)), np.log(spike)-np.log(event)


def interval(estimate, draws, statistic):
    require(np.isfinite(draws).all() and math.isfinite(estimate), 'Nonfinite bootstrap statistic')
    return {'estimate': float(estimate), 'descriptive_scene_percentile95': [float(x) for x in np.quantile(draws,[.025,.975])],
            'statistic': statistic, 'bootstrap_draws': DRAW_COUNT, 'bootstrap_seed': SEED,
            'resampling_unit': 'paired_scene', 'training_seeds': 1,
            'interpretation': 'conditional descriptive scene interval; not seed/selection uncertainty or final inference'}


def ratio_result(model, control, indices):
    estimate, draws = paired_ratio_draws(model,control,indices)
    return interval(estimate, draws, 'ratio_of_equal_scene_EPE_means')


def expected_population(selection):
    rows = [row for row in selection['scenes'] if row['split'] == 'val']
    require(len(rows) == 10 and len({row['scene_id'] for row in rows}) == 10, 'Require ten unique registered Val scenes')
    result = {row['scene_id']: sorted(row['scene_id']+'/'+frame for frame in row['target_source_frames']) for row in rows}
    require(all(len(ids) == len(set(ids)) == 2 for ids in result.values()), 'Require two unique target IDs per scene')
    require(len({sid for ids in result.values() for sid in ids}) == 20, 'Require twenty unique target IDs')
    return dict(sorted(result.items()))


def population(scene_rows, frame_rows, expected, *, method='checkpoint', conditions=CONDITIONS):
    """Check scene aggregation/support using supplied compact sums, no pixels."""
    all_ids = {sid: scene for scene,ids in expected.items() for sid in ids}
    groups, supports = {}, {}
    for condition in conditions:
        frames = [row for row in frame_rows if row['method'] == method and row['condition'] == condition]
        scenes = [row for row in scene_rows if row['method'] == method and row['condition'] == condition]
        require(len(frames) == len(all_ids) and len({x['sample_id'] for x in frames}) == len(all_ids), 'Missing/duplicate frame population')
        require({x['sample_id'] for x in frames} == set(all_ids), 'Unequal target populations')
        require(len(scenes) == len(expected) and len({x['scene_id'] for x in scenes}) == len(expected), 'Missing/duplicate scene population')
        require({x['scene_id'] for x in scenes} == set(expected), 'Unequal scene populations')
        by_id = {x['sample_id']:x for x in frames}
        for row in frames:
            require(row['scene_id'] == all_ids[row['sample_id']], 'Frame scene membership mismatch')
            n, total = row['valid_pixels'], row['total_pixels']
            require(type(n) is int and type(total) is int and 0 < n <= total and total == 135*240, 'Invalid frame support')
            numeric(row['epe']); numeric(row['absolute_error_sum'])
            require(near(row['epe'],row['absolute_error_sum']/n) and near(row['coverage'],n/total), 'Frame metric/support inconsistency')
            for key in ('bad1','bad3','bad5'): numeric(row[key],unit=True)
        by_scene = {x['scene_id']:x for x in scenes}
        for sid, ids in expected.items():
            row = by_scene[sid]
            require(sorted(row['sample_ids']) == ids and len(row['sample_ids']) == len(set(row['sample_ids'])) == row['frames'] == 2,
                    'Scene target population/duplicate ID mismatch')
            n, total = sum(by_id[x]['valid_pixels'] for x in ids),sum(by_id[x]['total_pixels'] for x in ids)
            require(row['valid_pixels'] == n and row['total_pixels'] == total and near(row['coverage'],n/total), 'Scene support mismatch')
            numeric(row['epe'])
            require(near(row['epe'],sum(by_id[x]['absolute_error_sum'] for x in ids)/n), 'Scene EPE aggregation mismatch')
            for key in ('bad1','bad3','bad5'): numeric(row[key],unit=True)
        groups[condition] = {sid:by_scene[sid] for sid in expected}
        supports[condition] = {sid:[by_id[sid]['valid_pixels'],by_id[sid]['total_pixels']] for sid in sorted(all_ids)}
    first = conditions[0]
    require(all(supports[x] == supports[first] for x in conditions), 'Condition support mismatch')
    return groups,supports[first]


def completed_training(summary, checkpoint, config, jobs, run):
    require(summary.get('run') == run and summary.get('status_attributed_to_current_attempt') is True, 'Training completion attribution missing')
    status = summary['attempt_status']; metadata = summary['metadata']
    require(status.get('status') == 'COMPLETED_FROZEN_HORIZON' and status.get('update') == 1000
            and status.get('presentations') == 8000 and status.get('total_horizon_updates') == 1000,
            'Partial training receipt cannot enter comparison')
    require(metadata.get('config',{}).get('seed') == SEED and metadata['config'].get('total_updates') == 1000
            and metadata.get('test_access') == 'sealed' and metadata.get('pretraining') is False, 'Training metadata seed/scope mismatch')
    for key,value in checkpoint['binding'].items():
        if key != 'gpu_name': require(metadata['binding'].get(key) == value, 'Completion/checkpoint binding mismatch: '+key)
    matching = [x for x in jobs if x.get('run') == run and str(x.get('job_id')) == str(summary['observed_job_id'])]
    require(len(matching) == 1 and matching[0].get('state') == 'COMPLETED'
            and matching[0].get('stop_at_update') == 1000 and matching[0].get('config_sha256') == checkpoint['binding']['config_sha256'],
            'No completed full-horizon scheduler receipt')
    require(status.get('best_update') == checkpoint['state']['best_update'] and status.get('best_val_epe') == checkpoint['state']['best_val_epe'],
            'Full-horizon best selection differs from diagnostic checkpoint')
    return {'job_id': str(summary['observed_job_id']), 'updates':1000,'presentations':8000,
            'fresh_seed_model_sha256':metadata['fresh_seed_model_sha256']}


def validate_receipt(report, cell, training_config, training_config_sha, summary, jobs, expected):
    require(report.get('status') == 'COMPLETED_REGISTERED_DIAGNOSTIC_NO_SCIENTIFIC_PASS_THRESHOLD'
            and report.get('mode') == 'checkpoint' and report.get('purpose') == 'engineering'
            and report.get('test_access') is False and report.get('fit_includes_val') is False
            and report.get('test_selection') is False and report.get('confidence_filter') is False,
            'Partial/unregistered/filtered functionality receipt rejected')
    cfg,checkpoint = report['config'],report['checkpoint']; binding,state = checkpoint['binding'],checkpoint['state']
    require(cfg.get('registered') is True and cfg.get('seed') == SEED and cfg.get('test_access') == 'sealed'
            and cfg.get('training_run') == cell['run'] and cfg.get('training_config') == cell['config']
            and cfg.get('training_source_snapshot') == 'source-prefetched-first-seed-pilot-v12'
            and cfg.get('shift_pixels') == 2, 'Diagnostic configuration/seed/lineage mismatch')
    pinned = {'source_identity_sha256':TRAIN_SOURCE,'source_registry_sha256':TRAIN_REGISTRY,'manifest_sha256':MANIFEST,
              'total_updates':1000,'effective_batch':8,'training_samples':320,'validation_samples':20,
              'train_sample_ids_sha256':TRAIN_IDS,'val_sample_ids_sha256':VAL_IDS,'precision':'float32','tf32':False,
              'config_sha256':training_config_sha}
    require(all(binding.get(k) == v for k,v in pinned.items()), 'Exact v12 checkpoint binding mismatch')
    require(report['binding']['manifest_sha256'] == cfg['data']['manifest_sha256'] == MANIFEST,
            'Diagnostic manifest mismatch')
    require(cfg['checkpoint']['path_sha256'] == checkpoint['checkpoint_sha256']
            and cfg['checkpoint']['training_config_sha256'] == training_config_sha
            and cfg['checkpoint']['training_source_registry_sha256'] == TRAIN_REGISTRY
            and cfg['checkpoint']['seed'] == SEED, 'Checkpoint/config SHA or seed mismatch')
    require(training_config['seed'] == SEED and training_config['total_updates'] == 1000
            and training_config['training']['evaluation_updates'] == [250,500,1000]
            and training_config['data']['modality'] == cell['sensor']
            and training_config['data']['manifest_sha256'] == MANIFEST
            and checkpoint['model'] == training_config['model'], 'Training config/model/modality mismatch')
    model = training_config['model']
    require(model['family'] == cell['family'] and model['config']['architecture'] == cell['architecture']
            and model['config']['input_channels'] == (1 if cell['sensor'] == 'spike' else 2), 'Wrong ten-cell model identity')
    if cell['family'] == 'study':
        require(model['config']['spiking'] == cell['spiking'] and model['config']['time_steps'] == 50
                and model['config']['feature_norm_eps'] == .01 and model['config']['attention_normalization'] == 'count_floor_v2',
                'Wrong study neuron/time/normalization recipe')
    selection = checkpoint['validation_selection']; visits = selection['trained_evaluations']
    require(checkpoint['selection'] == selection['selection'] == cfg['checkpoint']['selection'] == 'best-val.pt'
            and selection['trained_evaluation_updates'] == [250,500,1000]
            and [x['update'] for x in visits] == [250,500,1000]
            and selection.get('initial_update_excluded_from_selection') is True
            and selection['rule'] == 'minimum_trained_scene_mean_EPE;earliest_registered_update_on_ties', 'Incomplete final checkpoint selection receipt')
    for row in visits: numeric(row['mean_scene_epe'])
    best = min(visits,key=lambda row:row['mean_scene_epe'])
    require(state['update'] == state['best_update'] == selection['checkpoint_update'] == selection['best_update'] == best['update'] == cfg['checkpoint']['update']
            and state['presentations'] == state['update']*8 and state['best_val_epe'] == best['mean_scene_epe'] == selection['best_val_epe'],
            'Checkpoint was not selected over the complete final trained schedule')
    final = selection['final_full_validation_log']; prefix=selection['checkpoint_log_prefix']
    require(final == {'bytes':cfg['checkpoint']['selection_validation_log_bytes'],'sha256':cfg['checkpoint']['selection_validation_log_sha256']}
            and 0 < prefix['bytes'] <= final['bytes'] and all(len(x['sha256']) == 64 for x in (prefix,final)), 'Final selection log receipt mismatch')
    completion = completed_training(summary,checkpoint,training_config,jobs,cell['run'])
    diagnostic_jobs = [x for x in jobs if x.get('run') == 'functionality-'+cell['run'] and str(x.get('job_id')) == str(report['slurm_job_id'])]
    require(len(diagnostic_jobs) == 1 and diagnostic_jobs[0].get('state') == 'COMPLETED', 'Diagnostic scheduler completion receipt missing')
    require(report.get('val_observations_opened') is True and report.get('model_loaded') is True
            and report.get('train_observations_opened') is False and report.get('neural_training_seeds') == 1
            and report.get('aggregation') == 'pixels_within_scene_then_equal_scene;no_pixel_IID', 'Incomplete Val cohort/aggregation')
    groups,support = population(report['scene_metrics'],report['frame_metrics'],expected)
    shifted,shift_support = population(report['shifted_geometry_scene_metrics'],report['shifted_geometry_frame_metrics'],expected,
                                      conditions=('correct_common','shifted_geometry'))
    require(all(shift_support[k][0] <= support[k][0] for k in support), 'Shift intersection exceeds primary support')
    scenes = list(expected); values = {key:np.array([groups[key][sid]['epe'] for sid in scenes]) for key in CONDITIONS}
    for condition in CONDITIONS:
        rows=[x for x in report['scene_equal_summary'] if x['method'] == 'checkpoint' and x['condition'] == condition]
        require(len(rows) == 1 and rows[0]['scenes'] == 10 and near(rows[0]['mean_scene_epe'],values[condition].mean()), 'Scene summary mismatch')
    responses=[]
    for sid in scenes:
        a,b=shifted['correct_common'][sid],shifted['shifted_geometry'][sid]
        response=a['response_ratio']
        require(type(response) in (int,float) and math.isfinite(response) and response == b['response_ratio'], 'Invalid paired scene shift response')
        responses.append(response)
    shift_summary=report['shift_scene_response_summary']
    require(shift_summary['scene_count'] == 10 and near(shift_summary['median_scene_mean_response_ratio'],np.median(responses))
            and shift_summary['positive_mean_response_scenes'] == sum(x > 0 for x in responses), 'Shift response summary mismatch')
    for condition in ('shuffled','masked'):
        receipt=report['paired_scene_descriptive_ci'][condition+'_minus_correct_epe']
        require(receipt['scene_count'] == 10 and receipt['scenes'] == scenes and receipt['bootstrap_draws'] == DRAW_COUNT
                and receipt['bootstrap_seed'] == SEED and receipt['training_seeds'] == 1
                and near(receipt['paired_difference'],np.mean(values[condition]-values['correct'])), 'Existing paired-condition CI receipt mismatch')
        bounds=receipt['descriptive_ci95'];require(len(bounds) == 2 and all(math.isfinite(x) for x in bounds) and bounds[0] <= bounds[1], 'Invalid existing condition interval')
    return {'values':values, 'groups':groups, 'responses':np.array(responses), 'support':support,'shift_support':shift_support,
            'report':report,'completion':completion, 'selection':selection}


def missing_state(jobs, run):
    rows=[x for x in jobs if x.get('run') == run]
    if any(x.get('state') == 'COMPLETED' and x.get('stop_at_update') == 1000 for x in rows): return 'PENDING_FUNCTIONALITY_RECEIPT'
    if any(x.get('state') in ('RUNNING','PENDING','CONFIGURING') for x in rows): return 'PENDING_TRAINING'
    return 'UNMEASURED_MISSING_COMPLETED_RECEIPT'


def build_comparisons(cells, indices):
    """All10 rows survive, with unavailable numeric fields explicitly null."""
    lookup={(x['model'],x['sensor']):x for x in cells}; rows=[]
    for cell in cells:
        row={key:cell[key] for key in ('model','sensor','run','status')}
        row.update(mean_scene_epe=None,ratio_to_ALIF_I=None,constant_reference=None,functionality=None)
        if cell['status'] == 'MEASURED_COMPLETED_RECEIPT':
            values=cell['validated']['values']['correct'];row['mean_scene_epe']=float(values.mean())
            control=lookup[('original-I',cell['sensor'])]
            if control['status'] == 'MEASURED_COMPLETED_RECEIPT':
                row['ratio_to_ALIF_I']=ratio_result(values,control['validated']['values']['correct'],indices)
                row['ratio_to_ALIF_I']['role']='primary_A_or_B_vs_strong_control' if cell['model'] in ('A','B') else 'reference_self_ratio' if cell['model'] == 'original-I' else 'supporting_continuous_control_ratio'
            if 'constant' in cell:
                row['constant_reference']=ratio_result(values,cell['constant'],indices)
                row['functionality']={'constant_10pct_point_margin':bool(values.mean() <= .9*np.mean(cell['constant'])),
                    'correct_5pct_better_than_shuffled_point_margin':bool(values.mean() <= .95*cell['validated']['values']['shuffled'].mean()),
                    'median_scene_shift_in_registered_band':bool(.5 <= np.median(cell['validated']['responses']) <= 1.5),
                    'positive_scene_mean_response_majority':bool((cell['validated']['responses'] > 0).sum() > 5),
                    'scope':'unchanged registered functionality screening components; no final scientific verdict'}
        rows.append(row)
    interactions=[];counterparts=[]
    for model in ('A','B','continuous-A','continuous-B'):
        arms=[lookup[(model,'spike')],lookup[('original-I','spike')],lookup[(model,'event')],lookup[('original-I','event')]]
        row={'model':model,'status':'UNMEASURED_MISSING_ARMS','missing_runs':[x['run'] for x in arms if x['status'] != 'MEASURED_COMPLETED_RECEIPT'], 'interaction':None}
        if not row['missing_runs']:
            estimate,draws=joint_interaction_draws(*(x['validated']['values']['correct'] for x in arms),indices)
            row.update(status='DESCRIPTIVE_COMPLETED_ARMS',interaction=interval(estimate,draws,'log(mean_model_spike/mean_control_spike)-log(mean_model_event/mean_control_event)'))
            lo,hi=row['interaction']['descriptive_scene_percentile95'];margin=math.log(1.05)
            row.update(exp_interaction=math.exp(estimate), practical_point_flag='spike_direction_beyond_5pct' if estimate < -margin else 'event_direction_beyond_5pct' if estimate > margin else 'point_within_log_symmetric_5pct_band',
                       descriptive_interval_inside_practical_band=bool(-margin <= lo and hi <= margin),
                       practical_band_log=[-margin,margin],practical_band_ratio=[1/1.05,1.05],role='primary' if model in ('A','B') else 'supporting')
        interactions.append(row)
    for model in ('A','B'):
        for sensor in ('spike','event'):
            spike,continuous=lookup[(model,sensor)],lookup[('continuous-'+model,sensor)]
            row={'model':model,'sensor':sensor,'status':'UNMEASURED_MISSING_ARMS','missing_runs':[x['run'] for x in (spike,continuous) if x['status'] != 'MEASURED_COMPLETED_RECEIPT'],'ratio':None}
            if not row['missing_runs']:
                result=ratio_result(spike['validated']['values']['correct'],continuous['validated']['values']['correct'],indices)
                row.update(status='DESCRIPTIVE_COMPLETED_ARMS',ratio=result,
                           practical_point_within_10pct_counterpart_margin=bool(result['estimate'] <= 1.10),
                           descriptive_interval_upper_at_most_1_10=bool(result['descriptive_scene_percentile95'][1] <= 1.10))
            counterparts.append(row)
    return rows,interactions,counterparts


def summarize(study=STUDY):
    sources={};captured={}
    def read(member):
        value,content_sha,byte_count=captured_json(study/member)
        sources[str(member)]=content_sha
        captured[str(member)]={'sha256':content_sha,'bytes':byte_count,
                              'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        return value
    queue=read(Path('configs/pilot-first-seed-queue-v1.json'))
    require(len(queue['runs']) == 10 and queue['same_horizon'] == 1000 and queue['same_effective_batch'] == 8, 'Fixed ten-cell queue required')
    expected=expected_population(read(Path('configs/causal-pilot-scene-selection-v1.json')))
    jobs=read(Path('results/jobs.json'));indices=samples()
    cells=[];required={(name,sensor) for name in MODEL_NAMES for sensor in ('spike','event')}
    for registered in queue['runs']:
        name=registered['run'].removeprefix('pilot-').removesuffix('-'+registered['sensor']+'-seed9471-v1')
        require((name,registered['sensor']) in required, 'Unexpected/duplicate queue cell');required.remove((name,registered['sensor']))
        cell={'model':name,'sensor':registered['sensor'],'run':registered['run'],'status':missing_state(jobs,registered['run'])}
        member=Path('results')/('functionality-'+cell['run'])/'result.json'
        if (study/member).exists():
            report=read(member); config_member=Path('configs')/registered['config'];training=read(config_member)
            diagnostic_member=Path('configs')/('functionality-'+cell['run']+'.json')
            diagnostic_config=read(diagnostic_member)
            require(report['config'] == diagnostic_config and report['binding']['config_sha256'] == sources[str(diagnostic_member)],
                    'Diagnostic receipt differs from registered local checkpoint/config pin')
            summary_member=Path('results')/cell['run']/'training-summary.json';completion=read(summary_member)
            cell.update(status='MEASURED_COMPLETED_RECEIPT',validated=validate_receipt(report,registered,training,sources[str(config_member)],completion,jobs,expected))
        cells.append(cell)
    require(not required,'Missing registered queue cells')
    priors=read(Path('results/pilot-train-gt-priors-v1/result.json'))
    require(priors.get('status') == 'COMPLETED_REGISTERED_DIAGNOSTIC_NO_SCIENTIFIC_PASS_THRESHOLD'
            and priors.get('mode') == 'priors' and priors.get('test_access') is False and priors.get('fit_includes_val') is False
            and priors['binding']['manifest_sha256'] == MANIFEST and priors['prior_fit']['fit_split'] == 'train'
            and priors['prior_fit']['fit_includes_val'] is False and priors['prior_fit']['fit_includes_test'] is False
            and priors['prior_fit']['samples'] == 320, 'Train-only reference receipt mismatch')
    prior_groups,prior_support=population(priors['scene_metrics'],priors['frame_metrics'],expected,method='train_constant_median',conditions=('correct',))
    constant=np.array([prior_groups['correct'][sid]['epe'] for sid in expected]);shared_support=None;shared_shift=None;details=[]
    for cell in cells:
        if cell['status'] != 'MEASURED_COMPLETED_RECEIPT': continue
        data=cell['validated'];require(data['support'] == prior_support,'Model/prior trusted support differs')
        if shared_support is None: shared_support,shared_shift=data['support'],data['shift_support']
        require(data['support'] == shared_support and data['shift_support'] == shared_shift,'Cross-arm support differs')
        cell['constant']=constant;report=data['report'];responses=data['responses']
        detail={'run':cell['run'],'checkpoint_sha256':report['checkpoint']['checkpoint_sha256'],
            'training_binding':report['checkpoint']['binding'],'completion':data['completion'],
            'selected_update':data['selection']['checkpoint_update'],'final_validation_selection':data['selection'],
            'reload_minus_selection_epe':float(data['values']['correct'].mean()-data['selection']['best_val_epe']),
            'diagnostic_job_id':str(report['slurm_job_id']),'diagnostic_source_identity_sha256':report['binding']['source_identity_sha256'],
            'existing_paired_condition_intervals':report['paired_scene_descriptive_ci'],
            'scene_median_shift_response':interval(float(np.median(responses)),np.median(responses[indices],axis=1),'median_of_ten_scene_mean_signed_shift_response_ratios'),
            'positive_response_scenes':int((responses > 0).sum()),'per_scene':[]}
        for i,sid in enumerate(expected):
            detail['per_scene'].append({'scene_id':sid,'sample_ids':expected[sid],
                'correct_epe':float(data['values']['correct'][i]),'shuffled_epe':float(data['values']['shuffled'][i]),'masked_epe':float(data['values']['masked'][i]),
                'shuffled_minus_correct_epe':float(data['values']['shuffled'][i]-data['values']['correct'][i]),
                'masked_minus_correct_epe':float(data['values']['masked'][i]-data['values']['correct'][i]),
                'mean_signed_shift_response_ratio':float(responses[i])})
        details.append(detail)
    rows,interactions,counterparts=build_comparisons(cells,indices)
    for member in ('README.md','analysis/pilot-budget-24gpu-hours-v1.md','analysis/pilot-comparison-analysis-v1.md'):
        sources[member]=sha(study/member)
    scheduler_snapshot={'source_member':'results/jobs.json',**captured['results/jobs.json'],
        'semantics':'hash identifies the single captured ledger bytes at analysis time, not a future current-file invariant; full mutable ledger is not archived here',
        'expected_to_match_future_working_file':False,
        'retained_content':'completed training/functionality rows actually used, whitelisted below; active/unrelated jobs omitted',
        'missing_cell_statuses':'transient availability labels from the same captured ledger; no live outcome metrics used'}
    # Keep mutable-ledger capture separate from fixed receipt/source-file QA.
    del sources['results/jobs.json']
    scheduler_rows=scheduler_receipts_used(jobs,details)
    scheduler_snapshot['used_rows_sha256']=hashlib.sha256(json.dumps(scheduler_rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {'schema':'stereo-study-pilot-comparisons-v2','scope':'supplied compact completed receipts only; no new inference, GT arrays, live Val, Test or model experiment',
        'seed':SEED,'training_seeds':1,'population':expected,'scene_count':10,'sample_count':20,'units':'quarter-image horizontal pixels',
        'completed_cells':len(details),'registered_cells':10,'rows':rows,'interactions':interactions,'counterparts':counterparts,
        'cell_details':details,'constant_reference':{'source_run':'pilot-train-gt-priors-v1','mean_scene_epe':float(constant.mean()),'constant_disparity':priors['constant_median'],'fit':'Train only'},
        'bootstrap':{'draws':DRAW_COUNT,'seed':SEED,'generator':'numpy.default_rng','indices_sha256':hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
            'pairing':'one shared2000x10 scene index matrix across every ratio and all four arms of each interaction','percentiles':[.025,.975]},
        'practical_flags_only':{'counterpart_ratio_upper':1.10,'interaction_abs_log_upper':math.log(1.05),'interaction_exp_band':[1/1.05,1.05]},
        'limitations':['one trained seed; scene intervals omit seed and checkpoint-selection uncertainty',
            'ten historically exposed Val scenes; not untouched Test or external real-sensor validation',
            'scene means paired by exact IDs and support counts; no new bytewise GT/mask audit',
            'no multiplicity-adjusted final superiority/equivalence/significance or sensor-preference conclusion',
            'unequal/missing arms remain unmeasured; intermediate live validation does not fill missing cells'],
        'sources_sha256':sources,'source_snapshot_semantics':'JSON payloads and their SHA256 values come from one byte read per source; later working-copy changes do not revise this analysis snapshot',
        'scheduler_source_snapshot':scheduler_snapshot,'scheduler_receipts_used':scheduler_rows,
        'script_sha256':sha(Path(__file__)),'numpy_version':np.__version__,
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=STUDY)
    parser.add_argument('--output-prefix',type=Path,default=STUDY/'results/pilot-comparisons-v2')
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args();result=summarize(args.study)
    if args.validate_only:
        print(json.dumps({'status':'VALIDATED_SUPPLIED_RECEIPTS_ONLY','completed_cells':result['completed_cells'],'registered_cells':10}));return
    paths=[args.output_prefix.with_suffix('.'+suffix) for suffix in ('json','csv')]
    require(not any(path.exists() for path in paths),'Fresh output prefix required; preserve prior analysis receipts')
    args.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    with paths[1].open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['model','sensor','run','status','mean_scene_epe','ratio_to_ALIF_I','ratio_scene_p025','ratio_scene_p975'])
        writer.writeheader()
        for row in result['rows']:
            ratio=row['ratio_to_ALIF_I'];bounds=ratio['descriptive_scene_percentile95'] if ratio else [None,None]
            writer.writerow({**{key:row[key] for key in ('model','sensor','run','status','mean_scene_epe')},
                'ratio_to_ALIF_I':ratio['estimate'] if ratio else None,'ratio_scene_p025':bounds[0],'ratio_scene_p975':bounds[1]})
    result['csv_sha256']=sha(paths[1])
    paths[0].write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'ANALYZED_SUPPLIED_RECEIPTS_ONLY','completed_cells':result['completed_cells'],'registered_cells':10,'outputs':[str(x) for x in paths]},indent=2))


if __name__=='__main__':main()
