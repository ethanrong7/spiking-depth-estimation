"""Training lineage contracts, independent of PyTorch execution."""
import hashlib
import json
from pathlib import Path

def validate_execution_budget(config, registration, manifest_sha256, source, has_checkpoint=False):
    if registration is None:
        if config.get('campaign_id'):raise RuntimeError('campaign requires explicit registration')
        if config['updates']>3000:raise RuntimeError('registered maximum is3000 updates')
        return
    if registration['campaign_id']!=config.get('campaign_id') or registration['status']!='registered':
        raise RuntimeError('campaign registration mismatch')
    if registration['manifest_sha256']!=manifest_sha256:raise RuntimeError('campaign manifest mismatch')
    if has_checkpoint:raise RuntimeError('paired campaign requires fresh initialization')
    matches=[row for row in registration['runs'] if row['config']['model']==config['model'] and row['config']['seed']==config['seed']]
    if len(matches)!=1 or config!=matches[0]['config']:raise RuntimeError('configuration differs from registered campaign')
    model=registration['frozen_models'][config['model']]
    if source.get('source_override',{}).get('parent_source_identity_sha256')!=model['parent_source_identity_sha256']:
        raise RuntimeError('campaign source parent mismatch')
    actual={k:v for k,v in source['source_files'].items() if '/core/' in k}
    if actual!=model['core_source_files']:raise RuntimeError('campaign core differs from frozen model')

def training_contract(config, manifest_sha256, train_key, data_root):
    return {'version':1,'manifest_sha256':manifest_sha256,'train_split':train_key,
            'data_root':str(data_root.resolve()),'model':config['model'],
            'seed':config.get('seed',1234),'normalize_residual_current':config.get('normalize_residual_current',False),
            'physical_batch':2,'effective_batch':2,'api_train_iters':16,'api_eval_iters':32,'train_refinements':None if config['model']=='ssde_inspired' else 16,'selection_refinements':None if config['model']=='ssde_inspired' else 32,'actual_inference_passes':1 if config['model']=='ssde_inspired' else None,
            'input':[50,250,400],'padding_divisor':32,'calibration_inherited_upstream':{'f':289.740625,'B':.08},
            'validity':'finite positive depth and absolute disparity<512','flow_sign':'negative_display_disparity',
            'optimizer':{'name':'AdamW','max_lr':config.get('lr',.0002),'weight_decay':1e-5,'eps':1e-8,'clip_norm':1.},
            'scheduler':{'name':'OneCycleLR','total_steps':config.get('schedule_updates',config['updates'])+100,'pct_start':.01,'cycle_momentum':False,'anneal_strategy':'linear'},
            'loss':'final_smooth_l1' if config['model']=='ssde_inspired' else 'weighted_l1_gamma0.9_adjusted15'+('+neuron_rate0.05_lambda1e-4_voltage1e-3' if config['model']=='original_regularized' else ''),
            'encoders':'normalized_cumulative32_pyramid64' if config['model']=='ssde_inspired' else 'original_50_channel_cnn','context_norm':'group_layer' if config['model']=='ssde_inspired' else 'batch_frozen','precision':'fp32_cudnn_tf32_allowed_no_amp'}

def validate_parent_contract(checkpoint_path, checkpoint, contract, source):
    parent=json.loads((checkpoint_path.parent/'metadata.json').read_text())
    parent_config=parent['config']
    for key,expected_value in [('model',contract['model']),('seed',contract['seed']),('normalize_residual_current',contract['normalize_residual_current']),('lr',contract['optimizer']['max_lr'])]:
        default={'seed':1234,'normalize_residual_current':False,'lr':.0002}.get(key)
        if parent_config.get(key,default)!=expected_value:raise RuntimeError('parent metadata configuration mismatch: '+key)
    if parent_config.get('schedule_updates',parent_config['updates'])+100!=contract['scheduler']['total_steps']:raise RuntimeError('parent metadata schedule mismatch')
    checkpoint_sha=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if parent['source']['source_identity_sha256']!=checkpoint['source']['source_identity_sha256']:
        inherited_path=checkpoint_path.parent/'inherited-best.json'
        if not inherited_path.exists() or json.loads(inherited_path.read_text())['sha256']!=checkpoint_sha:raise RuntimeError('checkpoint source differs from parent run without inherited-best identity')
    expected=contract['manifest_sha256']
    if checkpoint.get('manifest_sha256',parent.get('manifest_sha256'))!=expected or parent.get('manifest_sha256')!=expected:
        raise RuntimeError('parent checkpoint/metadata manifest identity mismatch')
    if 'training_semantics' in checkpoint and checkpoint['training_semantics']!=contract:
        raise RuntimeError('parent training semantics mismatch')
    command=parent['command'];parent_data=Path(command[command.index('--data-root')+1]).resolve()
    if str(parent_data)!=contract['data_root']:raise RuntimeError('parent dataset root mismatch')
    common=parent['git_protocol']['common'];inputs=parent['git_protocol']['input']
    if [inputs['bins'],inputs['height'],inputs['width']]!=contract['input']:raise RuntimeError('parent input geometry mismatch')
    for key,value in [('physical_batch',2),('effective_batch',2),('train_refinements',16),('eval_refinements',32),('hidden_dims',[128]*3),('corr_levels',4),('corr_radius',4),('context_norm','batch'),('freeze_batchnorm',True),('optimizer','AdamW'),('lr',.0002),('weight_decay',1e-5),('clip_gradient_norm',1.)]:
        if common[key]!=value:raise RuntimeError('parent common protocol mismatch: '+key)
    parent_files=checkpoint['source']['source_files'];current_files=source['source_files']
    different=[name for name,digest in parent_files.items() if '/core/' in name and current_files.get(name)!=digest]
    if different:raise RuntimeError('continuation core source differs: '+str(different))
    return {'checkpoint':str(checkpoint_path),'checkpoint_sha256':checkpoint_sha,
            'run_directory':str(checkpoint_path.parent),'parent_run_source_identity_sha256':parent['source']['source_identity_sha256'],'source_identity_sha256':checkpoint['source']['source_identity_sha256'],
            'updates':checkpoint['updates'],'samples_seen':checkpoint['updates']*2,
            'manifest_sha256':expected,'legacy_contract_validated_from_parent_metadata':'training_semantics' not in checkpoint}


def validate_loaded_confirmation_checkpoint(checkpoint, entry):
    if checkpoint['updates']!=entry['selected_update']:
        raise RuntimeError('confirmation checkpoint update differs from frozen selection')
    expected_source=entry.get('checkpoint_source_identity_sha256')
    if expected_source is not None and checkpoint['source']['source_identity_sha256']!=expected_source:
        raise RuntimeError('confirmation checkpoint source differs from frozen selection')


def validate_confirmation_selection(selection_path,checkpoint_path,config,manifest_sha,output):
    record=json.loads(selection_path.read_text())
    if len(record['runs'])!=10 or record['confirmation_count']!=80 or record['primary_api_iters']!=32:
        raise RuntimeError('confirmation selection must bind all ten runs and the frozen80-frame primary protocol')
    if len({(r['model'],r['seed'],r['normalize_residual_current']) for r in record['runs']})!=10:raise RuntimeError('confirmation selections are not ten distinct planned model/seed runs')
    if record['manifest_sha256']!=manifest_sha:raise RuntimeError('confirmation manifest differs from selection')
    matches=[r for r in record['runs'] if Path(r['checkpoint']).resolve()==checkpoint_path.resolve()]
    if len(matches)!=1:raise RuntimeError('checkpoint is not a unique frozen selection')
    entry=matches[0]
    if 'evaluation_config_identity' in entry and config!=entry['evaluation_config_identity']['config']:
        raise RuntimeError('confirmation configuration differs from frozen evaluation configuration')
    for key,default in [('model',None),('seed',1234),('normalize_residual_current',False)]:
        if config.get(key,default)!=entry[key]:raise RuntimeError('confirmation selected configuration mismatch: '+key)
    if config.get('eval_iters',32)!=32:raise RuntimeError('only frozen primary32 confirmation is authorized')
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()!=entry['checkpoint_sha256']:raise RuntimeError('confirmation checkpoint identity differs')
    prior=checkpoint_path.parent/'priors.npz'
    if hashlib.sha256(prior.read_bytes()).hexdigest()!=entry['prior_sha256']:raise RuntimeError('confirmation training-prior identity differs')
    if output.resolve()!=Path(entry['confirmation_output']).resolve():raise RuntimeError('confirmation output differs from frozen selection')
    if record['preselected_case_ids']!=['scene_0150-0001','scene_0170-0001','scene_0189-0001']:raise RuntimeError('confirmation plot cases differ')
    return {'selection_sha256':hashlib.sha256(selection_path.read_bytes()).hexdigest(),'entry':entry}
