"""Checkpoint-bound logical arithmetic diagnostic; never an energy benchmark."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import socket
import statistics
import sys
import time

from arithmetic_counter import ArithmeticCounter
from evaluate_functionality import file_sha,load_bound_data,load_checkpoint,verify_source

PARITY_PROTOCOL={'api_warmup_forwards':1,'reference_forwards':2,
    'comparison':'all_recursive_prediction_leaves_excluding_root_diagnostics',
    'atol':1e-5,'rtol':1e-5,'preserve_cold_comparison':True,
    'warmup_scope':'first_selected_input_once_per_model'}


def validate_config(config,require_registered=True):
    if (config.get('schema')!='stereo-study-arithmetic-v1' or config.get('purpose')!='engineering'
            or config.get('driver')!='arithmetic_accounting' or config.get('stage')!='engineering_diagnostic'
            or config.get('test_access')!='sealed' or config.get('allocated_gpu_hour_cap')!=24):
        raise ValueError('Registered bounded engineering arithmetic protocol required')
    if require_registered and config.get('registered') is not True:
        raise ValueError('Execution requires registered configuration')
    if type(config.get('max_seconds')) is not int or not 30<=config['max_seconds']<=600:
        raise ValueError('Maximum30..600second diagnostic bound required')
    if config.get('sample_ordinals') not in ([0],list(range(20))):
        raise ValueError('Preselected first target diagnostic or complete20Val targets only')
    if type(config.get('seed')) is not int or not config.get('expected_hostname'):
        raise ValueError('Exact seed/authorized hostname required')
    data=config['data'];ckpt=config['checkpoint']
    for name,value in (('train_samples',320),('val_samples',20),('train_scenes',40),('val_scenes',10)):
        if data.get(name)!=value:
            raise ValueError('Full registered pilot cohort metadata required')
    if ckpt.get('selection')!='best-val.pt' or ckpt.get('seed')!=config['seed'] or type(ckpt.get('update')) is not int or ckpt['update']<1:
        raise ValueError('Exact trained best-Val selection required')
    identities=[data.get('manifest_sha256',''),data.get('audit_receipt_sha256','')]
    for name in ('path','metadata','training_config','training_source_registry','selection_validation_log'):
        if not ckpt.get(name+'_env'):
            raise ValueError('Required immutable member environment binding: '+name)
        identities.append(ckpt.get(name+'_sha256',''))
    for identity in identities:
        if len(identity)!=64 or any(c not in '0123456789abcdef' for c in identity) or (require_registered and identity=='0'*64):
            raise ValueError('Actual immutable SHA required')
    if type(ckpt.get('selection_validation_log_bytes')) is not int or ckpt['selection_validation_log_bytes']<1:
        raise ValueError('Complete final Val selection log bytes required')
    if config.get('parity_protocol') is not None and config['parity_protocol']!=PARITY_PROTOCOL:
        raise ValueError('Exact prospective warmup/full-output parity protocol required')
    return True


def inference_variant(model,training,device):
    """Rebuild only the permitted static diagnostics=False study variant."""
    if training['model']['family']!='study':
        return model,{}
    from core.study_models import build_model
    state={k:v.cpu() for k,v in model.state_dict().items()}
    model=model.cpu()
    config={**training['model']['config'],'collect_diagnostics':False}
    replacement=build_model(config)
    replacement.load_state_dict(state,strict=True)
    return replacement.to(device).eval(),{'collect_diagnostics':False}


class ArithmeticParityError(ValueError):
    def __init__(self, message, audit, ledger):
        super().__init__(message)
        self.audit, self.ledger = audit, ledger


def scripted_source_inventory(model):
    """Read raw scripted graphs only; no graph_for, tracing or extra inference."""
    import torch
    scripts={};flags={}
    for path,module in model.named_modules():
        selected={name:getattr(module,name) for name in ('spiking','step_mode','backend')
                  if hasattr(module,name) and isinstance(getattr(module,name),(bool,str))}
        if selected:flags[path or '<model>']=selected
        attributes={}
        for cls in type(module).__mro__:
            for name,value in vars(cls).items():attributes.setdefault(name,value)
        for name,raw in attributes.items():
            value=raw.__func__ if isinstance(raw,staticmethod) else raw
            if not isinstance(value,torch.jit.ScriptFunction):
                continue
            graph=value.graph
            text=str(graph)
            identity=hashlib.sha256(text.encode()).hexdigest()
            if identity not in scripts:
                def operations(block):
                    result=[]
                    for node in block.nodes():
                        result.append(node.kind())
                        for child in node.blocks():result.extend(operations(child))
                    return result
                scripts[identity]={'graph':text,'raw_graph_operations':operations(graph),'references':[]}
            scripts[identity]['references'].append((path or '<model>')+'.'+name)
    return {'schema':'stereo-study-scripted-source-inventory-v1','scripts':scripts,'module_execution_flags':flags,
            'scope':'Raw source graphs, not kernel/optimized-graph execution coverage; source-call reconciliation required'}


def observe(model,left,right,*,warmup_forwards=0):
    """Two references and observer agree on all prediction leaves after fixed warmup."""
    import torch
    import numpy as np
    from benchmark_energy import prediction_leaves,compare_prediction_leaves
    if warmup_forwards not in (0,1):
        raise ValueError('Only zero or one prospectively registered API warmup allowed')
    def rng():
        return (torch.get_rng_state(),torch.cuda.get_rng_state_all() if left.is_cuda else [],random.getstate(),np.random.get_state())
    def restore(state):
        torch.set_rng_state(state[0])
        if left.is_cuda:torch.cuda.set_rng_state_all(state[1])
        random.setstate(state[2]);np.random.set_state(state[3])
    def same(a,b):
        return (torch.equal(a[0],b[0]) and len(a[1])==len(b[1]) and all(torch.equal(x,y) for x,y in zip(a[1],b[1]))
                and a[2]==b[2] and a[3][0]==b[3][0] and np.array_equal(a[3][1],b[3][1]) and a[3][2:]==b[3][2:])
    with torch.inference_mode():
        input_left,input_right=left.clone(),right.clone();before=rng()
        input_checks=[];post_rng=[]
        def forward():
            restore(before)
            work_left,work_right=left.clone(),right.clone()
            output=model(work_left,work_right)
            primary=output['disparity']
            if primary.shape!=(1,1,*left.shape[-2:]) or not torch.isfinite(primary).all():
                raise ValueError('Complete finite primary disparity map required')
            input_checks.append(bool(torch.equal(work_left,input_left) and torch.equal(work_right,input_right)))
            post_rng.append(rng())
            return output
        def snapshot():
            return prediction_leaves(forward())
        cold=snapshot() if warmup_forwards else None
        expected=snapshot();repeated=snapshot()
        observer=ArithmeticCounter(model)
        try:
            # Input clones, finite checks and CPU snapshots stay outside the counted call.
            restore(before)
            work_left,work_right=left.clone(),right.clone()
            with observer:
                result=model(work_left,work_right)
            observed_rng=rng()
            input_checks.append(bool(torch.equal(work_left,input_left) and torch.equal(work_right,input_right)))
            actual=prediction_leaves(result)
            def comparison(a,b):
                try:
                    leaves=compare_prediction_leaves(a,b,1e-5,1e-5)
                    return {'prediction_tensors':leaves,'allclose':all(r['within_tolerance'] and not
                        r['reference_nonfinite_values'] and not r['candidate_nonfinite_values'] for r in leaves.values()),
                        'maximum_absolute_difference':max((r['max_absolute_error'] or 0 for r in leaves.values()),default=0)}
                except (RuntimeError,TypeError,ValueError) as error:
                    return {'allclose':False,'structure_error':str(error),'maximum_absolute_difference':None}
            comparisons={'reference_repeat':comparison(expected,repeated),
                         'observer_vs_first':comparison(expected,actual),
                         'observer_vs_repeat':comparison(repeated,actual)}
            audit={'maximum_absolute_difference':comparisons['observer_vs_first']['maximum_absolute_difference'],
                   'atol':1e-5,'rtol':1e-5,'api_warmup_forwards':warmup_forwards,
                   'reference_repeat_maximum_absolute_difference':comparisons['reference_repeat']['maximum_absolute_difference'],
                   'observer_vs_repeat_maximum_absolute_difference':comparisons['observer_vs_repeat']['maximum_absolute_difference'],
                   **{key+'_allclose':value['allclose'] for key,value in comparisons.items()},
                   'comparisons':comparisons,
                   'input_tensors_unchanged':all(input_checks) and bool(torch.equal(left,input_left) and torch.equal(right,input_right)),
                   'python_numpy_torch_cuda_rng_match_reference':all(same(state,post_rng[0]) for state in post_rng[1:]+[observed_rng]),
                   'scope':'all recursive prediction tensor/scalar/container leaves; root diagnostics metadata excluded; fixed tolerance',
                   'cold_vs_reference':comparison(cold,expected) if cold is not None else None}
            if not all(audit[key] for key in ('reference_repeat_allclose','observer_vs_first_allclose',
                                              'observer_vs_repeat_allclose','input_tensors_unchanged',
                                              'python_numpy_torch_cuda_rng_match_reference')):
                raise ArithmeticParityError('Full-output repeat/observer parity failed',audit,observer.report())
            return observer.report(),audit
        finally:
            observer.close()


def aggregate(rows):
    keys=sorted({k for row in rows for k in row['totals']})
    return {key:{'mean':statistics.mean(row['totals'].get(key,0) for row in rows),
                 'minimum':min(row['totals'].get(key,0) for row in rows),
                 'maximum':max(row['totals'].get(key,0) for row in rows)} for key in keys}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--source-registry',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--validate-config',action='store_true')
    args=parser.parse_args();config=json.loads(args.config.read_text())
    validate_config(config,not args.validate_config)
    if args.validate_config:
        print(json.dumps({'status':'VALID_CONFIG_ONLY','config_sha256':file_sha(args.config)}));return
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname()!=config['expected_hostname']:
        raise RuntimeError('Registered cluster Slurm allocation required')
    if args.source_registry is None or args.output is None:
        raise ValueError('Immutable source/config and fresh output required')
    import torch
    from benchmark_energy import load_sensor_only,tensor_state_sha,run_study_diagnostics_parity
    from core.study_training import atomic_json
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not any(x in torch.cuda.get_device_name(0) for x in ('4090','5090')):
        raise RuntimeError('Exactly one authorized RTX4090/5090 required')
    registry=verify_source(args.source_registry,args.config)
    member=str(Path(__file__).resolve().relative_to(args.source_registry.parent.resolve()))
    if registry['source_files'].get(member)!=file_sha(Path(__file__)):
        raise ValueError('Observer driver absent from verified source')
    root,manifest_path,manifest,audit=load_bound_data(config)
    if any(args.output.resolve().is_relative_to(path.resolve()) for path in (root,args.source_registry.parent)):
        raise ValueError('Output must be outside source/data')
    args.output.mkdir(parents=True,exist_ok=False)
    result={'schema':'stereo-study-arithmetic-result-v1','status':'INITIALIZING','samples':[],
        'slurm_job_id':os.environ['SLURM_JOB_ID'],'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
        'config_sha256':file_sha(args.config),'source_identity_sha256':registry['source_identity_sha256'],
        'source_registry_sha256':file_sha(args.source_registry),'test_access':False,
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'sample_ordinals':config['sample_ordinals'],'parity_protocol':config.get('parity_protocol'),
        'scope':'logical operator/arithmetic diagnostics; no timing/energy claim'}
    save=lambda:atomic_json(args.output/'result.json',result)
    save();started=time.monotonic()
    def expired(number,frame):
        raise TimeoutError('Registered arithmetic diagnostic time cap')
    previous=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,config['max_seconds']-5)
    try:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        import numpy as np
        random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
        model,training,checkpoint=load_checkpoint(config,manifest,torch.device('cuda:0'))
        model,overrides=inference_variant(model,training,torch.device('cuda:0'))
        before=tensor_state_sha(model)
        script_path=args.output/'scripted-source-inventory.json'
        scripts=scripted_source_inventory(model);atomic_json(script_path,scripts)
        result.update(status='OBSERVING',checkpoint=checkpoint,inference_overrides=overrides,
                      trained_state_sha256=before,modality=training['data']['modality'],
                      scripted_source_inventory_file=script_path.name,scripted_source_inventory_sha256=file_sha(script_path))
        for ordinal in config['sample_ordinals']:
            row=manifest['val'][ordinal]
            sample=load_sensor_only(root,row,training['data']['modality'])
            left,right=(sample[k].unsqueeze(0).to('cuda:0') for k in ('left','right'))
            if 'diagnostics_variant_parity' not in result:
                result['diagnostics_variant_parity']={}
                if training['model']['family']=='study':
                    run_study_diagnostics_parity(model,left,right,audit=result['diagnostics_variant_parity'])
                else:
                    result['diagnostics_variant_parity'].update(status='NOT_APPLICABLE_BASELINE',
                        reason='unchanged trained inference config; no extra forward')
                result['pre_warmup_context']={'prior_diagnostics_flag_gate_forwards':2 if training['model']['family']=='study' else 0,
                    'comparison_scope':'First arithmetic pre-warmup output; not necessarily process-cold after study diagnostic-flag gate'}
                save()
            warmup=1 if config.get('parity_protocol') and ordinal==config['sample_ordinals'][0] else 0
            ledger,parity=observe(model,left,right,warmup_forwards=warmup)
            ledger_path=args.output/('sample-'+str(ordinal)+'-operators.json');atomic_json(ledger_path,ledger)
            result['samples'].append({'ordinal':ordinal,'sample_id':row['sample_id'],'scene_id':row['scene_id'],
                'totals':ledger['totals'],'unclassified_operators':ledger['unclassified_operators'],
                'parity':parity,'ledger_file':ledger_path.name,'ledger_sha256':file_sha(ledger_path)})
            save();del sample,left,right
        if tensor_state_sha(model)!=before:
            raise ValueError('Inference observer changed trained weights/buffers')
        result.update(status='COMPLETED_LOGICAL_ARITHMETIC_DIAGNOSTIC',per_map_counts=aggregate(result['samples']),
            complete20=config['sample_ordinals']==list(range(20)),weights_buffers_unchanged=True,
            coverage_status='Observed ATen boundary vector; raw scripted inventory retained; no claim of universal kernel or hardware coverage',
            interpretation='Potential binary event ACs require sparse routing/weight access; dense binary pair tests retained; composite and unpriced terms separate; not measured GPU/neuromorphic joules')
    except Exception as error:
        result.update(status='INCOMPLETE_TIME_CAP' if isinstance(error,TimeoutError) else 'FAILED_ARITHMETIC_DIAGNOSTIC',error=repr(error))
        if isinstance(error,ArithmeticParityError):
            failure_path=args.output/'failed-parity-operators.json'
            atomic_json(failure_path,error.ledger)
            result.update(failed_output_parity=error.audit,failed_ledger_file=failure_path.name,
                          failed_ledger_sha256=file_sha(failure_path),counts_accepted=False)
        if 'model' in locals() and 'before' in locals():
            result['weights_buffers_unchanged_after_failure']=tensor_state_sha(model)==before
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        result['diagnostic_seconds_not_inference_latency']=time.monotonic()-started;save()
    print(json.dumps({'status':result['status'],'samples_completed':len(result['samples'])}))
    if not result['status'].startswith('COMPLETED_'):
        raise SystemExit(1)


if __name__=='__main__':main()
