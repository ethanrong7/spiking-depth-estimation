"""Static/config and tiny CPU initialization/scheduler checks; no experiment."""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import torch

CAMPAIGN=Path(__file__).resolve().parents[1]
EXPERIMENT=CAMPAIGN.parents[1]
sys.path.insert(0,str(EXPERIMENT/'scripts'))
import run_experiment as runner
from run_provenance import training_contract,validate_execution_budget


def main():
    torch.set_num_threads(1)
    registration=json.loads((CAMPAIGN/'configs/registration.json').read_text())
    baseline=json.loads((CAMPAIGN.parent/'paired-loss-v1/configs/registration.json').read_text())
    assert len(registration['runs'])==6 and registration['updates_per_run']==50000 and registration['samples_per_run']==100000
    for entry in registration['runs']:
        config=json.loads((CAMPAIGN/'configs'/entry['config_file']).read_text());assert config==entry['config']
        ref=next(x['config'] for x in baseline['runs'] if x['config']['model']==config['model'] and x['config']['seed']==config['seed'])
        assert {k:(ref.get(k),config.get(k)) for k in set(ref)|set(config) if ref.get(k)!=config.get(k)}=={'campaign_id':('paired-loss-v1','longer-horizon-v1'),'updates':(10000,50000),'schedule_updates':(10000,50000)}
        contract=training_contract(config,registration['manifest_sha256'],'train',Path('/documented-data-root'))
        assert contract['scheduler']['total_steps']==50100 and contract['physical_batch']==contract['effective_batch']==2
        assert config['updates']*2==100000 and config['eval_every']==500
        frozen=registration['frozen_models'][config['model']]
        source={'source_override':{'parent_source_identity_sha256':frozen['parent_source_identity_sha256']},'source_files':frozen['core_source_files']}
        validate_execution_budget(config,registration,registration['manifest_sha256'],source,False)
    for seed in registration['paired_seeds']:
        original=runner.build_model('original',seed,False);hard=runner.build_model('spiking',seed,True)
        assert sum(p.numel() for p in original.parameters())==12148246 and sum(p.numel() for p in hard.parameters())==8398902
        for name in ['cnet','fnet','context_zqr_convs','update_block.encoder','update_block.disp_head','update_block.mask']:
            a=original.get_submodule(name).state_dict();b=hard.get_submodule(name).state_dict();assert a.keys()==b.keys() and all(torch.equal(v,b[k]) for k,v in a.items())
    parameter=torch.nn.Parameter(torch.zeros(()));opt=torch.optim.AdamW([parameter],lr=.0002,weight_decay=1e-5,eps=1e-8)
    schedule=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.0002,total_steps=50100,pct_start=.01,cycle_momentum=False,anneal_strategy='linear')
    assert schedule.get_last_lr()[0]==8e-6
    used=[]
    for update in range(50000):
        used.append(schedule.get_last_lr()[0]);opt.step();schedule.step()
        assert schedule.last_epoch==update+1
    assert max(used)==.0002 and used.index(max(used))+1==501
    assert schedule.total_steps==50100 and len(used)==50000
    print(json.dumps(dict(status='CPU_STATIC_CHECKS_PASSED',configs=6,seeds=3,initial_shared_tensors_equal=True,models_unchanged=True,updates=50000,samples=100000,schedule_total_steps=50100,used_peak_update=501,used_first_lr=used[0],used_last_lr=used[-1],logged_next_lr=schedule.get_last_lr()[0],runtime_torch=torch.__version__,onecycle_signature=str(inspect.signature(torch.optim.lr_scheduler.OneCycleLR))),indent=2))
    print('CPU smoke/config only; no dataset access, full-model forward, training experiment or GPU workload')


if __name__=='__main__':main()
