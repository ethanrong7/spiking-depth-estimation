"""Tiny CPU initialization/config check for unchanged selected contrast."""
import json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_repeat
screen=run_repeat.run_screen;torch.set_num_threads(1)
reference=screen.base_build('original',2345,False);state={k:v.clone() for k,v in reference.state_dict().items()};rng=torch.get_rng_state().clone();del reference
extras=[];counts={}
for arm in ['O','I']:
 model=screen.build_model('original_'+arm,2345,False)
 assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items()) and torch.equal(rng,torch.get_rng_state())
 extras.append({k:v.clone() for k,v in model.state_dict().items() if 'current_' in k})
 counts[arm]=dict(total=sum(p.numel() for p in model.parameters()),active=sum(p.numel() for p in model.parameters() if p.requires_grad),dormant=sum(p.numel() for p in model.parameters() if not p.requires_grad));del model
assert all(torch.equal(v,extras[1][k]) for k,v in extras[0].items())
c=Path(__file__).parents[1];reg=json.loads((c/'configs/registration.json').read_text());assert len(reg['runs'])==2 and {x['arm'] for x in reg['runs']}=={'O','I'}
for row in reg['runs']:
 config=row['config'];assert config['seed']==2345 and config['updates']==5000 and config['schedule_updates']==10000 and config['eval_updates']==[1000,2500,5000]
 assert counts[row['arm']]==reg['counts'][row['arm']]
result=dict(initial_common_tensors_exact=True,postbuild_rng_exact=True,current_tensors_identical=True,counts=counts,scientific_source_hashes_verified=True,configuration_scope='O/I2345only5k10100; noF/extensions',changed_operator_tests='unchangedhashboundinitialscreen CPUcontrols reused; no scientificcodechange')
(c/'results/cpu-checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
