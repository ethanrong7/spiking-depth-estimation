"""Experiment-local input transform around the exact frozen original-I runner."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import torch
from input_codes import transform_stereo
EXPERIMENT=Path(__file__).resolve().parents[3]
PARENT=EXPERIMENT/'campaigns/alif-mechanism-screen-v1/scripts'
GUARDS={PARENT/'run_screen.py':'7247b190d3db59274600917b407e14b22cc7fcc2512115e2fb250cb9069605ca',PARENT/'mechanism.py':'f8a4147f682a4794ee7f814defcb3684d14764958d61b6606e527de9865f9ae0',EXPERIMENT/'scripts/run_experiment.py':'078d3342b88c8674c03910260aa8fc1aaad210430131822b9eee600e1605ab7c'}
for file,expected in GUARDS.items():
 if hashlib.sha256(file.read_bytes()).hexdigest()!=expected:raise RuntimeError('frozenparent changed: '+str(file))
sys.path.insert(0,str(PARENT))
# The parent's guarded source generation remains intact. These additions record
# initial tensor identity and the LR actually used, without changing operators.
parent_source=(PARENT/'run_screen.py').read_text()
marker="driver=types.ModuleType('alif_screen_driver');driver.__file__=str(parent)"
assert parent_source.count(marker)==1
extra='''old="model=build_model(c['model'],seed,c.get('normalize_residual_current',False)).cuda();parameter_count="
assert source.count(old)==1
source=source.replace(old,"model=build_model(c['model'],seed,c.get('normalize_residual_current',False)).cuda();initial_state_sha256=state_digest(model.state_dict());parameter_count=")
old="        opt.step();scheduler.step()"
assert source.count(old)==1
source=source.replace(old,"        used_lr=float(opt.param_groups[0]['lr']);opt.step();scheduler.step()")
old="'gradient_norm':grad,'lr':scheduler.get_last_lr()[0]"
assert source.count(old)==1
source=source.replace(old,"'gradient_norm':grad,'used_lr':used_lr,'lr':scheduler.get_last_lr()[0]")
old="    metadata.update(started_utc="
assert source.count(old)==1
source=source.replace(old,"    metadata.update(initial_state_sha256=initial_state_sha256)\\n    metadata.update(started_utc=")
'''
parent_source=parent_source.replace(marker,extra+marker)
spec=importlib.util.spec_from_loader('direction_parent_screen',loader=None);parent_module=importlib.util.module_from_spec(spec);parent_module.__file__=str(PARENT/'run_screen.py');exec(compile(parent_source,str(PARENT/'run_screen.py'),'exec'),parent_module.__dict__)
driver=parent_module.driver
base_dataset=driver.ManifestDataset;base_parent_build=parent_module.build_model;base_contract=parent_module.training_contract;base_budget=driver.validate_execution_budget
MODE=None
ALIASES={'original_I_raw':'raw','original_I_count_only':'count_only'}
PARENT_ID='cb43497fdc02bf612794dd48d516ee71e5658f10969e47ed6f4683591f3e8ec5'

def state_digest(state):
 h=hashlib.sha256()
 for name,value in sorted(state.items()):
  h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def set_mode(mode):
 global MODE
 if mode not in ['raw','count_only']:raise ValueError(mode)
 MODE=mode

class InputDataset(base_dataset):
 def __getitem__(self,index):
  if MODE is None:raise RuntimeError('input mode not bound')
  left,right,target,valid,idx=super().__getitem__(index)
  left,right=transform_stereo(left[None],right[None],MODE)
  return left[0],right[0],target,valid,idx

def build_model(name,seed,normalize_residual_current=False):
 if name not in ALIASES or normalize_residual_current:raise ValueError(name)
 return base_parent_build('original_I',seed,False)

def training_contract(config,*args,**kwargs):
 parent_config=dict(config,model='original_I');contract=base_contract(parent_config,*args,**kwargs);contract['model']=config['model']
 contract['information_screen']={'stage':'S1','input_code':config['input_code'],'transform_location':'datasetCPU_beforecuda_andunchanged2Sminus1','count_formula':'K_t=floor((t+1)*C/50)-floor(t*C/50); C=sum_tS','all_phases_use_same_transform':['training','warmup','initial_evaluation','scheduled_development','selected_checkpoint_export'],'frozen_scaffold':'exactoriginal_I_adaptiveALIFplusindependentnormalizedcurrent_outsidealpha','source_parent_identity_sha256':PARENT_ID,'interpretation':'information/encodingpackage;canonicaltimingandbinocularsynchronychanged;notpurechronologycausaltest','fresh_weights_no_trained_restore':True,'parameter_and_initialization_matched':True}
 contract['scheduler'].update(div_factor=25.,final_div_factor=10000.)
 return contract

def validate_budget(config,registration,manifest_sha,source,has_checkpoint=False):
 base_budget(config,registration,manifest_sha,source,has_checkpoint)
 fixed={'campaign_id':'spike-depth-direction-v1-S1','stage':'paired_training','updates':5000,'schedule_updates':10000,'seed':1234,'physical_batch':2,'effective_batch':2,'train_refinements':16,'eval_refinements':32,'lr':.0002,'train_split':'train','eval_split':'dev','final_extras':False,'normalize_residual_current':False}
 if any(config.get(k)!=v for k,v in fixed.items())or config['model']not in ALIASES or config['input_code']!=ALIASES[config['model']] or config['eval_updates']!=[1000,2500,5000]:raise RuntimeError('unauthorized S1configuration')
 if len(registration['runs'])!=2 or {r['config']['model']for r in registration['runs']}!=set(ALIASES)or registration['stage_updates_budget']!=10000 or registration['study_updates_cap']!=40000 or registration['study_GPU_hours_cap']!=40 or registration['max_allocated_study_gpus']!=2:raise RuntimeError('unauthorized S1stagebudget')
 if MODE!=config['input_code']:raise RuntimeError('dataset inputmode mismatch')

driver.ManifestDataset=InputDataset;driver.build_model=build_model;driver.training_contract=training_contract;driver.validate_execution_budget=validate_budget;driver.state_digest=state_digest
if __name__=='__main__':
 parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--config',type=Path,required=True);known,_=parser.parse_known_args();set_mode(json.loads(known.config.read_text())['input_code']);driver.main()
