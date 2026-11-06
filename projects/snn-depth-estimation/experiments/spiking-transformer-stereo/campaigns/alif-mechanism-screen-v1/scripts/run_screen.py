"""Guarded original driver with O/I/F builder and fixed screening checkpoints."""
import hashlib,sys,types
from pathlib import Path
from mechanism import adapt

EXPERIMENT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(EXPERIMENT/'scripts'))
parent=EXPERIMENT/'scripts/run_experiment.py'
if hashlib.sha256(parent.read_bytes()).hexdigest()!='078d3342b88c8674c03910260aa8fc1aaad210430131822b9eee600e1605ab7c':raise RuntimeError('parentdriver changed')
source=parent.read_text()
old="if (step+1)%c.get('eval_every',100)==0 or step+1==updates:"
assert source.count(old)==1
source=source.replace(old,"if step+1 in c['eval_updates']:")
old='    metadata.update(started_utc='
assert source.count(old)==1
source=source.replace(old,"    metadata.update(active_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),dormant_parameter_count=sum(p.numel() for p in model.parameters() if not p.requires_grad))\n    metadata.update(started_utc=")
old="    initial={**evaluate(model,evloader,32),**timing()}"
assert source.count(old)==1
source=source.replace(old,"    warmup_state={name:value.detach().cpu().clone() for name,value in model.state_dict().items()};warmup_batch=next(iter(evloader));model.eval()\n    with torch.no_grad():forward(model,warmup_batch[0].cuda(),warmup_batch[1].cuda(),32)\n    reset_model(model)\n    if not all(torch.equal(warmup_state[name],value.detach().cpu()) for name,value in model.state_dict().items()):raise RuntimeError('warmup changed learned/bufferstate')\n    (a.output/'warmup.json').write_text(json.dumps({'frames':2,'refinements':32,'training':False,'state_tensors_exactly_unchanged':True}))\n    del warmup_state,warmup_batch\n"+old)
driver=types.ModuleType('alif_screen_driver');driver.__file__=str(parent)
exec(compile(source,str(parent),'exec'),driver.__dict__)
base_build=driver.build_model
base_contract=driver.training_contract


def build_model(name,seed,normalize_residual_current=False):
 if name not in ('original_O','original_I','original_F') or normalize_residual_current:raise ValueError(name)
 return adapt(base_build('original',seed,False),name[-1],seed)


def training_contract(config,*args,**kwargs):
 contract=base_contract(config,*args,**kwargs)
 contract['screen_identity']=dict(arm=config['model'][-1],current_input='cat(h,cz,cr,cq,*original_x_list)',current_widths_fine_mid_coarse=[768,768,640],current='Conv1x1bias + GroupNorm(1,128,affine=True)',current_outside_alpha=True if config['model']!='original_O' else False,original_surrogate='unchanged ATan',original_reset_differentiation=True,charge={'original_O':'alpha*v+(1-alpha)*R','original_I':'alpha*v+(1-alpha)*R+c','original_F':'.5*v+.5*R+c'}[config['model']],threshold='beta' if config['model']!='original_F' else '.5',reset='gamma*beta*spike' if config['model']!='original_F' else '.5*spike',extra_module_init='isolatedCPUforkseed+19001; restores baseRNG',eval_updates=config['eval_updates'],total_stored_parameters=12427926,initial_evaluation_warmup='firstDevbatch2 at32 eval/no_grad thenreset, unchangedstatecheck',capacity='Ocurrentmodules dormant; Fadaptivegateconvs/paramsigmoid dormant; activecapacity differs')
 return contract

driver.build_model=build_model;driver.training_contract=training_contract
if __name__=='__main__':driver.main()
