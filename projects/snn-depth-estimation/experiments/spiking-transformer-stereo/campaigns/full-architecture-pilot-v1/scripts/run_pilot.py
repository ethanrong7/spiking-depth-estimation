"""Common counted driver; architecture/input/return adapters are explicit."""
import hashlib,sys,types
from pathlib import Path
EXPERIMENT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(EXPERIMENT/'scripts'))
parent=EXPERIMENT/'scripts/run_experiment.py'
if hashlib.sha256(parent.read_bytes()).hexdigest()!='078d3342b88c8674c03910260aa8fc1aaad210430131822b9eee600e1605ab7c':raise RuntimeError('parentdriver changed')
source=parent.read_text()
old="if (step+1)%c.get('eval_every',100)==0 or step+1==updates:"
assert source.count(old)==1;source=source.replace(old,"if step+1 in c['eval_updates']:")
old="    initial={**evaluate(model,evloader,32),**timing()}"
assert source.count(old)==1
source=source.replace(old,"    warmup_state={name:value.detach().cpu().clone() for name,value in model.state_dict().items()};warmup_batch=next(iter(evloader));model.eval()\n    with torch.no_grad():forward(model,warmup_batch[0].cuda(),warmup_batch[1].cuda(),32)\n    reset_model(model)\n    if not all(torch.equal(warmup_state[name],value.detach().cpu()) for name,value in model.state_dict().items()):raise RuntimeError('warmup changed learned/bufferstate')\n    (a.output/'warmup.json').write_text(json.dumps({'frames':2,'refinements':32,'training':False,'state_tensors_exactly_unchanged':True}))\n    del warmup_state,warmup_batch\n"+old)
driver=types.ModuleType('full_architecture_pilot_driver');driver.__file__=str(parent)
exec(compile(source,str(parent),'exec'),driver.__dict__)
from models import build_conventional
base_build=driver.build_model;base_contract=driver.training_contract


def build_model(name,seed,normalize_residual_current=False):
 if normalize_residual_current:raise ValueError('pilot has no normalizedreplacementarm')
 if name=='original':return base_build('original',seed,False)
 return build_conventional(name,seed)


def training_contract(config,*args,**kwargs):
 contract=base_contract(config,*args,**kwargs)
 contract['encoders']={'original':'original_50_channel_cnn','raft_stereo':'officialRAFT-Stereo50channelCNN','selective_raft':'officialSelective-RAFT50channelCNN'}[config['model']]
 contract['architecture_pilot']=dict(architecture={'original':'originalSpikeStereoNetALIF','raft_stereo':'officialRAFT-Stereo','selective_raft':'officialSelective-RAFT'}[config['model']],input_adapter='50independentchannels;2spike-1;feature/contextstems50',pretraining=None,common_loss='weightedsequenceL1only; no neuronregularizer',evaluation_updates=[0]+config['eval_updates'],pairing='dataorderandseed;NOTidenticalinitialtensorsacrossarchitectures',conventional_spike_voltage_diagnostics='notapplicable' if config['model']!='original' else 'originalfinestspikes/voltage',frozen_bn=True,common_validity='finitepositiveDepth and abs(coords1-coords0)<512',initial_warmup='firstDevbatch32eval/no_grad,reset,unchangedstatecheck',max_allocated_account_gpus=1,scope='adaptedcommonrecipe10kpilot,notpaperreproduction')
 return contract

def common_disparity_loss(preds,target,valid):
 if not all(driver.torch.isfinite(p).all() for p in preds):raise FloatingPointError('nonfinite predictions')
 mask=valid[:,None]&driver.torch.isfinite(target)&(target.abs()<512)
 if not mask.any():raise RuntimeError('no valid common-mask pixels')
 return sum((.9**(15/max(1,len(preds)-1)))**(len(preds)-i-1)*(p-target).abs()[mask].mean() for i,p in enumerate(preds))

driver.build_model=build_model;driver.training_contract=training_contract;driver.disparity_loss=common_disparity_loss
if __name__=='__main__':driver.main()
