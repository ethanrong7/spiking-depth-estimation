"""Original ALIF with exact finest-output regularizer and component logging.

The shared driver is immutable. Guarded experiment-local substitutions expose
its already-supported regularizer components and add registered Dev16 checks.
"""
import hashlib
from pathlib import Path
import sys,types

EXPERIMENT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(EXPERIMENT/'scripts'))
PARENT=EXPERIMENT/'scripts/run_experiment.py'
EXPECTED_PARENT_SHA256='078d3342b88c8674c03910260aa8fc1aaad210430131822b9eee600e1605ab7c'
source=PARENT.read_text()
if hashlib.sha256(PARENT.read_bytes()).hexdigest()!=EXPECTED_PARENT_SHA256:raise RuntimeError('parent driver identity differs')

def replace_once(old,new):
 global source
 if source.count(old)!=1:raise RuntimeError('parent driver changed critical insertion: '+old)
 source=source.replace(old,new)

replace_once("disparity_component=float(loss);neuron_component=0.","disparity_component=float(loss);neuron_component=0.;rate_component=0.;voltage_component=0.")
replace_once("neuron_loss=1e-4*(torch.stack(spikes,1).mean((0,1))-.05).square().sum()+1e-3*torch.stack(volts,1).square().mean();neuron_component=float(neuron_loss);loss=loss+neuron_loss;del neuron_loss", "rate_loss,voltage_loss=regularizer_components(spikes,volts);neuron_loss=rate_loss+voltage_loss;rate_component=float(rate_loss);voltage_component=float(voltage_loss);neuron_component=float(neuron_loss);loss=loss+neuron_loss;del neuron_loss,rate_loss,voltage_loss")
replace_once("'neuron_loss':neuron_component,'train_epe'", "'neuron_loss':neuron_component,'rate_loss':rate_component,'voltage_loss':voltage_component,'train_epe'")
replace_once("result={**evaluate(model,evloader,32),**timing()};history.append({'update':step+1,**result});print('EVAL',step+1,result['mean_scene_epe'],flush=True)", "result={**evaluate(model,evloader,32),**timing()};history.append({'update':step+1,**result});print('EVAL',step+1,result['mean_scene_epe'],flush=True)\n            if step+1 in (10000,20000,30000):\n                secondary_path=a.output/'secondary-history.json';secondary=json.loads(secondary_path.read_text()) if secondary_path.exists() else []\n                secondary.append({'update':step+1,'refinements':16,**evaluate(model,evloader,16),**timing()});secondary_path.write_text(json.dumps(secondary,indent=2));print('SECONDARY16',step+1,secondary[-1]['mean_scene_epe'],flush=True)")
driver=types.ModuleType('alif_repair_driver');driver.__file__=str(PARENT)
exec(compile(source,str(PARENT),'exec'),driver.__dict__)


def regularizer_components(spikes,volts):
 """Exact upstream core/loss.py tensors/reductions, including padded extent."""
 spike=driver.torch.stack(spikes,dim=1)
 rate=spike.mean(dim=(0,1))
 voltage=driver.torch.stack(volts,dim=1)
 return 1e-4*(rate-.05).square().sum(),1e-3*voltage.square().mean()

base_contract=driver.training_contract

def training_contract(config,*args,**kwargs):
 contract=base_contract(config,*args,**kwargs)
 contract['repair_identity']=dict(name='finest_output_rate_and_postreset_voltage',rate_target=.05,lambda_rate=1e-4,lambda_voltage=1e-3,rate_reduction='mean batch/time then sum channels/spatial',voltage_reduction='mean batch/time/channels/spatial',scope='finest outputs collected by unchanged model; original padded extent',detach=False,primary_dev_refinements=32,secondary_dev16_updates=[10000,20000,30000],training_horizon=16)
 return contract

driver.regularizer_components=regularizer_components
driver.training_contract=training_contract
if __name__=='__main__':driver.main()
