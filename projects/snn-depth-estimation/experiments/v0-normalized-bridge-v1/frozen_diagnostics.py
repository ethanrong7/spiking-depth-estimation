"""Frozen30kmechanismreplay;exactarchivedmembraneresetandRMS-hooksemantics."""
import argparse,hashlib,json,os,sys,random
from pathlib import Path
from contextlib import contextmanager,nullcontext
import numpy as np
import torch

@contextmanager
def reset_each_cell_call(model):
    """Reset Q/K/V/output voltages before each cell call; retain hidden spikes.

    This removes internal membrane carry, not all recurrence: hidden outputs,
    disparity, and cross-scale inputs still feed subsequent refinements.
    """
    handles = [cell.register_forward_pre_hook(
        lambda module, inputs: module.reset_state(clear_diagnostics=False))
        for cell in model.update_block.cells]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()

def register_branch_hooks(cells):
    """Read-only hooks distinguish the raw projection from the actual residual."""
    branches, handles = {}, []
    for i, cell in enumerate(cells):
        record = branches[f"scale_{i}"] = {name: [] for name in (
            "raw_projection_rms", "normalized_projection_rms", "residual_current_rms",
            "gated_attention_rms", "gated_ffn_rms")}
        def branch_hook(name, record, cell):
            def capture(module, inputs, value):
                scale = 1.0
                if name == "gated_attention_rms":
                    scale = cell.attention_gate.detach().sigmoid() if cell.attention_enabled else 0.0
                elif name == "gated_ffn_rms":
                    scale = cell.ffn_gate.detach().sigmoid()
                rms = float((value.detach() * scale).square().mean().sqrt())
                record[name].append(rms)
                if name == "normalized_projection_rms":
                    actual = rms if cell.normalize_residual_current else record["raw_projection_rms"][-1]
                    record["residual_current_rms"].append(actual)
            return capture
        for name, module in (("raw_projection_rms", cell.input_proj),
                             ("normalized_projection_rms", cell.norm),
                             ("gated_attention_rms", cell.output_proj),
                             ("gated_ffn_rms", cell.ffn)):
            handles.append(module.register_forward_hook(branch_hook(name, record, cell)))
    return branches, handles

def main():
 p=argparse.ArgumentParser()
 for key in ('source','root','data-root','manifest','output'):p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
 experiment=a.source/'projects/snn-depth-estimation/experiments/v0-normalized-bridge-v1';sys.path.insert(0,str(experiment));import recipe
 driver,mechanism,historical=recipe.load_frozen(a.source/'historical')
 registry=json.loads((a.source/'source-registry.json').read_text())
 assert all(hashlib.sha256((a.source/f).read_bytes()).hexdigest()==v for f,v in registry['source_files'].items())
 assert hashlib.sha256(a.manifest.read_bytes()).hexdigest()=='bad4369b6fd8b0cdf871a20b1727c2d9b59733a16bd33f75904227bdfef4510a'
 import cv2
 torch.set_num_threads(6);cv2.setNumThreads(1);random.seed(1234);np.random.seed(1234);torch.manual_seed(1234);torch.cuda.manual_seed_all(1234)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
 directory=a.root/'train/full';checkpoint=directory/'step-30000.pt';complete=json.loads((directory/'result.json').read_text());assert complete['status']=='COMPLETED' and complete['updates']==30000
 ck=torch.load(checkpoint,map_location='cpu',weights_only=False);assert ck['updates']==30000 and ck['config']['arm']=='full' and ck['config']['schedule_total']==30000
 assert ck['source_identity']==registry['source_identity_sha256']
 checkpoint_config=ck['config']
 # Audit the trained dormant controller tensors before GPU replay.
 initial=recipe.build_model(driver,mechanism,'spiking_normalized',1234)
 for cell in initial.update_block.cells:
  cell.attention_enabled=False;cell.qkv.requires_grad_(False);cell.output_proj.requires_grad_(False);cell.attention_gate.requires_grad_(False)
 control_directory=a.root/'train/no_attention';control_complete=json.loads((control_directory/'result.json').read_text());assert control_complete['updates']==30000 and control_complete['status']=='COMPLETED'
 control_checkpoint=control_directory/'step-30000.pt';control=torch.load(control_checkpoint,map_location='cpu',weights_only=False);assert control['updates']==30000 and control['source_identity']==registry['source_identity_sha256']
 dormant=[(name,param) for name,param in initial.named_parameters() if not param.requires_grad];assert sum(param.numel() for name,param in dormant)==49827
 assert all(torch.equal(param,control['model'][name]) for name,param in dormant),'dormantcontroltensorschanged'
 control_sha256=hashlib.sha256(control_checkpoint.read_bytes()).hexdigest();del initial,control
 model=recipe.build_model(driver,mechanism,'spiking_normalized',1234).cuda().eval();model.load_state_dict(ck['model'],strict=True);del ck
 rows=json.loads(a.manifest.read_text())['dev'];assert len(rows)==60;dataset=driver.ManifestDataset(a.data_root,rows)
 def loader():return torch.utils.data.DataLoader(dataset,batch_size=2,num_workers=0,shuffle=False,generator=torch.Generator().manual_seed(2225))
 official=next(json.loads(x) for x in (directory/'evaluations.jsonl').read_text().splitlines() if json.loads(x)['update']==30000)
 metadata=dict(status='RUNNING',no_attention_checkpoint_sha256=control_sha256,dormant_control_audit=dict(count=49827,unchanged=True),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_identity=registry['source_identity_sha256'],historical_source_identity=historical['source_identity_sha256'],checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),checkpoint=str(checkpoint),update=30000,checkpoint_config=checkpoint_config,torch_version=torch.__version__,cuda_version=torch.version.cuda,gpu_name=torch.cuda.get_device_name(0),precision='FP32;matmulTF32false;cuDNNTF32true',batch_size=2,iterations=32,manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),archived_reset_script_sha256='b4cb353dc4e9e2f93f2cef87c885b7ee40d995b721171d327d970a5f032ee5c7',archived_rms_script_sha256='42c41da061c2392ed812976f622b2864277a34b62146787239d887f285565c75',gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),slurm={k:v for k,v in os.environ.items() if k.startswith('SLURM_')},conditions={})
 a.output.parent.mkdir(parents=True,exist_ok=True)
 def save():a.output.write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
 control_official=next(json.loads(x) for x in (control_directory/'evaluations.jsonl').read_text().splitlines() if json.loads(x)['update']==30000)
 assert {(r['scene'],r['frame'],r['valid_pixels']) for r in official['per_frame']}=={(r['scene'],r['frame'],r['valid_pixels']) for r in control_official['per_frame']}
 metadata['retrained_control_official']=control_official
 predictions={};baseline_valid={}
 with torch.no_grad():
  for condition in ('full32','attention_off32','membrane_reset32','full32_rms'):
   cells=model.update_block.cells;before=[(c.spatial_mixing,c.attention_enabled,len(c._forward_pre_hooks)) for c in cells]
   context=model.update_block.attention_intervention(disable_attention=True) if condition=='attention_off32' else reset_each_cell_call(model) if condition=='membrane_reset32' else nullcontext()
   branches,handles=register_branch_hooks(cells) if condition=='full32_rms' else ({},[])
   frames=[];diagnostics=[]
   try:
    with context:
     for left,right,target,valid,indices in loader():
      assert torch.cuda.mem_get_info()[0]>3*1024**3,'diagnosticmemoryguard'
      left,right,target,valid=[x.cuda() for x in (left,right,target,valid)];prediction=driver.forward(model,left,right,32);assert torch.isfinite(prediction).all()
      diagnostics.append(model.update_block.diagnostics())
      for j,index in enumerate(indices.tolist()):
       row=rows[index];key=(row['scene'],row['frame']);error=(prediction[j,0]-target[j,0]).abs()[valid[j]];frame=dict(scene=key[0],frame=key[1],valid_pixels=int(valid[j].sum()),**driver.metric_values(error))
       if condition=='full32':predictions[key]=prediction[j,0].cpu();baseline_valid[key]=valid[j].cpu()
       else:
        delta=(prediction[j,0].cpu()-predictions[key]).abs();frame['mean_abs_prediction_change']=float(delta.mean());frame['valid_mean_abs_prediction_change']=float(delta[baseline_valid[key]].mean());frame['max_abs_prediction_change']=float(delta.max())
       frames.append(frame)
      driver.reset_model(model)
   finally:
    for handle in handles:handle.remove()
    driver.reset_model(model)
   after=[(c.spatial_mixing,c.attention_enabled,len(c._forward_pre_hooks)) for c in cells];assert before==after,'interventioncleanupfailed'
   scenes=sorted({r['scene'] for r in frames});scene_epe={s:float(np.mean([r['epe'] for r in frames if r['scene']==s])) for s in scenes};count=sum(r['valid_pixels'] for r in frames)
   result=dict(mean_scene_epe=float(np.mean(list(scene_epe.values()))),scene_epe=scene_epe,valid_pixels=count,**{k:sum(r[k]*r['valid_pixels'] for r in frames)/count for k in ('bad1','bad3','bad5')},per_frame=frames,cell_diagnostics={scale:{k:float(np.mean([d[scale][k] for d in diagnostics])) for k in diagnostics[0][scale]} for scale in diagnostics[0]})
   if condition=='full32':
    expected={(r['scene'],r['frame']):r for r in official['per_frame']};assert all(r['valid_pixels']==expected[(r['scene'],r['frame'])]['valid_pixels'] for r in frames)
    frame_diff=max(abs(r['epe']-expected[(r['scene'],r['frame'])]['epe']) for r in frames);scene_diff=max(abs(v-official['scene_epe'][s]) for s,v in scene_epe.items());aggregate_diff=abs(result['mean_scene_epe']-official['mean_scene_epe'])
    metadata['baseline_parity']=dict(passed=max(frame_diff,scene_diff,aggregate_diff)<1e-5,max_frame_epe_difference=frame_diff,max_scene_epe_difference=scene_diff,aggregate_epe_difference=aggregate_diff,tolerance=1e-5)
    metadata['conditions'][condition]=result;save()
    assert metadata['baseline_parity']['passed'],'strictofficialbaselineparityfailed;stopinterpretation'
   else:
    full=metadata['conditions']['full32'];result['delta_mean_scene_epe']=result['mean_scene_epe']-full['mean_scene_epe'];result['delta_bad3']=result['bad3']-full['bad3'];result['paired_scene_epe_delta']={s:v-full['scene_epe'][s] for s,v in scene_epe.items()};result['mean_abs_prediction_change']=float(np.mean([r['mean_abs_prediction_change'] for r in frames]))
    if branches:
     result['branch_rms']={scale:{k:float(np.mean(v)) for k,v in values.items()} for scale,values in branches.items()}
     metadata['rms_control_parity']=dict(max_prediction_difference=max(r['max_abs_prediction_change'] for r in frames),passed=max(r['max_abs_prediction_change'] for r in frames)<1e-5)
    metadata['conditions'][condition]=result;save()
    if condition=='full32_rms':assert metadata['rms_control_parity']['passed'],'RMS-controlpredictionparityfailed'
   print(condition,result['mean_scene_epe'],result['bad3'],flush=True)
 metadata['status']='COMPLETED';save()
if __name__=='__main__':main()
