"""Frozen CPU probe of signed spike-count inputs to NMRF transformer FFNs.

Forward passes use decoded values with dense PyTorch kernels. A small separate
event accumulator verifies the arithmetic interpretation. Event AC and sparse
ANN MAC counts are analytical workloads, not measured acceleration or energy.
"""
import argparse
import contextlib
import json
from pathlib import Path
import subprocess
import time

import torch
from torch import nn
from torch.nn import functional as F

from evaluate_nmrf import input_tensors,metric_counts,summarize
from nmrf_runtime import build_native_nmrf,UPSTREAM_REVISION
from probe_nmrf_bsa import calibrate,code
from runtime import NativeDataset

ROOT=Path(__file__).resolve().parent


def event_linear_row(module,integer,scale):
    """Execute one coded input row as repeated signed weight additions."""
    x=integer.reshape(-1).to(torch.int64)
    weights=module.weight*scale.repeat_interleave(module.in_features//len(scale))[None]
    out=module.bias.clone() if module.bias is not None else weights.new_zeros(module.out_features)
    for step in range(int(x.abs().max())):
        active=x.abs()>step
        out=out+(weights[:,active]*x[active].sign()).sum(-1)
    return out


@contextlib.contextmanager
def coding(modules,calibration,scope,steps,stats):
    handles=[]
    for name,module in modules.items():
        if scope == 'late':
            if not name.startswith(('parent.inference.','parent.refinement.')):continue
        elif scope != 'all' and not name.startswith('parent.'+scope+'.'): continue
        def hook(module,inputs,name=name):
            x=inputs[0]
            coded,integer=code(x,calibration[name],f'unary{steps}')
            stats.append(dict(module=name,dense_macs=x.numel()*module.out_features,
                              event_accumulations=int(integer.abs().sum())*module.out_features,
                              sparse_quant_ann_macs=int(integer.count_nonzero())*module.out_features,
                              input_elements=x.numel(),coding_element_steps=x.numel()*steps,
                              event_count=int(integer.abs().sum()),nonzero_count=int(integer.count_nonzero())))
            return (coded,)
        handles.append(module.register_forward_pre_hook(hook))
    try: yield
    finally:
        for h in handles:h.remove()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'configs/synthetic-split.json')
    p.add_argument('--crop',type=int,nargs=2,default=[96,192])
    p.add_argument('--variants',nargs='+',default=['inference:2','inference:4','inference:8','refinement:4','dpn:4','all:4','all:8'])
    a=p.parse_args()
    if a.output.resolve().is_relative_to(ROOT):p.error('Outputs must be outside Git')
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    model,_,_=build_native_nmrf(1234)
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if ckpt['model_name']!='nmrf-native-50' or ckpt['upstream_revision']!=UPSTREAM_REVISION:raise ValueError('Wrong checkpoint')
    model.load_state_dict(ckpt['model'],strict=True);model.eval()
    initial={k:v.clone() for k,v in model.state_dict().items()}
    modules={n:m for n,m in model.named_modules() if isinstance(m,nn.Linear) and '.nmp.mlp.fc' in n}
    manifest=json.loads(a.manifest.read_text())
    train=[next(r for r in manifest['train'] if r['scene']==f'scene_{s:04d}') for s in (0,40,80)]
    dev=[next(r for r in manifest['dev'] if r['scene']==f'scene_{s:04d}') for s in (120,135,149)]
    stress=next(r for r in manifest['dev'] if r['scene']=='scene_0134' and r['frame']=='0255')
    config=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},checkpoint_step=ckpt['step'],
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),upstream_revision=UPSTREAM_REVISION,
        calibration=train,primary_dev=dev,selected_stress=stress,modules=list(modules),device='cpu',torch=str(torch.__version__),
        hypothesis='Preserve precise geometric attention while coding the expensive transformer feed-forward inputs as signed events.',
        coding='Training-only per-head 99.5 percentile scale, 4 channel groups; static 2/4/8-step count code; equivalent low-bit ANN.',
        screening='Exploratory EPE increase <=0.05 px and bad3 increase <=0.5 pp; inspect bad1 and stress, event work versus sparse quantized ANN, and whole-model share.',
        execution='Dense decoded PyTorch forward, separate single-row event execution check. Counts are not measured FLOPs savings, speed, or energy.')
    (a.output/'config.json').write_text(json.dumps(config,indent=2))
    (a.output/Path(__file__).name).write_text(Path(__file__).read_text())
    del ckpt
    started=time.monotonic();calibration={};values={n:[] for n in modules};handles=[]
    for name,module in modules.items():
        def hook(module,inputs,name=name):values[name].append(inputs[0].detach().clone())
        handles.append(module.register_forward_pre_hook(hook))
    with torch.no_grad():
        for sample in NativeDataset(a.data_root,train):
            left,right,*_=input_tensors(sample,'cpu',a.crop);model(left,right)
        for h in handles:h.remove()
        event_error=0.;max_relative_event_error=0.;ann_error=0.
        for name,module in modules.items():
            calibration[name]=calibrate(values[name])
            row=values[name][0].reshape(-1,module.in_features)[0]
            for steps in (2,4,8):
                decoded,integer=code(row,calibration[name],f'unary{steps}')
                ann,_=code(row,calibration[name],f'quant{steps}')
                ann_error=max(ann_error,float((decoded-ann).abs().max()))
                expected=F.linear(decoded,module.weight,module.bias)
                actual=event_linear_row(module,integer,(calibration[name]['clip']/steps).clamp_min(1e-8))
                event_error=max(event_error,float((expected-actual).abs().max()))
                max_relative_event_error=max(max_relative_event_error,float((expected-actual).norm()/expected.norm().clamp_min(1e-8)))
        del values
        torch.save(calibration,a.output/'calibration.pt')
        results={}
        for index,sample in enumerate(NativeDataset(a.data_root,dev+[stress])):
            left,right,target,valid=input_tensors(sample,'cpu',a.crop);valid=valid & (target>0) & (target<192)
            reference=model(left,right)['disp'][0]
            results.setdefault('reference',[]).append(dict(counts=metric_counts(reference,target,valid)))
            for variant in a.variants:
                scope,steps=variant.split(':');stats=[]
                with coding(modules,calibration,scope,int(steps),stats):prediction=model(left,right)['disp'][0]
                results.setdefault(variant,[]).append(dict(counts=metric_counts(prediction,target,valid),accounting=stats))
            print(json.dumps(dict(frame=index+1,seconds=time.monotonic()-started)),flush=True)
        checks=dict(ann_code_max_abs_error=ann_error,event_linear_max_abs_error=event_error,event_linear_max_relative_l2=max_relative_event_error,
                    state_unchanged=all(torch.equal(v,initial[k]) for k,v in model.state_dict().items()))
        if ann_error!=0 or max_relative_event_error>1e-5 or not checks['state_unchanged']:raise RuntimeError(checks)
        summary={n:dict(primary=summarize([r['counts'] for r in rows[:3]]),selected_stress=summarize([rows[3]['counts']])) for n,rows in results.items()}
        report=dict(config=config,checks=checks,seconds=time.monotonic()-started,results=results,summary=summary)
        (a.output/'results.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(dict(checks=checks,summary=summary,seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()
