"""Bounded CPU feature/attention probes; no training or disparity evaluation.

Requires trusted local checkpoints. Four explanatory native frames, seven rows
per frame, both trained encoders. Reports intervention mechanics, not EPE effects.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT), str(HERE.parents[1])]
import torch
from core.extractor import BasicEncoder
from core.spiking_attention_placement import DescriptorAttention
from core.epipolar_spiking_stereo import signed_rate
from core.attention_encoder_controls import signed_value_and_derivative
from core.utils.utils import InputPadder
from runtime import load_spikes


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):h.update(chunk)
    return h.hexdigest()


def rms(x):
    return float(x.square().mean().sqrt())


def summary(x):
    v = x.flatten().float()
    return dict(mean=float(v.mean()), rms=rms(v), minimum=float(v.min()),
                maximum=float(v.max()), quantiles=torch.quantile(v, torch.tensor([.1,.5,.9,.99])).tolist())


def projections(module, left, right):
    def project(x):
        z = module.codes.norm(module.input(x.permute(0,2,3,1)))
        return tuple(layer(z) for layer in (module.codes.q,module.codes.k,module.codes.v))
    return project(left), project(right)


def exchange(module, left, right, currents, coding, gain=1., intervention='native'):
    def encode(x, branch):
        x = x * gain
        if coding == 'signed':return signed_rate(x, 4)
        if intervention == 'clip-all' or (intervention == 'clip-qk' and branch != 2) or (intervention == 'clip-v' and branch == 2):
            return x.clamp(-1, 1)
        return x
    (ql,kl,vl),(qr,kr,vr) = [tuple(encode(x,i) for i,x in enumerate(view)) for view in currents]
    al = ((ql @ kr.transpose(-1,-2))/4).softmax(-1)
    ar = ((qr @ kl.transpose(-1,-2))/4).softmax(-1)
    l,r = left.permute(0,2,3,1),right.permute(0,2,3,1)
    dl,dr = module.output(al @ vr),module.output(ar @ vl)
    out = l+dl,r+dr
    raw = l @ r.transpose(-1,-2)/16
    cost = out[0] @ out[1].transpose(-1,-2)/16
    delta = cost-raw
    row_shift = delta.mean(-1,keepdim=True)
    weights = torch.cat([al.flatten(0,-2),ar.flatten(0,-2)],0)
    entropy = -(weights * weights.clamp_min(1e-30).log()).sum(-1)
    metrics = dict(attention_entropy_normalized=float(entropy.mean()/math.log(weights.shape[-1])),
        effective_candidates=float(entropy.exp().mean()), candidates=weights.shape[-1],
        mean_top1_weight=float(weights.max(-1).values.mean()),
        fraction_top1_over_half=float((weights.max(-1).values>.5).float().mean()),
        residual_feature_rms_ratio=math.sqrt((rms(dl)**2+rms(dr)**2)/(rms(l)**2+rms(r)**2)),
        correlation_change_rms=rms(delta),
        row_offset_energy_fraction=float(row_shift.square().mean()/delta.square().mean().clamp_min(1e-30)),
        logits_std=float(torch.cat([(ql@kr.transpose(-1,-2)/4).flatten(),(qr@kl.transpose(-1,-2)/4).flatten()]).std(unbiased=False)))
    return out, metrics


def difference(first, second):
    return math.sqrt(sum(rms(x-y)**2 for x,y in zip(first,second))/sum(rms(y)**2 for y in second))


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--linear-checkpoint',type=Path,required=True)
    parser.add_argument('--hybrid-checkpoint',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,default=ROOT/'configs/synthetic-split.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    import cv2
    cv2.setNumThreads(1)
    started=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *unused:(_ for _ in ()).throw(TimeoutError('180-second CPU budget')))
    signal.alarm(180)
    manifest=json.loads(args.manifest.read_text())
    scenes=['scene_0120','scene_0123','scene_0134','scene_0137']
    cases=[next(r for r in manifest['dev'] if r['scene']==s and r['frame']=='0001') for s in scenes]
    result=dict(status='running',scope='four outcome-selected frames; encoder and seven full epipolar rows only; not representative sampling',
        gpu_used=False,optimizer_steps=0,disparity_forward_passes=0,torch=str(torch.__version__),
        source_sha256=digest(__file__),manifest_sha256=digest(args.manifest),models={},rows=[])
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(args.output)
    models={}
    for label,path in [('linear',args.linear_checkpoint),('signed',args.hybrid_checkpoint)]:
        ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['updates']==30000 and ck['config']['seed']==1234
        net=BasicEncoder(output_dim=256,norm_fn='instance',downsample=2).eval()
        module=DescriptorAttention().eval()
        net.load_state_dict({k[5:]:v for k,v in ck['model'].items() if k.startswith('fnet.')},strict=True)
        module.load_state_dict({k[10:]:v for k,v in ck['model'].items() if k.startswith('placement.')},strict=True)
        result['models'][label]=dict(checkpoint_sha256=digest(path),source_identity=ck['source_identity'])
        models[label]=(net,module)
        del ck
    save()
    for case in cases:
        left,right=[load_spikes(args.data_root/case[k])[None] for k in ('spike_l','spike_r')]
        left,right=InputPadder(left.shape,divis_by=32).pad(left,right)
        for label,(net,module) in models.items():
            l,r=net([2*left-1,2*right-1])
            indices=torch.linspace(8,l.shape[2]-9,7).round().long()
            l,r=l[:,:,indices,:],r[:,:,indices,:]
            currents=projections(module,l,r)
            allcurr=torch.cat([x.flatten() for view in currents for x in view])
            codes,localgrad=signed_value_and_derivative(allcurr)
            assert torch.equal(codes,signed_rate(allcurr,4))
            baseline,metrics=exchange(module,l,r,currents,label)
            # Reconstruction check against the actual trained block path.
            if label=='linear':
                from core.conventional_attention_placement import LinearCodes
                module.codes.__class__=LinearCodes
            actual=module.compute(l,r)
            for a,b in zip(actual,baseline):torch.testing.assert_close(a.permute(0,2,3,1),b,rtol=0,atol=0)
            interventions={}
            for gain in (.5,2.):
                out,m=exchange(module,l,r,currents,label,gain)
                m['relative_output_change']=difference(out,baseline)
                interventions['current_gain_'+str(gain)]=m
            if label=='linear':
                for kind in ['clip-qk','clip-v','clip-all']:
                    out,m=exchange(module,l,r,currents,label,intervention=kind)
                    m['relative_output_change']=difference(out,baseline);interventions[kind]=m
            row=dict(scene=case['scene'],frame=case['frame'],model=label,feature_rows=indices.tolist(),
                current=summary(allcurr),fraction_abs_current_over_1=float((allcurr.abs()>1).float().mean()),
                signed_code_saturation_fraction=float((codes.abs()==1).float().mean()),
                local_surrogate_derivative=summary(localgrad),fraction_derivative_over_identity=float((localgrad>1).float().mean()),
                baseline=metrics,interventions=interventions)
            result['rows'].append(row);save()
            print(case['scene'],label,json.dumps(metrics),flush=True)
    result.update(status='completed',cpu_wall_seconds=time.monotonic()-started)
    signal.alarm(0);save()
    print('COMPLETED',args.output,flush=True)


if __name__=='__main__':main()
