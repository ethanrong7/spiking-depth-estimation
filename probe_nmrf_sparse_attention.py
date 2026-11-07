"""CPU diagnostic: signed-code routing followed by selected geometry-aware attention.

This is a frozen hybrid attention experiment, not a trained SNN. All experiment
outputs belong outside Git. Upstream NMRF is patched only within a context.
"""
import argparse
import contextlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from einops import rearrange, repeat

from evaluate_nmrf import input_tensors, metric_counts, summarize
from nmrf_runtime import build_native_nmrf, UPSTREAM_REVISION
from probe_nmrf_bsa import attention, calibrate, capture, signed_unary
from runtime import NativeDataset

ROOT = Path(__file__).resolve().parent


def packed_dot(q, k, bits):
    """Exact signed-integer dot products using packed 32-bit AND/popcounts.

    Signed two's-complement bitplanes are an execution control: a quantized ANN
    can use this identical operation. No claim of spike-specific acceleration.
    """
    if q.shape[-1] != 32 or k.shape[-1] != 32:
        raise ValueError('This CPU diagnostic packs 32 channels per head')
    weights = [2**b for b in range(bits-1)] + [-2**(bits-1)]
    powers = np.left_shift(np.uint32(1), np.arange(32, dtype=np.uint32))
    def pack(x):
        a = x.detach().numpy().astype(np.int32)
        return [(((a >> b) & 1).astype(np.uint32)*powers).sum(-1, dtype=np.uint32)
                for b in range(bits)]
    pq, pk = pack(q), pack(k)
    out = np.zeros((*q.shape[:-1], k.shape[-2]), dtype=np.int32)
    for i, wi in enumerate(weights):
        for j, wj in enumerate(weights):
            out += wi*wj*np.bitwise_count(pq[i][..., :, None] & pk[j][..., None, :]).astype(np.int32)
    return torch.from_numpy(out)


def partition(module, qkv, mask):
    bs, ht, wd, n, _ = qkv.shape
    if module.shift_size:
        qkv = torch.roll(qkv, (-module.shift_size, -module.shift_size), (1, 2))
    q, k, v = [module.window_partition(x) for x in qkv.chunk(3, -1)]
    nw, heads, tokens, dim = q.shape
    if mask is None:
        mask = q.new_zeros(nw, heads, tokens, tokens)
    elif mask.ndim == 3:
        mask = mask.repeat(bs, 1, 1)[:, None].expand(nw, heads, tokens, tokens)
    elif mask.shape != (nw, heads, tokens, tokens):
        raise ValueError('Unexpected attention mask shape')
    return q, k, v, mask


def unpartition(module, message, shape):
    bs, ht, wd, n, _ = shape
    message = rearrange(message, '(b i j) h (a c n) d -> b (i a) (j c) n (h d)',
                        b=bs, i=ht//module.window_size[0], j=wd//module.window_size[1],
                        a=module.window_size[0], c=module.window_size[1], n=n)
    if module.shift_size:
        message = torch.roll(message, (module.shift_size, module.shift_size), (1, 2))
    return message


def selected_attention(module, qkv, mask, indices):
    """Compute the five attention dot/reduction terms only for selected edges."""
    q, k, v, mask = partition(module, qkv, mask)
    nw, heads, tokens, dim = q.shape
    n = qkv.shape[-2]
    w = torch.arange(nw)[:, None, None, None]
    h = torch.arange(heads)[None, :, None, None]
    ks, vs = k[w, h, indices], v[w, h, indices]
    pixel = torch.arange(tokens)[None, None, :, None]//n
    rindex = module.relative_position_index[pixel, indices//n]
    table = module.relative_position_enc_table.reshape(-1, heads, 3*dim)
    qe, ke, ve = table[rindex, h].chunk(3, -1)
    query = q[..., None, :]*module.scale
    logits = (query*ks).sum(-1) + (query*ke).sum(-1) + (ks*qe*module.scale).sum(-1)
    probs = (logits + mask.gather(-1, indices)).softmax(-1)
    msg = (probs[..., None]*vs).sum(-2) + (probs[..., None]*ve).sum(-2)
    return unpartition(module, msg, qkv.shape)


def dense_probabilities(module, qkv, mask):
    """Original scores and softmax, without either dense value aggregation."""
    q,k,_,mask=partition(module,qkv,mask)
    n=qkv.shape[-2]
    pixels=module.window_size[0]*module.window_size[1]
    rpe=module.relative_position_enc_table[module.relative_position_index.reshape(-1)].view(pixels,pixels,module.num_heads,-1)
    qe,ke,_=repeat(rpe,'i j h c -> (i a) (j b) h c',a=n,b=n).chunk(3,-1)
    q=q*module.scale
    logits=q@k.transpose(-2,-1)+torch.einsum('bhic,ijhc->bhij',q,ke)+torch.einsum('bhjc,ijhc->bhij',k,qe*module.scale)
    return (logits+mask).softmax(-1)


def event_indices(probabilities, count, seed=1234):
    """Systematic probability-to-event conversion with one phase per row.

    Counts sum exactly to the row budget. This preserves row normalization and
    samples across the distribution, rather than keeping only largest weights.
    It is equivalent to a stochastic quantized attention matrix, not unique to SNNs.
    """
    generator=torch.Generator().manual_seed(seed)
    phase=torch.rand(*probabilities.shape[:-1],1,generator=generator)
    positions=(torch.arange(count)+phase)/count
    cdf=probabilities.double().cumsum(-1)
    cdf[...,-1]=1.
    return torch.searchsorted(cdf.contiguous(),positions.double().contiguous(),right=True).clamp_max(probabilities.shape[-1]-1)


def event_message(module,qkv,mask,indices):
    """Accumulate selected V and relative V messages with unit event weights."""
    _,_,v,_=partition(module,qkv,mask)
    nw,heads,tokens,dim=v.shape
    w=torch.arange(nw)[:,None,None,None]
    h=torch.arange(heads)[None,:,None,None]
    n=qkv.shape[-2]
    pixel=torch.arange(tokens)[None,None,:,None]//n
    rindex=module.relative_position_index[pixel,indices//n]
    table=module.relative_position_enc_table.reshape(-1,heads,3*dim)
    ve=table[rindex,h][...,2*dim:]
    message=(v[w,h,indices]+ve).sum(-2)/indices.shape[-1]
    return unpartition(module,message,qkv.shape)


def quantized_event_message(module,qkv,mask,indices):
    """Dense ANN with the same event-count attention weights, for equivalence."""
    _,_,v,_=partition(module,qkv,mask)
    nw,heads,tokens,dim=v.shape
    probabilities=v.new_zeros(nw,heads,tokens,tokens)
    probabilities.scatter_add_(-1,indices,torch.ones_like(indices,dtype=v.dtype))
    probabilities=probabilities/indices.shape[-1]
    n=qkv.shape[-2];pixels=module.window_size[0]*module.window_size[1]
    rpe=module.relative_position_enc_table[module.relative_position_index.reshape(-1)].view(pixels,pixels,heads,-1)
    ve=repeat(rpe,'i j h c -> (i a) (j b) h c',a=n,b=n).chunk(3,-1)[2]
    message=probabilities@v+torch.einsum('bhij,ijhc->bhic',probabilities,ve)
    return unpartition(module,message,qkv.shape)


def route(module, qkv, mask, calibration, kind, fraction):
    q, k, v, mask = partition(module, qkv, mask)
    nw, heads, tokens, dim = q.shape
    count = max(1, math.ceil(tokens*fraction))
    info = dict(dense_pair_macs=5*nw*heads*tokens*tokens*dim,
                selected_pair_macs=5*nw*heads*tokens*count*dim,
                routing_macs=0, packed_and_popcount_pairs=0,
                packed_weighted_accumulations=0, coding_element_steps=0,
                packing_bit_tests=0, selection_score_elements=nw*heads*tokens*tokens)
    if kind == 'events':
        probs=dense_probabilities(module,qkv,mask)
        indices=event_indices(probs,count)
        info['routing_macs']=3*nw*heads*tokens*tokens*dim
        info['selected_pair_macs']=0
        info['message_additions']=nw*heads*tokens*(2*count-1)*dim
        info['message_rescalings']=nw*heads*tokens*dim
        info['cdf_additions']=nw*heads*tokens*(tokens-1)
        info['event_searches']=nw*heads*tokens*count
        return indices,info
    elif kind.startswith('spike') or kind.startswith('quant'):
        steps = int(kind[-1])
        integers = []
        for name, x in [('q', q), ('k', k)]:
            scale = (calibration[name]['clip']/steps).clamp_min(1e-8)[None, :, None, None]
            normalized = x/scale
            if kind.startswith('spike'):
                events = signed_unary(normalized, steps)
                integer = events.sum(0)
                info[name+'_events'] = float(events.abs().sum())
                info['coding_element_steps'] += x.numel()*steps
            else:
                integer = normalized.sign()*torch.floor(normalized.abs().clamp(max=steps).double()+.5).float()
            integers.append(integer)
        bits = math.ceil(math.log2(steps+1))+1
        if kind.startswith('spike'):
            score = packed_dot(*integers, bits).float()
            info['packed_and_popcount_pairs'] = nw*heads*tokens*tokens*bits*bits
            info['packed_weighted_accumulations'] = info['packed_and_popcount_pairs']
            info['packing_bit_tests'] = (q.numel()+k.numel())*bits
        else:
            score = integers[0] @ integers[1].transpose(-2, -1)
            info['routing_macs'] = nw*heads*tokens*tokens*dim
    elif kind == 'ann':
        score = q @ k.transpose(-2, -1)
        info['routing_macs'] = nw*heads*tokens*tokens*dim
    elif kind == 'uniform':
        # A fixed balanced spatial sampling order, independent of values/labels.
        i = torch.arange(tokens)[:, None]
        j = torch.arange(tokens)[None, :]
        score = -((j-i)*37 % tokens).float()[None, None].expand(nw, heads, -1, -1)
    elif kind == 'oracle':
        probs, _ = attention(module, qkv, mask[:, 0] if qkv.shape[0] == 1 else None)
        score = probs.clamp_min(1e-38).log()
        info['routing_macs'] = 5*nw*heads*tokens*tokens*dim
    else:
        raise ValueError(kind)
    # Stable sorting gives equal-level ANN and packed codes identical tie breaks.
    indices = (score+mask).argsort(dim=-1, descending=True, stable=True)[..., :count]
    return indices, info


@contextlib.contextmanager
def intervention(model, calibration, kind, fraction, accounting):
    originals = []
    for i, layer in enumerate(model.parent.inference.layers):
        module = layer.nmp.attn
        originals.append((module, module.forward))
        def forward(qkv, mask, module=module, i=i):
            indices, info = route(module, qkv, mask, calibration[i], kind, fraction)
            accounting.append(info)
            fn=event_message if kind == 'events' else selected_attention
            return fn(module, qkv, mask, indices)
        module.forward = forward
    try:
        yield
    finally:
        for module, forward in originals:
            module.forward = forward


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=ROOT/'configs/synthetic-split.json')
    p.add_argument('--crop', type=int, nargs=2, default=[48, 96])
    p.add_argument('--routers', nargs='+', default=['spike2', 'quant2', 'spike4', 'ann', 'uniform', 'oracle'])
    p.add_argument('--fractions', type=float, nargs='+', default=[.25, .5])
    p.add_argument('--profile', action='store_true', help='Profile supported baseline operator FLOPs once on CPU')
    args = p.parse_args()
    if args.output.resolve().is_relative_to(ROOT):
        p.error('Outputs must be outside Git')
    if any(not 0 < f <= 1 for f in args.fractions):
        p.error('Fractions must be in (0,1]')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    model, _, _ = build_native_nmrf(1234)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint['upstream_revision'] != UPSTREAM_REVISION or checkpoint['model_name'] != 'nmrf-native-50':
        raise ValueError('Wrong checkpoint architecture')
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    manifest = json.loads(args.manifest.read_text())
    train = [next(r for r in manifest['train'] if r['scene']==f'scene_{s:04d}') for s in (0,40,80)]
    dev = [next(r for r in manifest['dev'] if r['scene']==f'scene_{s:04d}') for s in (120,135,149)]
    stress = next(r for r in manifest['dev'] if r['scene']=='scene_0134' and r['frame']=='0255')
    config = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  checkpoint_step=checkpoint['step'], upstream_revision=UPSTREAM_REVISION,
                  revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  train_calibration=train, primary_dev=dev, selected_stress=stress,
                  hypothesis='Cheap signed-code routing can prune neighbor edges before precise geometry-aware attention.',
                  screening_rule='Exploratory: EPE increase <=0.05 px and bad3 increase <=0.5 percentage points on primary crops, with >=40% fewer precise pair MACs. No benchmark or energy claim.',
                  cost_scope='Five neighbor WindowAttention modules only; excludes unchanged model, gathers, sort comparisons, code division, softmax, memory and conversion costs. These are not zero-cost.',
                  torch=str(torch.__version__),numpy=np.__version__,device='cpu')
    if 'events' in args.routers:
        config['event_hypothesis']='Keep exact geometry-aware scores, encode each normalized attention row as a fixed budget of sampled events, and accumulate V plus relative-V without floating attention-weight multiplications.'
        config['event_interpretation']='Systematic sampling with seed 1234; full Q/K scoring and softmax retained; equivalent quantized ANN message checked; no sensor-time dynamics or energy claim.'
        config['event_screening_rule']='Exploratory: primary EPE increase <=0.05 px, bad3 increase <=0.5 pp; inspect bad1 and selected stress separately. Report arithmetic reductions with dense scoring, CDF, event selection and message additions included.'
    (args.output/'config.json').write_text(json.dumps(config, indent=2))
    (args.output/Path(__file__).name).write_text(Path(__file__).read_text())
    del checkpoint
    started = time.monotonic()
    with torch.no_grad():
        calibration_records=[]
        for sample in NativeDataset(args.data_root, train):
            left,right,*_=input_tensors(sample,'cpu',args.crop)
            _,records=capture(model,left,right)
            calibration_records.append(records)
        calibration=[{name:calibrate([r[i]['original'][...,j*128:(j+1)*128] for r in calibration_records])
                      for j,name in enumerate(('q','k'))} for i in range(5)]
        torch.save(calibration,args.output/'calibration.pt')
        del calibration_records
        results={}; identity_error=0.; paired_error=0.; packed_error=0.; event_ann_error=0.
        for frame_index,sample in enumerate(NativeDataset(args.data_root,dev+[stress])):
            left,right,target,valid=input_tensors(sample,'cpu',args.crop)
            valid=valid & (target>0) & (target<192)
            if args.profile and frame_index == 0:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
                    model(left,right)
                ops=[dict(operator=e.key,count=e.count,flops=e.flops) for e in prof.key_averages()]
                (args.output/'baseline-operator-flops.json').write_text(json.dumps(dict(
                    operators=ops,total_supported_flops=sum(e['flops'] for e in ops),
                    convention='Profiler counts supported operators only, generally two FLOPs per MAC. Unsupported operators are not free; this is not full-model latency or energy.'),indent=2))
            reference,records=capture(model,left,right)
            results.setdefault('reference',[]).append(dict(counts=metric_counts(reference,target,valid)))
            paired={}
            for kind in args.routers:
                for fraction in args.fractions:
                    name=f'{kind}_{fraction:g}'; local=[]
                    for i,record in enumerate(records):
                        module=model.parent.inference.layers[i].nmp.attn
                        qkv,mask=record['original'],record['mask']
                        idx,info=route(module,qkv,mask,calibration[i],kind,fraction)
                        fn=event_message if kind == 'events' else selected_attention
                        message=fn(module,qkv,mask,idx)
                        probs,original=attention(module,qkv,mask)
                        keep=torch.zeros_like(probs).scatter_(-1,idx,1.)
                        if kind == 'events':
                            event_ann_error=max(event_ann_error,float((message-quantized_event_message(module,qkv,mask,idx)).abs().max()))
                        local.append(dict(retained_attention_mass=float((probs*keep).sum(-1).mean()),
                                          unique_edge_fraction=float(keep.mean()),
                                          message_relative_l2=float((message-original).norm()/original.norm())))
                        if kind == args.routers[0] and fraction == args.fractions[0]:
                            q,k,*_=partition(module,qkv,mask)
                            allidx=torch.arange(q.shape[-2]).expand(*q.shape[:-1],q.shape[-2])
                            identity_error=max(identity_error,float((selected_attention(module,qkv,mask,allidx)-original).abs().max()))
                            qi=q.round().clamp(-2,2);ki=k.round().clamp(-2,2)
                            packed_error=max(packed_error,float((packed_dot(qi,ki,3)-(qi@ki.transpose(-2,-1))).abs().max()))
                    accounting=[]
                    t=time.monotonic()
                    with intervention(model,calibration,kind,fraction,accounting):
                        prediction=model(left,right)['disp'][0]
                    elapsed=time.monotonic()-t
                    if kind == 'spike2': paired[fraction]=prediction.clone()
                    if kind == 'quant2' and fraction in paired:
                        paired_error=max(paired_error,float((prediction-paired[fraction]).abs().max()))
                    results.setdefault(name,[]).append(dict(counts=metric_counts(prediction,target,valid),
                        local=local,accounting=accounting,forward_seconds=elapsed,
                        prediction_mae_change=float((prediction-reference).abs().mean())))
            print(json.dumps(dict(frame=frame_index+1,seconds=time.monotonic()-started)),flush=True)
        checks=dict(full_selection_message_max_abs_error=identity_error,packed_dot_max_abs_error=packed_error,
                    event_vs_quantized_ann_message_max_abs_error=event_ann_error,
                    paired_ann_prediction_max_abs_error=paired_error,
                    state_unchanged=all(torch.equal(v,initial[k]) for k,v in model.state_dict().items()))
        if identity_error>1e-4 or event_ann_error>1e-5 or packed_error!=0 or paired_error!=0 or not checks['state_unchanged']:
            raise RuntimeError(checks)
        summary={name:dict(primary=summarize([r['counts'] for r in rows[:3]]),
                           selected_stress=summarize([rows[3]['counts']])) for name,rows in results.items()}
        report=dict(config=config,checks=checks,seconds=time.monotonic()-started,summary=summary,results=results)
        (args.output/'results.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(dict(checks=checks,summary=summary,seconds=report['seconds'])),flush=True)


if __name__ == '__main__':
    main()
