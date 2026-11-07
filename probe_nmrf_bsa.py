"""Tiny CPU-only frozen NMRF Q/K coding probe; no training or GPU use.

Ternary codes simulate one coding step. Optional signed count encoders use
2/4/8 steps, with attention run once after decoding, not a full temporal SNN.
Scales come from training activations only. All outputs must be outside Git.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import subprocess
import time
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import numpy as np
import torch
from torch.nn import functional as F
from einops import rearrange, repeat
from nmrf_runtime import build_native_nmrf, UPSTREAM_REVISION, weighted_loss
from runtime import NativeDataset
from evaluate_nmrf import input_tensors, metric_counts, summarize
ROOT = Path(__file__).resolve().parent
VARIANTS = ['split_fp32', 'content_int8', 'content_int4', 'content_ternary_05',
            'content_ternary_10', 'whole_ternary_05']
RICH_VARIANTS = ['split_fp32', 'content_int4', 'content_ternary_05'] + [
    f'content_{kind}{steps}' for steps in (2, 4, 8) for kind in ('unary', 'quant')]


def signed_unary(normalized, steps):
    """Encode a static value with signed integrate-and-fire count coding.

    A half-threshold initial potential implements nearest-integer rounding.
    This encoder has state, but attention is evaluated ONCE after decoding;
    it is not a temporally executed spiking transformer or a sensor-time model.
    """
    magnitude = normalized.abs().clamp(max=steps).double()
    potential = torch.full_like(magnitude, .5)
    spikes = []
    for _ in range(steps):
        potential = potential + magnitude / steps
        fired = potential >= 1
        potential = potential - fired.to(potential.dtype)
        spikes.append(fired.to(normalized.dtype)*normalized.sign())
    return torch.stack(spikes)


def split_projection(module, inputs, channels):
    content = F.linear(inputs[..., :channels], module.weight[:, :channels], module.bias)
    geometry = F.linear(inputs[..., channels:], module.weight[:, channels:], None)
    return content, geometry


def head_view(tensor, heads=4):
    return tensor.reshape(*tensor.shape[:-1], heads, tensor.shape[-1]//heads)


def calibrate(values, heads=4):
    x = torch.cat([head_view(v, heads).reshape(-1, heads, v.shape[-1]//heads) for v in values])
    a = x.abs().permute(1, 0, 2).reshape(heads, -1)
    sigma = x.square().mean((0, 2)).sqrt()
    result = dict(clip=torch.quantile(a, .995, dim=1))
    for name, ratio in [('05', .5), ('10', 1.)]:
        threshold = ratio*sigma
        active = a > threshold[:, None]
        scale = (a*active).sum(1)/active.sum(1).clamp_min(1)
        result[name] = dict(threshold=threshold, scale=scale.clamp_min(1e-8))
    return result


def code(tensor, calibration, kind, ste=False):
    x = head_view(tensor, len(calibration['clip']))
    if kind.startswith('ternary'):
        cfg = calibration[kind.rsplit('_', 1)[-1]]
        threshold, scale = cfg['threshold'][:, None], cfg['scale'][:, None]
        integer = torch.sign(x)*(x.abs() > threshold)
        decoded = integer*scale
    elif kind.startswith(('unary', 'quant')):
        steps = int(kind.removeprefix('unary').removeprefix('quant'))
        scale = (calibration['clip']/steps).clamp_min(1e-8)[:, None]
        normalized = x/scale
        if kind.startswith('unary'):
            integer = signed_unary(normalized, steps).sum(0)
        else:
            integer = normalized.sign()*torch.floor(normalized.abs().clamp(max=steps).double()+.5).to(normalized.dtype)
        decoded = integer*scale
    else:
        levels = 127 if kind == 'int8' else 7
        scale = (calibration['clip']/levels).clamp_min(1e-8)[:, None]
        integer = torch.round(x/scale).clamp(-levels, levels)
        decoded = integer*scale
    decoded = decoded.reshape_as(tensor)
    if ste:
        decoded = tensor + (decoded-tensor).detach()
    return decoded, integer


def modify(output, content, geometry, calibration, variant, ste=False):
    if variant == 'split_fp32':
        return torch.cat((content[..., :256]+geometry[..., :256], output[..., 256:]), dim=-1)
    whole = variant.startswith('whole')
    kind = variant.split('_', 1)[1]
    source = output if whole else content
    base = torch.zeros_like(geometry) if whole else geometry
    decoded = []
    for j, key in enumerate(('q', 'k')):
        value, _ = code(source[..., j*128:(j+1)*128], calibration[key], kind, ste)
        decoded.append(value+base[..., j*128:(j+1)*128])
    return torch.cat((*decoded, output[..., 256:]), dim=-1)


def attention(module, qkv, mask):
    """Diagnostic mirror of WindowAttention, checked against upstream output."""
    bs, ht, wd, n, _ = qkv.shape
    if module.shift_size:
        qkv = torch.roll(qkv, (-module.shift_size, -module.shift_size), (1, 2))
    q, k, v = [module.window_partition(x) for x in qkv.chunk(3, -1)]
    pixels = module.window_size[0]*module.window_size[1]
    rpe = module.relative_position_enc_table[module.relative_position_index.reshape(-1)].view(
        pixels, pixels, module.num_heads, -1)
    qe, ke, ve = repeat(rpe, 'i j h c -> (i a) (j b) h c', a=n, b=n).chunk(3, -1)
    q, qe = q*module.scale, qe*module.scale
    logits = q @ k.transpose(-2, -1) + torch.einsum('bhic,ijhc->bhij', q, ke) + torch.einsum('bhjc,ijhc->bhij', k, qe)
    logits = rearrange(logits, '(b j) h m n -> b j h m n', b=bs)
    if mask is not None:
        logits = logits+mask[None, :, None]
    probs = rearrange(logits, 'b j h m n -> (b j) h m n').softmax(-1)
    message = probs@v+torch.einsum('bhij,ijhc->bhic', probs, ve)
    message = rearrange(message, '(b i j) h (a c n) d -> b (i a) (j c) n (h d)',
                        i=ht//module.window_size[0], j=wd//module.window_size[1],
                        a=module.window_size[0], c=module.window_size[1])
    if module.shift_size:
        message = torch.roll(message, (module.shift_size, module.shift_size), (1, 2))
    return probs, message


@contextlib.contextmanager
def interventions(model, calibrations, variant, ste=False):
    handles = []
    for i, layer in enumerate(model.parent.inference.layers):
        def hook(module, inputs, output, i=i):
            content, geometry = split_projection(module, inputs[0], 128)
            source = 'whole' if variant.startswith('whole') else 'content'
            return modify(output, content, geometry, calibrations[i][source], variant, ste)
        handles.append(layer.nmp.qkv.register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles: h.remove()


def capture(model, left, right):
    records, handles = [dict() for _ in model.parent.inference.layers], []
    for i, layer in enumerate(model.parent.inference.layers):
        def projection(module, inputs, output, i=i):
            content, geometry = split_projection(module, inputs[0], 128)
            records[i].update(content=content.detach(), geometry=geometry.detach(), original=output.detach())
        def attn(module, inputs, output, i=i):
            records[i].update(mask=inputs[1], message=output.detach())
        handles.extend([layer.nmp.qkv.register_forward_hook(projection), layer.nmp.attn.register_forward_hook(attn)])
    try:
        with torch.no_grad(): prediction = model(left, right)['disp'][0]
    finally:
        for h in handles: h.remove()
    return prediction, records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=ROOT/'configs/synthetic-split.json')
    p.add_argument('--richer', action='store_true', help='Compare 2/4/8-step signed unary encoders with equal-level ANN controls')
    a = p.parse_args()
    variants = RICH_VARIANTS if a.richer else VARIANTS
    if a.output.resolve().is_relative_to(ROOT): p.error('Outputs must be outside Git')
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2); torch.manual_seed(1234)
    manifest = json.loads(a.manifest.read_text())
    train = [next(r for r in manifest['train'] if r['scene']==f'scene_{s:04d}') for s in (0, 40, 80)]
    dev = [next(r for r in manifest['dev'] if r['scene']==f'scene_{s:04d}') for s in (120, 135, 149)]
    stress = next(r for r in manifest['dev'] if r['scene']=='scene_0134' and r['frame']=='0255')
    selected = dict(calibration=train, fixed_dev=dev, selected_stress=[stress])
    model, criterion, cfg = build_native_nmrf(1234)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint['upstream_revision'] != UPSTREAM_REVISION or checkpoint['model_name'] != 'nmrf-native-50':
        raise ValueError('Wrong checkpoint architecture')
    model.load_state_dict(checkpoint['model'], strict=True); model.eval()
    step = checkpoint['step']; del checkpoint
    initial_parameters = {k: v.detach().clone() for k,v in model.named_parameters()}
    crop = [48, 96]
    config = dict(checkpoint=str(a.checkpoint), step=step, device='cpu', torch=str(torch.__version__),
                  crop=crop, selected=selected, upstream_revision=UPSTREAM_REVISION,
                  revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  protocol='Frozen tiny center crops; single-step ternary simulation; no optimizer; not held-out benchmark accuracy',
                  calibration='Training-only per-head RMS thresholds 0.5/1.0; active absolute mean for ternary scale; 99.5 percentile clip for int4/int8',
                  geometry='Content-only Q/K splitting retains continuous geometry and all cross terms. Whole-Q/K arm deliberately quantizes geometry too.',
                  controls='Same one-step ternary values equal an ordinary ternary ANN. This is not evidence of a dynamical SNN advantage.')
    config['variants'] = variants
    if a.richer:
        config['protocol'] = 'Frozen tiny center crops; 2/4/8-step signed count encoders, attention once after decoding; no optimizer or GPU'
        config['richer_encoding'] = 'Training-only per-head 99.5 percentile clip / step budget; half-threshold initial state; at most one signed event per element per coding step'
        config['controls'] = 'Independent equal-level ANN round-to-nearest controls. Code count and complete predictions must agree exactly. No sparse kernel or time-unrolled attention.'
    (a.output/'config.json').write_text(json.dumps(config,indent=2))
    (a.output/'probe_nmrf_bsa.py').write_text(Path(__file__).read_text())
    started=time.monotonic(); captures=[]
    for sample in NativeDataset(a.data_root,train):
        left,right,*_=input_tensors(sample,'cpu',crop)
        _, rec=capture(model,left,right); captures.append(rec)
    calibrations=[]
    for i in range(len(model.parent.inference.layers)):
        cal={}
        for source,key in [('content','content'),('whole','original')]:
            cal[source]={name:calibrate([r[i][key][...,j*128:(j+1)*128] for r in captures]) for j,name in enumerate(('q','k'))}
        calibrations.append(cal)
    torch.save(calibrations,a.output/'calibration.pt')
    del captures
    summaries={v:dict(frames=[], local=[]) for v in ['reference']+variants}
    equality_error=0.; mirror_error=0.
    paired_prediction_error=0.
    for index,row in enumerate(dev+[stress]):
        left,right,target,valid=input_tensors(NativeDataset(a.data_root,[row])[0],'cpu',crop)
        mask=valid & (target>0) & (target<192)
        ref,rec=capture(model,left,right)
        ref_counts=metric_counts(ref,target,mask)
        summaries['reference']['frames'].append(dict(scene=row['scene'],frame=row['frame'],counts=ref_counts))
        paired_predictions={}
        for variant in variants:
            local=[]
            with torch.no_grad():
                for i,layer in enumerate(model.parent.inference.layers):
                    record=rec[i]; original=record['original']
                    calibration=calibrations[i]['whole' if variant.startswith('whole') else 'content']
                    changed=modify(original,record['content'],record['geometry'],calibration,variant)
                    pa,ma=attention(layer.nmp.attn,original,record['mask'])
                    pb,mb=attention(layer.nmp.attn,changed,record['mask'])
                    mirror_error=max(mirror_error,float((ma-record['message']).abs().max()))
                    kl=(pa*(pa.clamp_min(1e-12).log()-pb.clamp_min(1e-12).log())).sum(-1).mean()
                    metric=dict(layer=i,kl=float(kl),row_l1=float((pa-pb).abs().sum(-1).mean()),
                                top1_agreement=float((pa.argmax(-1)==pb.argmax(-1)).float().mean()),
                                message_relative_l2=float((ma-mb).norm()/ma.norm().clamp_min(1e-8)))
                    if variant!='split_fp32':
                        src=original if variant.startswith('whole') else record['content']
                        rates=[]
                        for j,name in enumerate(('q','k')):
                            decoded,integer=code(src[...,j*128:(j+1)*128],calibration[name],variant.split('_',1)[1])
                            rates.append(float((integer!=0).float().mean()))
                            if 'unary' in variant:
                                steps=int(variant.rsplit('unary',1)[-1])
                                metric[name+'_events_per_element']=float(integer.abs().mean())
                                metric[name+'_firing_per_step']=float(integer.abs().mean())/steps
                                ann,_=code(src[...,j*128:(j+1)*128],calibration[name],f'quant{steps}')
                                equality_error=max(equality_error,float((decoded-ann).abs().max()))
                            if 'ternary' in variant:
                                # Alternate signed event decoding must equal the same ternary ANN.
                                s=calibration[name][variant.rsplit('_',1)[-1]]['scale'][:,None]
                                event_decoded=((integer>0).float()-(integer<0).float())*s
                                equality_error=max(equality_error,float((event_decoded.reshape_as(decoded)-decoded).abs().max()))
                        metric['q_active'],metric['k_active']=rates
                    local.append(metric)
                with interventions(model,calibrations,variant): pred=model(left,right)['disp'][0]
                if 'unary' in variant:
                    paired_predictions[variant.replace('unary','quant')]=pred.clone()
                elif 'quant' in variant:
                    paired_prediction_error=max(paired_prediction_error,float((pred-paired_predictions[variant]).abs().max()))
            stats=dict(scene=row['scene'],frame=row['frame'],counts=metric_counts(pred,target,mask),
                       prediction_mae_change=float((pred-ref).abs().mean()), prediction_max_change=float((pred-ref).abs().max()))
            summaries[variant]['frames'].append(stats)
            summaries[variant]['local'].extend(local)
        print(json.dumps(dict(completed=index+1,scene=row['scene'],reference_epe=summarize([ref_counts])['epe'])),flush=True)
    result={}
    for variant,data in summaries.items():
        agg=dict(fixed_dev=summarize([r['counts'] for r in data['frames'][:3]]),
                 selected_stress=summarize([data['frames'][3]['counts']]),frames=data['frames'])
        if data['local']:
            agg['local_mean']={k:float(np.mean([r[k] for r in data['local']])) for k in data['local'][0] if k!='layer'}
            agg['local_records']=data['local']
        result[variant]=agg
    # Single backward through the frozen-shape model, no optimizer or parameter mutation.
    left,right,target,valid=input_tensors(NativeDataset(a.data_root,[dev[0]])[0],'cpu',crop)
    model.zero_grad(set_to_none=True)
    # The native refinement detaches its coarse disparity input. Its final loss
    # alone cannot test gradient flow into candidate inference: use the actual
    # auxiliary training losses that supervise this branch.
    model.train()
    model.parent.freeze_bn()
    gradient_variant='content_unary4' if a.richer else 'content_ternary_05'
    with interventions(model,calibrations,gradient_variant,ste=True):
        outputs=model(left,right)
        loss,_=weighted_loss(criterion,outputs,target[None],valid[None])
        loss.backward()
    gradients=[]
    for layer in model.parent.inference.layers:
        g=layer.nmp.qkv.weight.grad
        if g is None:
            raise RuntimeError('Missing candidate-inference gradient under native training loss')
        gradients.append(dict(q_norm=float(g[:128,:128].norm()),k_norm=float(g[128:256,:128].norm()),finite=bool(torch.isfinite(g).all())))
    checks=dict(mirror_max_abs_error=mirror_error,ternary_ann_vs_event_max_abs_error=equality_error,
                paired_ann_prediction_max_abs_error=paired_prediction_error,gradient_variant=gradient_variant,
                identity_prediction_max_abs_error=max(x['prediction_max_change'] for x in result['split_fp32']['frames']),
                gradients=gradients,parameters_unchanged=all(torch.equal(v,initial_parameters[k]) for k,v in model.named_parameters()))
    if mirror_error>1e-5 or equality_error!=0 or paired_prediction_error!=0 or checks['identity_prediction_max_abs_error']>0.005 or not checks['parameters_unchanged'] or not all(g['finite'] and g['q_norm']>0 and g['k_norm']>0 for g in gradients):
        raise RuntimeError(f'Probe verification failed: {checks}')
    report=dict(config=config,checks=checks,seconds=time.monotonic()-started,results=result)
    (a.output/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(completed=True,seconds=report['seconds'],checks=checks)),flush=True)


if __name__=='__main__':main()
