"""CPU integration check of independent forward/backward rules in trained A.

Uses tiny synthetic cost rows and the admitted A checkpoint's filter weights.
No optimizer steps, native data, decoder inference, GPU or EPE evaluation.
The fixed output cotangent tests local derivatives, not the stereo loss.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from types import MethodType
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SOURCE = REPO / 'storage/experiment-runs/snn-depth-estimation/trained-cost-filter-cpu-controls-v1/source'
PROJECT = SOURCE / 'projects/snn-depth-estimation'
sys.path[:0] = [str(PROJECT), str(PROJECT/'core')]
from core.spiking_attention_placement import CostFilter
from core.epipolar_spiking_stereo import signed_rate
from forward_backward_controls import controlled_code, signed_value_and_derivative


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_code(module, forward, backward, native=False):
    def code(self, x):
        z = self.norm(x)
        currents = [projection(z) for projection in (self.q,self.k,self.v)]
        if native:
            return tuple(signed_rate(c,4) if forward=='signed' else c.tanh() for c in currents)
        return tuple(controlled_code(c,forward,backward) for c in currents)
    module.codes.forward = MethodType(code,module.codes)


def evaluate(template, x, cotangent, forward, backward, native=False, checkpointed=False):
    model = copy.deepcopy(template)
    set_code(model,forward,backward,native)
    model.train(checkpointed)
    cost = x.detach().clone().requires_grad_()
    prediction = model(cost)
    params = dict(model.named_parameters())
    gradients = torch.autograd.grad(prediction,(cost,*params.values()),cotangent)
    gradients = {k:g.detach() for k,g in zip(('cost',*params),gradients)}
    assert all(torch.isfinite(g).all() for g in gradients.values())
    return dict(output=prediction.detach(),gradients=gradients)


def compare(reference, actual, atol, rtol):
    assert torch.equal(reference['output'],actual['output'])
    errors={}
    for name,g in reference['gradients'].items():
        torch.testing.assert_close(actual['gradients'][name],g,atol=atol,rtol=rtol)
        errors[name]=(actual['gradients'][name]-g).abs().max().item()
    return max(errors.values())


def qkv_gradient(row):
    return torch.cat([g.flatten() for n,g in row['gradients'].items()
                      if n.startswith(('codes.q.','codes.k.','codes.v.'))])


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    started=time.monotonic()
    checkpoint=REPO/'storage/experiment-runs/snn-depth-estimation/spiking-transformer-stereo/spiking-attention-placement-v1/train/A/step-30000.pt'
    expected='d5dad669ba1ba521f36f7ede66142f163afca896cf677f7586a4f4b53829031b'
    assert digest(checkpoint)==expected
    registry=json.loads((SOURCE/'source-registry.json').read_text())
    for relative in ('projects/snn-depth-estimation/core/spiking_attention_placement.py',
                     'projects/snn-depth-estimation/core/epipolar_spiking_stereo.py'):
        assert digest(SOURCE/relative)==registry['source_files'][relative]
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['updates']==30000 and ck['config']['seed']==1234 and ck['config']['arm']=='A'
    state={n[len('placement.'):]:v for n,v in ck['model'].items() if n.startswith('placement.')}
    del ck
    model=CostFilter().eval(); model.load_state_dict(state,strict=True)
    assert model.output.weight.abs().sum()>0
    before={n:t.clone() for n,t in model.state_dict().items()}
    result=dict(status='running',scope=__doc__,checkpoint_sha256=expected,source_identity=registry['source_identity_sha256'],
                helper_sha256=digest(HERE/'forward_backward_controls.py'),test_sha256=digest(Path(__file__)),
                torch=torch.__version__,cpu_threads=2,optimizer_steps=0,results=[])
    for dtype,atol,rtol in ((torch.float64,2e-12,2e-11),(torch.float32,3e-6,3e-5)):
        # Three deterministic profiles include local peaks, alternating values,
        # and a broad ramp. Width17 exercises both padded window boundaries.
        t=torch.linspace(-1,1,17,dtype=dtype)
        x=torch.stack((3*t,2*torch.sin(5*t),4*torch.exp(-9*t*t)-1)).reshape(1,1,3,1,17)
        upstream=torch.cos(torch.arange(x.numel(),dtype=dtype)*.73).reshape_as(x)
        template=copy.deepcopy(model).to(dtype)
        cells={}
        errors=[]
        checkpoint_errors=[]
        for forward in ('signed','tanh'):
            reference=evaluate(template,x,upstream,forward,forward,native=True)
            diagonal=evaluate(template,x,upstream,forward,forward)
            errors.append(compare(reference,diagonal,atol,rtol))
            for backward in ('signed','tanh'):
                row=evaluate(template,x,upstream,forward,backward)
                recomputed=evaluate(template,x,upstream,forward,backward,checkpointed=True)
                checkpoint_errors.append(compare(row,recomputed,atol,rtol))
                cells[(forward,backward)]=row
        contrasts=[]
        for forward in ('signed','tanh'):
            a,b=(cells[(forward,g)] for g in ('signed','tanh'))
            assert torch.equal(a['output'],b['output'])
            ga,gb=qkv_gradient(a),qkv_gradient(b)
            assert (ga-gb).norm()>1e-8
            # Downstream parameter gradients must not change under the same
            # forward function and same externally fixed output cotangent.
            for name in ('output.weight','output.bias'):
                assert torch.equal(a['gradients'][name],b['gradients'][name])
            contrasts.append(dict(forward=forward,outputs_bitwise_identical=True,
                qkv_gradient_norm_signed=ga.norm().item(),qkv_gradient_norm_tanh=gb.norm().item(),
                qkv_gradient_cosine=torch.nn.functional.cosine_similarity(ga,gb,dim=0).item(),
                qkv_gradient_difference_norm=(ga-gb).norm().item(),
                output_projection_gradients_identical=True))
        forward_difference=(cells[('signed','signed')]['output']-cells[('tanh','signed')]['output']).abs().max().item()
        assert forward_difference>0
        result['results'].append(dict(dtype=str(dtype),diagonal_max_gradient_error=max(errors),
            checkpoint_recompute_max_gradient_error=max(checkpoint_errors),
            signed_vs_tanh_output_max_difference=forward_difference,backward_only_contrasts=contrasts))
    sample=torch.tensor([0.,1.,2.,4.,8.],dtype=torch.float64)
    value,gradient=signed_value_and_derivative(sample)
    result['scalar_examples']=[dict(current=x.item(),signed_value=s.item(),tanh_value=x.tanh().item(),
        signed_backward=g.item(),tanh_backward=(1-x.tanh().square()).item()) for x,s,g in zip(sample,value,gradient)]
    assert all(torch.equal(before[n],v) for n,v in model.state_dict().items())
    result.update(status='passed',elapsed_seconds=time.monotonic()-started,template_weights_unchanged=True,
                  cuda_initialized=torch.cuda.is_initialized(),native_rows=0,synthetic_cost_rows=3)
    (HERE/'results').mkdir(exist_ok=True)
    (HERE/'results/forward-backward-block-cpu-20261105.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
