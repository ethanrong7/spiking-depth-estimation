"""One small CPU-reference/NPU neuron forward-and-gradient check."""
import argparse
import json
from pathlib import Path
import torch
from neuron import MultiStepLIFNode


def validate(device):
    from spikingjelly.clock_driven.neuron import MultiStepLIFNode as Reference
    torch.manual_seed(917)
    records=[]
    for threshold in (1.,.5):
        x=(torch.randn(5,2,7)*.7+.8).requires_grad_()
        r=Reference(tau=2.,v_threshold=threshold,detach_reset=True,backend='torch')
        a=r(x);g=torch.autograd.grad(a.square().sum(),x)[0]
        y=x.detach().to(device).requires_grad_()
        m=MultiStepLIFNode(tau=2.,v_threshold=threshold,detach_reset=True).to(device)
        b=m(y);h=torch.autograd.grad(b.square().sum(),y)[0]
        assert torch.equal(a,b.cpu()),'Spike outputs differ from CPU reference'
        torch.testing.assert_close(g,h.cpu(),rtol=2e-5,atol=2e-6)
        records.append(dict(threshold=threshold,max_gradient_error=float((g-h.cpu()).abs().max())))
    return dict(status='passed',checks=records,torch=torch.__version__,device=str(device))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'b-attention-mechanism-v1'))
    from npu_backend import configure,environment
    # configure imports the native model.
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    result=validate(configure(0));result['environment']=environment()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
