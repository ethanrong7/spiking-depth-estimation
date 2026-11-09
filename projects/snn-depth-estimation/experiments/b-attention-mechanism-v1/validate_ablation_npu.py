"""Small on-device encoder VJP and B checkpoint tests; no training data needed."""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP.parents[1]))
from core.attention_encoder_controls import VARIANTS, controlled_code, configure_codes
from core.epipolar_spiking_stereo import signed_rate
from core.spiking_attention_placement import DescriptorAttention
from core.conventional_attention_placement import LinearCodes


def validate(device):
    """Also runs on CPU for local tests. CPU float64 is the scalar oracle."""
    torch.manual_seed(123)
    current = torch.cat((torch.linspace(-6, 6, 257), torch.tensor([-1., 0., 1., 4.])))
    incoming = torch.randn_like(current)
    oracle = current.double().requires_grad_()
    expected = signed_rate(oracle, 4)
    derivative, = torch.autograd.grad(expected, oracle, incoming.double())
    # Explicit FP32 tolerance: recurrence arithmetic may differ by backend.
    rtol, atol = 1e-4, 1e-5
    for variant, (forward, backward) in VARIANTS.items():
        x = current.to(device).requires_grad_()
        y = controlled_code(x, forward, backward)
        reference = signed_rate(x, 4) if forward == 'signed' else x
        assert torch.equal(y, reference), variant
        gradient, = torch.autograd.grad(y, x, incoming.to(device))
        wanted = derivative.float() if backward == 'surrogate' else incoming
        torch.testing.assert_close(gradient.cpu(), wanted, rtol=rtol, atol=atol)

    base = DescriptorAttention()
    with torch.no_grad():
        base.output.weight.normal_(std=.02)
    inputs = [torch.randn(1, 256, 2, 7).to(device) for _ in range(2)]
    outputs, gradients = {}, {}
    for variant in VARIANTS:
        pairs = []
        models = []
        for checkpointed in (False, True):
            model = copy.deepcopy(base).to(device).train()
            configure_codes(model.codes, variant)
            x = [value.clone().requires_grad_() for value in inputs]
            out = model(*x) if checkpointed else model.compute(*x)
            sum(value.square().sum() for value in out).backward()
            pairs.append(([v.detach() for v in out],
                          [p.grad for p in model.parameters()], [v.grad for v in x]))
            models.append(model)
        for direct, checked in zip(pairs[0], pairs[1]):
            for actual, wanted in zip(direct, checked):
                torch.testing.assert_close(actual, wanted, rtol=rtol, atol=atol)
        for name in ('q', 'k', 'v'):
            grad = getattr(models[0].codes, name).weight.grad
            assert torch.isfinite(grad).all() and grad.norm() > 0, (variant, name)
        if variant in ('linear-identity', 'signed-surrogate'):
            reference = copy.deepcopy(base).to(device).train()
            if variant == 'linear-identity':
                reference.codes.__class__ = LinearCodes
            out = reference.compute(*inputs)
            sum(value.square().sum() for value in out).backward()
            for actual, wanted in zip(pairs[0][0], out):
                torch.testing.assert_close(actual, wanted, rtol=rtol, atol=atol)
            for actual, wanted in zip(models[0].parameters(), reference.parameters()):
                torch.testing.assert_close(actual.grad, wanted.grad, rtol=rtol, atol=atol)
        outputs[variant] = pairs[0][0]
        gradients[variant] = models[0].codes.q.weight.grad
    for forward in ('linear', 'signed'):
        for actual, wanted in zip(outputs[forward+'-identity'], outputs[forward+'-surrogate']):
            torch.testing.assert_close(actual, wanted, rtol=rtol, atol=atol)
        assert not torch.allclose(gradients[forward+'-identity'], gradients[forward+'-surrogate'])
    return dict(status='passed', device=str(device), dtype='float32',
                scalar_oracle='cpu_float64', rtol=rtol, atol=atol,
                checks=['encoder_forward_and_vjp', 'opened_readout_qkv_gradients',
                        'checkpoint_recompute', 'historical_diagonal_operators'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from npu_backend import configure, environment
    result = validate(configure(0))
    result['environment'] = environment()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
