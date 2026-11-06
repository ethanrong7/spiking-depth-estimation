"""Tiny equation checks using archived modules; not trained accuracy evidence."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib
import json
import math
from pathlib import Path
import sys
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PROJECT = REPO / 'storage/experiment-runs/snn-depth-estimation/trained-cost-filter-cpu-controls-v1/source/projects/snn-depth-estimation'
sys.path[:0] = [str(PROJECT), str(PROJECT / 'core')]
from core.epipolar_spiking_stereo import signed_rate
from core.guided_stereo_attention import GuidedCorrection
from core.spiking_attention_update import RefinementLIF


def main():
    torch.set_num_threads(2)
    torch.manual_seed(61)
    result = {'scope':'synthetic equation/gradient checks only; no trained accuracy inference', 'checks':{}}
    checks = result['checks']
    x = torch.linspace(-3, 3, 601)
    y = signed_rate(x, 4)
    assert torch.equal(y, signed_rate(x, 4))
    assert torch.equal(y*4, (y*4).round()) and y.abs().max() <= 1
    checks['static_code'] = {'levels_on_grid': y.unique().tolist(), 'reset_repeat_exact':True,
        'meaning':'T4 defines a scalar quantizer of a fixed current; no sensor-time inputs enter this call'}

    guide = GuidedCorrection(8).eval()
    with torch.no_grad():
        guide.query.weight.zero_(); guide.query.bias.zero_()
        guide.key.weight.zero_(); guide.key.bias.zero_(); guide.gate.fill_(1)
        readout = torch.randn(1,8,2,12)
        costs = torch.randn(1,36,2,12)
        disparity = torch.full((1,2,2,12), -2.)
        correction = guide(readout, costs, disparity)
        assert correction.abs().max() < 1e-5
    checks['guided_uniform'] = {'max_abs_correction':correction.abs().max().item(),
        'meaning':'subtracting valid-offset mean removes correction under uniform weights, including truncated support'}
    # Each head has four binary bits, with dot-product divided by two.
    odds = math.exp(2)
    checks['guided_score_bound'] = {'min_logit':0, 'max_logit':2, 'max_valid_weight_ratio':odds,
        'max_one_candidate_probability_if_36_valid':odds/(odds+35),
        'meaning':'soft displacement correction, not a sharp 36-way selector; upper bound depends on valid support'}
    q = torch.tensor([[0.,0.,0.,0.],[1.,1.,1.,1.]])
    score = q@q.T/2
    assert score.min() == 0 and score.max() == 2

    current = torch.randn(1,8,3,5)
    norm = torch.nn.GroupNorm(1,8)
    transformed = 10*current+7
    normalized = norm(current)
    normalized_transformed = norm(transformed)
    difference = (normalized-normalized_transformed).abs().max().item()
    assert difference < 1e-4
    def evolve(c):
        cell = RefinementLIF()
        for _ in range(4):
            _, v = cell(c)
        return v
    nv, nt = evolve(normalized), evolve(normalized_transformed)
    rv, rt = evolve(current), evolve(transformed)
    checks['normalized_current'] = {'normalized_current_max_difference':difference,
        'normalized_four_step_membrane_MAE':(nv-nt).abs().mean().item(),
        'raw_four_step_membrane_MAE':(rv-rt).abs().mean().item(),
        'meaning':'GroupNorm can remove shared gain/offset at this interface; this does not identify the trained gain or imply invariance to arbitrary sensor changes'}
    assert (nv-nt).abs().mean() < 1e-4

    # A direct input-current path changes sensitivity independently of attention.
    u = torch.tensor([.2], requires_grad=True)
    gate = .01
    attenuated = gate*u
    independent = gate*u+u
    g0, = torch.autograd.grad(attenuated, u, retain_graph=True)
    g1, = torch.autograd.grad(independent, u)
    checks['direct_current_path_algebra'] = {'gated_local_derivative':g0.item(), 'with_independent_path':g1.item(),
        'meaning':'illustration of a direct path bypassing multiplicative attenuation; not a measurement of actual ALIF gates or gradients'}
    result['source_files'] = {str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (PROJECT/'core/epipolar_spiking_stereo.py', PROJECT/'core/guided_stereo_attention.py', PROJECT/'core/spiking_attention_update.py')}
    result['all_checks_passed'] = True
    (HERE/'results').mkdir(exist_ok=True)
    (HERE/'results/mechanics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['checks'], indent=2))


if __name__ == '__main__':
    main()
