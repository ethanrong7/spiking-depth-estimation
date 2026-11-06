"""Read-only CPU audit of a retained phase-2 endpoint and parameter changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--endpoint', type=Path, required=True)
    a = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only audit requires CUDA hidden'
    project = a.source_root / 'projects/snn-depth-estimation'
    sys.path[:0] = [str(project), str(project / 'experiments/spiking-attention-placement-phase2')]
    import torch
    from protocol import SEEDS, tensor_hash
    from core.spiking_attention_phase2 import MatchedCostFilter
    torch.set_num_threads(1)
    endpoint = json.loads(a.endpoint.read_text())
    config = endpoint['config']
    update = endpoint['update']
    assert update in (7500, 30000)
    checkpoint = Path(endpoint['checkpoint'])
    digest = hashlib.sha256()
    with checkpoint.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == endpoint['checkpoint_sha256']
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert ck['updates'] == update and ck['samples_seen'] == update * 8
    assert ck['scheduler']['last_epoch'] == update and update in ck['evaluated']
    assert ck['config'] == config and ck['source_identity'] == endpoint['source_identity']
    steps = sorted({int(value['step']) for value in ck['optimizer']['state'].values() if 'step' in value})
    assert steps == [update]
    assert all(bool(torch.isfinite(value).all()) for value in ck['model'].values())
    assert all(bool(torch.isfinite(value).all()) for state in ck['optimizer']['state'].values()
               for value in state.values() if torch.is_tensor(value))
    assert {'python', 'numpy', 'torch', 'cuda', 'loader'} <= set(ck['rng'])
    metadata = json.loads((checkpoint.parent / 'metadata.json').read_text())
    assert metadata['source_registry']['source_identity_sha256'] == ck['source_identity']
    assert endpoint['common_initial_state_sha256'] == endpoint['native_initial_state_sha256']
    fields, count = {}, 0
    if config['arm'] != 'Original':
        assert config['branch_seed'] == SEEDS[config['seed']]
        torch.manual_seed(config['branch_seed'])
        initial = MatchedCostFilter(config['arm']).state_dict()
        assert tensor_hash(initial) == metadata['branch_initial_state_sha256']
        count = sum(value.numel() for value in initial.values())
        for name, value in initial.items():
            trained = ck['model']['placement.' + name]
            delta = trained - value
            fields[name] = dict(elements=value.numel(), changed_elements=int((delta != 0).sum()),
                                max_absolute_change=float(delta.abs().max()),
                                l2_change=float(torch.linalg.vector_norm(delta)), finite=True)
    print(json.dumps(dict(observed_utc=datetime.now(timezone.utc).isoformat(), cpu_only=True,
          arm=config['arm'], seed=config['seed'], update=update, presentations=update * 8,
          source_identity=ck['source_identity'], checkpoint_sha256=digest.hexdigest(),
          scheduler_last_epoch=update, optimizer_step_counts=steps, complete_rng_state=True,
          finite_model_optimizer=True, placement_parameter_count=count,
          parameter_deltas_from_recorded_initialization=fields,
          auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2))


if __name__ == '__main__':
    main()
