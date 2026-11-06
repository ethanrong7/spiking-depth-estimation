"""Read-only archived reference audit; write a compact explicit reuse receipt.

Run from the frozen phase2 source capsule. A receipt admits complete endpoints,
never resumes or rewrites historical runs. Missing/mismatched evidence fails.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
from protocol import MANIFEST_SHA, SEEDS, audit_history, tensor_hash, sha256
from common import atomic_json
EXP = Path(__file__).resolve().parent
PROJECT = EXP.parents[1] if len(EXP.parents)>1 else Path.cwd()


def verify(reference, arm, seed, output, source_root=None):
    global EXP,PROJECT
    if source_root is not None:
        PROJECT=source_root/'projects/snn-depth-estimation';EXP=PROJECT/'experiments/spiking-attention-placement-phase2'
    import torch
    assert arm in ('SC','Original') and seed in SEEDS
    current=json.loads((PROJECT.parents[1]/'source-registry.json').read_text())
    metadata=json.loads((reference/'metadata.json').read_text())
    old=metadata['source_registry']; config=metadata['config']
    assert config['seed']==seed and config['arm']==('A' if arm=='SC' else 'original')
    for field,value in dict(schedule_total=30000,effective_batch=8,physical_batch=4,augmentation='hv',manifest_sha256=MANIFEST_SHA).items():
        assert config[field]==value,field
    # Exact native model/loader/loss/forward and all core dependencies recorded
    # in the old capsule must remain unchanged. The opt-in matched SC module is
    # separately checked by exact initial tensor identity and model parity tests.
    required=[name for name in old['source_files'] if '/core/' in name or name.endswith('/scripts/run_experiment.py')]
    assert required
    for name in required:
        assert old['source_files'][name]==current['source_files'][name],name
        assert sha256(PROJECT.parents[1]/name)==current['source_files'][name],name
    sys.path.insert(0,str(PROJECT)); driver_path=EXP.parent/'spiking-transformer-stereo/scripts/run_experiment.py'
    sys.path.insert(0,str(driver_path.parent))
    spec=importlib.util.spec_from_file_location('reuse_native_driver',driver_path);driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    model=driver.build_model('original',seed)
    native_hash=tensor_hash(model.state_dict())
    if arm=='SC':
        from core.spiking_attention_phase2 import install
        model=install(model,cell='SC',seed=SEEDS[seed])
    assert tensor_hash(model.state_dict())==metadata['initial_state_sha256'],'initial tensors differ'
    common=tensor_hash({n:v for n,v in model.state_dict().items() if not n.startswith('placement.')})
    assert common==native_hash
    order=torch.randperm(960,generator=torch.Generator().manual_seed(seed)).tolist()
    assert metadata['order']==order
    rows=[json.loads(line) for line in (reference/'evaluations.jsonl').read_text().splitlines()]
    manifest=json.loads((EXP.parent/'spiking-transformer-stereo/configs/manifest.json').read_text())
    frame_ids={(r['scene'],r['frame']) for r in manifest['dev']}
    endpoints=[]
    for update in (7500,30000):
        path=reference/f'step-{update}.pt'
        if not path.exists() and update==30000:path=reference/'last.pt'
        checkpoint_verified=path.exists()
        assert checkpoint_verified or (arm=='Original' and update==7500),'required model endpoint missing'
        if checkpoint_verified:
            ck=torch.load(path,map_location='cpu',weights_only=False)
            assert ck['updates']==update and ck['samples_seen']==update*8 and ck['order']==order
            assert ck['config']==config and ck['source_identity']==old['source_identity_sha256']
            assert ck['scheduler']['last_epoch']==update and update in ck['evaluated']
            assert set(('python','numpy','torch','cuda','loader'))<=set(ck['rng'])
            assert all(int(state['step'])==update for state in ck['optimizer']['state'].values() if 'step' in state)
            model.load_state_dict(ck['model'],strict=True)
            assert all(torch.isfinite(tensor).all() for tensor in ck['model'].values())
            assert all(torch.isfinite(tensor).all() for state in ck['optimizer']['state'].values() for tensor in state.values() if torch.is_tensor(tensor))
            del ck
        matches=[row for row in rows if row['update']==update];assert len(matches)==1
        evaluation=matches[0]
        assert len(evaluation['per_frame'])==60 and len(evaluation['scene_epe'])==30
        assert {(r['scene'],r['frame']) for r in evaluation['per_frame']}==frame_ids
        assert all(r['valid_pixels']>0 for r in evaluation['per_frame'])
        endpoints.append(dict(update=update,checkpoint_verified=checkpoint_verified,checkpoint=str(path.resolve()) if checkpoint_verified else None,checkpoint_sha256=sha256(path) if checkpoint_verified else None,archived_source_identity=old['source_identity_sha256'],evaluation=evaluation,evidence_scope='complete checkpoint and official metrics' if checkpoint_verified else 'official metrics only; checkpoint unavailable and replay prohibited'))
    audit=audit_history(reference/'train.jsonl',order,seed,30000)
    receipt=dict(status='VERIFIED_REUSE',verifier_sha256=sha256(Path(__file__)),time_utc=datetime.now(timezone.utc).isoformat(),seed=seed,branch_seed=SEEDS[seed],arm=arm,historical_arm=config['arm'],reference=str(reference.resolve()),current_source_identity=current['source_identity_sha256'],archived_source_identity=old['source_identity_sha256'],initial_state_sha256=metadata['initial_state_sha256'],native_initial_state_sha256=native_hash,common_initial_state_sha256=common,endpoints=endpoints,stream_audit=audit,checked_dependency_files=required,confirmation='checkpoint replay parity remains required before diagnostics')
    atomic_json(output,receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--arm',choices=('SC','Original'),required=True);p.add_argument('--seed',choices=tuple(SEEDS),type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-root',type=Path)
    a=p.parse_args();verify(a.reference,a.arm,a.seed,a.output,a.source_root)
