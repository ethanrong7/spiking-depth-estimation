"""Reconcile copied logical ledgers; never import Torch, decode data or infer."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def total(rows):
    answer=Counter()
    for row in rows:
        assert all(type(v) is int and v>=0 for v in row.values())
        answer.update(row)
    return dict(answer)


def close(a,b):
    assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-9),(a,b)


def source_reset_terms(ledger,result,inventory):
    """Pinned source equations, explicitly separate from observed boundaries."""
    if result['checkpoint']['model']['family']!='baseline':
        assert not inventory['scripts']
        return dict(source_derived_unobserved_reset_mul=0,source_derived_unobserved_reset_sub=0)
    provenance=result['checkpoint']['baseline_provenance']['module_sources']
    assert provenance['projects/snn-depth-estimation/core/ALIF.py']['source_sha256']=='c26d96e17d5e56e9e8b71e7d9d784ccb8481d259aa8d077a21fbcbd13be3e5aa'
    mechanism='projects/snn-depth-estimation/experiments/spiking-transformer-stereo/campaigns/alif-mechanism-screen-v1/scripts/mechanism.py'
    assert provenance[mechanism]['source_sha256']=='f8a4147f682a4794ee7f814defcb3684d14764958d61b6606e527de9865f9ae0'
    resets=[r for r in inventory['scripts'].values() if any(p.endswith('.jit_soft_reset') for p in r['references'])]
    assert len(resets)==1
    ops=Counter(op for op in resets[0]['raw_graph_operations'] if op.startswith('aten::'))
    assert ops==Counter({'aten::mul':2,'aten::sub':1})
    signature=ledger['first_observed_signatures']['aten.einsum.default']
    batch,channels,h,w=signature['input_tensors'][0]['shape']
    assert batch==1 and channels==256 and h%4==w%4==0
    refinements=result['checkpoint']['model']['config']['eval_refinements'];assert refinements==32
    missing_mul=missing_sub=0
    for layer,divisor in [('08',1),('16',2),('32',4)]:
        node='model.update_block.snn'+layer+'.snn'
        n=128*(h//divisor)*(w//divisor)*refinements
        assert ledger['components'][node+'.surrogate_function']['element_ge']==n
        row=ledger['components'][node]
        # Charge/threshold:3mul+1sub; active reset adds2mul+1sub per element.
        mul=5*n-row.get('element_mul',0);sub=2*n-row.get('element_sub',0)
        assert 0<=sub<=n and mul==2*sub
        missing_mul+=mul;missing_sub+=sub
    return dict(source_derived_unobserved_reset_mul=missing_mul,source_derived_unobserved_reset_sub=missing_sub)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--progress',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text());progress=json.loads(args.progress.read_text())
    registry_path=args.snapshot/'source-registry.json';registry=json.loads(registry_path.read_text())
    assert len(plan['cases'])==10 and len(progress['cases'])==10 and progress['active'] is None
    assert args.snapshot.name==progress['source_snapshot']
    source_hashes={str(args.plan):sha(args.plan),str(args.progress):sha(args.progress),
                   str(registry_path):sha(registry_path)}
    cases=[];all_targets=[];reference_ids=None;reference_scenes=None
    for case,record in zip(plan['cases'],progress['cases']):
        assert case['case']==record['case'] and case['run']==record['run']
        assert record['state']=='COMPLETED'
        root=Path(record['raw_root']);inv_path=Path(record['inventory'])
        inv=json.loads(inv_path.read_text());source_hashes[str(inv_path)]=sha(inv_path)
        assert inv['exact_file_set_verified'] and inv['remote_before_after_equal']
        assert inv['job_id']==record['job_id']
        assert {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}==set(inv['files'])
        for name,member in inv['files'].items():
            path=root/name
            assert not path.is_symlink() and path.stat().st_size==member['bytes'] and sha(path)==member['sha256']
            source_hashes[str(path)]=member['sha256']
        result=json.loads((root/'result.json').read_text())
        script_inventory=json.loads((root/result['scripted_source_inventory_file']).read_text())
        assert sha(root/result['scripted_source_inventory_file'])==result['scripted_source_inventory_sha256']
        assert result['status']=='COMPLETED_LOGICAL_ARITHMETIC_DIAGNOSTIC' and result['complete20']
        assert result['weights_buffers_unchanged'] and result['test_access'] is False
        assert result['source_identity_sha256']==registry['source_identity_sha256']
        assert result['source_registry_sha256']==sha(registry_path)
        assert result['config_sha256']==case['config_sha256']
        member='projects/snn-depth-estimation/experiments/stereo-spiking-transformer-study/configs/'+case['config']
        assert registry['source_files'][member]==case['config_sha256']
        assert result['checkpoint']['checkpoint_sha256']==case['checkpoint_sha256']
        assert result['checkpoint']['binding']['source_identity_sha256']=='e84cbdafeeb332ac33d6ae509168cd418bae1958bcc8d47b4fe2cc4c7349149a'
        flag=result['diagnostics_variant_parity']['status']
        assert flag==('NOT_APPLICABLE_BASELINE' if case['case'].startswith('original-I') else 'PASSED_CHECKPOINT_DIAGNOSTICS_PARITY')
        samples=result['samples'];assert len(samples)==20 and [r['ordinal'] for r in samples]==list(range(20))
        ids=[r['sample_id'] for r in samples];scenes=[r['scene_id'] for r in samples]
        if reference_ids is None:reference_ids,reference_scenes=ids,scenes
        assert ids==reference_ids and scenes==reference_scenes
        assert len(set(ids))==20 and len(set(scenes))==10 and set(Counter(scenes).values())=={2}
        for row in samples:
            assert not row['unclassified_operators']
            parity=row['parity']
            assert parity['atol']==parity['rtol']==1e-5
            assert all(parity[k] for k in ('reference_repeat_allclose','observer_vs_first_allclose',
                       'observer_vs_repeat_allclose','input_tensors_unchanged','python_numpy_torch_cuda_rng_match_reference'))
            for comparison in parity['comparisons'].values():
                assert comparison['allclose']
                assert all(p['within_tolerance'] and not p['reference_nonfinite_values'] and not p['candidate_nonfinite_values']
                           for p in comparison['prediction_tensors'].values())
            path=root/row['ledger_file'];assert sha(path)==row['ledger_sha256']
            ledger=json.loads(path.read_text());assert ledger['totals']==row['totals'] and not ledger['unclassified_operators']
            assert total(ledger['operators'].values())==ledger['totals']
            assert total(ledger['components'].values())==ledger['totals']
            vector=row['totals'];dense=vector.get('dense_mac',0);analog=vector.get('analog_mac',0)
            binary_conv=vector.get('binary_input_convolution_mac',0)
            matrix_replaced=0
            for op,counts in ledger['operators'].items():
                if op.split('.')[1] in ('mm','bmm','matmul','einsum','addmm'):
                    matrix_replaced+=counts.get('dense_mac',0)-counts.get('analog_mac',0)
            assert dense==analog+binary_conv+matrix_replaced
            source_terms=source_reset_terms(ledger,result,script_inventory)
            all_targets.append(dict(case=case['case'],sample_id=row['sample_id'],scene_id=row['scene_id'],
                                    **vector,binary_matrix_mac_opportunities_replaced=matrix_replaced,**source_terms))
        vectors=[r for r in all_targets if r['case']==case['case']]
        keys=sorted({k for r in samples for k in r['totals']})
        means={key:math.fsum(r['totals'].get(key,0) for r in samples)/20 for key in keys}
        for key,value in means.items():close(value,result['per_map_counts'][key]['mean'])
        per_scene=[]
        for scene in sorted(set(scenes)):
            selected=[r for r in samples if r['scene_id']==scene]
            per_scene.append(dict(scene_id=scene,targets=2,mean={key:math.fsum(r['totals'].get(key,0) for r in selected)/2 for key in keys}))
        for key,value in means.items():close(value,math.fsum(r['mean'][key] for r in per_scene)/10)
        cases.append(dict(case=case['case'],job_id=record['job_id'],allocated_seconds=record['allocated_seconds'],
                          checkpoint_sha256=case['checkpoint_sha256'],selected_update=case['selected_update'],
                          targets=20,scenes=10,mean=means,per_scene=per_scene,
                          binary_matrix_mac_opportunities_replaced_mean=math.fsum(r['binary_matrix_mac_opportunities_replaced'] for r in vectors)/20,
                          source_derived_state_mean={key:math.fsum(r[key] for r in vectors)/20 for key in
                              ('source_derived_unobserved_reset_mul','source_derived_unobserved_reset_sub')},
                          maximum_prediction_difference=max(r['parity']['maximum_absolute_difference'] for r in samples),
                          pre_warmup_comparison=samples[0]['parity']['cold_vs_reference'],
                          pre_warmup_context=result.get('pre_warmup_context'),unclassified_operator_names=0,
                          diagnostic_seconds_not_latency=result['diagnostic_seconds_not_inference_latency']))
    output=dict(schema='stereo-study-arithmetic-full20-analysis-v1',status='PASS_SAVED_VECTOR_AND_COHORT_RECONCILIATION',
                cases=cases,source_snapshot=progress['source_snapshot'],source_identity_sha256=registry['source_identity_sha256'],
                source_hashes=source_hashes,analyzer_sha256=sha(Path(__file__)),
                common_sample_ids=reference_ids,common_scene_ids=sorted(set(reference_scenes)),
                predictions=200,aggregation='mean_two_targets_within_each_scene_then_equal_ten_scenes',
                allocated_seconds=sum(r['allocated_seconds'] for r in cases),
                scope='Pilot selected checkpoints, warmed complete API, full50 bins; legacy baseline time-as-channels/K32; logical counts only',
                limitations=['No physical GPU/neuromorphic joules or timing estimate','Composite vectors are proxy sizes; logical bytes are not DRAM traffic',
                             'Dense and hypothetical binary-event scenarios are alternatives','Saved ledger reconciliation does not independently replay operators or prove physical/foreign-kernel coverage',
                             'Pinned baseline reset bypasses observer; source-derived logical mul/sub retained separately, not physical executed instructions',
                             'RNG checks match post-forward reference states; zero RNG consumption is not inferred',
                             'First-call output limitation retained; no training/gradient parity claim'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(output,stream,indent=2);stream.write('\n')
    keys=sorted({k for row in all_targets for k in row if k not in ('case','sample_id','scene_id')})
    with args.output.with_suffix('.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['case','sample_id','scene_id',*keys],restval=0)
        writer.writeheader();writer.writerows(all_targets)
    print(json.dumps(dict(status=output['status'],cases=len(cases),predictions=200,allocated_seconds=output['allocated_seconds'])))


if __name__=='__main__':main()
