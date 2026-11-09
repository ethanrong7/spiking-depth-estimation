"""Derive fixed-30k matched H200 CUDA and separately labeled historical CUDA comparisons."""
import argparse,csv,json,statistics,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
models=('original-non-i','hybrid-b');rows={};metadata={}
for model in models:
 audit=json.loads((a.root/f'endpoint-audit-seed1234-{model}-step30000.json').read_text());assert audit['status']=='passed'
 folder=a.root/'seed-1234'/model
 rows[model]=[json.loads(s) for s in (folder/'evaluations.jsonl').read_text().splitlines()]
 assert rows[model][-1]['update']==30000
 metadata[model]=json.loads((folder/'metadata.json').read_text())
original,hybrid=[rows[m][-1] for m in models]
assert metadata[models[0]]['config']['initial_native_state_sha256']==metadata[models[1]]['config']['initial_native_state_sha256']
assert metadata[models[0]]['source_registry']['source_identity_sha256']==metadata[models[1]]['source_registry']['source_identity_sha256']
model_keys={'model','arm','placement_seed','initial_state_sha256'}
assert {k:v for k,v in metadata[models[0]]['config'].items() if k not in model_keys}=={k:v for k,v in metadata[models[1]]['config'].items() if k not in model_keys}
root=a.root/'summaries';root.mkdir(exist_ok=True)
def write(name,values,fields):
 with (root/name).open('w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(values)
endpoints=[];curves=[]
for model in models:
 config=metadata[model]['config']
 for r in rows[model]:
  for metric in ('mean_scene_epe','bad3'):
   curves.append(dict(model=model,seed=1234,backend='cuda',update=r['update'],metric=metric,value=r[metric],unit='px' if metric=='mean_scene_epe' else 'percent'))
  if r['update']==30000:endpoints.append(dict(model=model,seed=1234,backend='cuda',update=30000,schedule_total=30000,physical_batch=4,accumulation=2,effective_batch=8,mean_scene_epe=r['mean_scene_epe'],bad3_percent=r['bad3'],source_identity=metadata[model]['source_registry']['source_identity_sha256'],execution_policy='native_CUDA_FP32_cudnn_TF32_on_matmul_TF32_off_cudnn_deterministic_on_strict_off'))
write('endpoints.csv',endpoints,list(endpoints[0]));write('curves.csv',curves,list(curves[0]))
assert original['scene_epe'].keys()==hybrid['scene_epe'].keys()
scene=[]
for name in sorted(original['scene_epe']):
 def bad3(r):
  frames=[f for f in r['per_frame'] if f['scene']==name];return sum(f['bad3']*f['valid_pixels'] for f in frames)/sum(f['valid_pixels'] for f in frames)
 scene.append(dict(update=30000,scene=name,original_epe=original['scene_epe'][name],hybrid_b_epe=hybrid['scene_epe'][name],original_minus_hybrid_epe=original['scene_epe'][name]-hybrid['scene_epe'][name],original_bad3_percent=bad3(original),hybrid_b_bad3_percent=bad3(hybrid)))
write('scene-deltas.csv',scene,list(scene[0]))
history={'original-non-i':dict(epe=2.691127,bad3=10.712),'hybrid-b':dict(epe=2.034117,bad3=9.626333)}
comparisons=[]
for model,r in zip(models,(original,hybrid)):
 h=history[model];comparisons.append(dict(model=model,seed=1234,update=30000,h200_epe=r['mean_scene_epe'],historical_cuda_epe=h['epe'],h200_minus_historical_epe=r['mean_scene_epe']-h['epe'],h200_bad3_percent=r['bad3'],historical_cuda_bad3_percent=h['bad3'],scope='Historical reference aggregates from Markdown; fresh H200 pair is a separate native CUDA replicate; no cross-backend kernel-parity claim'))
write('historical-cuda-comparison.csv',comparisons,list(comparisons[0]))
delta=original['mean_scene_epe']-hybrid['mean_scene_epe'];deltas=[s['original_minus_hybrid_epe'] for s in scene];assert math.isclose(statistics.mean(deltas),delta,rel_tol=1e-10,abs_tol=1e-10)
summary=dict(status='audited_30k_pair',seed=1234,endpoint=30000,original=original['mean_scene_epe'],hybrid_b=hybrid['mean_scene_epe'],original_minus_hybrid_epe=delta,hybrid_relative_epe_reduction_percent=100*delta/original['mean_scene_epe'],original_minus_hybrid_bad3_pp=original['bad3']-hybrid['bad3'],hybrid_scene_wins=sum(x>0 for x in deltas),scene_count=len(scene),median_paired_scene_difference=statistics.median(deltas),largest_scene_differences=sorted(scene,key=lambda r:abs(r['original_minus_hybrid_epe']),reverse=True)[:5],historical_comparisons=comparisons,limitations=['One training seed; scenes are not independent training replicates','Comparison tests the complete B addition, not a uniquely spiking mechanism','Reference CUDA values are historical aggregates; H200 values are fresh native CUDA runs','No Test/confirmation data accessed; no best-checkpoint selection'])
largest=summary['largest_scene_differences']
hardest=sorted(scene,key=lambda r:r['original_epe'],reverse=True)[:5]
absolute_gap_sum=sum(abs(x) for x in deltas)
share=sum(abs(r['original_minus_hybrid_epe']) for r in largest)/absolute_gap_sum if absolute_gap_sum else 0.
assert 0<=share<=1+1e-12
concentration=dict(top5_absolute_gap_share_percent=100*share,top5_absolute_gap_signed_mean_contribution=sum(r['original_minus_hybrid_epe'] for r in largest)/len(scene),hardest_original_5_scenes=hardest,hardest_original_5_signed_mean_contribution=sum(r['original_minus_hybrid_epe'] for r in hardest)/len(scene))
summary['gap_concentration']=concentration
gap_table='\n'.join(f"| {r['scene']} | {r['original_epe']:.6f} | {r['hybrid_b_epe']:.6f} | {r['original_minus_hybrid_epe']:.6f} |" for r in largest)
(root/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
text=f"""# Fixed-30k Original versus hybrid B on H200 CUDA

Both seed-1234 runs completed 30,000 updates with passed independent endpoint audits.

| Model | EPE, px | bad3, percent |
|---|---:|---:|
| Original non-I | {original['mean_scene_epe']:.6f} | {original['bad3']:.6f} |
| Hybrid B | {hybrid['mean_scene_epe']:.6f} | {hybrid['bad3']:.6f} |

Original-minus-hybrid EPE: {delta:.6f} px; relative reduction {summary['hybrid_relative_epe_reduction_percent']:.3f}%. Hybrid wins {summary['hybrid_scene_wins']}/30 scenes; median paired difference {summary['median_paired_scene_difference']:.6f} px. Original-minus-hybrid bad3: {summary['original_minus_hybrid_bad3_pp']:.6f} percentage points.

All scenes, including large errors, are retained in scene-deltas.csv. Full planned Dev curves are in curves.csv. The five largest absolute paired scene gaps account for {100*share:.2f}% of the total absolute gap and contribute {concentration['top5_absolute_gap_signed_mean_contribution']:.6f} px to the overall signed mean difference. The five hardest scenes ranked by Original endpoint EPE contribute {concentration['hardest_original_5_signed_mean_contribution']:.6f} px; their identities and all retained scenes are in comparison.json and scene-deltas.csv.\n\n| Scene (five largest absolute gaps) | Original EPE | Hybrid B EPE | Original-minus-B |\n|---|---:|---:|---:|\n{gap_table}\n\n One seed does not establish robust superiority; the comparison concerns the entire B addition and does not isolate spiking.

Historical CUDA values from spikestereonet-npu-baseline.md: Original EPE 2.691127 / bad3 10.712%; hybrid B EPE 2.034117 / bad3 9.626333%. They are separately labeled context in historical-cuda-comparison.csv. The reference CUDA values are historical aggregates; the two H200 endpoints are fresh native CUDA runs. No same-weight cross-backend prediction/gradient comparison was performed. All primary H200 CUDA comparisons use the fixed 30k endpoint, not a best interim checkpoint. No additional seeds or controls are launched by this analysis.
"""
(root/'interpretation.md').write_text(text);print(json.dumps(summary))
