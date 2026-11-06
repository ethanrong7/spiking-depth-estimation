"""Render all real diagnostic values only after the complete log/checkpoint audits."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

p=argparse.ArgumentParser()
p.add_argument('--raw',type=Path,required=True)
p.add_argument('--results',type=Path,required=True)
a=p.parse_args()
s=json.loads((a.results/'summary.json').read_text())
audit=json.loads((a.raw/'checkpoint-audit.json').read_text())
assert s['complete'] and len(s['cases'])==15 and audit['status']=='PASS' and len(audit['cases'])==15
lookup={(r['model'],r['batch'],r['augmentation'],r['schedule_total']):r for r in s['cases']}
labels={'original':'Original ALIF','original_I':'ALIF + I','spiking_normalized':'Normalized SNN transformer'}
conditions=[(2,'none',10100),(8,'none',10100),(2,'hv',10100),(8,'hv',10100),(8,'hv',300100)]
lines=['# Historical recipe-gap diagnostic','',
 'This bounded screen completed15 fresh cases, each480 optimizer updates:7200 updates and40320 optimizer-training sample presentations in total. It tests early recipe sensitivity with the complete unchanged historical models. It cannot identify the cause of final10k convergence differences on its own. No previous campaign, model source, dataset, or monitor was changed.','',
 '## Observed Dev error after480 updates','',
 'Mean-scene EPE in pixels; lower is better. All cases use seed1234. Batch8 means physical4×2 with one pooled-valid gradient/clip/AdamW/scheduler update. Batch2 is physical2.','',
 '| Model | B2, no flips,10k schedule | B8, no flips,10k schedule | B2,HV,10k schedule | B8,HV,10k schedule | B8,HV,300k schedule |',
 '| --- | ---: | ---: | ---: | ---: | ---: |']
for model,label in labels.items():
    lines.append('| '+label+' | '+' | '.join(f'{lookup[(model,*c)]["dev480"]:.6f}' for c in conditions)+' |')
lines+=['','The10k and300k schedule labels mean OneCycle total_steps10100 and300100, respectively. Every case stops480, before either schedule finishes. In particular,480 is still in the300k warmup. Batch8 receives3840 presentations versus960 for batch2 at this fixed-update endpoint.','','## Controlled recipe contrasts','',
 'Positive values below increase error. These are one-seed early endpoint differences, not population estimates.','',
 '| Model | B8−B2 without flips | HV−none atB2 | HV−none atB8 | Batch×HV interaction | Long−short atB8/HV |',
 '| --- | ---: | ---: | ---: | ---: | ---: |']
for r in s['contrasts']:
    lines.append('| '+labels[r['model']]+' | '+' | '.join(f'{r[k]:+.6f}' for k in ('batch_effect_without_flips','flips_effect_batch2','flips_effect_batch8','interaction','long_schedule_effect'))+' |')
lines+=['','The interaction is [HV cost atB8]−[HV cost atB2]. A negative value means a larger batch reduces the augmentation penalty in this trajectory. The fixed-update batch contrast includes the resulting difference in sample presentations; that is part of the intervention, not an equal-exposure comparison.','','## Model advantage depends on the recipe and learning phase','',
 'Method minus Original EPE at480; negative favors the method.','',
 '| Model | B2/none | B8/none | B2/HV | B8/HV | B8/HV/long |',
 '| --- | ---: | ---: | ---: | ---: | ---: |']
for model in ('original_I','spiking_normalized'):
    lines.append('| '+labels[model]+' | '+' | '.join(f'{lookup[(model,*c)]["dev480"]-lookup[("original",*c)]["dev480"]:+.6f}' for c in conditions)+' |')
lines+=['','The early screen does not reproduce the later loss of advantage. Under the revisedB8/HV/short recipe, I and transformer lead Original by1.328542px and2.273178px at480. UnderB2/no-flips/short, I trails by2.801304px and transformer leads by only0.326437px. The joint revised recipe therefore favors the proposed models more at this early point, opposite to the smaller advantages at the historical5k/10k endpoints. This argues against a simple immediate trainability failure as the explanation.','',
 'Batch8 without augmentation improves I by3.559861px and transformer by1.587260px while worsening Original by0.774978px at480. Thus a larger effective batch alone is not supported as an early cause of the proposed models deteriorating. AddingH/V atB8 increases errors for all three, by1.248828/1.453822/1.664325px for Original/I/transformer. These penalties are model-dependent but do not erase the proposed models’ early lead.','',
 'The long-schedule contrast raises480-update error by0.026038px for Original,0.806646px for I and1.592805px for transformer. It narrows their early advantages. This isolates the warmup-phase effect inB8/HV; it does not predict their final300k performance or establish the cause of late high-LR fluctuations.','',
 'At equal960 presentations, compare batch2/update480 with batch8/update120. Optimizer and LR clocks then differ. These views answer different questions; neither should be silently substituted for the other.','',
 '| Model | B2/no flips at480 | B8/no flips at120 | B2/HV at480 | B8/HV at120 |',
 '| --- | ---: | ---: | ---: | ---: |']
for model,label in labels.items():
    values=[lookup[(model,2,'none',10100)]['dev480'],lookup[(model,8,'none',10100)]['dev120'],lookup[(model,2,'hv',10100)]['dev480'],lookup[(model,8,'hv',10100)]['dev120']]
    lines.append('| '+label+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
lines+=['','## Local gradient diagnostics','',
 'The probe uses the first eight fixed Train presentations at each final state, with16 refinements and no optimizer updates. Cosines compare whole-model gradients on the identical examples. Negative H cosine denotes a locally competing direction, not proof that augmentation is invalid or that it caused the final10k gap.','',
 '| Model underB8/HV/short | Identity EPE | H EPE | V EPE | H gradient cosine | V gradient cosine | H correct-sign fraction |',
 '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for model,label in labels.items():
    case=lookup[(model,8,'hv',10100)]['case']
    q=json.loads((a.raw/'cases'/case/'perturbations.json').read_text())
    values=[q['identity']['epe'],q['H']['epe'],q['V']['epe'],q['H']['cosine_to_identity'],q['V']['cosine_to_identity'],q['H']['correct_sign_fraction']]
    lines.append('| '+label+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
lines+=['','H reflects both views without swapping and changes signed horizontal flow from u to−u. This is geometrically consistent, but adds the opposite correspondence sign to training; unaugmented Dev retains the original convention. V preserves the horizontal sign. The factorial tests jointH/V augmentation, so it cannot isolate H from V as a training cause. The orientation probes are supporting local diagnostics only.','',
 'Every training update had finite loss and a finite nonzero global gradient. All final model/AdamW tensors are finite, and every populated AdamW counter and scheduler counter equals480. Frequent clipping is recorded in the numeric evidence and is not itself evidence of unstable training.','',
 '## Relation to the historical advantage','',
 'See the [matched historical comparison](historical-comparison.md) and [actual historical trajectories](historical-trajectories.csv). At paired seed1234, the transformer10k advantage shrank from21.15% to5.29%; comparing its historical three-seed37.3% mean directly with a new one-seed result exaggerates the apparent change. The I comparison must use a matched5k endpoint:22.80% to3.37%, rather than mixing historical5k and revised10k.','',
 'The reduced gap has two arithmetic components: Original improves under the revised joint recipe, while I/transformer develop some larger scene-tail errors. For I at5k, baseline catch-up accounts for75.6% of the gap reduction; for transformer at10k it accounts for45.9%. This is an arithmetic decomposition, not causal separation of augmentation and batch. Both models still improve many scenes in the revised study.','',
 'The completed revised10k experiment did not change LR horizon: both it and v0 use10100 steps. The latest fresh planned300k experiment is a separate comparison. At update10000 its used LR is1.95288458e-4, versus2.00099202e-6 in the completed10k recipe (about97.6× higher); its intermediate error is not an endpoint after comparable cooling. The stopped continuation with an LR restart is another separate experiment.','',
 'The loss of advantage occurred after useful early learning, rather than at initialization. Under the revised recipe, I leads Original by38.46% at1k,22.22% at2.5k,3.37% at5k and7.63% at10k. The transformer leads by20.09%,30.60%,−1.71% and5.29% at those checkpoints. In particular, transformer Dev EPE changes only3.277306→3.254275 from2.5k to5k, while Original improves4.722503→3.199417. I improves3.673166→3.091679, but Original catches up faster.','',
 'This is not a stopped optimizer: in complete960-update windows ending2400/4920/7440/9960, transformer training loss decreases29.4469→22.3295→17.5005→14.1155 and I decreases30.6773→22.7113→17.4872→13.6351. The training-versus-Dev divergence is evidence of a later optimization/generalization tradeoff, without isolating its specific cause. See [advantage over time](advantage-over-time.csv).','',
 'These results support recipe-sensitive learning rather than a demonstrated implementation failure. The short screen cannot establish which factor explains the eventual5k/10k ranking, nor whether the current architectures are intrinsically inferior. A final conclusion must retain the completed historical endpoints, early controlled contrasts, schedule-phase differences, and mixed tail/bad-pixel behavior together. No additional experiment is launched by this report.','',
 '## Verification and artifacts','',
 '- All15 initial-state hashes match their corresponding frozen recipe-aligned10k starts and match across recipe cells for each model. Shared code, manifest, sample order and statelessH/V decisions were checked.','- Exactly480 contiguous updates per case, exact presentation counters, and analytic LR used/next values passed the independent saved-log audit. All three scheduled Dev observations contain60 frames/30 scenes.','- A separate remote CPU-only inspection verified15 saved checkpoints, finite tensors, exact configuration/source/order lineage and optimizer/scheduler counters. No model was constructed and no inference or extra Slurm job was used for this audit.','- The actual execution used one GPU sequentially. The original two-GPU job2032 was cancelled before starting because other users occupied two GPUs; no unrelated job was changed. The resource-only amendment and sole replacement2033 are recorded under operations/ and raw submission receipts.','- Diagnostic gradients add480 Train probe presentations with zero optimizer steps;45 Dev evaluations and15 two-frame warmups are separate from the40320 optimizer-training presentations. No Test/confirmation data were loaded.','',
 'The B2/no-flip controls reproduce the exact historical first loss for all three models and the exact first gradient norm for Original/I. Transformer first norm differs by0.0001220703125 on a1513.807373 norm (about8.06e-8 relative). The complete trajectories are not claimed to be bitwise historical replays; all recipe cells use the same current runner and exact matching initial tensors.','',
 'Slurm2033 completed0:0 in4694 allocated seconds (78min14s,1.3039 GPU-hours). The pending predecessor2032 consumed0 allocated seconds. The experiment is terminal; no follow-up training, retry, extension or scheduled task is launched.','',
 '[Case table](cases.csv) · [Factor contrasts](contrasts.csv) · [Audited summary](summary.json) · [Checkpoint audit](checkpoint-audit.json) · [Prospective design](../README.md) · [Resource amendment](../operations/README.md)','']
body='\n'.join(lines)
body=re.sub(r'(?<=[a-z])(?=\d)', ' ', body)
body=re.sub(r'(?<=\d)(?=px\b)', ' ', body)
body=re.sub(r',(?=\S)', ', ', body)
for before,after in [('atB','at B'),('underB','under B'),('UnderB','Under B'),('revisedB','revised B'),('AddingH/V','Adding H/V'),('Batch8','Batch 8'),('to−u','to −u')]:
    body=body.replace(before,after)
(a.results/'final-report.md').write_text(body)
(a.results/'checkpoint-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print('Rendered final-report.md')
