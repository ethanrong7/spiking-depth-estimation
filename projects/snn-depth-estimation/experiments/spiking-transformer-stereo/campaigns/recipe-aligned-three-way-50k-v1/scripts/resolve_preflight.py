"""Bind the three zero-update resumed Dev reproduction checks."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
names=['original-seed1234-50k-v1','original_i-seed1234-50k-v1','spiking_normalized-seed1234-50k-v1']
rows=[json.loads((a.root/name/'resume-check.json').read_text()) for name in names]
for row in rows:
 assert row['status']=='PASS' and row['scientific_optimizer_updates']==0 and row['optimizer_step_min_max']==[10000,10000]
 assert row['scheduler_local_epoch']==0 and row['parent_rng_restored'] and row['state_exactly_unchanged']
value=dict(status='PASS',scope='all3 actual-parent zero-update resume + strict parent10k Dev reproduction; no repeated batch8/memory search',checks=rows)
temp=a.root/'status.tmp';temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(a.root/'status.json')
