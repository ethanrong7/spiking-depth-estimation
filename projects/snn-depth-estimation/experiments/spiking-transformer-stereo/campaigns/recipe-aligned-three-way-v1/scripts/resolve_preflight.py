"""Resolve complete native8 outcomes, then the single predeclared common fallback."""
import argparse
import json
from pathlib import Path
from recipe import ARMS
from run_recipe import write
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=('native8','fallback4'),required=True);a=p.parse_args()
batch=8 if a.phase=='native8' else 4
rows=[json.loads((a.root/f'{arm}-b{batch}.json').read_text()) for arm in ARMS]
if any(row['status'] not in ('PASS','OOM') for row in rows):raise RuntimeError('non-memory preflight failure; inspect before continuing')
if a.phase=='native8' and any(row['status']=='OOM' for row in rows):
 (a.root/'needs-physical4').write_text('Prospective fallback: any native8 OOM => all3 physical4/effective8.\n')
 write(a.root/'status.json',dict(status='FALLBACK_REQUIRED',physical_batch=4,effective_batch=8,native8=rows))
else:
 if not all(row['status']=='PASS' for row in rows):raise RuntimeError('physical4 fallback did not fit; stop')
 hashes={row['initialization']['common_state_sha256'] for row in rows}
 if len(hashes)!=1:raise RuntimeError('shared initialization differs across arms')
 write(a.root/'status.json',dict(status='PASS',physical_batch=batch,effective_batch=8,checks=rows,shared_initialization_sha256=next(iter(hashes))))
