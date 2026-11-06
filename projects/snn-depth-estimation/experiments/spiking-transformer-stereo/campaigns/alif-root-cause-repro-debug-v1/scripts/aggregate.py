import json

def aggregate(root, reg):
 workers=[]
 for seed in reg['paired_seeds']:
  path=root/('seed'+str(seed))/'result.json'
  if path.exists():workers.append(json.loads(path.read_text()))
 status='COMPLETED' if len(workers)==2 and all(w['status']=='COMPLETED' for w in workers) else 'RUNNING' if workers else 'PENDING'
 state=json.loads((root/'state.json').read_text())
 if any(any(word in j.get('accounting','') for word in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY']) for j in state['jobs']):status='FAILED'
 result=dict(status=status,phase='REPRODUCTION_DEBUG_ONLY',workers=workers,jobs=state['jobs'])
 temp=root/'result.tmp';temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(root/'result.json');return result
