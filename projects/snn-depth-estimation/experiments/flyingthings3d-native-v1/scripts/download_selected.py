"""Cluster-only resumable selected PNG/PFM transfer; no decode, training or GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import copy
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import struct
import threading
from range_archive import RangeArchive,load,tar_header


def validate_payload(path,payload):
 if path.endswith('.png'):
  if payload[:8]!=b'\x89PNG\r\n\x1a\n' or payload[12:16]!=b'IHDR' or struct.unpack('>II',payload[16:24])!=(960,540):raise ValueError('NativePNG header/geometry differs')
 elif path.endswith('.pfm'):
  lines=payload.split(b'\n',3)
  if len(lines)!=4 or lines[0].strip()!=b'Pf' or lines[1].split()!=[b'960',b'540'] or len(lines[3])!=960*540*4:raise ValueError('NativePFM header/geometry differs')
 else:raise ValueError('Unexpected source format')


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--expected-host',required=True);p.add_argument('--approved-transfer-bytes',type=int,required=True)
 p.add_argument('--workers',type=int,default=4)
 p.add_argument('--endpoint',choices=['https://hf-mirror.com'],default=None,
                help='Explicit transport-only endpoint; frozen provider/revision/manifest remain unchanged')
 a=p.parse_args()
 if platform.system()!='Linux' or socket.gethostname()!=a.expected_host:raise RuntimeError('Bulk source transfer must run on configured cluster host, never Mac')
 manifest=load(a.manifest);budget=manifest['budget']
 if a.approved_transfer_bytes!=budget['http_range_bytes']:raise ValueError('Explicit transfer budget must equal frozen manifest')
 if not 1<=a.workers<=4:raise ValueError('Use1..4 network workers')
 root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
 lockfile=(root/'.download.lock').open('a');fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
 receipt=root/'download-state.json';digest=hashlib.sha256(a.manifest.read_bytes()).hexdigest()
 state=load(receipt) if receipt.exists() else dict(status='PREPARED',manifest_sha256=digest,completed={},transferred_bytes=0)
 if state['manifest_sha256']!=digest:raise ValueError('Existing output bound to different manifest')
 state.setdefault('attempts',[]).append({k:state.get(k) for k in ['status','pid','command','updated_utc','transport_endpoint']})
 state.update(status='DOWNLOADING',pid=os.getpid(),hostname=socket.gethostname(),command=__import__('sys').argv,expected_files=4020,transport_endpoint=a.endpoint or 'https://huggingface.co')
 sources=copy.deepcopy(manifest['sources'])
 if a.endpoint:
  for config in sources.values():
   for part in config['parts']:
    if not part['url'].startswith('https://huggingface.co/'):raise ValueError('Unexpected original source provider')
    part['url']=a.endpoint+part['url'][len('https://huggingface.co'):]
 mutex=threading.Lock();local=threading.local();members={m['path']:m for m in manifest['members']}
 def save():
  state['updated_utc']=datetime.now(timezone.utc).isoformat();temporary=receipt.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(receipt)
 save()
 def destination(path):
  if Path(path).is_absolute() or '..' in Path(path).parts or not (path.startswith('frames_cleanpass/TRAIN/C/') or path.startswith('disparity/TRAIN/C/')):raise ValueError('Path outside selectedTRAIN/C source tree')
  out=(root/path).resolve()
  if root not in out.parents:raise ValueError('Output escapes source-data')
  return out
 def verified_existing(path):
  out=destination(path)
  with mutex:record=state['completed'].get(path)
  if record:
   if not out.is_file() or out.stat().st_size!=record['size'] or hashlib.sha256(out.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Recorded payload changed: '+path)
   return True
  if out.exists():raise ValueError('Unattributed existing payload; preserve/reconcile before retry: '+path)
  return False
 def store(path,payload):
  info=members[path]
  if len(payload)!=info['size']:raise ValueError('Source member byte size differs')
  validate_payload(path,payload);out=destination(path);out.parent.mkdir(parents=True,exist_ok=True)
  temporary=out.with_suffix(out.suffix+'.part');temporary.write_bytes(payload);temporary.replace(out)
  with mutex:
   state['completed'][path]=dict(size=len(payload),sha256=hashlib.sha256(payload).hexdigest(),role=info['role'],source_id=info['source_id']);save()
 def transfer(block):
  paths=block['paths'];existing={path:verified_existing(path) for path in paths}
  if all(existing.values()):return
  role=block['role']
  if not hasattr(local,'archives'):local.archives={k:RangeArchive(v) for k,v in sources.items()}
  payload=local.archives[role].read(block['start'],block['size'])
  with mutex:state['transferred_bytes']+=len(payload)
  for path in paths:
   if existing[path]:continue
   info=members[path]
   if role=='rgb':
    at=info['header_offset']-block['start'];header=tar_header(payload[at:at+512],info['header_offset'])
    if header is None or header['path']!=path or header['size']!=info['size']:raise ValueError('FrozenRGB header identity differs')
    start=info['data_offset']-block['start'];part=payload[start:start+info['size']]
   else:part=payload
   store(path,part)
  print(role,'files',len(paths),'completed',len(state['completed']),flush=True)
 try:
  # Keep at most workers ranges scheduled, so an unreachable endpoint cannot
  # leave thousands of queued retries running after the first failed range.
  blocks=iter(manifest['ranges'])
  with ThreadPoolExecutor(max_workers=a.workers) as pool:
   pending={pool.submit(transfer,block) for block in [next(blocks,None) for _ in range(a.workers)] if block is not None}
   while pending:
    done,pending=wait(pending,return_when=FIRST_COMPLETED)
    for future in done:future.result()
    for _ in done:
     block=next(blocks,None)
     if block is not None:pending.add(pool.submit(transfer,block))
  if len(state['completed'])!=4020:raise ValueError('Final source file count differs')
  state['status']='COMPLETE';save()
 except Exception as error:
  # Do not preserve signed redirect URLs from HTTP exception strings.
  state.update(status='FAILED',error_type=type(error).__name__);save();raise


if __name__=='__main__':main()
