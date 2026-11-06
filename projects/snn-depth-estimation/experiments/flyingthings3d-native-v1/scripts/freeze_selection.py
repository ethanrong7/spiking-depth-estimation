"""Freeze first150 complete TRAIN/C sequences and exact targeted transfer budget."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import fcntl
import os
import json
from pathlib import Path
import random
import threading
from range_archive import RangeArchive,load,tar_header


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['rgb-index','gt-index','rgb-source','gt-source','output']:
  p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--selection-seed',type=int,default=1234);a=p.parse_args()
 a.output.parent.mkdir(parents=True,exist_ok=True)
 lock=a.output.with_suffix('.freeze.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 rgb,gt=load(a.rgb_index),load(a.gt_index)
 candidates=rgb['complete_scenes'][:150]
 if len(candidates)!=150:raise RuntimeError('WAITING_RGB150_COMPLETE')
 missing=[s for s in candidates if s not in gt['gt_scene_headers']]
 if missing:raise RuntimeError('WAITING_MATCHING_GT_HEADERS: '+','.join(missing))
 if a.output.exists():
  old=load(a.output)
  if old['candidate_header_order']!=candidates:raise ValueError('Existing frozen selection differs')
  print('Selection already frozen; preserve existing manifest');return
 configs={'rgb':load(a.rgb_source),'gt':load(a.gt_source)}
 cache_path=a.output.with_suffix('.headers.json')
 cache=load(cache_path) if cache_path.exists() else {}
 local=threading.local()
 def scene_headers(source_id):
  if source_id in cache:return source_id,cache[source_id]
  if not hasattr(local,'archive'):local.archive=RangeArchive(configs['gt'])
  info=gt['gt_scene_headers'][source_id];entries={}
  for i in range(10):
   at=info['first_header']+i*info['stride'];payload=local.archive.read(at,512);entry=tar_header(payload,at)
   prefix='FlyingThings3D/disparity/'+source_id+'/left/'
   if entry is None or not entry['path'].startswith(prefix) or entry['size']!=2073618:raise ValueError('Selected GT header mismatch: '+source_id)
   entry['header_sha256']=hashlib.sha256(payload).hexdigest();entries[Path(entry['path']).stem]=entry
  if set(entries)!={f'{i:04d}' for i in range(6,16)}:raise ValueError('Incomplete native10frame GT sequence')
  return source_id,entries
 with ThreadPoolExecutor(max_workers=4) as pool:
  for source_id,entries in pool.map(scene_headers,candidates):
   cache[source_id]=entries
   temporary=cache_path.with_suffix('.tmp');temporary.write_text(json.dumps(cache,indent=2)+'\n');temporary.replace(cache_path)
   print('verified GT scene headers',len(cache),source_id,flush=True)
 gt_entries={source_id:cache[source_id] for source_id in candidates}
 train=set(random.Random(a.selection_seed).sample(candidates,120));sequences=[];members=[];ranges=[]
 for source_id in candidates:
  split='train' if source_id in train else 'dev';centers=set(range(1,9)) if split=='train' else {3,6};frames=[]
  for i,native in enumerate(range(6,16)):
   frame=f'{native:04d}';paths={}
   for side,field in [('left','rgb_l'),('right','rgb_r')]:
    entry=rgb['scenes'][source_id][side][frame];path=entry['path']
    if not path.startswith('frames_cleanpass/TRAIN/C/'):raise ValueError('RGB outside approved source cohort')
    members.append(dict(role='rgb',path=path,source_id=source_id,**{k:entry[k] for k in ['size','header_offset','data_offset']}));paths[field]=path
   paths['disparity_l']=None
   if i in centers:
    entry=gt_entries[source_id][frame];path=entry['path'].removeprefix('FlyingThings3D/') if hasattr(str,'removeprefix') else entry['path'][len('FlyingThings3D/'):]
    paths['disparity_l']=path
    members.append(dict(role='gt',path=path,source_id=source_id,source_member=entry['path'],**{k:entry[k] for k in ['size','header_offset','data_offset','header_sha256']}))
    ranges.append(dict(role='gt',start=entry['data_offset'],size=entry['size'],paths=[path]))
   frames.append(dict(index=i,source_frame=frame,**paths))
  for side in ('left','right'):
   entries=[rgb['scenes'][source_id][side][f'{n:04d}'] for n in range(6,16)]
   start=min(e['header_offset'] for e in entries);end=max(e['data_offset']+e['size'] for e in entries)
   ranges.append(dict(role='rgb',start=start,size=end-start,paths=[e['path'] for e in entries]))
  sequences.append(dict(source_id=source_id,split=split,reference_centers=sorted(centers),frames=frames))
 rgb_bytes=sum(m['size'] for m in members if m['role']=='rgb');gt_bytes=sum(m['size'] for m in members if m['role']=='gt')
 manifest=dict(version=1,dataset='FlyingThings3D native cleanpass TRAIN/C availability cohort',selection_seed=a.selection_seed,
  selection_rule='first150 complete stereo10frame TRAIN/C sequences in RGB TAR header encounter order; objective GT availability only; then Python Random(seed).sample120 train, remaining30 dev; no outcome/quality/motion filtering',candidate_header_order=candidates,replacements=[],
  sequences=sequences,members=members,ranges=ranges,sources=configs,
  source_configs_sha256={k:hashlib.sha256(path.read_bytes()).hexdigest() for k,path in [('rgb',a.rgb_source),('gt',a.gt_source)]},
  budget=dict(sequences=150,train_sequences=120,dev_sequences=30,rgb_files=3000,gt_files=1020,train_references=960,dev_references=60,rgb_payload_bytes=rgb_bytes,gt_payload_bytes=gt_bytes,disk_payload_bytes=rgb_bytes+gt_bytes,http_range_bytes=sum(r['size'] for r in ranges),http_ranges=len(ranges)),
  provenance='Pinned mirror revisions/providerLFSarchiveSHA256; all selected GT TAR headers individually verified, RGB TAR headers verified during inventory. Full archives not downloaded/rehashed. PayloadSHA256 recorded by downloader.',native_geometry=[960,540],processing='Download preservedPNG/PFM bytes only; EMA/spike conversion separate and cluster-only; no training/Test/Synthetic mutation.')
 assert len([m for m in members if m['role']=='gt'])==1020 and len(train)==120
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps(manifest['budget'],indent=2));print('manifest_sha256',hashlib.sha256(a.output.read_bytes()).hexdigest())


if __name__=='__main__':main()
