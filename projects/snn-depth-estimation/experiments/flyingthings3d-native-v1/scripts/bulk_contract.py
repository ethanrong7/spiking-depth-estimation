"""Frozen cohort/split and atomic committed-sequence verification."""
import hashlib,json
from pathlib import Path


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(4194304),b''):h.update(chunk)
 return h.hexdigest()


def centers_for_split(split):
 if split=='train':return list(range(1,9))
 if split=='dev':return [3,6]
 raise ValueError('only frozen TRAIN-derived train/dev splits are authorized')


def verify_sequence(seq,derived_root,result,manifest_sha):
 directory=Path(derived_root)/seq['source_id'];manifest=json.loads((directory/'manifest.json').read_text());centers=centers_for_split(seq['split']);rows=manifest['rows']
 if manifest['source_index_sha256']!=manifest_sha or result['status']!='COMPLETED' or result['source_id']!=seq['source_id']:raise RuntimeError('committed sequence identity mismatch')
 if len(rows)!=len(centers) or len(result['windows'])!=2*len(centers) or result['completed_intervals']!=18 or not result['all450_samples_processed_per_view']:raise RuntimeError('incomplete committed sequence')
 verify_rows(seq,rows)
 keys=[(w['center'],w['view']) for w in result['windows']]
 if len(set(keys))!=len(keys) or set(keys)!={(c,v) for c in centers for v in ['l','r']}:raise RuntimeError('windowcenter/view coverage mismatch')
 for window in result['windows']:
  expected=f'spike_{window["view"]}/{seq["frames"][window["center"]]["source_frame"]}.dat'
  if window['relative_path']!=expected:raise RuntimeError('windowfilename/sourceframe mismatch')
  p=directory/window['relative_path']
  if p.stat().st_size!=window['bytes'] or window['bytes']!=3240000 or sha(p)!=window['sha256']:raise RuntimeError('committed DAT hash/length mismatch')
 return rows


def verify_rows(seq,rows):
 centers=centers_for_split(seq['split'])
 if len(rows)!=len(centers):raise RuntimeError('wrong selectedcenter count')
 for row,center in zip(rows,centers):
  frame=seq['frames'][center]
  expected=dict(source_id=seq['source_id'],split=seq['split'],center_index=center,source_frame=frame['source_frame'],source_time=center,spike_shape=[50,540,960],global_indices=list(range(50*center-24,50*center+26)),vertical_flip=False,half_frame_future_lookahead=True)
  if any(row.get(k)!=v for k,v in expected.items()) or any(row[k]!=frame[k] for k in ['rgb_l','rgb_r','disparity_l']):raise RuntimeError('center/source/view/GT mapping mismatch')
  for view in ['l','r']:
   if row['spike_'+view]!=str(Path(seq['source_id'])/f'spike_{view}/{frame["source_frame"]}.dat'):raise RuntimeError('wrong canonical derived relative path')
 return True
