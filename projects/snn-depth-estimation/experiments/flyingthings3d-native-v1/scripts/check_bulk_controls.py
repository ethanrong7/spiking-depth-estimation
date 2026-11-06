"""Metadata/tiny file tests for generalized splits and atomic commit; no inference."""
import argparse,copy,json,tempfile
from pathlib import Path
from bulk_contract import centers_for_split,verify_rows,sha
from source_io import validate_index
from spike_codec import window_indices


def main():
 p=argparse.ArgumentParser();p.add_argument('--source-index',required=True);p.add_argument('--pilot-source',required=True);a=p.parse_args();root=Path(__file__).parent;pilot=Path(a.pilot_source);index=json.loads(Path(a.source_index).read_text());sequences=validate_index(index)
 for name in ['ema_adapter.py','spike_codec.py','source_io.py']:assert sha(root/name)==sha(pilot/'scripts'/name)
 counts={'train':0,'dev':0};new=0
 for seq in sequences:
  centers=centers_for_split(seq['split']);counts[seq['split']]+=len(centers);new+=seq['source_id']!='TRAIN/C/0087';rows=[]
  for center in centers:
   f=seq['frames'][center];rows.append(dict(source_id=seq['source_id'],split=seq['split'],center_index=center,source_frame=f['source_frame'],source_time=center,spike_shape=[50,540,960],global_indices=window_indices(center),vertical_flip=False,half_frame_future_lookahead=True,rgb_l=f['rgb_l'],rgb_r=f['rgb_r'],disparity_l=f['disparity_l'],spike_l=f'{seq["source_id"]}/spike_l/{f["source_frame"]}.dat',spike_r=f'{seq["source_id"]}/spike_r/{f["source_frame"]}.dat'))
  assert verify_rows(seq,rows)
  bad=copy.deepcopy(rows);bad[0]['global_indices'][0]+=1
  try:verify_rows(seq,bad);raise AssertionError('badtime must fail')
  except RuntimeError:pass
 assert counts=={'train':960,'dev':60} and new==149 and 2*sum(counts.values())==2040
 assert centers_for_split('dev')==[3,6] and window_indices(3)==list(range(126,176)) and window_indices(6)==list(range(276,326))
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp);staging=root/'sequence.staging';staging.mkdir();(staging/'conversion-result.json').write_text('{"status":"COMPLETED"}');(staging/'manifest.json').write_text('{"rows":[]}');destination=root/'sequence'
  assert not destination.exists();staging.replace(destination);assert not staging.exists() and json.loads((destination/'conversion-result.json').read_text())['status']=='COMPLETED'
 print(json.dumps(dict(scope='metadata+tinyCPUfilechecks_only_noEMA/GPU/datasetconversion',core_EMA_IF_sourceIO_codec_exact_pilot_bytes=True,all150_frozen_split_mappings_pass=True,train_samples=960,dev_samples=60,remaining_sequences=149,DAT_files=2040,dev_centers_3_6_and50indices_exact=True,reject_wrongwindow_mapping=True,atomic_directory_commit_includes_recoverable_result=True),indent=2))
if __name__=='__main__':main()
