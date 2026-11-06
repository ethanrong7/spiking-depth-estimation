"""Assess predeclared selector gates and paired scene differences, never launch."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();evaluations=[json.loads(s) for s in (a.root/'evaluations.jsonl').read_text().splitlines()];final=evaluations[-1];m=final['metrics_by_mode'];sp=m['spatial'];summary=dict(update=final['update'],metrics_by_mode=m,gate=dict(candidate_beats_top1=sp['candidate_epe']<m['top1']['candidate_epe'],candidate_beats_pointwise=sp['candidate_epe']<m['pointwise']['candidate_epe'],final_beats_top1=sp['full_epe']<m['top1']['full_epe']),beats_native=sp['full_epe']<m['native']['full_epe'])
summary['positive_gate']=all(summary['gate'].values());deltas={}
for other in ('top1','pointwise','native'):
 scenes={}
 for scene in sorted({r['scene'] for r in final['per_frame']}):
  def mean(mode,key):
   v=[r[key] for r in final['per_frame'] if r['scene']==scene and r['mode']==mode and r.get(key) is not None];return sum(v)/len(v) if v else None
  scenes[scene]=dict(final_delta=mean('spatial','full_epe')-mean(other,'full_epe'))
  if other!='native':scenes[scene]['candidate_delta']=mean('spatial','candidate_epe')-mean(other,'candidate_epe')
 deltas[other]=dict(scenes=scenes,final_scene_wins=sum(r['final_delta']<0 for r in scenes.values()),scene_count=len(scenes))
summary['paired_deltas']=deltas;(a.root/'analysis.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='paired_deltas'},indent=2))
