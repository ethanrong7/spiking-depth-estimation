"""Operational saved-log-only S9 curves; no remote polling, model or dataset access.
Visual contract: observed fixed-grid Dev metrics and raw/400-update cycle L1;
Vblue circles/Fvermilion squares; pending arms empty; final5000 primary.
"""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path

CAMP=Path(__file__).resolve().parents[1]
REPO=CAMP.parents[5]
DEFAULT=REPO/'storage/experiment-runs/snn-depth-estimation/spiking-transformer-stereo/spike-depth-direction-v1/stage9'
COLORS={'V':'#0072B2','F':'#D55E00'}
ACTIONS=[f'p{p}-r{r}' for p in (24,32,49) for r in (4,8,16)]
GRID=(0,2500,5000)
def finite(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def jsonfile(path,default):return json.loads(path.read_text()) if path.exists() else default

def collect(root):
    status=jsonfile(root/'training-watch-status.json',{})
    data={'status':status,'arms':{},'input_sha256':{}}
    for arm in ('V','F'):
        directory=root/f'{arm}-seed1234-5k-v1';history=jsonfile(directory/'history.json',[])
        train=[];tail_ignored=False;p=directory/'train.jsonl'
        if p.exists():
            lines=p.read_text().splitlines()
            for i,line in enumerate(lines):
                if not line.strip():continue
                try:row=json.loads(line)
                except json.JSONDecodeError:
                    if i==len(lines)-1:tail_ignored=True;continue
                    raise
                if finite(row.get('loss')) and isinstance(row.get('update'),int):train.append(row)
        assert len({r['update'] for r in train})==len(train),'duplicate update'
        assert all(1<=r['update']<=5000 for r in train)
        history=[{'update':entry['update'],'actions':{key:{metric:values[metric] for metric in ('mean_scene_epe','mean_scene_bad3_percent','worst5_scene_epe')} for key,values in entry['actions'].items()}} for entry in history]
        history=sorted(history,key=lambda r:r['update'])
        assert len({r['update'] for r in history})==len(history),'duplicate evaluation'
        assert all(r['update'] in GRID for r in history),'unexpected evaluation grid'
        by={r['update']:r for r in train};cycles=[];partial=None
        latest=max(by,default=0)
        for start in range(1,latest+1,400):
            end=min(start+399,latest);rows=[by[u] for u in range(start,end+1) if u in by]
            if len(rows)!=end-start+1:continue
            point={'x':end,'y':sum(v['loss'] for v in rows)/len(rows),'first':start,'last':end,'count':len(rows)}
            if len(rows)==400:cycles.append(point)
            else:partial=point
        saved=status.get('arms',{}).get(arm,{})
        state=saved.get('state','NOT_SUBMITTED' if not history and not train else 'SAVED_LOGS_PRESENT')
        if arm=='F' and not saved:state=status.get('F_pending',state)
        data['arms'][arm]={'state':state,'job_id':saved.get('job_id'),'latest_update':latest,'latest_samples':by[latest].get('samples_seen') if latest else None,'history':history,'train':[{'x':r['update'],'y':r['loss']} for r in train],'cycles':cycles,'partial':partial,'incomplete_jsonl_tail_ignored':tail_ignored}
        for p in (directory/'train.jsonl',directory/'history.json'):
            if p.exists():data['input_sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    p=root/'training-watch-status.json'
    if p.exists():data['input_sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return data

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--runs-root',type=Path,default=DEFAULT);ap.add_argument('--output',type=Path,default=CAMP/'results/stage9-curves/index.html');a=ap.parse_args();data=collect(a.runs_root)
    data['renderer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload=json.dumps(data,allow_nan=False,separators=(',',':')).replace('<','\\u003c')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>S9 observed training curves</title>
<style>body{font:15px system-ui,sans-serif;color:#222;background:#f5f7fa;margin:0}main{max-width:1180px;margin:auto;padding:26px}h1{font-size:26px;margin:0 0 8px}h2{font-size:19px;margin:0 0 10px}.note{color:#52606d;line-height:1.5}section{background:white;border:1px solid #dce2e8;border-radius:9px;padding:18px;margin:18px 0}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}svg{width:100%;height:auto;display:block}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;text-align:left;border-bottom:1px solid #e5e9ed}select{padding:6px;margin:3px 12px 3px 0}small{font-size:12px}pre{overflow:auto;font-size:11px}button{padding:5px 10px}@media(max-width:850px){.grid{grid-template-columns:1fr}main{padding:14px}}.v{color:#0072B2}.f{color:#D55E00}</style>
<main><h1>S9 · FlyingThings3D native observation / refinement pilot</h1><div class="note">Fresh seed 1234 · 800 model-fit / 160 calibration / 60 Dev frames. Native 540×960, all finite positive source disparity. <strong>Fixed 5000-update endpoint is primary.</strong> Curves do not implement early stopping or a policy decision.</div>
<section><h2>Saved operational status</h2><div id="status"></div><p class="note" id="stamp"></p><div class="note">One existing watcher publishes mirrored logs. This page performs no polling. Unrelated account jobs are outside these curves.</div></section>
<section><h2>Training disparity loss</h2><div id="train"></div><p class="note">Faint lines: observed per-update pooled-valid L1 at refinements4/8/16. Solid markers: means over <strong>complete 400-update cycles</strong> (800 fitting examples at effective batch2); hollow marker and dashed segment: latest partial cycle, labelled with its exact bounds. No EMA or interpolated observations. <strong>V mixes 25/33/50-bin supports; F uses 50.</strong> Their aggregate training losses are not a matched-input fit comparison. AdamW peak2e−4; OneCycle 5100, pct_start .05, linear; one optimizer step per two micro-samples.</p><div id="cycle"></div></section>
<section><h2>Observed Dev validation</h2><label>V support <select id="prefix"><option value="49">50bins (p49)</option><option value="24">25bins (p24)</option><option value="32">33bins (p32)</option></select></label><label>Refinements <select id="refine"><option>16</option><option>8</option><option>4</option></select></label><label><input id="initial" type="checkbox" checked> Include observed initialization</label><div class="note" id="comparison"></div><div class="grid"><div id="epe"></div><div id="bad3"></div><div id="tail"></div></div><div id="observations"></div><p class="note">Only saved points at 0/2500/5000. Initialization0 is a measured untrained prediction, not a forecast. No line extends to an unobserved checkpoint. F supplies full50-bin support only. Default p49/r16 compares matched full-support V/F; availability price remains a final 5000 question. Equal-frame → equal-scene EPE and bad3; worst-five = mean of five largest scene EPEs. Pixels and percent retain native units.</p></section>
<section><h2>Evidence and scope</h2><div class="note">These are neural training/validation curves, not final adaptive-policy results. Observation lookahead bins and measured GPU costs are distinct axes; neither is physical robot latency. No old Synthetic data or historical curves are pooled here.</div><details><summary>Read-only provenance / renderer contract</summary><pre id="provenance"></pre></details></section></main>
<script id="records" type="application/json">PAYLOAD</script><script>
'use strict';const D=JSON.parse(document.getElementById('records').textContent);const C={V:'#0072B2',F:'#D55E00'};const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function marker(arm,x,y,color,hollow=false){return arm==='V'?`<circle cx="${x}" cy="${y}" r="4" fill="${hollow?'white':color}" stroke="${color}" stroke-width="2"/>`:`<rect x="${x-4}" y="${y-4}" width="8" height="8" fill="${hollow?'white':color}" stroke="${color}" stroke-width="2"/>`;}
function chart(title,unit,series,width=390,height=290){const left=57,right=15,top=34,bottom=46;let values=series.flatMap(s=>s.points.map(p=>p.y));const max=values.length?Math.max(...values)*1.08:1;const ymax=unit==='percent'?100:(max>0?max:1);const xx=x=>left+x/5000*(width-left-right),yy=y=>height-bottom-y/ymax*(height-top-bottom);let out=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(title)}"><text x="${left}" y="19" font-size="14" fill="#222">${esc(title)} (${esc(unit)})</text>`;
for(let i=0;i<=4;i++){let value=ymax*i/4,y=yy(value);out+=`<line x1="${left}" y1="${y}" x2="${width-right}" y2="${y}" stroke="#e5e9ed"/><text x="${left-8}" y="${y+4}" text-anchor="end" font-size="11" fill="#556">${value<10?value.toFixed(2):value.toFixed(1)}</text>`;}
for(const x of[0,1000,2500,4000,5000])out+=`<text x="${xx(x)}" y="${height-24}" text-anchor="middle" font-size="11" fill="#556">${x}</text>`;out+=`<text x="${(width+left)/2}" y="${height-5}" text-anchor="middle" font-size="12">Optimizer updates</text>`;
for(const s of series){if(!s.points.length)continue;const xy=s.points.map(p=>`${xx(p.x)},${yy(p.y)}`).join(' ');if(s.points.length>1)out+=`<polyline points="${xy}" fill="none" stroke="${C[s.arm]}" stroke-width="${s.raw?1:2}" opacity="${s.raw?.18:1}" ${s.partial?'stroke-dasharray="5 4"':''}/>`;if(!s.raw)for(const p of (s.partial?s.points.slice(-1):s.points))out+=marker(s.arm,xx(p.x),yy(p.y),C[s.arm],s.partial);}
if(!values.length)out+=`<text x="${width/2}" y="${height/2}" text-anchor="middle" fill="#667">No observed points</text>`;return out+'</svg>';}
const labels={V:'V · variable support',F:'F · full support'};let status='<table><tr><th>Arm</th><th>Saved state</th><th>Job</th><th>Latest observed update</th><th>Presentations</th></tr>';for(const arm of['V','F']){let r=D.arms[arm];status+=`<tr><td class="${arm.toLowerCase()}">${labels[arm]} ${arm==='V'?'●':'■'}</td><td>${esc(r.state)}</td><td>${r.job_id??'—'}</td><td>${r.latest_update||'No training records'}</td><td>${r.latest_samples??'—'}</td></tr>`;}document.getElementById('status').innerHTML=status+'</table>';document.getElementById('stamp').textContent='Saved watcher snapshot: '+(D.status.utc??'not available')+' · budget 5000 updates / 10000 presentations per arm';
let trainseries=[],cycle='<table><tr><th>Arm</th><th>Complete 400-update cycles</th><th>Latest complete mean</th><th>Partial cycle (separate)</th></tr>';for(const arm of['V','F']){const r=D.arms[arm];trainseries.push({arm,raw:true,points:r.train},{arm,points:r.cycles});if(r.partial){const last=r.cycles.at(-1);trainseries.push({arm,partial:true,points:[...(last?[last]:[]),r.partial]});}const last=r.cycles.at(-1);cycle+=`<tr><td>${arm}</td><td>${r.cycles.length}</td><td>${last?`${last.first}–${last.last}: ${last.y.toFixed(4)}`:'—'}</td><td>${r.partial?`${r.partial.first}–${r.partial.last} (${r.partial.count}/400): ${r.partial.y.toFixed(4)}`:'—'}</td></tr>`;}document.getElementById('train').innerHTML=chart('Observed training L1','pixels',trainseries,1100,330);document.getElementById('cycle').innerHTML=cycle+'</table>';
function update(){const p=document.getElementById('prefix').value,r=document.getElementById('refine').value,key=`p${p}-r${r}`,initial=document.getElementById('initial').checked;const metrics=[['epe','mean_scene_epe','Dev mean-scene EPE','pixels'],['bad3','mean_scene_bad3_percent','Dev bad3','percent'],['tail','worst5_scene_epe','Dev worst-five scenes','pixels']];let rows=[];for(const arm of['V','F'])for(const e of D.arms[arm].history){if(!initial&&e.update===0)continue;const m=e.actions?.[key];if(m)rows.push({arm,update:e.update,...m});}for(const[id,field,title,unit]of metrics){const series=['V','F'].map(arm=>({arm,points:rows.filter(e=>e.arm===arm&&Number.isFinite(e[field])).map(e=>({x:e.update,y:e[field]}))}));document.getElementById(id).innerHTML=chart(title,unit,series);}document.getElementById('comparison').textContent=p==='49'?`Matched full 50-bin support at ${r} refinements. V blue circles; F vermilion squares.`:`V${Number(p)+1}-bin support at${r} refinements. F has no registered metric at this support; no F curve drawn.`;let table='<table><tr><th>Arm</th><th>Observed update</th><th>EPE (px)</th><th>Bad3 (%)</th><th>Worst5 (px)</th></tr>';for(const x of rows)table+=`<tr><td>${x.arm}</td><td>${x.update}${x.update===5000?' · primary':x.update===0?' · initialization':''}</td><td>${x.mean_scene_epe.toFixed(4)}</td><td>${x.mean_scene_bad3_percent.toFixed(3)}</td><td>${x.worst5_scene_epe.toFixed(4)}</td></tr>`;document.getElementById('observations').innerHTML=table+'</table>';}
for(const id of['prefix','refine','initial'])document.getElementById(id).addEventListener('change',update);update();document.getElementById('provenance').textContent=JSON.stringify({renderer_sha256:D.renderer_sha256,input_sha256:D.input_sha256,tail_ignored:Object.fromEntries(Object.entries(D.arms).map(([k,v])=>[k,v.incomplete_jsonl_tail_ignored])),visual_contract:'Observed points only; rawloss faint, complete400-cyclemeans solid, partialhollow/dashed; final5000primary; no performance conclusions.'},null,2);
</script></html>'''.replace('PAYLOAD',payload)
    a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.html.part');tmp.write_text(page);tmp.replace(a.output)
    print(json.dumps({'output':str(a.output),'arms':{k:{'update':v['latest_update'],'evaluations':len(v['history']),'full_cycles':len(v['cycles']),'partial_count':v['partial']['count'] if v['partial'] else 0} for k,v in data['arms'].items()}},indent=2))
if __name__=='__main__':main()
