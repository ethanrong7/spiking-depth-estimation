"""Independent read-only actual DAT/generator audit; CPU, no EMA/model inference."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import socket
import sys
import time
import numpy as np
from PIL import Image,ImageDraw


def gray(rgb):
    x=np.asarray(rgb,dtype=np.float32)/np.float32(255)
    return np.float32(.299)*x[...,0]+np.float32(.587)*x[...,1]+np.float32(.114)*x[...,2]


def manual_decode(payload):
    if len(payload)!=3240000:raise ValueError('wrong native50-bin byte length')
    raw=np.frombuffer(payload,dtype=np.uint8);out=np.empty((raw.size,8),dtype=np.uint8)
    for bit in range(8):out[:,bit]=(raw>>bit)&1
    return out.reshape(50,540,960)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['source-root','derived-root','source-index','prep-root','upstream-reader','output-root']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--expected-host',required=True);a=p.parse_args()
    if platform.system()!='Linux' or socket.gethostname()!=a.expected_host:raise RuntimeError('actual data audit cluster CPU only')
    out=a.output_root;out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    manifest=json.loads((a.derived_root/'manifest.json').read_text());index=json.loads(a.source_index.read_text());source_index={s['source_id']:s for s in index['sequences']}
    manifest_sha=hashlib.sha256((a.derived_root/'manifest.json').read_bytes()).hexdigest();assert manifest_sha=='eb52ec22890423efc2d5455904c3155d6aaa8c8a96199eb8fcd371305d62534d'
    selected=[]
    for split in ['train','dev']:
        sid=manifest[split][0]['source_id'];rows=[r for r in manifest[split] if r['source_id']==sid];selected.extend([rows[0],rows[-1]])
    (out/'selection.json').write_text(json.dumps({'rule':'first Train + first Dev sequence in frozen manifest, first and last selected target center; fixed before DAT decode','samples':[r['sample_id'] for r in selected],'dark_bright_rule':'bottom/top target-native-RGB luminance decile, computed before spike decode; diagnostic regions only, no data filtering'},indent=2)+'\n')
    count=0
    for split in ['train','dev']:
        for row in manifest[split]:
            center=row['center_index'];source=source_index[row['source_id']]['frames'][center]
            assert row['global_indices']==list(range(50*center-24,50*center+26)) and row['global_indices'][24]==50*center
            assert row['spike_shape']==[50,540,960] and row['vertical_flip'] is False
            assert row['source_frame']==source['source_frame'] and row['source_time']==center
            for key in ['rgb_l','rgb_r','disparity_l']:assert row[key]==source[key]
            for view in ['l','r']:assert row['spike_'+view]==str(Path(row['source_id'])/('spike_'+view)/(row['source_frame']+'.dat'))
            count+=1
    assert count==1020
    spec=importlib.util.spec_from_file_location('pinned_upstream_reader',a.upstream_reader);upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    sequences=[];windows=[];panels=[];upstream_frames=0
    for sid in dict.fromkeys(r['source_id'] for r in selected):
        trace_root=a.prep_root/'pilot-v1/results' if sid=='TRAIN/C/0087' else a.prep_root/'bulk-v1/results/sequences'/sid
        traces=json.loads((trace_root/'voltage-traces.json').read_text());intervals=json.loads((trace_root/'interval-timing.json').read_text());conversion=json.loads((trace_root/'result.json').read_text());assert len(traces)==900 and len(intervals)==18
        assert all(r['interpolated_images']==49 and r['exact_native_endpoint_images']==1 for r in intervals)
        side={v:{r['global_index']:r for r in traces if r['view']==v} for v in ['l','r']};endpoint_checks=[];state_checks=[]
        for view in ['l','r']:
            assert list(side[view])==list(range(1,451));previous=0.;inferred=[]
            for n,t in side[view].items():
                assert t['source_time']==n/50 and 0<=t['voltage_min']<=t['voltage_max']<5.00001 and 0<=t['spike_fraction']<=1
                inferred.append(t['voltage_mean']-previous+5*t['spike_fraction']);previous=t['voltage_mean']
                if n%50==0:
                    f=source_index[sid]['frames'][n//50];rgb=np.array(Image.open(a.source_root/f['rgb_'+view]).convert('RGB'));assert rgb.shape==(540,960,3)
                    expected=float(gray(rgb).mean(dtype=np.float64));bgr=float(gray(rgb[...,::-1]).mean(dtype=np.float64));other='r' if view=='l' else 'l';other_rgb=np.array(Image.open(a.source_root/f['rgb_'+other]).convert('RGB'));opposite=float(gray(other_rgb).mean(dtype=np.float64))
                    endpoint_checks.append(dict(view=view,global_index=n,source_frame=f['source_frame'],inferred_intensity_mean=inferred[-1],RGB_gray_mean=expected,BGR_swapped_gray_mean=bgr,opposite_view_gray_mean=opposite,absolute_RGB_error=abs(inferred[-1]-expected),absolute_BGR_swapped_error=abs(inferred[-1]-bgr),absolute_opposite_view_error=abs(inferred[-1]-opposite)))
            assert side[view][1]['spike_fraction']==0 and side[view][1]['voltage_max']<=1.00001
            state_checks.append(dict(view=view,samples=450,inferred_intensity_min=min(inferred),inferred_intensity_max=max(inferred),out_of_01_transitions=sum(x< -2e-6 or x>1+2e-6 for x in inferred),initial_spike_fraction=side[view][1]['spike_fraction'],initial_voltage_mean=side[view][1]['voltage_mean'],last_voltage_mean=side[view][450]['voltage_mean']))
        assert max(r['absolute_RGB_error'] for r in endpoint_checks)<2e-6 and all(r['out_of_01_transitions']==0 for r in state_checks)
        sequences.append(dict(source_id=sid,state_checks=state_checks,endpoint_checks=endpoint_checks,interval_records=18,interior_images_per_view=441,native_endpoints_per_view=9,checkpoint=conversion['checkpoint'],overshoot_channel_values_clamped=conversion['overshoot_channel_values_clamped']))
        sample_rows=[r for r in selected if r['source_id']==sid]
        for row in sample_rows:
            row_panels=[]
            for view in ['l','r']:
                rgb=np.array(Image.open(a.source_root/row['rgb_'+view]).convert('RGB'));g=gray(rgb);lo,hi=np.quantile(g,[.1,.9]);dark=g<=lo;bright=g>=hi
                path=a.derived_root/row['spike_'+view];spikes=manual_decode(path.read_bytes());per_bin=spikes.sum(axis=(1,2),dtype=np.int64);trace_counts=np.array([round(side[view][n]['spike_fraction']*540*960) for n in row['global_indices']]);assert np.array_equal(per_bin,trace_counts)
                mismatches=None
                if row==sample_rows[-1]:
                    mismatches=0
                    for t in range(50):
                        reference=upstream.get_block_spikes(str(path),begin_idx=t,spike_height=540,spike_width=960,block_len=1,flipud=False,with_head=False)
                        mismatches+=int(np.count_nonzero(reference[0]!=spikes[t]));upstream_frames+=1
                    assert mismatches==0
                count_image=spikes.sum(axis=0,dtype=np.int16);other='r' if view=='l' else 'l';other_counts=np.array([round(side[other][n]['spike_fraction']*540*960) for n in row['global_indices']]);reversed_counts=trace_counts[::-1]
                raw=np.frombuffer(path.read_bytes(),dtype=np.uint8);big=np.unpackbits(raw,bitorder='big').reshape(50,540,960)
                record=dict(sample_id=row['sample_id'],split=row['split'],view=view,shape=list(spikes.shape),bytes=path.stat().st_size,global_indices=[row['global_indices'][0],row['global_indices'][-1]],target_bin_zero_based=24,past_including_target_bins=25,future_bins=25,per_bin_spike_counts=per_bin.tolist(),spike_density=float(spikes.mean()),spikes_per_pixel_mean=float(count_image.mean()),spikes_per_pixel_min=int(count_image.min()),spikes_per_pixel_max=int(count_image.max()),target_gray_mean=float(g.mean(dtype=np.float64)),dark=dict(gray_threshold=float(lo),pixels=int(dark.sum()),mean_gray=float(g[dark].mean()),spike_density=float(count_image[dark].mean()/50)),bright=dict(gray_threshold=float(hi),pixels=int(bright.sum()),mean_gray=float(g[bright].mean()),spike_density=float(count_image[bright].mean()/50)),upstream_full_frame_bits_checked=50 if mismatches is not None else 0,upstream_mismatched_bits=mismatches,trace_bin_counts_exact=True,wrong_MSB_decode_mismatch_fraction=float(np.mean(big!=spikes)),wrong_vertical_flip_mismatch_fraction=float(np.mean(spikes[:,::-1,:]!=spikes)),wrong_time_reverse_count_mismatched_bins=int(np.count_nonzero(per_bin!=reversed_counts)),wrong_other_view_count_mismatched_bins=int(np.count_nonzero(per_bin!=other_counts)))
                windows.append(record);row_panels.extend([rgb,np.repeat(np.rint(count_image/10*255).clip(0,255).astype(np.uint8)[...,None],3,axis=2)])
                print('AUDITED',row['sample_id'],view,record['spike_density'],flush=True)
            panels.append((row['sample_id']+' '+row['split'],row_panels))
    canvas=Image.new('RGB',(960,180*len(panels)),'white');draw=ImageDraw.Draw(canvas)
    for i,(name,images) in enumerate(panels):
        draw.text((6,i*180+3),name+' | left RGB / left spike-count / right RGB / right spike-count (0..10 spikes)',fill='black')
        for col,image in enumerate(images):canvas.paste(Image.fromarray(image).resize((240,135)),(col*240,i*180+23))
    canvas.save(out/'examples.png')
    result=dict(status='PASS_NO_ENCODING_DEFECT_DETECTED_IN_CHECKED_PATHS',audited_utc=datetime.now(timezone.utc).isoformat(),manifest_sha256=manifest_sha,all_manifest_rows_checked=count,selected_source_ids=list(dict.fromkeys(r['source_id'] for r in selected)),selected_DAT_files=len(windows),upstream_actual_frames_compared=upstream_frames,upstream_actual_bits_compared=upstream_frames*540*960,upstream_reader_sha256=hashlib.sha256(a.upstream_reader.read_bytes()).hexdigest(),windows=windows,sequences=sequences,
        interpretation=dict(coverage='all1020mappingrows;2predeterminedsequences/1800actualstate-trace rows/36exactnativeendpoints/8actualDATs;lastcenters4DATs fully compared to pinnedupstreamreader,firstcenters separately corroborated by model-path audit',IF='FP32 V=0 once persequence/view; V+=normalizedBT601RGB; S=(V>=5); V-=5S. Actualnativeendpoint mean massbalance and allstate transition bounds checked; no dt/50 factor, no window reset, no leak/noise in declaredcode',encoding='headerless [50,540,960],chronological,rowmajor,LSB-first,noflip; index24 contains target and25 binsfuture',EMA='Frozenadapter BGR input,official arbitrarytime ours_t49interiorfractions j/50; outputs flipped channels backtoRGB,exactnativeendpoint appendedonce; batch1/view,FP32,noTTA,pad/unpad32. Code review+savedinterval/trace evidence,not freshEMA replay',limits='InteriorEMA fullRGB andper-pixelvoltage notpersisted; no newinference/re-simulation allowed, so byte-exact regeneration andeveryinteriorpixelcolor equation are not independently replayed. Paper exactnoise/amplitude,fps/exposure,gamma,initialphase andcodeimplementation variants remain unspecified/unverified; this is the declared A2.3-style dataset,notreleased-data reproduction.'),
        provenance=dict(command=sys.argv,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),hostname=socket.gethostname(),pid=os.getpid(),elapsed_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,NumPy=np.__version__,gpu_used=False,source_mutated=False,training_touched=False))
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('COMPLETE',len(windows),upstream_frames,flush=True)


if __name__=='__main__':main()
