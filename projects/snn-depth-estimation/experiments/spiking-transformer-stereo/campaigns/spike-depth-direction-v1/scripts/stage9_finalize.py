"""Final-only S9 extraction and Train-calibration freeze; no standalone job API."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump_json(path, value, serializer=None):
    path=Path(path);part=path.with_suffix(path.suffix+'.part')
    if path.exists() or part.exists():raise FileExistsError(path)
    part.write_text(json.dumps(value,default=serializer,indent=2,allow_nan=False)+'\n')
    part.replace(path)


def finalize(model,arm,storage_config,output,T_gpu_ms,provenance):
    """Fresh final5000 model; stream calibration then frozen actual Dev policies.

    Caller owns the registered training allocation and exact final checkpoint.
    Output must be a new directory; failures retain partial artifacts for review.
    No retry/resume, source changes, parameter update, Dev fit or extra rollout.
    """
    if not (os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='debug'):
        raise RuntimeError('final extraction requires registered Slurm allocation')
    if arm not in ('V','F') or provenance.get('updates')!=5000:
        raise ValueError('only registered V/F final5000 extraction')
    required=('final_checkpoint_sha256','source_identity_sha256','config_sha256','preflight_result_sha256','split_sha256')
    if any(not isinstance(provenance.get(k),str) or len(provenance[k])!=64 for k in required):
        raise ValueError('complete final source/config/checkpoint/preflight/split identities required')
    if not math.isfinite(T_gpu_ms) or T_gpu_ms<=0:raise ValueError('fixed preflight T required')
    import torch
    from stage9_data import NativeData,device_pairs
    from stage9_native import digest_state,FEATURE_NAMES
    from stage9_actions import extract_frame
    from stage9_controls import numerical_environment,ordered_calibration_binding
    from stage9_analyze import fit_calibration_policies,serial
    runtime=numerical_environment()
    if any(v!='1' for v in runtime['threads'].values()):raise RuntimeError('numeric thread binding changed')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    begun=time.perf_counter();model.eval();model.reset();before=digest_state(model)
    packet=None;packet_sha=None;cal_binding={};cal_fit_seconds=0.;cal_count=0
    table=out/'action-table.jsonl';table_part=table.with_suffix('.jsonl.part')
    def stream_group(group,path,policies=None):
        dataset=NativeData(storage_config,group)
        expected=160 if group=='policy_calibration' else 60
        if len(dataset)!=expected:raise RuntimeError('certified extraction cohort changed')
        ids=[]
        with Path(path).open('x') as handle:
            for i in range(len(dataset)):
                item=dataset[i];pairs,cpu25=device_pairs(item,prefixes=(24,32,49) if arm=='V' else (49,))
                row=extract_frame(model,item,pairs,cpu25,'calibration' if group=='policy_calibration' else 'dev',arm,
                                  policies=policies,T_gpu_ms=T_gpu_ms,policy_sha=packet_sha)
                if row['sample_id']!=dataset.rows[i]['sample_id']:raise RuntimeError('extraction row order changed')
                if row['arm']!=arm:raise RuntimeError('extraction arm changed')
                ids.append(row['sample_id'])
                handle.write(json.dumps(row,default=serial,allow_nan=False,separators=(',',':'))+'\n');handle.flush()
                del item,pairs,cpu25,row
        if len(set(ids))!=expected:raise RuntimeError('duplicate extraction frame')
        return expected
    with torch.no_grad():
        if arm=='V':
            calibration=out/'calibration.jsonl';cal_part=calibration.with_suffix('.jsonl.part')
            cal_count=stream_group('policy_calibration',cal_part);cal_part.replace(calibration)
            # Reload the exact frozen bytes and original order for the sole calibration fit.
            cal_binding=ordered_calibration_binding(calibration)
            cal_rows=[json.loads(line) for line in calibration.read_text().splitlines() if line.strip()]
            fit_started=time.perf_counter();packet=fit_calibration_policies(cal_rows,T_gpu_ms);cal_fit_seconds=time.perf_counter()-fit_started
            dump_json(out/'policy-packet.json',packet,serial);packet_sha=file_sha(out/'policy-packet.json')
            packet=json.loads((out/'policy-packet.json').read_text())
            del cal_rows
        dev=out/'dev.jsonl';dev_part=dev.with_suffix('.jsonl.part')
        dev_count=stream_group('dev',dev_part,packet);dev_part.replace(dev)
        # Copy exact original lines; never reserialize calibration records in a combined table.
        with table_part.open('xb') as dest:
            if arm=='V':
                with (out/'calibration.jsonl').open('rb') as src:shutil.copyfileobj(src,dest)
            with dev.open('rb') as src:shutil.copyfileobj(src,dest)
        table_part.replace(table)
        model.reset()
        after=digest_state(model)
        if after!=before:raise RuntimeError('final extraction changed learned tensors or buffers')
        for cell in model.native.update_block.cells:
            if any(getattr(cell,name).membrane is not None for name in ('q_lif','k_lif','v_lif','output_lif')):
                raise RuntimeError('final extraction retained membrane state')
    metadata={'status':'COMPLETE','stage':'S9_FINAL5000_EXTRACTION','arm':arm,'scientific_optimizer_updates_during_extraction':0,
              'updates':5000,'provenance':dict(provenance),'T_gpu_ms':float(T_gpu_ms),'feature_names':list(FEATURE_NAMES),
              'analysis_runtime':runtime,**cal_binding,'policy_packet_sha256':packet_sha,
              'action_table_sha256':file_sha(table),'dev_jsonl_sha256':file_sha(dev),'calibration_rows':cal_count,'dev_rows':dev_count,
              'action_table_rows':cal_count+dev_count,'full_state_sha256_before':before,'full_state_sha256_after':after,
              'learned_and_buffer_tensors_unchanged':True,'recurrent_state_reset':True,'calibration_fit_cpu_wall_seconds':cal_fit_seconds,
              'complete_extraction_wall_seconds':time.perf_counter()-begun,'Dev_fitting':False,
              'timing_interpretation':'All device paths/costs and CPUselector/I/O/transfer times are retained perframe; T fixed from zero-update preflight; no latency or energy conversion.'}
    dump_json(out/'extraction-metadata.json',metadata)
    return metadata
