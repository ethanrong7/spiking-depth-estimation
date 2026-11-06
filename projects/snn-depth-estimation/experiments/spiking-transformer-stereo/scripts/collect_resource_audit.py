"""Collect scheduler accounting for this experiment's registered jobs only."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess


def category(label):
    if label == 'setup':return 'environment_setup'
    if 'smoke' in label:return 'gpu_smoke'
    if 'probe' in label or 'correspondence' in label:return 'development_probe'
    if label.startswith('confirmation-'):return 'confirmation_evaluation'
    return 'scientific_training'


def main():
    p=argparse.ArgumentParser()
    for name in ['cluster-config','jobs','output','raw-output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();cluster=json.loads(a.cluster_config.read_text());ledger=json.loads(a.jobs.read_text())
    def remote(command):return subprocess.check_output(['ssh','-o','BatchMode=yes',cluster['ssh_host'],command],text=True)
    assert remote('id -un; hostname').strip().splitlines()==[cluster['expected_user'],cluster['expected_hostname']]
    entries={str(row['job_id']):row for row in ledger}
    assert all(re.fullmatch(r'\d+',job) for job in entries)
    fields=['JobIDRaw','State','ExitCode','ElapsedRaw','Start','End','AllocTRES','ReqTRES']
    accounting=remote('sacct -n -X -P -j '+','.join(entries)+' --format=JobIDRaw,State,ExitCode,ElapsedRaw,Start,End,AllocTRES%100,ReqTRES%100')
    a.raw_output.parent.mkdir(parents=True,exist_ok=True);a.raw_output.write_text('|'.join(fields)+'\n'+accounting)
    active=remote('squeue -h -u '+shlex.quote(cluster['expected_user'])+' -o "%i|%T|%b"').strip().splitlines()
    active_ids={line.split('|')[0] for line in active}
    rows=[];groups={}
    for line in accounting.splitlines():
        values=line.split('|');row=dict(zip(fields,values));entry=entries[row['JobIDRaw']]
        label=entry.get('run') or entry.get('stage','unlabeled');kind=category(label)
        count=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',row['AllocTRES']);gpus=int(count.group(1)) if count else 0
        seconds=int(row['ElapsedRaw']);item=dict(job_id=row['JobIDRaw'],label=label,category=kind,state=row['State'],exit_code=row['ExitCode'],elapsed_seconds=seconds,start=row['Start'],end=row['End'],allocated_tres=row['AllocTRES'],requested_tres=row['ReqTRES'],allocated_gpus=gpus,gpu_allocation_seconds=seconds*gpus)
        rows.append(item);group=groups.setdefault(kind,dict(jobs=0,gpu_allocation_seconds=0,states={}))
        group['jobs']+=1;group['gpu_allocation_seconds']+=seconds*gpus;group['states'][row['State']]=group['states'].get(row['State'],0)+1
    if {row['job_id'] for row in rows}!=set(entries):raise RuntimeError('scheduler accounting is missing registered jobs')
    pending=sorted(set(entries)&active_ids)
    record=dict(collected_at=datetime.now(timezone.utc).isoformat(),status='interim' if pending else 'terminal',registered_jobs=len(rows),registered_jobs_pending_or_running=pending,account_queue_snapshot=active,unrelated_jobs_untouched=True,allocation_time_definition='scheduler ElapsedRaw multiplied by allocated GPU count; includes setup within GPU jobs, loading, decoding, evaluation and idle time; not measured energy',groups=groups,total_gpu_allocation_seconds=sum(row['gpu_allocation_seconds'] for row in rows),jobs=rows)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ['status','registered_jobs','registered_jobs_pending_or_running','total_gpu_allocation_seconds']}))


if __name__=='__main__':main()
