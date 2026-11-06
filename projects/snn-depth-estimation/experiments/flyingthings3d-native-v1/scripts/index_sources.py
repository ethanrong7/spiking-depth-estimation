"""Resumable metadata-only RGB and GT TAR indexing; never extracts Test payload."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from range_archive import RangeArchive, load, tar_header


def atomic(path, value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['rgb','gt'],required=True);p.add_argument('--max-seconds',type=int,default=3600)
    p.add_argument('--max-metadata-bytes',type=int,default=16*1024*1024);p.add_argument('--scenes',type=int,default=150)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'index.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=a.output/'state.json';source=load(a.source)
    identity=hashlib.sha256(a.source.read_bytes()).hexdigest()
    state=load(state_path) if state_path.exists() else dict(status='INDEXING',cursor=0,headers=0,metadata_bytes=0,source_sha256=identity,complete_scenes=[],scenes={},gt_scene_headers={},created_utc=datetime.now(timezone.utc).isoformat())
    if state['source_sha256']!=identity:raise ValueError('Source identity changed')
    archive=RangeArchive(source);started=time.monotonic();base_bytes=state['metadata_bytes']
    state.update(pid=os.getpid(),command=__import__('sys').argv,status='INDEXING')
    def save():
        state.update(updated_utc=datetime.now(timezone.utc).isoformat(),metadata_bytes=base_bytes+archive.bytes_read,current_session_requests=archive.requests)
        atomic(state_path,state)
    save()
    log=(a.output/'members.jsonl').open('a',buffering=1)
    try:
        while state['cursor'] < source['total_size']:
            if time.monotonic()-started>a.max_seconds or archive.bytes_read>=a.max_metadata_bytes:
                state['status']='PAUSED_BUDGET';break
            entry=tar_header(archive.read(state['cursor'],512),state['cursor'])
            if entry is None:
                state['status']='COMPLETE_ARCHIVE';break
            path=entry['path'];state['headers']+=1
            if a.mode=='gt' and entry['type'] in ('0','\x00') and path.endswith('.pfm'):
                # Native left-only GT TAR has ten equal-size PFMs per scene.
                # Verify first and last TAR headers; selected-file headers are
                # all individually verified before manifest freeze/download.
                parts=path.split('/')
                if len(parts)!=7 or parts[1]!='disparity' or parts[5]!='left' or entry['size']!=2073618:
                    raise ValueError('GT stride schema differs: '+path)
                stride=entry['next_offset']-entry['header_offset']
                last_offset=entry['header_offset']+9*stride
                last=tar_header(archive.read(last_offset,512),last_offset)
                if last is None or str(Path(last['path']).parent)!=str(Path(path).parent) or last['size']!=entry['size']:
                    raise ValueError('Ten-frame GT block differs: '+path)
                source_id='/'.join(parts[2:5])
                if source_id.startswith('TRAIN/C/'):
                    state['gt_scene_headers'][source_id]=dict(first_header=entry['header_offset'],stride=stride,frame_count=10,size=entry['size'],verification='first/last headers; all selected headers verified before freeze')
                state['cursor']=entry['header_offset']+10*stride
            else:
                state['cursor']=entry['next_offset']
            if a.mode=='rgb' and entry['type'] in ('0','\x00') and path.startswith('frames_cleanpass/TRAIN/C/') and path.endswith('.png'):
                parts=path.split('/');source_id='/'.join(parts[1:4]);view=parts[4];frame=Path(parts[5]).stem
                if view not in ('left','right'):raise ValueError('Unexpected RGB view')
                scene=state['scenes'].setdefault(source_id,{'left':{},'right':{},'first_header':entry['header_offset']})
                scene[view][frame]=entry
                log.write(json.dumps(entry)+'\n')
                expected={f'{i:04d}' for i in range(6,16)}
                if set(scene['left'])==expected and set(scene['right'])==expected and source_id not in state['complete_scenes']:
                    state['complete_scenes'].append(source_id);print('complete RGB',len(state['complete_scenes']),source_id,flush=True)
                if len(state['complete_scenes'])>=a.scenes:
                    state['status']='COMPLETE_SELECTION_INVENTORY';save();break
            if state['headers']%20==0:
                save();print(a.mode,'headers',state['headers'],'cursor',state['cursor'],'RGBscenes',len(state['complete_scenes']),'GTscenes',len(state['gt_scene_headers']),flush=True)
            if a.mode=='gt' and len(state['gt_scene_headers'])>=746:
                state['status']='COMPLETE_C_GT_INVENTORY';break
        save()
    except Exception as error:
        state.update(status='FAILED',error=repr(error));save();raise
    finally:log.close()


if __name__=='__main__':main()
