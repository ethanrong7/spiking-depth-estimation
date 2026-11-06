"""Bounded metadata-only TAR inventory for new unexposed paired source scenes.

Consumes pinned existing source configs. Requests ONLY exact 512-byte known TAR
headers, skips payload via TAR size, and records target-split member identities.
No RGB/PFM pixels, geometry statistics or performance selection are accessed.
Use a fresh ignored output directory; --resume continues this tool's own state.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT/'experiments/flyingthings3d-native-v1/scripts'))
from range_archive import RangeArchive, tar_header


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--mode',choices=['rgb','gt'],required=True)
    p.add_argument('--target-split',choices=['TRAIN','TEST'],default='TEST')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cursor',type=int,default=0,help='Known verified TAR header offset, never arbitrary payload offset')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--max-headers',type=int,default=512)
    p.add_argument('--max-seconds',type=int,default=300)
    p.add_argument('--scenes',type=int,default=30)
    p.add_argument('--transport',choices=['original','hf-mirror'],default='original')
    a=p.parse_args()
    if min(a.max_headers,a.max_seconds,a.scenes)<=0 or a.cursor<0 or a.cursor%512:
        raise ValueError('Positive budgets and a known aligned header offset required')
    raw=a.source.read_bytes();identity=hashlib.sha256(raw).hexdigest();config=json.loads(raw)
    if a.transport=='hf-mirror':
        # Previously verified transport fallback; payload identity still bound
        # to the immutable source revision and recorded provider LFS hashes.
        for part in config['parts']:
            part['url']=part['url'].replace('https://huggingface.co/','https://hf-mirror.com/')
    state_path=a.output/'state.json'
    if a.resume:
        state=json.loads(state_path.read_text())
        if (state['source_config_sha256']!=identity or state['mode']!=a.mode
                or state['target_split']!=a.target_split):
            raise ValueError('Cannot resume a different inventory contract')
    else:
        a.output.mkdir(parents=True,exist_ok=False)
        state=dict(source_config_sha256=identity,mode=a.mode,target_split=a.target_split,
                   cursor=a.cursor,headers=0,metadata_bytes=0,scenes={},status='STARTED',
                   source_config=config,source_pixels_read=False)
    archive=RangeArchive(config);started=time.monotonic();session_headers=0
    def save(status):
        state.update(status=status,transport=a.transport,session_headers=session_headers,
                     session_metadata_bytes=archive.bytes_read,session_seconds=time.monotonic()-started,
                     complete_scenes=[sid for sid,scene in state['scenes'].items()
                        if (all(set(scene[v])=={f'{i:04d}' for i in range(6,16)} for v in ('left','right'))
                            if a.mode=='rgb' else set(scene['left'])=={f'{i:04d}' for i in range(6,16)})])
        temporary=state_path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n')
        temporary.replace(state_path)
    save('INDEXING')
    try:
        while session_headers<a.max_headers and time.monotonic()-started<a.max_seconds:
            offset=state['cursor'];entry=tar_header(archive.read(offset,512),offset)
            session_headers+=1;state['headers']+=1;state['metadata_bytes']+=512
            if entry is None:
                save('ARCHIVE_COMPLETE');break
            state['cursor']=entry['next_offset']
            parts=entry['path'].strip('/').split('/')
            if parts and parts[0]=='FlyingThings3D':parts=parts[1:]
            prefix='frames_cleanpass' if a.mode=='rgb' else 'disparity'
            suffix='.png' if a.mode=='rgb' else '.pfm'
            if (len(parts)==6 and parts[0]==prefix and parts[1]==a.target_split
                    and parts[4] in ('left','right') and parts[5].endswith(suffix)
                    and entry['type'] in ('0','\x00')):
                sid='/'.join(parts[1:4]);view=parts[4];frame=Path(parts[5]).stem
                scene=state['scenes'].setdefault(sid,{'left':{},'right':{}})
                scene[view][frame]=entry
            if session_headers%10==0:save('INDEXING')
            if len(state.get('complete_scenes',[]))>=a.scenes:
                save('TARGET_INVENTORY_COMPLETE');break
        else:save('PAUSED_METADATA_BUDGET')
        # Final receipt includes current headers even if target reached between saves.
        save(state['status'])
        print(json.dumps(dict(status=state['status'],cursor=state['cursor'],
                              complete_scenes=state['complete_scenes'],
                              session_headers=session_headers,metadata_bytes=archive.bytes_read,
                              source_pixels_read=False)))
    except Exception as error:
        state['error']=str(error);save('FAILED');raise

if __name__=='__main__':main()
