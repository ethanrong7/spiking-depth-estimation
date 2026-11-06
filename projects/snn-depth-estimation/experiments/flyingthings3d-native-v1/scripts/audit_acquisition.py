"""Fresh cluster-only file/header/hash audit of a completed source acquisition."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import socket
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','source-root','control-root','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-host',required=True);a=p.parse_args()
    assert platform.system()=='Linux' and socket.gethostname()==a.expected_host
    manifest=json.loads(a.manifest.read_text());receipt_path=a.source_root/'download-state.json'
    receipt=json.loads(receipt_path.read_text());digest=hashlib.sha256(a.manifest.read_bytes()).hexdigest()
    assert receipt['status']=='COMPLETE' and receipt['manifest_sha256']==digest
    source_hashes=json.loads((a.control_root/'snapshot.json').read_text())
    for name,expected in source_hashes.items():
        assert hashlib.sha256((a.control_root/name).read_bytes()).hexdigest()==expected
    sys.path.insert(0,str(a.control_root/'scripts'))
    from download_selected import validate_payload
    members={m['path']:m for m in manifest['members']}
    assert len(members)==4020 and set(members)==set(receipt['completed'])
    actual={str(p.relative_to(a.source_root)) for p in a.source_root.rglob('*') if p.is_file() and p.suffix in ['.png','.pfm']}
    assert actual==set(members) and not list(a.source_root.rglob('*.part'))
    sizes=0;roles=Counter();canonical=[]
    for path,member in sorted(members.items()):
        assert '/TRAIN/C/' in path and '/TEST/' not in path
        payload=(a.source_root/path).read_bytes();record=receipt['completed'][path]
        sha=hashlib.sha256(payload).hexdigest()
        assert len(payload)==member['size']==record['size'] and sha==record['sha256']
        assert record['role']==member['role'] and record['source_id']==member['source_id']
        validate_payload(path,payload);sizes+=len(payload);roles[member['role']]+=1
        canonical.append([path,len(payload),sha])
    assert roles=={'rgb':3000,'gt':1020} and sizes==manifest['budget']['disk_payload_bytes']
    references=Counter(s['split'] for s in manifest['sequences'] for f in s['frames'] if f['disparity_l'] is not None)
    assert references=={'train':960,'dev':60}
    result=dict(status='PASS',verified_utc=datetime.now(timezone.utc).isoformat(),manifest_sha256=digest,
        fresh_hash_verified_files=len(members),counts=dict(roles),reference_counts=dict(references),disk_payload_bytes=sizes,
        recorded_full_range_transfer_bytes=receipt['transferred_bytes'],download_receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        canonical_file_list_sha256=hashlib.sha256(json.dumps(canonical,separators=(',',':')).encode()).hexdigest(),
        control_snapshot_sha256=hashlib.sha256((a.control_root/'snapshot.json').read_bytes()).hexdigest(),
        checks='Every selected source file freshly rehashed; native PNG/PFM headers checked without pixel decoding; exact manifest set, no extra PNG/PFM or partials; no Test source files.',
        scope='File integrity/source preparation only; no conversion, model evaluation, GPU or training.')
    with a.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
