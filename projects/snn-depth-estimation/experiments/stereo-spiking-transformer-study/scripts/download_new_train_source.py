"""Cluster-only exact-range new TRAIN source acquisition; no GPU or decoding.

Two explicit phases: headers writes a bound transport receipt for all840 native
TAR members; payload requires that exact receipt on the same endpoint. Each
subrange has one attempt, at most two workers, no full archives or auto fallback.
Partial failures preserve receipts and original bytes; output is never reused.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import datetime
import email.utils
import fcntl
import getpass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import struct
import sys
import threading
import time
from urllib.parse import urljoin,urlsplit

sys.path.insert(0,str(Path(__file__).resolve().parent))
from prepare_new_train_acquisition import (NATIVE,REPOSITORY,RELATIVE_ROOT,FAMILY_ROOT,file_sha,
    segments,validate_manifest,recovery_attestation)

ENDPOINTS=('https://huggingface.co','https://hf-mirror.com')


class RangeResponseError(ValueError):
    """A rejected response with only allowlisted, sanitized diagnostic fields."""
    def __init__(self,diagnostic):
        super().__init__('Exact identity-encoded HTTP206 range/length required; body not accepted')
        self.diagnostic=diagnostic


def sanitized_response_receipt(response,piece,part_size):
    def field(name,pattern):
        value=response.headers.get(name)
        if value is None:return None
        return value if isinstance(value,str) and re.fullmatch(pattern,value) else '<nonstandard-redacted>'
    def http_date(name):
        value=response.headers.get(name)
        if value is None:return None
        try:
            parsed=email.utils.parsedate_to_datetime(value)
            if parsed.tzinfo is None:raise ValueError('Timezone required')
            return parsed.astimezone(datetime.timezone.utc).isoformat()
        except (ValueError,TypeError,OverflowError):return '<nonstandard-redacted>'
    retry=response.headers.get('Retry-After')
    retry=field('Retry-After',r'[0-9]{1,10}') if retry is None or re.fullmatch(r'[0-9]{1,10}',retry) else http_date('Retry-After')
    rate_headers={name:field(name,r'[0-9]{1,20}') for name in ('RateLimit-Limit','RateLimit-Remaining','RateLimit-Reset',
        'X-RateLimit-Limit','X-RateLimit-Remaining','X-RateLimit-Reset') if name in response.headers}
    lo=piece['start'];n=piece['size']
    return dict(response_status=response.status_code if type(response.status_code) is int else None,
        response_headers={
            'Content-Range':field('Content-Range',r'bytes (?:[0-9]{1,20}-[0-9]{1,20}|\*)/[0-9]{1,20}'),
            'Content-Length':field('Content-Length',r'[0-9]{1,20}'),
            'Content-Encoding':field('Content-Encoding',r'[A-Za-z0-9_, -]{1,64}')},
        rate_limit_diagnostics=dict(retry_after=retry,date=http_date('Date'),numeric_fields=rate_headers,
            provider_units_or_quota_inferred=False),
        expected_numeric_range=dict(part_index=piece['part'],start=lo,end=lo+n-1,part_size=part_size,bytes=n),
        response_body_read=False)


def failed_member_receipt(member,mode,error):
    record=dict(path=member['path'],role=member['role'],source_id=member['source_id'],mode=mode,
        archive_start=member['header_offset'] if mode=='headers' else member['data_offset'],
        requested_bytes=512 if mode=='headers' else member['size'],error_type=type(error).__name__)
    if isinstance(error,RangeResponseError):record['response_guard']=error.diagnostic
    return record


def acquire_phase_lock(parent):
    parent.mkdir(parents=True,exist_ok=True)
    stream=(parent/'.acquisition-phase.lock').open('a')
    try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:
        stream.close();raise
    return stream


def validate_native_payload(path,raw,legacy_validator):
    """Original native checks plus explicit encoded-RGB/PFM scale semantics."""
    legacy_validator(path,raw)
    if path.endswith('.png'):
        if (len(raw)<33 or struct.unpack('>I',raw[8:12])[0]!=13 or raw[24]!=8 or raw[25]!=2):
            raise ValueError('Expected original8bit truecolor RGB PNG IHDR; preserve unexpected format')
        return dict(format='PNG',height=540,width=960,channels=3,bit_depth=8,color_type=2,pixels_decoded=False)
    if path.endswith('.pfm'):
        try:scale=float(raw.split(b'\n',3)[2].strip())
        except (ValueError,IndexError):raise ValueError('Malformed native PFM scale; preserve unexpected format') from None
        if not math.isfinite(scale) or abs(scale)!=1.:
            raise ValueError('Native float32 PFM scale must be finite+/-1; preserve unexpected format')
        return dict(format='PFM',height=540,width=960,channels=1,scale=scale,
            endian='little' if scale<0 else 'big',bytes_per_float=4,values_decoded=False)
    raise ValueError('Unselected native source format')


def load_native_helpers(manifest):
    for member,sha in manifest['controls_sha256'].items():
        if file_sha(REPOSITORY/member)!=sha:raise ValueError('Frozen acquisition control changed: '+member)
    sys.path.insert(0,str(NATIVE/'scripts'))
    ranges=importlib.import_module('range_archive');download=importlib.import_module('download_selected')
    if Path(ranges.__file__).resolve()!=NATIVE/'scripts/range_archive.py' or Path(download.__file__).resolve()!=NATIVE/'scripts/download_selected.py':
        raise ValueError('Original helper import resolved to another control tree')
    return ranges.tar_header,download.validate_payload


def verify_header(member,raw,parse_header):
    entry=parse_header(raw,member['header_offset'])
    if (entry is None or entry['path']!=member['source_member'] or entry['size']!=member['size']
            or entry['data_offset']!=member['data_offset'] or entry['type']!='0'):
        raise ValueError('Native TAR header differs from frozen selected TRAIN member')
    sha=hashlib.sha256(raw).hexdigest()
    if member.get('header_sha256') and sha!=member['header_sha256']:raise ValueError('Previously verified GT header digest changed')
    return dict(path=member['path'],role=member['role'],header_offset=member['header_offset'],header_sha256=sha,size=entry['size'])


def verify_transport_receipt(manifest,manifest_sha,receipt,endpoint):
    if (receipt.get('status')!='COMPLETE_ALL840_TRAIN_HEADERS' or receipt.get('manifest_sha256')!=manifest_sha
            or receipt.get('endpoint')!=endpoint or receipt.get('test_access') is not False
            or receipt.get('payload_bytes_read')!=0 or receipt.get('returned_bytes')!=manifest['budget']['header_http_bytes']
            or set(receipt.get('completed',{}))!={m['path'] for m in manifest['members']}):
        raise ValueError('Complete same-manifest/same-endpoint TRAIN-only transport receipt required before payload')
    for m in manifest['members']:
        row=receipt['completed'][m['path']]
        if (row['path']!=m['path'] or row['role']!=m['role'] or row['size']!=m['size']
                or row['header_offset']!=m['header_offset'] or not re.fullmatch(r'[0-9a-f]{64}',row['header_sha256'])
                or (m.get('header_sha256') and m['header_sha256']!=row['header_sha256'])):
            raise ValueError('Transport receipt member/header identity changed')


def verify_header_reuse(manifest,manifest_sha,header,header_sha,old,old_sha,failed,failed_sha,
        attestation,registry,registry_sha,endpoint):
    if endpoint!=manifest.get('recovery',{}).get('endpoint'):
        raise ValueError('Recovery endpoint changed')
    expected=recovery_attestation(old,manifest,old_sha,manifest_sha,header,header_sha,failed,failed_sha,
        registry_sha,registry['source_identity_sha256'],registered=True)
    if attestation!=expected:raise ValueError('Exact registered header-reuse attestation required')
    for member,sha in manifest['controls_sha256'].items():
        if registry['source_files'].get(member)!=sha:raise ValueError('Recovery controls differ from executing frozen source')
    return expected


def acquisition_paths(manifest,deployment,output,mode):
    """All versions share a family lock; output/root validation does no writes."""
    run_root=Path(deployment['run_root']).resolve();family=(run_root/FAMILY_ROOT).resolve()
    root=(run_root/manifest['output_root_relative']).resolve();output=Path(output).resolve()
    native=Path(deployment['native_source_root']).resolve()
    if (not root.is_relative_to(family) or root.is_relative_to(native) or root.exists() and mode=='payload'
            or Path(output).exists() or (mode=='payload' and output!=root)
            or (mode=='headers' and output.parent!=root.parent)):
        raise ValueError('Fresh dedicated run-root output only; existing roots are immutable')
    if manifest.get('recovery'):
        if mode!='payload':raise ValueError('Recovery reuses840 headers; no repeated header phase permitted')
        old_root=run_root/manifest['recovery']['failed_root_relative']
        if (old_root.is_symlink() or not old_root.is_dir()
                or {p.name for p in old_root.iterdir()}!={'download-state.json'}
                or (old_root/'download-state.json').is_symlink()
                or file_sha(old_root/'download-state.json')!=manifest['recovery']['failed_payload_receipt_sha256']):
            raise ValueError('Preserved failed root must contain only the exact zero-progress state')
    return root,family


class ExactRangeReader:
    """Streaming bounded HTTP206 only; multipart ranges split at exact boundaries."""
    def __init__(self,sources,endpoint,budget,deadline,*,session_factory=None,minimum_interval=0,clock=None,waiter=None):
        if endpoint not in ENDPOINTS:raise ValueError('Explicit approved transport endpoint required')
        self.sources=sources;self.endpoint=endpoint;self.budget=budget;self.deadline=deadline
        self.lock=threading.Lock();self.local=threading.local();self.stop=threading.Event()
        self.reserved_bytes=0;self.returned_bytes=0;self.logical_requests=0;self.redirect_requests=0
        self.session_factory=session_factory
        if type(minimum_interval) not in (int,float) or not math.isfinite(minimum_interval) or minimum_interval<0:
            raise ValueError('Finite nonnegative request spacing required')
        self.minimum_interval=minimum_interval;self.clock=clock or time.monotonic
        self.waiter=waiter or self.stop.wait;self.pacing_lock=threading.Lock();self.last_request_start=None

    def request(self,url,headers):
        # Redirects are explicit, so every actual HTTP send passes this same gate.
        for redirect in range(4):
            parsed=urlsplit(url)
            if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password:
                self.stop.set();raise ValueError('HTTPS-only redirect with no URL credentials required')
            with self.pacing_lock:
                delay=0 if self.last_request_start is None else max(0,self.last_request_start+self.minimum_interval-self.clock())
                if self.stop.is_set() or self.clock()+delay>=self.deadline:
                    raise TimeoutError('Acquisition deadline or peer failure')
                if delay:self.waiter(delay)
                if self.stop.is_set() or self.clock()>=self.deadline:raise TimeoutError('Acquisition deadline or peer failure')
                self.last_request_start=self.clock()
                remaining=self.deadline-self.clock()
                try:
                    response=self.local.session.get(url,headers=headers,stream=True,
                        timeout=(min(15,remaining),min(20,remaining)),allow_redirects=False)
                except Exception:
                    self.stop.set();raise RuntimeError('Ranged HTTP request failed; no retry/fallback; URL omitted') from None
            if response.status_code not in (301,302,303,307,308):return response
            location=response.headers.get('Location');response.close()
            if redirect==3 or not isinstance(location,str):
                self.stop.set();raise ValueError('Bounded explicit HTTPS redirect required; URL omitted')
            url=urljoin(url,location)
            with self.lock:self.redirect_requests+=1

    def read(self,role,start,size):
        if self.stop.is_set() or time.monotonic()>=self.deadline:raise TimeoutError('Acquisition deadline or peer failure')
        pieces=segments(self.sources[role],start,size)
        with self.lock:
            if self.reserved_bytes+size>self.budget:raise ValueError('Approved returned-range byte budget exceeded')
            self.reserved_bytes+=size
        if not hasattr(self.local,'session'):
            if self.session_factory:self.local.session=self.session_factory()
            else:
                import requests
                self.local.session=requests.Session()
                self.local.session.mount('https://',requests.adapters.HTTPAdapter(max_retries=0))
            self.local.session.max_redirects=3
            self.local.urls={}
        result=[]
        for piece in pieces:
            remaining=self.deadline-time.monotonic()
            if self.stop.is_set() or remaining<=0:raise TimeoutError('Acquisition deadline or peer failure')
            part=self.sources[role]['parts'][piece['part']];url=part['url']
            if not url.startswith('https://huggingface.co/'):raise ValueError('Frozen original provider URL required')
            url=self.local.urls.get((role,piece['part']),self.endpoint+url[len('https://huggingface.co'):])
            lo=piece['start'];n=piece['size'];hi=lo+n-1
            with self.lock:self.logical_requests+=1
            response_context=self.request(url,{'Range':f'bytes={lo}-{hi}','Accept-Encoding':'identity'})
            with response_context as response:
                if (response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {lo}-{hi}/{part["size"]}'
                        or response.headers.get('Content-Length')!=str(n) or response.headers.get('Content-Encoding','identity')!='identity'):
                    self.stop.set()
                    raise RangeResponseError(sanitized_response_receipt(response,piece,part['size']))
                # Same validated pinned part; short-lived signed URLs never enter receipts/logs.
                parsed=urlsplit(response.url)
                if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password:
                    self.stop.set();raise ValueError('Validated final URL must remain HTTPS; URL omitted')
                self.local.urls[(role,piece['part'])]=response.url
                data=[];count=0
                while count<n:
                    if self.stop.is_set() or time.monotonic()>=self.deadline:raise TimeoutError('Acquisition deadline or peer failure')
                    try:block=response.raw.read(min(65536,n-count))
                    except Exception:raise RuntimeError('Ranged HTTP body failed; no retry; URL omitted') from None
                    if not block:raise ValueError('Truncated HTTP range')
                    if len(block)>n-count:raise ValueError('Server exceeded requested range')
                    count+=len(block);data.append(block)
                    with self.lock:self.returned_bytes+=len(block)
                result.append(b''.join(data))
        return b''.join(result)


def bounded_map(items,worker,workers=2):
    if type(workers) is not int or not 1<=workers<=2:raise ValueError('At most two range workers')
    items=iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending=set()
        for _ in range(workers):
            item=next(items,None)
            if item is not None:pending.add(pool.submit(worker,item))
        while pending:
            done,pending=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:yield future.result()
            for _ in done:
                item=next(items,None)
                if item is not None:pending.add(pool.submit(worker,item))


def verify_source_registry(path):
    registry=json.loads(path.read_text())
    if hashlib.sha256(json.dumps(registry['source_files'],sort_keys=True).encode()).hexdigest()!=registry['source_identity_sha256']:
        raise ValueError('Frozen source registry identity changed')
    for member,sha in registry['source_files'].items():
        p=(path.parent/member).resolve()
        if not p.is_relative_to(path.parent.resolve()) or file_sha(p)!=sha:raise ValueError('Frozen source changed')
    for p in (Path(__file__),Path(__file__).with_name('prepare_new_train_acquisition.py')):
        if registry['source_files'].get(str(p.resolve().relative_to(path.parent.resolve())))!=file_sha(p):
            raise ValueError('Acquisition controls absent from frozen source')
    return registry


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['headers','payload'],required=True)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--deployment',type=Path,required=True);parser.add_argument('--source-registry',type=Path,required=True)
    parser.add_argument('--endpoint',choices=ENDPOINTS,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--approved-transfer-bytes',type=int,required=True)
    parser.add_argument('--header-receipt',type=Path);parser.add_argument('--header-receipt-sha256')
    parser.add_argument('--old-manifest',type=Path);parser.add_argument('--old-manifest-sha256')
    parser.add_argument('--failed-payload-receipt',type=Path);parser.add_argument('--failed-payload-receipt-sha256')
    parser.add_argument('--header-reuse-attestation',type=Path);parser.add_argument('--header-reuse-attestation-sha256')
    args=parser.parse_args();manifest=json.loads(args.manifest.read_text());validate_manifest(manifest,execution=True)
    if file_sha(args.manifest)!=args.manifest_sha256:raise ValueError('Approved acquisition manifest identity changed')
    deployment=json.loads(args.deployment.read_text())
    if file_sha(args.deployment)!=manifest['input_receipts_sha256']['deployment']:
        raise ValueError('Reviewed deployment/root identity changed')
    if (platform.system()!='Linux' or socket.gethostname()!=manifest['expected_hostname']
            or getpass.getuser()!=manifest['expected_user'] or deployment['expected_hostname']!=manifest['expected_hostname']
            or deployment['expected_user']!=manifest['expected_user']):
        raise RuntimeError('Source bulk/header transport must run on the configured cluster, never Mac')
    root,family=acquisition_paths(manifest,deployment,args.output,args.mode);output=args.output.resolve()
    registry=verify_source_registry(args.source_registry)
    parse_header,validate_payload=load_native_helpers(manifest)
    budget=manifest['budget']['header_http_bytes' if args.mode=='headers' else 'payload_http_bytes']
    if args.approved_transfer_bytes!=budget:raise ValueError('Explicit exact phase transfer budget required')
    reuse=None
    if args.mode=='payload':
        if not args.header_receipt or file_sha(args.header_receipt)!=args.header_receipt_sha256:
            raise ValueError('Exact completed header receipt SHA required')
        header=json.loads(args.header_receipt.read_text())
        if manifest.get('recovery'):
            for path,sha in ((args.old_manifest,args.old_manifest_sha256),
                    (args.failed_payload_receipt,args.failed_payload_receipt_sha256),
                    (args.header_reuse_attestation,args.header_reuse_attestation_sha256)):
                if path is None or file_sha(path)!=sha:raise ValueError('Exact immutable recovery input SHA required')
            reuse=verify_header_reuse(manifest,args.manifest_sha256,header,args.header_receipt_sha256,
                json.loads(args.old_manifest.read_text()),args.old_manifest_sha256,
                json.loads(args.failed_payload_receipt.read_text()),args.failed_payload_receipt_sha256,
                json.loads(args.header_reuse_attestation.read_text()),registry,file_sha(args.source_registry),args.endpoint)
        else:
            if any((args.old_manifest,args.failed_payload_receipt,args.header_reuse_attestation)):
                raise ValueError('Explicit recovery manifest required; legacy gate cannot accept header reuse')
            verify_transport_receipt(manifest,args.manifest_sha256,header,args.endpoint)
    phase_lock=acquire_phase_lock(family)  # common across legacy and every fresh recovery attempt
    # Repeat no-write freshness/preserved-root checks while holding the family lock.
    acquisition_paths(manifest,deployment,args.output,args.mode)
    if args.mode=='payload':
        root.mkdir(parents=True,exist_ok=False);receipt=root/'download-state.json'
    else:
        output.parent.mkdir(parents=True,exist_ok=True);receipt=output
    # Exclusive receipt creation is a durable attempt ticket before network IO.
    started=time.monotonic();state=dict(status='STARTED',mode=args.mode,manifest_sha256=args.manifest_sha256,
        deployment_sha256=file_sha(args.deployment),source_registry_sha256=file_sha(args.source_registry),
        source_identity_sha256=registry['source_identity_sha256'],git_commit=registry['git_commit'],
        script_sha256=file_sha(__file__),controls_sha256=manifest['controls_sha256'],endpoint=args.endpoint,
        source_configs_sha256=manifest['source_configs_sha256'],source_object_sha256=manifest['source_object_sha256'],
        host=socket.gethostname(),user=getpass.getuser(),pid=os.getpid(),command=sys.argv,completed={},
        approved_transfer_bytes=budget,test_access=False,pixel_arrays_decoded=False,model_inference=False,GPU_used=False,
        payload_bytes_read=0,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if reuse:
        state.update(header_preflight_evidence='REUSED_VERIFIED_ALL840_NOT_REEXECUTED',
            header_reuse_attestation_sha256=args.header_reuse_attestation_sha256,header_reuse_attestation=reuse,
            old_header_receipt_sha256=args.header_receipt_sha256,recovery=manifest['recovery'])
    with receipt.open('x') as stream:json.dump(state,stream,indent=2);stream.write('\n')
    reader=ExactRangeReader(manifest['sources'],args.endpoint,budget,started+manifest['max_seconds']-30,
        minimum_interval=manifest.get('recovery',{}).get('minimum_request_interval_seconds',0))
    state_lock=threading.RLock()
    def save():
        with state_lock:
            state.update(returned_bytes=reader.returned_bytes,logical_HTTP_subrange_requests=reader.logical_requests,
                redirect_HTTP_requests=reader.redirect_requests,elapsed_seconds=time.monotonic()-started,
                payload_bytes_read=reader.returned_bytes if args.mode=='payload' else 0)
            temporary=receipt.with_suffix('.receipt-tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(receipt)
    def worker(member):
        try:
            if args.mode=='headers':return member['path'],verify_header(member,reader.read(member['role'],member['header_offset'],512),parse_header)
            raw=reader.read(member['role'],member['data_offset'],member['size'])
            try:format_receipt=validate_native_payload(member['path'],raw,validate_payload)
            except ValueError as error:
                # Retain received bytes for review; never coerce or silently replace a source format.
                preserved=root/'unexpected-format'/member['path'];preserved.parent.mkdir(parents=True,exist_ok=True)
                with preserved.open('xb') as stream:stream.write(raw)
                with state_lock:
                    state.setdefault('unexpected_formats',{})[member['path']]=dict(preserved_path=str(preserved.relative_to(root)),
                        size=len(raw),sha256=hashlib.sha256(raw).hexdigest(),reason=str(error),pixel_arrays_decoded=False)
                raise
            destination=(root/member['path']).resolve()
            if not destination.is_relative_to(root):raise ValueError('Source output escapes fresh root')
            destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('xb') as stream:stream.write(raw)
            return member['path'],dict(size=len(raw),sha256=hashlib.sha256(raw).hexdigest(),role=member['role'],source_id=member['source_id'],format=format_receipt)
        except BaseException as error:
            with state_lock:
                state.setdefault('failed_members',[]).append(failed_member_receipt(member,args.mode,error))
            reader.stop.set();raise
    try:
        for path,row in bounded_map(manifest['members'],worker,manifest['workers']):
            state['completed'][path]=row;save()
        if len(state['completed'])!=840 or reader.returned_bytes!=budget:raise ValueError('Complete840-file/header exact budget required')
        if args.mode=='payload':
            # Fresh readback, headers and SHA only; no native image/GT array decode.
            for member in manifest['members']:
                if time.monotonic()-started>=manifest['max_seconds']-2:raise TimeoutError('Final header/hash audit deadline')
                path=root/member['path'];raw=path.read_bytes();format_receipt=validate_native_payload(member['path'],raw,validate_payload)
                if (len(raw)!=member['size'] or hashlib.sha256(raw).hexdigest()!=state['completed'][member['path']]['sha256']
                        or format_receipt!=state['completed'][member['path']]['format']):
                    raise ValueError('Final source payload readback identity changed')
            actual={str(p.relative_to(root)) for p in root.rglob('*') if p.suffix in ('.png','.pfm')}
            if actual!={m['path'] for m in manifest['members']}:raise ValueError('Unexpected/missing source files')
            state.update(status='COMPLETE840_NATIVE_TRAIN_SOURCE_FILES',payload_bytes_read=reader.returned_bytes,
                disk_payload_bytes=sum(m['size'] for m in manifest['members']),header_receipt_sha256=args.header_receipt_sha256,readback_all840_PASS=True)
        else:state['status']='COMPLETE_ALL840_TRAIN_HEADERS'
    except BaseException as error:
        reader.stop.set();state.update(status='FAILED_INCOMPLETE_PRESERVE_FOR_REVIEW',error_type=type(error).__name__)
        raise
    finally:save()
    print(json.dumps(dict(status=state['status'],receipt=str(receipt),receipt_sha256=file_sha(receipt),returned_bytes=reader.returned_bytes)))


if __name__=='__main__':main()
