"""Freeze metadata-only targeted acquisition for the already selected30 TRAIN/C scenes.

Default output is unregistered. --approve is reserved for a later explicit
parent-reviewed freeze; this script never accesses the network or dataset pixels.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import tarfile

STUDY=Path(__file__).resolve().parents[1]
REPOSITORY=STUDY.parents[3]
NATIVE=STUDY.parent/'flyingthings3d-native-v1'
PROPOSAL_SHA='ef69a1ce953f83bcbeab4c79e81898ee0f366a82547d4beee2155afe908de796'
SOURCE_SHAS={'rgb':'74eadf86c3c8da50eeb22ef29fb916c95f0322901472820ea1f1a8d030b2fa05',
    'gt':'846223d27104f607c8ede938c4b5ec7fd09c4a3ab42d65b0d586f0d63baf92f6'}
RELATIVE_ROOT='stereo-spiking-transformer-study/new-train-source-acquisition-v1/source-data'
FAMILY_ROOT=str(Path(RELATIVE_ROOT).parent)
RECOVERY_SCHEMA='stereo-study-zero-progress-recovery-v1'
OPERATIONAL_KEYS={'registered','workers','output_root_relative','controls_sha256','recovery'}


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def segments(source,start,size):
    total=sum(part['size'] for part in source['parts'])
    if type(start) is not int or type(size) is not int or start<0 or size<=0 or start+size>total:
        raise ValueError('Exact nonempty bounded archive interval required')
    result=[];base=0
    for index,part in enumerate(source['parts']):
        lo=max(start,base);hi=min(start+size,base+part['size'])
        if hi>lo:result.append(dict(part=index,start=lo-base,size=hi-lo))
        base+=part['size']
    return result


def validate_member(member):
    role=member['role'];scene=member['source_id'];frame=member['source_frame']
    if not re.fullmatch(r'TRAIN/C/\d{4}',scene) or not re.fullmatch(r'00(?:0[6-9]|1[0-5])',frame):
        raise ValueError('Only fixed TRAIN/C native0006..0015 members allowed')
    if role=='rgb':
        side=member['view']
        if side not in ('left','right'):raise ValueError('Stereo view required')
        expected=f'frames_cleanpass/{scene}/{side}/{frame}.png'
    elif role=='gt':expected=f'disparity/{scene}/left/{frame}.pfm'
    else:raise ValueError('Only PNG RGB and positive-left PFM source roles allowed')
    if (member['path']!=expected or member['source_member']!=('FlyingThings3D/'+expected if role=='gt' else expected)
            or member['data_offset']!=member['header_offset']+512 or member['header_offset']%512
            or type(member['size']) is not int or not 32<=member['size']<=2073618 or member['type']!='0'
            or (role=='gt' and (member['size']!=2073618 or not re.fullmatch(r'[0-9a-f]{64}',member.get('header_sha256',''))))):
        raise ValueError('Exact native file/member/size/TAR offset identity required')


def budgets(members,sources):
    roles={role:[m for m in members if m['role']==role] for role in ('rgb','gt')}
    payload=sum(m['size'] for m in members);headers=512*len(members)
    return dict(rgb_files=len(roles['rgb']),gt_files=len(roles['gt']),rgb_payload_bytes=sum(m['size'] for m in roles['rgb']),
        gt_payload_bytes=sum(m['size'] for m in roles['gt']),disk_payload_bytes=payload,header_http_bytes=headers,
        payload_http_bytes=payload,total_header_plus_payload_http_bytes=headers+payload,
        header_http_ranges=sum(len(segments(sources[m['role']],m['header_offset'],512)) for m in members),
        payload_http_ranges=sum(len(segments(sources[m['role']],m['data_offset'],m['size'])) for m in members),
        maximum_payload_range_bytes=max(m['size'] for m in members),network_attempts_per_subrange=1)


def validate_manifest(manifest,*,execution=False):
    if (manifest.get('schema')!='stereo-study-new-train-acquisition-v1' or manifest.get('proposal_sha256')!=PROPOSAL_SHA
            or manifest.get('test_access') is not False or manifest.get('scene_count')!=30
            or manifest.get('max_seconds')!=3600
            or manifest.get('source_configs_sha256')!=SOURCE_SHAS
            or manifest.get('source_object_sha256')!={key:digest(value) for key,value in manifest['sources'].items()}):
        raise ValueError('Exact bounded30-scene new TRAIN-only acquisition contract required')
    recovery=manifest.get('recovery')
    if recovery is None:
        if manifest.get('workers')!=2 or manifest.get('output_root_relative')!=RELATIVE_ROOT:
            raise ValueError('Legacy root/two-worker contract changed')
    else:
        keys={'schema','attempt_id','old_manifest_sha256','old_header_receipt_sha256','failed_payload_receipt_sha256',
            'failed_root_relative','endpoint','minimum_request_interval_seconds'}
        if (set(recovery)!=keys or recovery['schema']!=RECOVERY_SCHEMA
                or not re.fullmatch(r'attempt-[0-9]{3}',recovery['attempt_id']) or recovery['attempt_id']=='attempt-000'
                or recovery['failed_root_relative']!=RELATIVE_ROOT or recovery['endpoint']!='https://hf-mirror.com'
                or type(recovery['minimum_request_interval_seconds']) not in (int,float)
                or not 2<=recovery['minimum_request_interval_seconds']<=10
                or manifest.get('workers')!=1
                or manifest.get('output_root_relative')!=f"{FAMILY_ROOT}/{recovery['attempt_id']}/source-data"
                or any(not re.fullmatch(r'[0-9a-f]{64}',recovery[k]) for k in
                    ('old_manifest_sha256','old_header_receipt_sha256','failed_payload_receipt_sha256'))):
            raise ValueError('Explicit bounded fresh zero-progress recovery contract required')
    if execution and manifest.get('registered') is not True:raise ValueError('Parent-reviewed registered acquisition manifest required')
    scenes=manifest['sequences'];members=manifest['members']
    if manifest.get('scene_ids')!=[s['source_id'] for s in scenes]:raise ValueError('Frozen new-scene ordering changed')
    if len(scenes)!=30 or len({s['source_id'] for s in scenes})!=30 or len(members)!=840 or len({m['path'] for m in members})!=840:
        raise ValueError('Exactly30 scenes/600 RGB/240 GT files required')
    expected=set()
    for scene in scenes:
        if scene['split']!='train' or scene['reference_centers']!=list(range(1,9)) or len(scene['frames'])!=10:
            raise ValueError('Only fixed Train source indices1..8 are disparity targets')
        for index,frame in enumerate(scene['frames']):
            if frame['index']!=index or frame['source_frame']!=f'{index+6:04d}':raise ValueError('Native frame order changed')
            expected.update((frame['rgb_l'],frame['rgb_r']))
            if index in range(1,9):
                if not frame['disparity_l']:raise ValueError('Required target disparity missing')
                expected.add(frame['disparity_l'])
            elif frame['disparity_l'] is not None:raise ValueError('Non-target GT must not be acquired')
    for member in members:
        validate_member(member)
        if member['source_id'] not in {s['source_id'] for s in scenes}:raise ValueError('Unselected scene member')
        segments(manifest['sources'][member['role']],member['header_offset'],512)
        segments(manifest['sources'][member['role']],member['data_offset'],member['size'])
    if expected!={m['path'] for m in members}:raise ValueError('Extra/unselected/Test member or missing frame')
    budget=budgets(members,manifest['sources'])
    if budget!=manifest['budget'] or budget['rgb_files']!=600 or budget['gt_files']!=240:
        raise ValueError('Exact derived transfer/disk budget changed')
    return True


def build_plan(proposal,sources,rgb_cache,gt_cache,historical):
    scenes=[s for s in proposal['scenes'] if s['split']=='train' and s['original_split']=='TRAIN' and s['historical_exposure'] is False]
    if len(scenes)!=30 or {s['scene_id'] for s in scenes}&set(historical):raise ValueError('Exactly the fixed30 new scenes disjoint from historical150 required')
    sequences=[];members=[]
    for scene in scenes:
        sid=scene['scene_id'];frames=[]
        if scene['target_center_indices']!=list(range(1,9)):raise ValueError('Frozen eight Train target indices changed')
        for index,frame in enumerate(scene['frames']):
            native=f'{index+6:04d}';paths=dict(index=index,source_frame=native,disparity_l=None)
            for view,key in [('left','rgb_l'),('right','rgb_r')]:
                entry=frame[key+'_tar']
                if entry!=rgb_cache['scenes'][sid][view][native]:raise ValueError('Proposal differs from cached RGB header inventory')
                paths[key]=frame[key]
                members.append(dict(role='rgb',source_id=sid,source_frame=native,view=view,path=frame[key],source_member=entry['path'],
                    **{k:entry[k] for k in ('size','header_offset','data_offset','type')}))
            if index in range(1,9):
                entry=frame['disparity_l_tar']
                if entry!=gt_cache['scenes'][sid][native]:raise ValueError('Proposal differs from individually verified GT headers')
                paths['disparity_l']=frame['disparity_l']
                members.append(dict(role='gt',source_id=sid,source_frame=native,path=frame['disparity_l'],source_member=entry['path'],
                    **{k:entry[k] for k in ('size','header_offset','data_offset','type','header_sha256')}))
            frames.append(paths)
        sequences.append(dict(source_id=sid,split='train',reference_centers=list(range(1,9)),frames=frames))
    plan=dict(schema='stereo-study-new-train-acquisition-v1',registered=False,proposal_sha256=PROPOSAL_SHA,
        scene_count=30,scene_ids=[s['scene_id'] for s in scenes],test_access=False,workers=2,max_seconds=3600,
        output_root_relative=RELATIVE_ROOT,sequences=sequences,members=members,sources=copy.deepcopy(sources),
        source_configs_sha256=SOURCE_SHAS,source_object_sha256={k:digest(v) for k,v in sources.items()},budget=budgets(members,sources),
        transport_policy='Explicit huggingface.co or hf-mirror.com; all840 selected TRAIN TAR headers verified on chosen cluster transport before any bulk; no automatic fallback/retry; no whole archive',
        scope='Source byte acquisition only; preserve PNG/PFM bytes; no pixel decoding/simulation/models/GPU/Test; historical source-data immutable')
    validate_manifest(plan);return plan


def protected_identity(manifest):
    """Only these enumerated top-level operational fields may differ."""
    return {k:v for k,v in manifest.items() if k not in OPERATIONAL_KEYS}


def verify_zero_progress(old,old_sha,header,failed,endpoint):
    # Import at call time: downloader imports this module for the shared contract.
    from download_new_train_source import verify_transport_receipt
    validate_manifest(old,execution=True)
    if old.get('recovery') is not None:raise ValueError('Only original zero-progress acquisition may be recovered')
    verify_transport_receipt(old,old_sha,header,endpoint)
    for receipt,mode in ((header,'headers'),(failed,'payload')):
        if (receipt.get('mode')!=mode or receipt.get('manifest_sha256')!=old_sha
                or receipt.get('endpoint')!=endpoint or receipt.get('test_access') is not False
                or receipt.get('controls_sha256')!=old['controls_sha256']
                or receipt.get('source_configs_sha256')!=old['source_configs_sha256']
                or receipt.get('source_object_sha256')!=old['source_object_sha256']
                or receipt.get('deployment_sha256')!=old['input_receipts_sha256']['deployment']
                or receipt.get('source_registry_sha256')!=header.get('source_registry_sha256')
                or receipt.get('source_identity_sha256')!=header.get('source_identity_sha256')):
            raise ValueError('Old receipts must bind exact original sources/controls/deployment')
    if (header.get('logical_HTTP_subrange_requests')!=old['budget']['header_http_ranges']
            or header.get('approved_transfer_bytes')!=old['budget']['header_http_bytes'] or header.get('failed_members')
            or any(not isinstance(header.get(k),str) or not re.fullmatch(r'[0-9a-f]{64}',header[k])
                for k in ('source_registry_sha256','source_identity_sha256'))):
        raise ValueError('Original complete bounded header transport provenance required')
    if (failed.get('status')!='FAILED_INCOMPLETE_PRESERVE_FOR_REVIEW' or failed.get('completed')!={}
            or failed.get('returned_bytes')!=0 or failed.get('payload_bytes_read')!=0
            or failed.get('unexpected_formats')):
        raise ValueError('Only preserved zero-byte zero-file payload failure can recover')


def verify_recovery_compatibility(old,new,old_sha,header,header_sha,failed,failed_sha):
    validate_manifest(new)
    recovery=new.get('recovery')
    if not recovery:raise ValueError('Explicit header-reuse recovery manifest required')
    verify_zero_progress(old,old_sha,header,failed,recovery['endpoint'])
    if (protected_identity(old)!=protected_identity(new)
            or recovery['old_manifest_sha256']!=old_sha or recovery['old_header_receipt_sha256']!=header_sha
            or recovery['failed_payload_receipt_sha256']!=failed_sha):
        raise ValueError('Protected ordered source/member/input identity or old artifact binding changed')
    old_controls=old['controls_sha256'];new_controls=new['controls_sha256']
    mutable={str((STUDY/'scripts'/name).relative_to(REPOSITORY)) for name in
        ('prepare_new_train_acquisition.py','download_new_train_source.py')}
    if (set(old_controls)!=set(new_controls) or any(old_controls[k]!=new_controls[k] for k in old_controls if k not in mutable)
            or any(not re.fullmatch(r'[0-9a-f]{64}',v) for v in new_controls.values())):
        raise ValueError('Only enumerated new acquisition controls may change; native helpers immutable')


def recovery_attestation(old,new,old_sha,new_sha,header,header_sha,failed,failed_sha,
        registry_sha,source_identity,*,registered=False):
    verify_recovery_compatibility(old,new,old_sha,header,header_sha,failed,failed_sha)
    if registered and (not new.get('registered') or any(not isinstance(v,str) or not re.fullmatch(r'[0-9a-f]{64}',v)
            for v in (registry_sha,source_identity))):
        raise ValueError('Registered attestation requires actual frozen executing source registry')
    return dict(schema='stereo-study-header-reuse-attestation-v1',registered=registered,
        status='VALIDATED_REUSED_HEADER_PREFLIGHT' if registered else 'UNREGISTERED_REVIEW_ONLY_NO_EXECUTING_SNAPSHOT',
        old_manifest_sha256=old_sha,new_manifest_sha256=new_sha,old_header_receipt_sha256=header_sha,
        failed_payload_receipt_sha256=failed_sha,endpoint=new['recovery']['endpoint'],
        old_source_registry_sha256=header['source_registry_sha256'],old_source_identity_sha256=header['source_identity_sha256'],
        executing_source_registry_sha256=registry_sha,executing_source_identity_sha256=source_identity,
        protected_identity_sha256=digest(protected_identity(new)),reused_headers=840,new_header_requests=0,
        operational_changes={k:dict(old=old.get(k),new=new.get(k)) for k in sorted(OPERATIONAL_KEYS) if old.get(k)!=new.get(k)})


def current_controls():
    return {str(p.relative_to(REPOSITORY)):file_sha(p) for p in [Path(__file__),Path(__file__).with_name('download_new_train_source.py'),
        NATIVE/'scripts/range_archive.py',NATIVE/'scripts/download_selected.py']}


def verify_planning_registry(registry_path,controls):
    """Read a standard local archive capsule or an extracted tree, never extract."""
    registry_path=Path(registry_path);registry=json.loads(registry_path.read_text())
    files=registry['source_files']
    def safe_name(name):
        return (isinstance(name,str) and bool(name) and '\\' not in name
            and not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts
            and str(PurePosixPath(name))==name and name!='.')
    if (not isinstance(files,dict) or not files or any(not safe_name(name)
            or not isinstance(sha,str) or not re.fullmatch(r'[0-9a-f]{64}',sha) for name,sha in files.items())
            or hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()!=registry['source_identity_sha256']):
        raise ValueError('Frozen source registry identity or member paths changed')
    capsule=registry_path.parent.resolve()
    if (capsule/'source.tar.gz').exists() or (capsule/'overlay.tar.gz').exists():
        overlay=registry.get('overlay_files')
        if (not isinstance(overlay,dict) or any(name not in files or files[name]!=sha for name,sha in overlay.items())):
            raise ValueError('Frozen overlay registry differs from full source')
        for filename,hash_key,expected in (('source.tar.gz','archive_sha256',files),
                ('overlay.tar.gz','overlay_sha256',overlay)):
            archive=capsule/filename
            if not archive.is_file() or archive.is_symlink() or file_sha(archive)!=registry.get(hash_key):
                raise ValueError('Frozen archive bytes changed')
            seen=set()
            with tarfile.open(archive,'r|gz') as stream:
                for member in stream:
                    if (not safe_name(member.name) or member.name in seen or member.name not in expected
                            or member.type not in (tarfile.REGTYPE,tarfile.AREGTYPE) or member.size<0):
                        raise ValueError('Exact regular bounded source archive members required')
                    seen.add(member.name);hasher=hashlib.sha256();count=0
                    with stream.extractfile(member) as payload:
                        for block in iter(lambda:payload.read(65536),b''):
                            count+=len(block);hasher.update(block)
                    if count!=member.size or hasher.hexdigest()!=expected[member.name]:
                        raise ValueError('Frozen archive source member changed')
            if seen!=set(expected):raise ValueError('Frozen archive member set changed')
    else:
        for member,sha in files.items():
            path=(capsule/member).resolve()
            if not path.is_relative_to(capsule) or not path.is_file() or file_sha(path)!=sha:
                raise ValueError('Frozen extracted source changed')
    for member,sha in controls.items():
        if files.get(member)!=sha:raise ValueError('New controls not in executing snapshot')
    return registry


def recovery_main(argv):
    parser=argparse.ArgumentParser(description='Metadata-only explicit zero-progress recovery; no source root creation')
    parser.add_argument('--recovery-from',type=Path,required=True);parser.add_argument('--old-header-receipt',type=Path,required=True)
    parser.add_argument('--failed-payload-receipt',type=Path,required=True);parser.add_argument('--attempt-id',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--attestation-output',type=Path,required=True)
    parser.add_argument('--executing-source-registry',type=Path);parser.add_argument('--approve',action='store_true')
    args=parser.parse_args(argv)
    for path in (args.output,args.attestation_output):
        if path.exists() or not any(path.resolve().is_relative_to(REPOSITORY/k) for k in ('storage','scratch')):
            raise ValueError('Fresh ignored metadata output required')
    if args.output.resolve()==args.attestation_output.resolve():raise ValueError('Distinct metadata outputs required')
    old=json.loads(args.recovery_from.read_text());header=json.loads(args.old_header_receipt.read_text())
    failed=json.loads(args.failed_payload_receipt.read_text());new=copy.deepcopy(old)
    old_sha=file_sha(args.recovery_from);header_sha=file_sha(args.old_header_receipt);failed_sha=file_sha(args.failed_payload_receipt)
    new.update(registered=args.approve,workers=1,output_root_relative=f'{FAMILY_ROOT}/{args.attempt_id}/source-data',
        controls_sha256=current_controls(),recovery=dict(schema=RECOVERY_SCHEMA,attempt_id=args.attempt_id,
            old_manifest_sha256=old_sha,old_header_receipt_sha256=header_sha,failed_payload_receipt_sha256=failed_sha,
            failed_root_relative=RELATIVE_ROOT,endpoint='https://hf-mirror.com',minimum_request_interval_seconds=2))
    verify_recovery_compatibility(old,new,old_sha,header,header_sha,failed,failed_sha)
    registry_sha=identity=None
    if args.executing_source_registry:
        # Planning may run from the control checkout against an already frozen
        # capsule. Actual payload execution additionally verifies its own path.
        registry=verify_planning_registry(args.executing_source_registry,new['controls_sha256'])
        registry_sha=file_sha(args.executing_source_registry);identity=registry['source_identity_sha256']
    # Serialize exact bytes first so attestation binds the real new manifest SHA.
    raw=json.dumps(new,indent=2)+'\n';new_sha=hashlib.sha256(raw.encode()).hexdigest()
    attestation=recovery_attestation(old,new,old_sha,new_sha,header,header_sha,failed,failed_sha,
        registry_sha,identity,registered=args.approve)
    for path,text in ((args.output,raw),(args.attestation_output,json.dumps(attestation,indent=2)+'\n')):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as stream:stream.write(text)
    print(json.dumps(dict(status=attestation['status'],manifest_sha256=new_sha,
        attestation_sha256=file_sha(args.attestation_output),budget=new['budget']),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgb-index',type=Path,required=True);parser.add_argument('--gt-headers',type=Path,required=True)
    parser.add_argument('--historical-index',type=Path,required=True);parser.add_argument('--deployment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--approve',action='store_true')
    args=parser.parse_args()
    if args.output.exists() or not any(args.output.resolve().is_relative_to(REPOSITORY/k) for k in ('storage','scratch')):
        raise ValueError('Fresh ignored metadata output required')
    proposal_path=STUDY/'configs/causal-benchmark-source-proposal-v1.json'
    if file_sha(proposal_path)!=PROPOSAL_SHA:raise ValueError('Reviewed proposal identity changed')
    sources={role:json.loads((NATIVE/'configs'/name).read_text()) for role,name in [('rgb','acquisition-rgb-source.json'),('gt','acquisition-source.json')]}
    for role,name in [('rgb','acquisition-rgb-source.json'),('gt','acquisition-source.json')]:
        if file_sha(NATIVE/'configs'/name)!=SOURCE_SHAS[role]:raise ValueError('Original pinned source config changed')
    proposal=json.loads(proposal_path.read_text())
    history=json.loads(args.historical_index.read_text());rgb=json.loads(args.rgb_index.read_text());gt=json.loads(args.gt_headers.read_text())
    if (len(history['sequences'])!=150 or rgb['source_config_sha256']!=SOURCE_SHAS['rgb']
            or gt['source_config_sha256']!=SOURCE_SHAS['gt'] or gt['status']!='COMPLETE'
            or rgb.get('source_pixels_read') is not False or gt.get('source_pixels_read') is not False
            or {s['source_index_sha256'] for s in proposal['scenes'] if s['historical_exposure']}!={file_sha(args.historical_index)}):
        raise ValueError('Original historical150 and complete pinned source inventories required')
    plan=build_plan(proposal,sources,rgb,gt,[s['source_id'] for s in history['sequences']])
    deployment=json.loads(args.deployment.read_text())
    plan.update(registered=args.approve,expected_hostname=deployment['expected_hostname'],expected_user=deployment['expected_user'],
        input_receipts_sha256={k:file_sha(p) for k,p in [('rgb_index',args.rgb_index),('gt_headers',args.gt_headers),('historical_index',args.historical_index),('deployment',args.deployment)]},
        controls_sha256={str(p.relative_to(REPOSITORY)):file_sha(p) for p in [Path(__file__),Path(__file__).with_name('download_new_train_source.py'),
            NATIVE/'scripts/range_archive.py',NATIVE/'scripts/download_selected.py']})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(plan,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status='REGISTERED_METADATA_ONLY' if args.approve else 'PENDING_PARENT_REVIEW_NO_TRANSFER',
        manifest_sha256=file_sha(args.output),budget=plan['budget']),indent=2))


if __name__=='__main__':
    import sys
    if '--recovery-from' in sys.argv:recovery_main(sys.argv[1:])
    else:main()
