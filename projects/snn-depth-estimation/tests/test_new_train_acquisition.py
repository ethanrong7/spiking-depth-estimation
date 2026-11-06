"""Metadata/header/HTTP mocks only; no source pixels, downloads or GPU jobs."""
import copy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import struct
import tarfile
import tempfile
import time
import unittest
from unittest import mock

SCRIPTS=Path(__file__).resolve().parents[1]/'experiments/stereo-spiking-transformer-study/scripts'
sys.path.insert(0,str(SCRIPTS))
import prepare_new_train_acquisition as plan
import download_new_train_source as download


def fixture():
    proposal=json.loads((SCRIPTS.parent/'configs/causal-benchmark-source-proposal-v1.json').read_text())
    selected=[s for s in proposal['scenes'] if s['split']=='train' and not s['historical_exposure']]
    rgb={'scenes':{}};gt={'scenes':{}}
    for scene in selected:
        sid=scene['scene_id'];rgb['scenes'][sid]={view:{} for view in ('left','right')};gt['scenes'][sid]={}
        for frame in scene['frames']:
            for view,key in [('left','rgb_l'),('right','rgb_r')]:rgb['scenes'][sid][view][frame['source_frame']]=frame[key+'_tar']
            gt['scenes'][sid][frame['source_frame']]=frame['disparity_l_tar']
    sources={role:json.loads((plan.NATIVE/'configs'/name).read_text()) for role,name in
        [('rgb','acquisition-rgb-source.json'),('gt','acquisition-source.json')]}
    historical=[s['scene_id'] for s in proposal['scenes'] if s['historical_exposure']]
    return plan.build_plan(proposal,sources,rgb,gt,historical)


def recovery_fixture():
    old=fixture();old.update(registered=True,expected_hostname='fixture-cluster',expected_user='fixture-user',
        input_receipts_sha256={'deployment':'d'*64,'rgb_index':'1'*64,'gt_headers':'2'*64,'historical_index':'3'*64},
        controls_sha256=plan.current_controls())
    old_sha=plan.digest(old);rows={m['path']:dict(path=m['path'],role=m['role'],size=m['size'],
        header_offset=m['header_offset'],header_sha256=m.get('header_sha256','a'*64)) for m in old['members']}
    common=dict(manifest_sha256=old_sha,endpoint='https://hf-mirror.com',test_access=False,payload_bytes_read=0,
        controls_sha256=old['controls_sha256'],source_configs_sha256=old['source_configs_sha256'],
        source_object_sha256=old['source_object_sha256'],deployment_sha256='d'*64,
        source_registry_sha256='e'*64,source_identity_sha256='f'*64)
    header=dict(common,status='COMPLETE_ALL840_TRAIN_HEADERS',mode='headers',returned_bytes=430080,completed=rows,
        logical_HTTP_subrange_requests=840,approved_transfer_bytes=430080)
    failed=dict(common,status='FAILED_INCOMPLETE_PRESERVE_FOR_REVIEW',mode='payload',returned_bytes=0,completed={})
    new=copy.deepcopy(old);new.update(workers=1,output_root_relative=plan.FAMILY_ROOT+'/attempt-002/source-data',
        recovery=dict(schema=plan.RECOVERY_SCHEMA,attempt_id='attempt-002',old_manifest_sha256=old_sha,
            old_header_receipt_sha256=plan.digest(header),failed_payload_receipt_sha256=plan.digest(failed),
            failed_root_relative=plan.RELATIVE_ROOT,endpoint='https://hf-mirror.com',minimum_request_interval_seconds=2))
    return old,new,header,failed


def archive_capsule(root,payloads,*,entries=None):
    root.mkdir(parents=True,exist_ok=True)
    def archive(path,rows):
        with tarfile.open(path,'w:gz') as stream:
            for name,raw,kind in rows:
                info=tarfile.TarInfo(name);info.type=kind;info.size=len(raw)
                stream.addfile(info,io.BytesIO(raw))
    rows=[(name,raw,tarfile.REGTYPE) for name,raw in payloads.items()]
    archive(root/'source.tar.gz',rows if entries is None else entries)
    archive(root/'overlay.tar.gz',rows)
    files={name:plan.hashlib.sha256(raw).hexdigest() for name,raw in payloads.items()}
    registry=dict(schema=1,git_commit='a'*40,source_files=files,overlay_files=copy.deepcopy(files),
        source_identity_sha256=plan.hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
        archive_sha256=plan.file_sha(root/'source.tar.gz'),overlay_sha256=plan.file_sha(root/'overlay.tar.gz'))
    path=root/'source-registry.json';path.write_text(json.dumps(registry))
    return path,registry


class AcquisitionTests(unittest.TestCase):
    def test_exact30_scene_payload_budget_and_no_test_or_nonreference_GT(self):
        value=fixture();self.assertTrue(plan.validate_manifest(value))
        self.assertEqual(value['budget']['rgb_payload_bytes'],441231505)
        self.assertEqual(value['budget']['gt_payload_bytes'],497668320)
        self.assertEqual(value['budget']['disk_payload_bytes'],938899825)
        self.assertEqual(value['budget']['total_header_plus_payload_http_bytes'],939329905)
        self.assertEqual(value['budget']['header_http_ranges'],840);self.assertEqual(value['budget']['payload_http_ranges'],840)
        self.assertEqual(len(value['members']),840)
        self.assertTrue(all('/TRAIN/C/' in m['path'] and '/TEST/' not in m['path'] for m in value['members']))
        self.assertFalse(any(m['role']=='gt' and m['source_frame'] in ('0006','0015') for m in value['members']))
        with self.assertRaises(ValueError):plan.validate_manifest(value,execution=True)
        for change in [lambda p:p['members'][0].update(path='frames_cleanpass/TEST/C/0001/left/0006.png'),
                lambda p:p['members'][0].update(path='../escape.png'),lambda p:p['members'][0].update(size=10**12),
                lambda p:p['sequences'][0]['frames'][0].update(disparity_l='unselected.pfm'),
                lambda p:p.update(workers=3)]:
            bad=copy.deepcopy(value);change(bad)
            with self.assertRaises(ValueError):plan.validate_manifest(bad)

    def test_multipart_exact_boundary_splitting(self):
        source={'parts':[{'size':1024},{'size':1024}]}
        self.assertEqual(plan.segments(source,1000,48),[dict(part=0,start=1000,size=24),dict(part=1,start=0,size=24)])
        for start,size in [(-1,10),(2000,100),(0,0)]:
            with self.assertRaises(ValueError):plan.segments(source,start,size)

    def test_native_TAR_headers_and_GT_digest_before_payload(self):
        sys.path.insert(0,str(plan.NATIVE/'scripts'))
        from range_archive import tar_header
        member=next(m for m in fixture()['members'] if m['role']=='gt')
        info=tarfile.TarInfo(member['source_member']);info.size=member['size'];raw=info.tobuf(format=tarfile.USTAR_FORMAT)
        member=copy.deepcopy(member);member['header_sha256']=plan.hashlib.sha256(raw).hexdigest()
        result=download.verify_header(member,raw,tar_header);self.assertEqual(result['path'],member['path'])
        bad=bytearray(raw);bad[0]^=1
        with self.assertRaises(ValueError):download.verify_header(member,bytes(bad),tar_header)
        member['header_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'digest'):download.verify_header(member,raw,tar_header)

    def test_explicit_RGB_depth_color_and_PFM_scale_without_pixel_decode(self):
        # Header-only fixtures; the original validator is separately retained.
        png=b'\x89PNG\r\n\x1a\n'+struct.pack('>I',13)+b'IHDR'+struct.pack('>II',960,540)+bytes([8,2,0,0,0])+b'CRC!'
        legacy=mock.Mock()
        self.assertEqual(download.validate_native_payload('frame.png',png,legacy)['color_type'],2)
        legacy.assert_called_once_with('frame.png',png)
        for at,value in [(24,16),(25,0),(25,6)]:
            bad=bytearray(png);bad[at]=value
            with self.assertRaisesRegex(ValueError,'truecolor RGB'):download.validate_native_payload('frame.png',bytes(bad),legacy)
        for scale,endian in [(b'-1.0','little'),(b'+1.0','big')]:
            pfm=b'Pf\n960 540\n'+scale+b'\n'
            row=download.validate_native_payload('frame.pfm',pfm,legacy)
            self.assertEqual(row['endian'],endian);self.assertFalse(row['values_decoded'])
        for scale in (b'nan',b'inf',b'-inf',b'0',b'.5',b'-2',b'invalid'):
            with self.subTest(scale=scale):
                with self.assertRaises(ValueError):download.validate_native_payload('frame.pfm',b'Pf\n960 540\n'+scale+b'\n',legacy)

    def test_single_attempt_range_HTTP200_fails_before_body_and_no_retry(self):
        source={'rgb':{'parts':[dict(size=4096,url='https://huggingface.co/pinned/source')]}}
        response=mock.MagicMock();response.__enter__.return_value=response;response.status_code=200;response.history=[]
        response.headers={};response.raw=mock.Mock();session=mock.Mock();session.get.return_value=response
        reader=download.ExactRangeReader(source,'https://hf-mirror.com',512,time.monotonic()+10,session_factory=lambda:session)
        with self.assertRaises(ValueError):reader.read('rgb',512,512)
        response.raw.read.assert_not_called();self.assertEqual(session.get.call_count,1);self.assertEqual(reader.returned_bytes,0)

    def test_failed_response_records_exact_member_and_safe_headers_without_relaxing_guard(self):
        source={'rgb':{'parts':[dict(size=4096,url='https://huggingface.co/pinned/source')]}}
        response=mock.MagicMock();response.__enter__.return_value=response;response.status_code=206;response.history=[]
        # A plausible CDN response with the requested range but absent length
        # must still fail, with enough evidence to identify the missing field.
        response.headers={'Content-Range':'bytes 512-1023/4096','Content-Encoding':'identity',
            'Location':'https://fixture.invalid/?private=secret','Authorization':'secret','Set-Cookie':'secret'}
        response.url='https://fixture.invalid/?private=secret';response.raw=mock.Mock()
        session=mock.Mock();session.get.return_value=response
        reader=download.ExactRangeReader(source,'https://hf-mirror.com',512,time.monotonic()+10,session_factory=lambda:session)
        with self.assertRaises(download.RangeResponseError) as caught:reader.read('rgb',512,512)
        member=dict(path='frames_cleanpass/TRAIN/C/0349/left/0006.png',role='rgb',source_id='TRAIN/C/0349',
            header_offset=512,data_offset=1024,size=1024)
        record=download.failed_member_receipt(member,'headers',caught.exception)
        self.assertEqual(record['path'],member['path']);self.assertEqual(record['archive_start'],512)
        detail=record['response_guard'];self.assertEqual(detail['response_status'],206)
        self.assertEqual(detail['response_headers'],{'Content-Range':'bytes 512-1023/4096','Content-Length':None,'Content-Encoding':'identity'})
        self.assertEqual(detail['expected_numeric_range'],dict(part_index=0,start=512,end=1023,part_size=4096,bytes=512))
        self.assertFalse(detail['response_body_read']);response.raw.read.assert_not_called()
        self.assertEqual(session.get.call_count,1);self.assertEqual(reader.returned_bytes,0)
        serialized=json.dumps(record)+str(caught.exception)
        for forbidden in ('https://','private=','secret','Authorization','Location','Set-Cookie'):
            self.assertNotIn(forbidden,serialized)
        response.headers['Content-Range']='https://fixture.invalid/?private=secret'
        response.headers['Content-Length']='private=secret'
        response.headers['Content-Encoding']='https://fixture.invalid/'
        sanitized=download.sanitized_response_receipt(response,dict(part=0,start=512,size=512),4096)
        self.assertTrue(all(v=='<nonstandard-redacted>' for v in sanitized['response_headers'].values()))

    def test_exact_HTTP206_and_bound_byte_budget(self):
        source={'rgb':{'parts':[dict(size=4096,url='https://huggingface.co/pinned/source')]}}
        response=mock.MagicMock();response.__enter__.return_value=response;response.status_code=206;response.history=[]
        response.url='https://fixture.invalid/signed-object'
        response.headers={'Content-Range':'bytes 512-1023/4096','Content-Length':'512'};response.raw=io.BytesIO(b'x'*512)
        session=mock.Mock();session.get.return_value=response
        reader=download.ExactRangeReader(source,'https://hf-mirror.com',512,time.monotonic()+10,session_factory=lambda:session)
        self.assertEqual(reader.read('rgb',512,512),b'x'*512);self.assertEqual(reader.returned_bytes,512)
        self.assertEqual(session.get.call_args.args[0],'https://hf-mirror.com/pinned/source')
        self.assertEqual(session.get.call_args.kwargs['headers']['Range'],'bytes=512-1023')
        with self.assertRaises(ValueError):reader.read('rgb',512,512)
        self.assertEqual(session.get.call_count,1)

    def test_validated_final_redirect_is_cached_per_part_without_receipt_leak(self):
        source={'rgb':{'parts':[dict(size=4096,url='https://huggingface.co/pinned/source')]}}
        calls=[]
        def get(url,**kwargs):
            calls.append(url);start=512 if len(calls)==1 else 1024
            response=mock.MagicMock();response.__enter__.return_value=response;response.status_code=206
            response.url='https://fixture.invalid/signed-object?private=fixture';response.history=[]
            response.headers={'Content-Range':f'bytes {start}-{start+511}/4096','Content-Length':'512'}
            response.raw=io.BytesIO(b'x'*512);return response
        session=mock.Mock();session.get.side_effect=get
        reader=download.ExactRangeReader(source,'https://hf-mirror.com',1024,time.monotonic()+10,session_factory=lambda:session)
        reader.read('rgb',512,512);reader.read('rgb',1024,512)
        self.assertEqual(calls,['https://hf-mirror.com/pinned/source','https://fixture.invalid/signed-object?private=fixture'])
        self.assertEqual(reader.returned_bytes,1024)
        self.assertFalse(any('private' in key for key in vars(reader)))

    def test_transport_receipt_requires_all840_same_endpoint_and_GT_hashes(self):
        value=fixture();rows={m['path']:dict(path=m['path'],role=m['role'],size=m['size'],header_offset=m['header_offset'],
            header_sha256=m.get('header_sha256','a'*64)) for m in value['members']}
        receipt=dict(status='COMPLETE_ALL840_TRAIN_HEADERS',manifest_sha256='b'*64,endpoint='https://hf-mirror.com',
            test_access=False,payload_bytes_read=0,returned_bytes=value['budget']['header_http_bytes'],completed=rows)
        download.verify_transport_receipt(value,'b'*64,receipt,'https://hf-mirror.com')
        for key,changed in [('endpoint','https://huggingface.co'),('payload_bytes_read',1),('test_access',True)]:
            bad=copy.deepcopy(receipt);bad[key]=changed
            with self.assertRaises(ValueError):download.verify_transport_receipt(value,'b'*64,bad,'https://hf-mirror.com')
        bad=copy.deepcopy(receipt);bad['completed'].pop(next(iter(rows)))
        with self.assertRaises(ValueError):download.verify_transport_receipt(value,'b'*64,bad,'https://hf-mirror.com')

    def test_at_most_two_workers_no_whole_queue_after_failure(self):
        calls=[]
        def fail(item):calls.append(item);raise ValueError('fixture')
        with self.assertRaises(ValueError):list(download.bounded_map(range(1000),fail,2))
        self.assertLessEqual(len(calls),2)
        with self.assertRaises(ValueError):list(download.bounded_map([1],lambda x:x,3))

    def test_fresh_header_filename_cannot_duplicate_active_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            first=download.acquire_phase_lock(Path(tmp))
            try:
                with self.assertRaises(BlockingIOError):download.acquire_phase_lock(Path(tmp))
            finally:first.close()
            second=download.acquire_phase_lock(Path(tmp));second.close()


    def test_recovery_attestation_reuses_exact_headers_and_binds_current_registry_without_network(self):
        old,new,header,failed=recovery_fixture()
        args=(old,new,plan.digest(old),plan.digest(new),header,plan.digest(header),failed,plan.digest(failed),'7'*64,'8'*64)
        attestation=plan.recovery_attestation(*args,registered=True)
        registry=dict(source_identity_sha256='8'*64,source_files=new['controls_sha256'])
        with mock.patch.object(download.ExactRangeReader,'read',side_effect=AssertionError('No network/header repeat')):
            download.verify_header_reuse(new,plan.digest(new),header,plan.digest(header),old,plan.digest(old),
                failed,plan.digest(failed),attestation,registry,'7'*64,'https://hf-mirror.com')
        self.assertEqual(attestation['new_header_requests'],0);self.assertEqual(attestation['reused_headers'],840)
        # The legacy gate still refuses the new manifest SHA.
        with self.assertRaises(ValueError):download.verify_transport_receipt(new,plan.digest(new),header,'https://hf-mirror.com')
        for changed in ('executing_source_registry_sha256','old_header_receipt_sha256','protected_identity_sha256'):
            bad=copy.deepcopy(attestation);bad[changed]='0'*64
            with self.assertRaises(ValueError):download.verify_header_reuse(new,plan.digest(new),header,plan.digest(header),
                old,plan.digest(old),failed,plan.digest(failed),bad,registry,'7'*64,'https://hf-mirror.com')
        review=plan.recovery_attestation(*args[:-2],None,None,registered=False)
        with self.assertRaises(ValueError):download.verify_header_reuse(new,plan.digest(new),header,plan.digest(header),
            old,plan.digest(old),failed,plan.digest(failed),review,registry,'7'*64,'https://hf-mirror.com')

    def test_recovery_rejects_order_selection_sources_inputs_native_changes_and_failed_header(self):
        old,new,header,failed=recovery_fixture()
        changes=[lambda x:x['members'].reverse(),lambda x:x['sequences'].reverse(),
            lambda x:x['input_receipts_sha256'].update(rgb_index='0'*64),
            lambda x:x['sources']['rgb']['parts'][0].update(url='https://huggingface.co/other'),
            lambda x:x.update(max_seconds=7200),lambda x:x.update(workers=2),
            lambda x:x['recovery'].update(minimum_request_interval_seconds=0),
            lambda x:x['recovery'].update(endpoint='https://huggingface.co'),
            lambda x:x['controls_sha256'].update({str((plan.NATIVE/'scripts/range_archive.py').relative_to(plan.REPOSITORY)):'0'*64})]
        for change in changes:
            bad=copy.deepcopy(new);change(bad)
            with self.assertRaises(ValueError):plan.verify_recovery_compatibility(old,bad,plan.digest(old),header,
                plan.digest(header),failed,plan.digest(failed))
        for role in ('header','failed'):
            for field,value in [('status','STARTED'),('payload_bytes_read',1),('returned_bytes',1),('completed',{'partial':{}})]:
                h,f=copy.deepcopy(header),copy.deepcopy(failed);(h if role=='header' else f)[field]=value
                with self.assertRaises(ValueError):plan.verify_recovery_compatibility(old,new,plan.digest(old),h,
                    plan.digest(header),f,plan.digest(failed))

    def test_recovery_paths_preserve_failed_root_reject_collisions_and_share_legacy_family_lock(self):
        old,new,header,failed=recovery_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            deployment=dict(run_root=tmp,native_source_root=str(Path(tmp)/'historical-native'))
            failed_root=Path(tmp)/plan.RELATIVE_ROOT;failed_root.mkdir(parents=True)
            state=failed_root/'download-state.json';state.write_text(json.dumps(failed))
            new['recovery']['failed_payload_receipt_sha256']=plan.file_sha(state)
            output=Path(tmp)/new['output_root_relative']
            root,family=download.acquisition_paths(new,deployment,output,'payload')
            self.assertFalse(root.exists())
            legacy_family=download.acquisition_paths(old,deployment,failed_root.parent/'fresh-header.json','headers')[1]
            self.assertEqual(family,legacy_family)
            lock=download.acquire_phase_lock(legacy_family)
            try:
                with self.assertRaises(BlockingIOError):download.acquire_phase_lock(family)
            finally:lock.close()
            with self.assertRaises(ValueError):download.acquisition_paths(new,deployment,output.parent/'headers.json','headers')
            output.mkdir(parents=True)
            with self.assertRaises(ValueError):download.acquisition_paths(new,deployment,output,'payload')
            other=copy.deepcopy(new);other['output_root_relative']=plan.FAMILY_ROOT+'/attempt-003/source-data'
            (failed_root/'unexpected-partial.png').write_bytes(b'fixture')
            with self.assertRaises(ValueError):download.acquisition_paths(other,deployment,Path(tmp)/other['output_root_relative'],'payload')
            self.assertEqual(state.read_text(),json.dumps(failed))

    def test_all_HTTP_starts_including_redirects_paced_and_429_stops_without_body_or_retry(self):
        clock=[0.];starts=[];responses=[]
        def response(status,headers,url,body=b''):
            r=mock.MagicMock();r.__enter__.return_value=r;r.status_code=status;r.headers=headers;r.url=url
            r.raw=mock.Mock(wraps=io.BytesIO(body));responses.append(r);return r
        def get(url,**kwargs):
            starts.append(clock[0]);self.assertFalse(kwargs['allow_redirects'])
            if len(starts)==1:return response(302,{'Location':'https://fixture.invalid/object?private=secret'},url)
            if len(starts)==2:return response(206,{'Content-Range':'bytes 512-1023/4096','Content-Length':'512'},url,b'x'*512)
            return response(429,{'Content-Length':'317','Retry-After':'120','Date':'Thu, 29 Oct 2026 04:00:00 GMT',
                'RateLimit-Remaining':'0','X-RateLimit-Reset':'https://fixture.invalid/?secret'},url)
        session=mock.Mock();session.get.side_effect=get
        source={'rgb':{'parts':[dict(size=4096,url='https://huggingface.co/pinned')]}}
        reader=download.ExactRangeReader(source,'https://hf-mirror.com',1536,100,minimum_interval=2,
            clock=lambda:clock[0],waiter=lambda seconds:clock.__setitem__(0,clock[0]+seconds),session_factory=lambda:session)
        # read's physical deadline uses monotonic; align it with the synthetic clock.
        with mock.patch.object(download.time,'monotonic',side_effect=lambda:clock[0]):
            self.assertEqual(reader.read('rgb',512,512),b'x'*512)
            with self.assertRaises(download.RangeResponseError) as caught:reader.read('rgb',1024,512)
            with self.assertRaises(TimeoutError):reader.read('rgb',1536,512)
        self.assertEqual(starts,[0.,2.,4.]);self.assertEqual(session.get.call_count,3)
        responses[0].raw.read.assert_not_called();responses[-1].raw.read.assert_not_called()
        self.assertEqual(reader.returned_bytes,512);self.assertEqual(reader.redirect_requests,1)
        detail=caught.exception.diagnostic['rate_limit_diagnostics']
        self.assertEqual(detail['retry_after'],'120');self.assertEqual(detail['numeric_fields']['RateLimit-Remaining'],'0')
        self.assertEqual(detail['numeric_fields']['X-RateLimit-Reset'],'<nonstandard-redacted>')
        self.assertNotIn('secret',json.dumps(caught.exception.diagnostic))
        self.assertNotIn('https://',json.dumps(caught.exception.diagnostic))

    def test_redirect_limit_HTTPS_only_and_pacing_deadline_without_retry(self):
        source={'rgb':{'parts':[dict(size=4096,url='https://huggingface.co/pinned')]}}
        for location,expected_calls in [('http://fixture.invalid/insecure',1),('https://fixture.invalid/loop',4)]:
            r=mock.MagicMock();r.__enter__.return_value=r;r.status_code=302;r.headers={'Location':location}
            session=mock.Mock();session.get.return_value=r
            reader=download.ExactRangeReader(source,'https://hf-mirror.com',512,time.monotonic()+10,session_factory=lambda:session)
            with self.assertRaises(ValueError):reader.read('rgb',512,512)
            self.assertEqual(session.get.call_count,expected_calls);r.raw.read.assert_not_called()
        clock=[0.];session=mock.Mock()
        reader=download.ExactRangeReader(source,'https://hf-mirror.com',512,1,session_factory=lambda:session,
            minimum_interval=2,clock=lambda:clock[0]);reader.last_request_start=0.
        reader.local.session=session
        with self.assertRaises(TimeoutError):reader.request('https://fixture.invalid',{})
        session.get.assert_not_called()


    def test_standard_archive_capsule_acceptance_and_extracted_tree_fallback(self):
        controls=plan.current_controls();payloads={name:(plan.REPOSITORY/name).read_bytes() for name in controls}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,registry=archive_capsule(root,payloads)
            before={p.name for p in root.iterdir()}
            self.assertEqual(plan.verify_planning_registry(path,controls),registry)
            self.assertEqual({p.name for p in root.iterdir()},before)  # no extraction
            (root/'source.tar.gz').unlink();(root/'overlay.tar.gz').unlink()
            for name,raw in payloads.items():
                target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
            self.assertEqual(plan.verify_planning_registry(path,controls),registry)
            (root/next(iter(payloads))).write_bytes(b'changed')
            with self.assertRaises(ValueError):plan.verify_planning_registry(path,controls)

    def test_archive_capsule_rejects_archive_source_member_path_type_and_control_changes(self):
        payloads={'scripts/control.py':b'control','scripts/other.py':b'other'}
        controls={'scripts/control.py':plan.hashlib.sha256(b'control').hexdigest()}
        base=[(name,raw,tarfile.REGTYPE) for name,raw in payloads.items()]
        variants=[base+base[:1],base[:1],base+[('extra.py',b'x',tarfile.REGTYPE)],
            [('scripts/control.py',b'changed',tarfile.REGTYPE)]+base[1:],
            [('scripts/control.py',b'',tarfile.SYMTYPE)]+base[1:],
            [('../scripts/control.py',b'control',tarfile.REGTYPE)]+base[1:]]
        with tempfile.TemporaryDirectory() as tmp:
            for index,entries in enumerate(variants):
                path,_=archive_capsule(Path(tmp)/str(index),payloads,entries=entries)
                with self.subTest(index=index):
                    with self.assertRaises(ValueError):plan.verify_planning_registry(path,controls)
            root=Path(tmp)/'valid';path,registry=archive_capsule(root,payloads)
            with self.assertRaisesRegex(ValueError,'controls'):plan.verify_planning_registry(path,{'scripts/control.py':'0'*64})
            with (root/'source.tar.gz').open('ab') as stream:stream.write(b'tampered')
            with self.assertRaisesRegex(ValueError,'archive bytes'):plan.verify_planning_registry(path,controls)
            path,registry=archive_capsule(root,payloads)
            registry['source_identity_sha256']='0'*64;path.write_text(json.dumps(registry))
            with self.assertRaisesRegex(ValueError,'registry identity'):plan.verify_planning_registry(path,controls)
            path,registry=archive_capsule(root,payloads)
            registry['overlay_files']['scripts/control.py']='0'*64;path.write_text(json.dumps(registry))
            with self.assertRaisesRegex(ValueError,'overlay registry'):plan.verify_planning_registry(path,controls)

    def test_approve_positive_workflow_with_standard_local_capsule_no_extraction(self):
        # Metadata-only fixture approval, never an actual acquisition registration.
        old,_,header,failed=recovery_fixture();controls=plan.current_controls()
        payloads={name:(plan.REPOSITORY/name).read_bytes() for name in controls}
        with tempfile.TemporaryDirectory(dir=plan.REPOSITORY/'scratch',prefix='acquisition-metadata-test-') as tmp:
            root=Path(tmp);registry_path,registry=archive_capsule(root/'capsule',payloads)
            old_path=root/'old.json';old_path.write_text(json.dumps(old))
            header['manifest_sha256']=failed['manifest_sha256']=plan.file_sha(old_path)
            header_path=root/'header.json';header_path.write_text(json.dumps(header))
            failed_path=root/'failed.json';failed_path.write_text(json.dumps(failed))
            output=root/'approved-fixture.json';attestation_path=root/'attestation-fixture.json'
            with contextlib.redirect_stdout(io.StringIO()):
                plan.recovery_main(['--recovery-from',str(old_path),'--old-header-receipt',str(header_path),
                    '--failed-payload-receipt',str(failed_path),'--attempt-id','attempt-002','--output',str(output),
                    '--attestation-output',str(attestation_path),'--approve','--executing-source-registry',str(registry_path)])
            new=json.loads(output.read_text());attestation=json.loads(attestation_path.read_text())
            self.assertTrue(new['registered']);self.assertTrue(attestation['registered'])
            self.assertEqual(attestation['new_manifest_sha256'],plan.file_sha(output))
            self.assertEqual(attestation['executing_source_registry_sha256'],plan.file_sha(registry_path))
            download.verify_header_reuse(new,plan.file_sha(output),header,plan.file_sha(header_path),old,
                plan.file_sha(old_path),failed,plan.file_sha(failed_path),attestation,registry,plan.file_sha(registry_path),'https://hf-mirror.com')
            self.assertFalse((root/'capsule'/'projects').exists());self.assertFalse((root/new['output_root_relative']).exists())


if __name__=='__main__':unittest.main()
