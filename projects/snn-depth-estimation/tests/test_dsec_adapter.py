"""Tiny synthetic geometry/HDF5/count fixtures only; no real-data experiment."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest import mock

import numpy as np

SCRIPT=Path(__file__).resolve().parents[1]/'experiments/stereo-spiking-transformer-study/scripts/dsec_adapter.py'
spec=importlib.util.spec_from_file_location('dsec_adapter',SCRIPT)
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)


class DSECAdapterTests(unittest.TestCase):
    def test_causal_bin_endpoints_polarity_rectification_tie_and_conservation(self):
        yy,xx=np.mgrid[:8,:16]
        mapping=np.stack((xx,yy),axis=-1).astype(np.float32)
        mapping[0,1]=[3.5,0] # exact quarter-pixel tie deposits to higher index1
        mapping[0,2]=[np.nan,0];mapping[0,3]=[-.1,0]
        target=2**35+100000;start=target-50000
        events=dict(x=np.array([0,0,1,1,20,0,2,3,0,0]),y=np.zeros(10,dtype=int),
            p=np.array([1,1,0,1,1,2,1,1,1,1]),
            t=np.array([start,start+999,start+1000,target-1,start,start,start,start,target,start-1],dtype=np.int64))
        counts,stats=adapter.bin_rectified_counts(events,mapping,target)
        self.assertEqual(counts.shape,(50,2,2,4));self.assertEqual(counts.dtype,np.int64)
        self.assertEqual(counts[0,0,0,0],2);self.assertEqual(counts[1,1,0,1],1)
        self.assertEqual(counts[49,0,0,1],1);self.assertEqual(int(counts.sum()),4)
        self.assertEqual(stats,dict(input_events=10,time_outside=2,bad_raw_coordinate=1,bad_polarity=1,
            nonfinite_rectification=1,outside_rectified_support=1,accepted_events=4,on_events=3,off_events=1))
        for event in (dict(events,t=events['t'].astype(np.float32)),dict(events,p=np.array([0]))):
            with self.assertRaises(ValueError):adapter.bin_rectified_counts(event,mapping,target)

    @unittest.skipUnless(importlib.util.find_spec('h5py'),'task runtime h5py needed for synthetic HDF5 fixture')
    def test_HDF5_offset_duplicate_boundaries_and_chunked_reads(self):
        import h5py
        local=np.array([0,999,1000,49999,50000,50000,50001,100000],dtype=np.uint32)
        offset=2**35
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tiny.h5'
            with h5py.File(path,'w') as stream:
                for key,value in dict(t=local,x=np.arange(len(local),dtype=np.uint16),y=np.zeros(len(local),dtype=np.uint16),p=np.ones(len(local),dtype=np.uint8)).items():
                    stream.create_dataset('events/'+key,data=value)
                stream.create_dataset('t_offset',data=np.int64(offset))
                stream.create_dataset('ms_to_idx',data=np.searchsorted(local,np.arange(102,dtype=np.int64)*1000).astype(np.uint64))
            metadata=adapter.h5_event_metadata(path)
            self.assertEqual(metadata['first_global_us'],offset)
            self.assertEqual(metadata['last_global_us'],offset+100000)
            with adapter.H5EventWindow(path) as reader:
                early=list(reader.iter_window(offset+50000,chunk_events=2))
                late=list(reader.iter_window(offset+100000,chunk_events=2))
                np.testing.assert_array_equal(np.concatenate([x['x'] for x in early]),[0,1,2,3])
                np.testing.assert_array_equal(np.concatenate([x['x'] for x in late]),[4,5,6])
                self.assertTrue(all(len(x['t'])<=2 for x in early+late))
                np.testing.assert_array_equal(np.concatenate([x['t'] for x in late]),offset+local[[4,5,6]].astype(np.int64))
                self.assertEqual([r['index'] for r in reader.boundary_receipts],[0,4,4,7])
                self.assertTrue(all(r['lower_bound_verified'] for r in reader.boundary_receipts))
                with self.assertRaises(ValueError):reader.lower_bound(float(offset))
            with h5py.File(path,'r+') as stream:
                stream['ms_to_idx'][50]=7
            with adapter.H5EventWindow(path) as reader:
                with self.assertRaisesRegex(ValueError,'neighboring timestamps'):
                    reader.lower_bound(offset+50000)

    def test_prepared_sensor_readback_preserves_each_bin_and_rejects_lossy_casts(self):
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(adapter,'OUTPUT_SHAPE',(2,4)):
            root=Path(tmp);asset=root/'sensor.npz'
            sensors={view:np.zeros((50,2,2,4),dtype=np.uint32) for view in ('left','right')}
            sensors['left'][0,0,0,1]=301;sensors['right'][49,1,1,3]=19
            def write():
                np.savez_compressed(asset,**sensors)
                return dict(scene_id=adapter.SCENES[0],original_split='train',sensor_asset=asset.name,
                    sensor_sha256=adapter.file_sha(asset),count_statistics={view:dict(
                        per_bin_ON_OFF_counts=value.sum(axis=(2,3),dtype=np.uint64).tolist(),
                        accepted_events=int(value.sum(dtype=np.uint64))) for view,value in sensors.items()})
            row=write()
            report=adapter.verify_sensor_readback(root,row,sensors)
            self.assertTrue(report['float32_count_cast_exact']);self.assertFalse(report['ground_truth_opened'])
            row['count_statistics']['left']['per_bin_ON_OFF_counts'][0][0]+=1
            with self.assertRaisesRegex(ValueError,'per-bin'):
                adapter.verify_sensor_readback(root,row,sensors)
            sensors['left'][0,0,0,0]=2**24+1;row=write()
            with self.assertRaisesRegex(ValueError,'exact FP32'):
                adapter.verify_sensor_readback(root,row,sensors)

    def test_integral_storage_and_native_GT_mask_keep_every_positive_label(self):
        counts=np.array([0,300,2**32-1],dtype=np.int64)
        stored=adapter.checked_uint32(counts)
        np.testing.assert_array_equal(stored.astype(np.int64),counts)
        with self.assertRaises(ValueError):adapter.checked_uint32(np.array([2**32],dtype=np.int64))
        with self.assertRaises(ValueError):adapter.checked_uint32(np.array([-1],dtype=np.int64))
        encoded=np.array([[0,256],[257,65535]],dtype=np.uint16)
        target,valid=adapter.decode_uint16_gt(encoded)
        np.testing.assert_array_equal(valid,[[[False,True],[True,True]]])
        np.testing.assert_array_equal(target,[[[0,1],[257/256,65535/256]]])
        self.assertEqual(target.dtype,np.float32)
        with self.assertRaises(ValueError):adapter.decode_uint16_gt(encoded.astype(np.float32))

    @unittest.skipUnless(importlib.util.find_spec('torch'),'torch needed for tiny interpolation geometry')
    def test_native_disparity_horizontal_units_fractional_shift_and_no_clamp(self):
        import torch
        # A two-output-pixel right shift reduces disparity by8 native pixels.
        correct=adapter.predict_to_native(torch.full((1,1,2,4),3.),(8,16))
        shifted=adapter.predict_to_native(torch.full((1,1,2,4),1.),(8,16))
        torch.testing.assert_close(correct-shifted,torch.full((1,1,8,16),8.))
        fractional=adapter.predict_to_native(torch.full((1,1,2,4),.125),(8,16))
        torch.testing.assert_close(fractional,torch.full((1,1,8,16),.5))
        raw=adapter.predict_to_native(torch.full((1,1,2,4),-1.),(8,16))
        self.assertEqual(raw.max().item(),-4.)
        # Bilinear pixel-center mapping: native interior x5 samples q=.875;
        # q+.5 quarter disparity becomes native5.5, without corner alignment.
        ramp=torch.arange(4,dtype=torch.float32)[None,None,None,:].expand(1,1,2,4)+.5
        self.assertEqual(adapter.predict_to_native(ramp,(8,16))[0,0,3,5].item(),5.5)

    def test_metadata_selection_uses_only_complete_two_view_time_spans(self):
        times=np.arange(12,dtype=np.int64)*100000
        names=[str(i).zfill(6)+'.png' for i in range(12)]
        views=dict(left=dict(first_global_us=1,last_global_us=1100000),right=dict(first_global_us=2000,last_global_us=1099999))
        rows,n=adapter.uniform_eligible_targets(times,names,views)
        self.assertEqual(n,10);self.assertEqual([r['timestamp_ordinal'] for r in rows],list(range(1,11)))
        self.assertEqual(rows[0]['filename'],'000001.png')
        self.assertEqual(rows[-1]['target_us'],1000000)
        with self.assertRaises(ValueError):adapter.uniform_eligible_targets(times[:5],names[:5],views)
        broken=times.copy();broken[2]=broken[1]
        with self.assertRaises(ValueError):adapter.uniform_eligible_targets(broken,names,views)

    def test_prospective_configuration_and_no_real_GT_or_conversion_onMac(self):
        config=adapter.example_config();self.assertTrue(adapter.validate_config(config))
        self.assertFalse(config['registered'])
        with self.assertRaises(ValueError):adapter.validate_config(config,execution=True)
        for key,value in [('window_us',100000),('T',15),('scenes',['unverified']),('output_shape',[135,240])]:
            broken=copy.deepcopy(config);broken[key]=value
            with self.assertRaises(ValueError):adapter.validate_config(broken)
        with mock.patch.dict(adapter.os.environ,{},clear=True):
            with self.assertRaises(RuntimeError):adapter.load_native_gt(Path('/unused'),{})
        with self.assertRaises(ValueError):adapter.safe('/source','test/scene/file')

    @unittest.skipUnless(importlib.util.find_spec('h5py') and importlib.util.find_spec('yaml'),'tiny metadata fixture needs h5py/PyYAML')
    def test_preflight_freezes_IDs_without_GT_decoding_and_reports_missing_map(self):
        import h5py
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            provenance=[]
            for scene in adapter.SCENES:
                downloaded=root/'download'/scene;downloaded.mkdir(parents=True)
                times=np.arange(12,dtype=np.int64)*100000
                np.savetxt(downloaded/(scene+'_disparity_timestamps.txt'),times,fmt='%d')
                with zipfile.ZipFile(downloaded/(scene+'_disparity_event.zip'),'w') as archive:
                    for index in range(12):archive.writestr(str(index*2).zfill(6)+'.png',b'NOT_A_PNG;METADATA_MUST_NOT_DECODE')
                intrinsics={}
                for camera,view in ((0,'left'),(3,'right')):
                    intrinsics['cam'+str(camera)]=dict(camera_type='event',camera_location=view,resolution=[640,480])
                    intrinsics['camRect'+str(camera)]=dict(is_rectified=True,resolution=[640,480],camera_matrix=[500.,500.,320.,240.])
                matrix=np.eye(4);matrix[3,2]=2.
                with zipfile.ZipFile(downloaded/(scene+'_calibration.zip'),'w') as archive:
                    archive.writestr('cam_to_cam.yaml',json.dumps(dict(intrinsics=intrinsics,disparity_to_depth={'cams_03':matrix.tolist()})))
                    archive.writestr('cam_to_lidar.yaml','{}')
                for view in ('left','right'):
                    folder=root/'train'/scene/'events'/view;folder.mkdir(parents=True)
                    with h5py.File(folder/'events.h5','w') as stream:
                        for key,values in dict(t=np.array([0,2000000],dtype=np.uint32),x=[0,0],y=[0,0],p=[1,1]).items():
                            stream.create_dataset('events/'+key,data=values)
                        stream.create_dataset('t_offset',data=np.int64(0));stream.create_dataset('ms_to_idx',shape=(2001,),dtype=np.uint64)
                    # Unallocated map dataset: shape/header inspected; no real map values.
                    with h5py.File(folder/'rectify_map.h5','w') as stream:
                        stream.create_dataset('rectify_map',shape=(480,640,2),dtype=np.float32)
                    with zipfile.ZipFile(downloaded/(scene+'_events_'+view+'.zip'),'w') as archive:
                        for name in ('events.h5','rectify_map.h5'):archive.write(folder/name,arcname=name)
                for suffix in ('events_left.zip','events_right.zip','disparity_event.zip','calibration.zip','disparity_timestamps.txt'):
                    path=downloaded/(scene+'_'+suffix)
                    provenance.append(dict(sequence=scene,suffix=suffix,size_bytes=path.stat().st_size,sha256=adapter.file_sha(path),
                        url='https://download.ifi.uzh.ch/rpg/DSEC/train/'+scene+'/'+path.name))
            (root/'provenance.json').write_text(json.dumps({'files':provenance}))
            original=zipfile.ZipFile.read
            def deny_GT(archive,name,*args,**kwargs):
                self.assertFalse(str(name).endswith('.png'))
                return original(archive,name,*args,**kwargs)
            with mock.patch.object(zipfile.ZipFile,'read',deny_GT),mock.patch.object(adapter,'H5EventWindow',side_effect=AssertionError('no conversion')):
                report=adapter.preflight(adapter.example_config(),root)
            self.assertEqual(report['status'],'PREFLIGHT_READY_METADATA_ONLY')
            self.assertEqual(len(report['targets']),20)
            self.assertFalse(report['gt_pixels_opened']);self.assertFalse(report['event_conversion'])
            self.assertTrue(all(row['timestamp_ordinal']>0 for row in report['targets']))
            self.assertTrue(all(len(row['native_gt']['archive_sha256'])==64 for row in report['targets']))
            (root/'train'/adapter.SCENES[-1]/'events/right/rectify_map.h5').unlink()
            missing=adapter.preflight(adapter.example_config(),root)
            self.assertEqual(missing['status'],'BLOCKED_METADATA_NOT_READY')
            self.assertIn('rectification map unavailable',missing['failures'][0]['error'])


if __name__=='__main__':unittest.main()
