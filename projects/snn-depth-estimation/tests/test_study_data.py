"""Tiny CPU contract checks; no model training or benchmark evaluation."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.study_data import (LegacyNativeEngineeringDataset, PairedStereoDataset,
                             read_packed_spikes, read_pfm, safe_path, validate_paired_manifest)


def fixture():
    rows = {}
    for split in ('train', 'val', 'test'):
        rows[split] = [dict(sample_id=split+'-1', scene_id=split+'-scene',
            target_time=1., bin_edges=np.linspace(0, 1, 51).tolist(), source_support_end=1.,
            event_channels=['ON', 'OFF'], spike='spike.npz', event='event.npz', disparity='disp.npy')]
    return dict(schema='stereo-paired-v1', T=50, disparity='positive_left_pixels', rectified=True,
                time_unit='source_frame_interval', source_identity='fixture-only', license='fixture',
                simulators={m:dict(implementation='fixture', version='v1', configuration={'test':True})
                            for m in ('spike', 'event')}, **rows)


class StudyDataTests(unittest.TestCase):
    def test_scene_leakage_and_causality(self):
        m = fixture(); self.assertTrue(validate_paired_manifest(m))
        for change in ('scene', 'bins', 'future', 'polarity', 'seconds'):
            broken = copy.deepcopy(m)
            if change == 'scene': broken['test'][0]['scene_id'] = 'train-scene'
            if change == 'bins': broken['train'][0]['bin_edges'] = list(range(50))
            if change == 'future': broken['train'][0]['source_support_end'] = 1.02
            if change == 'polarity': broken['train'][0]['event_channels'] = ['signed_sum']
            if change == 'seconds': broken['time_unit'] = 'seconds'
            with self.assertRaises(ValueError, msg=change): validate_paired_manifest(broken)

    def test_codec_exact_temporal_order_and_pfm_orientation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spikes = np.zeros((50,1,2,4), dtype=np.uint8)
            spikes[0,0,0,0] = 1; spikes[49,0,1,3] = 1
            (root/'s.dat').write_bytes(np.packbits(spikes.reshape(-1), bitorder='little').tobytes())
            np.testing.assert_array_equal(read_packed_spikes(root/'s.dat', (2,4)), spikes)
            (root/'bad.dat').write_bytes(b'\x00')
            with self.assertRaises(ValueError): read_packed_spikes(root/'bad.dat', (2,4))
            target = np.arange(1,9,dtype=np.float32).reshape(2,4)
            (root/'d.pfm').write_bytes(b'Pf \n4 2\n-1\n' + np.flipud(target).astype('<f4').tobytes())
            np.testing.assert_array_equal(read_pfm(root/'d.pfm',(2,4)),target)

    def test_polarity_counts_positive_target_and_test_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest_path=root/'manifest.json'
            m=fixture(); manifest_path.write_text(json.dumps(m))
            events=np.zeros((50,2,2,4),dtype=np.uint16)
            events[0,0,0,0]=3; events[0,1,0,0]=2
            np.savez(root/'event.npz',left=events,right=events)
            np.save(root/'disp.npy',np.array([[2,np.nan,0,-1],[1,3,4,5]],dtype=np.float32))
            item=PairedStereoDataset(root,manifest_path,'event','train')[0]
            self.assertEqual(tuple(item['left'].shape),(50,2,2,4))
            self.assertEqual(item['left'][0,0,0,0].item(),3)
            self.assertEqual(item['left'][0,1,0,0].item(),2)
            self.assertEqual(item['target'][0,0,0].item(),2)
            self.assertEqual(item['valid'].sum().item(),5)
            # Fail before reading any Test asset.
            with self.assertRaises(ValueError): PairedStereoDataset(root,manifest_path,'event','test')
            release=dict(manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                         frozen_registration_sha256='fixture',test_access_authorized=True)
            self.assertEqual(len(PairedStereoDataset(root,manifest_path,'event','test',test_release=release)),1)
            with self.assertRaises(ValueError): LegacyNativeEngineeringDataset(root,root,purpose='science')

    def test_engineering_pilot_has_no_fake_test(self):
        m=fixture();m['schema']='stereo-paired-engineering-pilot-v1'
        m['evidence_scope']='engineering_only';m['test']=[]
        with self.assertRaises(ValueError):validate_paired_manifest(m)
        self.assertTrue(validate_paired_manifest(m,allow_engineering_pilot=True))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'manifest.json';p.write_text(json.dumps(m))
            self.assertEqual(len(PairedStereoDataset(root,p,'spike','train',purpose='engineering')),1)
            with self.assertRaises(ValueError):PairedStereoDataset(root,p,'spike','test',purpose='engineering')
        m['test']=fixture()['test']
        with self.assertRaises(ValueError):validate_paired_manifest(m,allow_engineering_pilot=True)

    def test_paths_and_binary_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for bad in ('../escaped','/absolute'):
                with self.assertRaises(ValueError): safe_path(root,bad)
            m=fixture(); p=root/'manifest.json'; p.write_text(json.dumps(m))
            spikes=np.ones((50,1,2,4),dtype=np.uint8)*2
            np.savez(root/'spike.npz',left=spikes,right=spikes)
            np.save(root/'disp.npy',np.ones((2,4),dtype=np.float32))
            with self.assertRaises(ValueError): PairedStereoDataset(root,p,'spike','train')[0]

    def test_explicit_subset_preserves_registered_order_and_split_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); p = root/'manifest.json'; m = fixture()
            second = copy.deepcopy(m['train'][0]); second['sample_id'] = 'train-2'
            m['train'].append(second); p.write_text(json.dumps(m))
            dataset = PairedStereoDataset(root,p,'spike','train',sample_ids=['train-2','train-1'])
            self.assertEqual([r['sample_id'] for r in dataset.rows], ['train-2','train-1'])
            for ids in ([], ['train-1','train-1'], ['val-1']):
                with self.assertRaises(ValueError):
                    PairedStereoDataset(root,p,'spike','train',sample_ids=ids)
            with self.assertRaisesRegex(ValueError, 'Test remains sealed'):
                PairedStereoDataset(root,p,'spike','test',sample_ids=['test-1'])


if __name__ == '__main__': unittest.main()
