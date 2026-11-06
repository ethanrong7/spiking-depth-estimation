"""Tiny source-byte/metadata fixtures only; no real pixels, simulator, model or GPU."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

S=Path(__file__).resolve().parents[1]/'experiments/stereo-spiking-transformer-study/scripts'
sys.path.insert(0,str(S))
import multiroot_source_adapter as M


def fixture(root):
    aliases=M.ALIASES;roots={aliases[0]:str(root/'old'),aliases[1]:str(root/'new')};receipts={a:{} for a in aliases}
    index=dict(schema='stereo-study-multiroot-source-index-v1',test=[],test_values_opened=False,sensor_generation_performed=False,
        training_compatible=False,legacy_split_mapping={'dev':'val'},root_aliases={a:{} for a in aliases},sequences=[])
    for sid,split,alias in [('TRAIN/C/0001','train',aliases[0]),('TRAIN/C/0002','val',aliases[0]),('TRAIN/C/0003','train',aliases[1])]:
        centers=list(range(1,9)) if split=='train' else [3,6]
        row=dict(source_id=sid,split=split,legacy_source_split='dev' if split=='val' else 'train',original_split='TRAIN',
            root_alias=alias,historical_exposure=alias==aliases[0],reference_centers=centers,frames=[])
        for i in range(10):
            stem=f'{i+6:04d}';f=dict(index=i,source_frame=stem,rgb_l=f'frames_cleanpass/{sid}/left/{stem}.png',
                rgb_r=f'frames_cleanpass/{sid}/right/{stem}.png',disparity_l=f'disparity/{sid}/left/{stem}.pfm' if i in centers else None)
            for field in ('rgb_l','rgb_r','disparity_l'):
                if f[field] is None:continue
                raw=(sid+':'+field+':'+stem).encode();p=Path(roots[alias])/f[field];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
                receipts[alias][f[field]]=dict(role='gt' if field=='disparity_l' else 'rgb',source_id=sid,size=len(raw),sha256=hashlib.sha256(raw).hexdigest())
            row['frames'].append(f)
        index['sequences'].append(row)
    return index,roots,receipts


class SourceAdapterTests(unittest.TestCase):
    def test_both_roots_sparse_val_targets_and_verified_reader(self):
        with tempfile.TemporaryDirectory() as d:
            index,roots,receipts=fixture(Path(d));c=M.bind_catalog(index,roots,receipts);r=M.NativeSourceReader(c)
            self.assertEqual(c['counts']['files'],78);self.assertEqual(len(r.targets('train')),16);self.assertEqual(len(r.targets('val')),2)
            self.assertEqual([t['center'] for t in r.targets('val')],[3,6])
            self.assertEqual(r.read_bytes('TRAIN/C/0003','rgb','right',9),b'TRAIN/C/0003:rgb_r:0015')
            self.assertTrue(r.path('TRAIN/C/0002','gt','left',3).is_relative_to(Path(roots['historical_native'])))
            self.assertFalse(c['training_compatible']);self.assertFalse(c['registered_for_generation'])

    def test_no_test_leakage_scene_duplication_or_split_reassignment(self):
        with tempfile.TemporaryDirectory() as d:
            index,roots,receipts=fixture(Path(d))
            for mode in ('test','duplicate','dev','new_val','exposure'):
                value=copy.deepcopy(index)
                if mode=='test':value['sequences'][0]['source_id']='TEST/C/0001'
                if mode=='duplicate':value['sequences'].append(copy.deepcopy(value['sequences'][0]))
                if mode=='dev':value['sequences'][1]['legacy_source_split']='train'
                if mode=='new_val':value['sequences'][2].update(split='val',legacy_source_split='dev',reference_centers=[3,6])
                if mode=='exposure':value['sequences'][2]['historical_exposure']=True
                with self.assertRaises(ValueError):M.bind_catalog(value,roots,receipts)

    def test_missing_receipts_views_centers_and_unacquired_gt_refused(self):
        with tempfile.TemporaryDirectory() as d:
            index,roots,receipts=fixture(Path(d))
            for mode in ('missing','view','center','extra_gt','traversal'):
                value=copy.deepcopy(index);rec=copy.deepcopy(receipts)
                if mode=='missing':rec['new_train_attempt002'].pop(value['sequences'][2]['frames'][4]['disparity_l'])
                if mode=='view':value['sequences'][0]['frames'][0]['rgb_l']=value['sequences'][0]['frames'][0]['rgb_r']
                if mode=='center':value['sequences'][1]['reference_centers']=[1,3,6]
                if mode=='extra_gt':value['sequences'][1]['frames'][1]['disparity_l']='disparity/TRAIN/C/0002/left/0007.pfm'
                if mode=='traversal':value['sequences'][0]['frames'][0]['rgb_l']='../escape'
                with self.assertRaises(ValueError):M.bind_catalog(value,roots,rec)
            r=M.NativeSourceReader(M.bind_catalog(index,roots,receipts))
            for args in [('TRAIN/C/0002','gt','left',1),('TRAIN/C/0001','rgb','right',10),('TEST/C/0001','rgb','left',0)]:
                with self.assertRaises(ValueError):r.member(*args)
            with self.assertRaises(ValueError):r.targets('test')

    def test_catalog_cannot_redirect_source_members_or_target_roles(self):
        with tempfile.TemporaryDirectory() as d:
            index,roots,receipts=fixture(Path(d));c=M.bind_catalog(index,roots,receipts)
            for mode in ('member','target','center','source','frame'):
                value=copy.deepcopy(c)
                if mode=='member':value['sequences'][0]['files'][0]['path']=value['sequences'][1]['files'][0]['path']
                if mode=='target':value['targets'][0]['split']='val'
                if mode=='center':value['sequences'][1]['reference_centers']=[1,3,6]
                if mode=='source':value['sequences'][2]['root_alias']='historical_native'
                if mode=='frame':value['sequences'][0]['frames'][0]['rgb_l']='different'
                with self.assertRaises(ValueError):M.NativeSourceReader(value)

    def test_missing_or_changed_payload_has_no_other_root_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            index,roots,receipts=fixture(Path(d));r=M.NativeSourceReader(M.bind_catalog(index,roots,receipts))
            p=r.path('TRAIN/C/0003','rgb','left',0);raw=p.read_bytes();other=Path(roots['historical_native'])/r.member('TRAIN/C/0003','rgb','left',0)['path']
            other.parent.mkdir(parents=True,exist_ok=True);other.write_bytes(raw)
            p.write_bytes(b'x'*len(raw))
            with self.assertRaisesRegex(ValueError,'bytes changed'):r.read_bytes('TRAIN/C/0003','rgb','left',0)
            p.unlink()
            with self.assertRaisesRegex(ValueError,'missing'):r.read_bytes('TRAIN/C/0003','rgb','left',0)

    def test_symlinks_and_overlapping_roots_refused(self):
        with tempfile.TemporaryDirectory() as d:
            index,roots,receipts=fixture(Path(d));r=M.NativeSourceReader(M.bind_catalog(index,roots,receipts));p=r.path('TRAIN/C/0003','rgb','left',0)
            raw=p.read_bytes();outside=Path(d)/'outside';outside.write_bytes(raw);p.unlink();p.symlink_to(outside)
            with self.assertRaisesRegex(ValueError,'Symlinked'):r.read_bytes('TRAIN/C/0003','rgb','left',0)
            bad={**roots,'new_train_attempt002':roots['historical_native']+'/nested'}
            with self.assertRaises(ValueError):M.bind_catalog(index,bad,receipts)

if __name__=='__main__':unittest.main()
