"""Synthetic sparse HDF5 headers and mocked HTTP ranges only; no real pixels."""
import importlib.util
import io
from pathlib import Path
import tempfile
import types
import unittest

SCRIPT=Path(__file__).resolve().parents[1]/'experiments/stereo-spiking-transformer-study/scripts/audit_source_hdf5_metadata.py'
spec=importlib.util.spec_from_file_location('source_metadata_audit',SCRIPT)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class MetadataAuditTests(unittest.TestCase):
    def test_selected_identity_population_and_path_rejection(self):
        def source(i):
            return dict(repository='public/pinned',revision='a'*40,path=f'individual_quantities/test/{i}_{i+127}.h5',
                        size=100000,lfs_sha256='b'*64,left_dataset='frame_cleanpass_left',right_dataset='frame_cleanpass_right',file_index=3)
        scenes=[dict(scene_id=f'TEST/C/{i:04}',split='test',frames=[dict(rgb_hdf5=source(i))]) for i in range(12)]
        self.assertEqual(len(audit.selected_shards(dict(scenes=scenes))),12)
        scenes[0]['frames'][0]['rgb_hdf5']['path']='../private.h5'
        with self.assertRaisesRegex(ValueError,'preselected'):audit.selected_shards(dict(scenes=scenes))

    def test_bounded_exact_ranges_and_rejected_unhonored_response(self):
        class Response:
            status_code=206
            headers={'Content-Range':'bytes 2-5/10'}
            raw=io.BytesIO(b'cdef');url='https://pinned/object'
            def __enter__(self):return self
            def __exit__(self,*_):pass
        response=Response();calls=[]
        def get(*args,**kwargs):calls.append(kwargs);return response
        source=dict(size=10,repository='public/pinned',revision='a'*40,path='object')
        remote=audit.MetadataRanges(source,types.SimpleNamespace(get=get),max_requests=1)
        remote.seek(2);self.assertEqual(remote.read(4),b'cdef');self.assertEqual(calls[0]['headers']['Range'],'bytes=2-5')
        with self.assertRaisesRegex(RuntimeError,'bound'):remote.read(1)
        with self.assertRaisesRegex(RuntimeError,'Unbounded'):remote.read()
        with self.assertRaises(ValueError):remote.seek(-1)
        response.status_code=200
        remote=audit.MetadataRanges(source,types.SimpleNamespace(get=get))
        with self.assertRaisesRegex(RuntimeError,'not honored'):remote.read(4)

    def test_sparse_headers_without_values_and_guard_restoration(self):
        import h5py
        source=dict(left_dataset='frame_cleanpass_left',right_dataset='frame_cleanpass_right',selected_indices=[0,127])
        with tempfile.TemporaryDirectory() as raw:
            path=Path(raw)/'synthetic-sparse.h5'
            with h5py.File(path,'w') as f:
                for name in ('frame_cleanpass_left','frame_cleanpass_right'):
                    f.create_dataset(name,shape=(128,540,960,3),dtype='u1',chunks=(1,540,960,3),compression='gzip')
                tiny=f.create_dataset('tiny',shape=(1,),dtype='u1')
                self.assertEqual(len(audit.inspect_rgb_metadata(f,source,h5py)),2)
                with audit.forbid_dataset_values(h5py):
                    with self.assertRaisesRegex(RuntimeError,'value access'):tiny[0]
                self.assertEqual(int(tiny[0]),0)
                source['selected_indices']=[128]
                with self.assertRaisesRegex(ValueError,'index bounds'):audit.inspect_rgb_metadata(f,source,h5py)
            self.assertLess(path.stat().st_size,32768)


if __name__=='__main__':unittest.main()
