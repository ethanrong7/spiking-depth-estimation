"""Small scalar/file fixtures only; no SSH, GPUs, checkpoints or dataset decoding."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT=Path(__file__).resolve().parents[1]
SCRIPTS=PROJECT/'experiments/spiking-stereo-correspondence/scripts'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('terminal_collector',SCRIPTS/'collect_terminal_scalars.py')
collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(collector)
import compare_saved_scalars as comparison


class CollectorTest(unittest.TestCase):
    def fixture(self,root,optional=True):
        remote=root/'remote';remote.mkdir()
        metadata=dict(config=dict(stage='main',run_id=collector.P2_RUN,model='correspondence',seed=1234,updates=2500,effective_batch=8,time_steps=50),
            manifest_sha256=collector.MANIFEST_SHA,base_source=dict(source_identity_sha256=comparison.BASE_ID),
            source=dict(source_identity_sha256=collector.P2_ID),config_sha256='c'*64)
        visits=[dict(update=u) for u in (0,1000,2500)]
        docs={'metadata.json':metadata,'result.json':dict(collector.REQUIRED_RESULT,final=visits[-1]),'sample-order.json':dict(order=[])}
        if optional:docs.update({'training-summary.json':dict(updates=2500),'validation-summary.json':dict(rows=visits)})
        for name,doc in docs.items():(remote/name).write_text(json.dumps(doc)+'\n')
        (remote/'training.jsonl').write_text(''.join(json.dumps(dict(update=u,presentations=u*8))+'\n' for u in range(1,2501)))
        (remote/'validation.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in visits))
        if optional:(remote/'checkpoints.jsonl').write_text('{}\n')
        # These are inert byte fixtures; collector must never copy/open them locally.
        for name in collector.CHECKPOINTS:(remote/name).write_bytes(b'not tensor data')
        (remote/'unselected-note.txt').write_text('inventory metadata only')
        entry=dict(job_id='2598',run=collector.P2_RUN,state='COMPLETED',exit_code='0:0',allocated_seconds=17,allocated_gpus=1,
            source_identity=collector.P2_ID,source='/source/stereo-study/spiking-stereo-correspondence/main-v3',
            output='/runs/spiking-stereo-correspondence/'+collector.P2_RUN,registry_sha256='b'*64,config_sha256='c'*64,git_commit='d'*40)
        cluster=dict(run_root='/runs',source_root='/source')
        files=[dict(path=p.name,size=p.stat().st_size,sha256=collector.sha(p)) for p in sorted(remote.iterdir()) if p.name in collector.SCALARS|collector.CHECKPOINTS]
        observation=dict(files=files,all_files=[dict(path=p.name,size=p.stat().st_size) for p in sorted(remote.iterdir())],
            accounting={k:entry[k] for k in ('state','exit_code','allocated_seconds','allocated_gpus')},registry_sha256='b'*64,
            source_identity=collector.P2_ID,embedded_source_equals_registry=True)
        calls=[]
        def copy_fn(_cluster,source,destination):
            calls.append(Path(source).name);shutil.copyfile(remote/Path(source).name,destination)
        return cluster,entry,observation,copy_fn,calls

    def test_remote_inventory_program_compiles(self):
        compile(collector.INVENTORY.replace('PAYLOAD',repr(json.dumps({}))),'<remote-terminal-inventory>','exec')

    def test_active_or_incomplete_entry_refused_before_remote_or_output(self):
        for state in ('RUNNING','PENDING','FAILED'):
            entry=dict(job_id='2598',run=collector.P2_RUN,source_identity=collector.P2_ID,state=state)
            with self.assertRaisesRegex(ValueError,'active/incomplete'):
                collector.collect({},entry,Path('/unused'),Path('/unused.json'),inventory_fn=lambda *_:self.fail('remote must not be called'))

    def test_complete_capture_hashes_exact_scalar_subset_and_matches_comparison_receipt(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(collector,'ROOT',Path(folder)):
            root=Path(folder);cluster,entry,observation,copy_fn,calls=self.fixture(root)
            destination=root/'scratch/capture';receipt=root/'receipt.json'
            doc=collector.collect(cluster,entry,destination,receipt,inventory_fn=lambda *_:copy.deepcopy(observation),copy_fn=copy_fn)
            self.assertEqual(set(calls),collector.SCALARS)
            self.assertEqual({p.name for p in destination.iterdir()},collector.SCALARS)
            self.assertFalse(any(destination.glob('*.pt')))
            self.assertEqual({r['path'] for r in doc['remote_checkpoint_hashes_only']},collector.CHECKPOINTS)
            self.assertIn('unselected-note.txt',{r['path'] for r in doc['remote_all_file_inventory']})
            self.assertEqual(doc['optional_scalar_logs_missing'],[])
            # The existing comparison parser accepts this exact receipt schema.
            (root/'results').mkdir();(root/'results/jobs.json').write_text(json.dumps([entry]))
            with patch.object(comparison,'STUDY',root):comparison.verify_receipt(destination,receipt)
            with self.assertRaises(FileExistsError):collector.collect(cluster,entry,destination,receipt,inventory_fn=lambda *_:self.fail('no overwrite'))

    def test_optional_scalar_logs_can_be_absent(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(collector,'ROOT',Path(folder)):
            root=Path(folder);cluster,entry,observation,copy_fn,calls=self.fixture(root,optional=False)
            doc=collector.collect(cluster,entry,root/'scratch/capture',root/'receipt.json',inventory_fn=lambda *_:observation,copy_fn=copy_fn)
            self.assertEqual(set(calls),collector.REQUIRED_SCALARS)
            self.assertEqual(doc['optional_scalar_logs_missing'],sorted(collector.OPTIONAL_SCALARS))

    def test_required_file_absence_refused_and_remote_change_preserves_partial_journal(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(collector,'ROOT',Path(folder)):
            root=Path(folder);cluster,entry,observation,copy_fn,_=self.fixture(root)
            bad=copy.deepcopy(observation);bad['files']=[r for r in bad['files'] if r['path']!='result.json']
            with self.assertRaisesRegex(ValueError,'Required terminal'):
                collector.collect(cluster,entry,root/'scratch/missing',root/'missing.json',inventory_fn=lambda *_:bad,copy_fn=copy_fn)
            self.assertFalse((root/'scratch/missing').exists())
            changed=copy.deepcopy(observation);changed['files'][0]['sha256']='f'*64
            sequence=iter((observation,changed));destination=root/'scratch/changed';receipt=root/'changed.json'
            with self.assertRaisesRegex(ValueError,'changed during transfer'):
                collector.collect(cluster,entry,destination,receipt,inventory_fn=lambda *_:next(sequence),copy_fn=copy_fn)
            self.assertFalse(receipt.exists());self.assertTrue(destination.exists())
            journal=destination.with_name(destination.name+'.collection.jsonl')
            self.assertEqual(json.loads(journal.read_text().splitlines()[-1])['phase'],'COLLECTION_FAILED')

    def test_incomplete_copied_result_cannot_receive_verification_receipt(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(collector,'ROOT',Path(folder)):
            root=Path(folder);cluster,entry,observation,copy_fn,_=self.fixture(root)
            source=root/'remote/result.json';result=json.loads(source.read_text());result['updates']=2499;source.write_text(json.dumps(result))
            for row in observation['files']:
                if row['path']=='result.json':row.update(size=source.stat().st_size,sha256=collector.sha(source))
            for row in observation['all_files']:
                if row['path']=='result.json':row['size']=source.stat().st_size
            receipt=root/'receipt.json'
            with self.assertRaisesRegex(ValueError,'Incomplete copied'):
                collector.collect(cluster,entry,root/'scratch/capture',receipt,inventory_fn=lambda *_:observation,copy_fn=copy_fn)
            self.assertFalse(receipt.exists())


if __name__=='__main__':unittest.main()
