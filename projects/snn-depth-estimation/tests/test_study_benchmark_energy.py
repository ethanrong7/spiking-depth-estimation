"""Tiny CPU controls/mocks only; no hardware timing, inference campaign or NVML."""
import copy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

STUDY = Path(__file__).resolve().parents[1]/'experiments/stereo-spiking-transformer-study'
SPEC = importlib.util.spec_from_file_location('study_energy_control_test', STUDY/'scripts/benchmark_energy.py')
ENERGY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENERGY)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


@dataclass(frozen=True)
class ParityConfig:
    collect_diagnostics: bool = False


class ParityStudy(torch.nn.Module):
    """Mock full-output API with a real telemetry-bearing neuron, no experiment."""
    def __init__(self,*,mismatch=False,fail=False,mutate_input=False,consume_rng=False):
        super().__init__()
        from core.study_models import TemporalNeuron
        self.weight=torch.nn.Parameter(torch.tensor([-1.]))
        self.neuron=TemporalNeuron();self.neuron.collect_diagnostics=False
        self.config=ParityConfig();self.calls=[]
        self.mismatch,self.fail,self.mutate_input,self.consume_rng=mismatch,fail,mutate_input,consume_rng
        self.eval()

    def forward(self,left,right):
        enabled=self.config.collect_diagnostics
        self.calls.append((enabled,self.neuron.collect_diagnostics))
        if self.mutate_input:left.add_(1)
        value=left.sum(1)*self.weight
        if self.consume_rng:value=value+torch.rand_like(value)
        if self.fail:raise RuntimeError('mock forward exception')
        confidence=value*.1+(0.01 if self.mismatch and not enabled else 0.)
        return {'disparity':value,'right_disparity':value+1,'occlusion_logits':value*0,
                'aux_disparities':{4:value/2},'aux_disparities_native':{4:value/3},
                'right_aux_disparities_native':{4:value/4},
                'match_confidence_native':{4:{'max_probability':confidence,'normalized_entropy':value*0}},
                'diagnostics':{'mode':enabled,'different_metadata_is_expected':torch.tensor(float(enabled))}}


class EnergyControlTests(unittest.TestCase):
    def test_registration_bounds_and_interleaved_plan(self):
        config = ENERGY.example_config()
        ENERGY.validate_energy_config(config, require_registered=False)
        with self.assertRaisesRegex(ValueError, 'registered'):
            ENERGY.validate_energy_config(config)
        config['registered'] = True
        config['resource_forecast']['sha256'] = '1'*64
        for key in ENERGY.SHA_FIELDS:
            config['cases'][0][key] = '1'*64
        ENERGY.validate_energy_config(config)
        other = copy.deepcopy(config['cases'][0]); other['case_id'] = 'B-spike'
        config['cases'].append(other)
        plan = [(r,b,c['case_id']) for r,b,c in ENERGY.interval_plan(config)]
        self.assertEqual(plan[:4],[(1,'pipeline','A-spike'),(1,'gpu_resident','A-spike'),(1,'pipeline','B-spike'),(1,'gpu_resident','B-spike')])
        self.assertEqual([(b,c) for r,b,c in plan[4:8]],list(reversed([(b,c) for r,b,c in plan[:4]])))
        self.assertEqual(len(plan),12)
        for key, value in [('measurement_seconds',29),('max_seconds',2701),('val_samples',2),('repeats',2),('test_access','released')]:
            bad = copy.deepcopy(config); bad[key] = value
            with self.assertRaises(ValueError):
                ENERGY.validate_energy_config(bad)
        for key,value in [('cache_priming','none'),('warmup_all_samples',True)]:
            bad=copy.deepcopy(config);bad[key]=value
            with self.assertRaises(ValueError):
                ENERGY.validate_energy_config(bad)
        compatible=copy.deepcopy(config);compatible.pop('study_diagnostics_parity')
        ENERGY.validate_energy_config(compatible)
        compatible['study_diagnostics_parity']={'atol':1e-3,'rtol':1e-3}
        with self.assertRaisesRegex(ValueError,'parity'):
            ENERGY.validate_energy_config(compatible)
        bad = copy.deepcopy(config); bad['max_seconds'] = 300
        with self.assertRaisesRegex(ValueError, 'fit'):
            ENERGY.validate_energy_config(bad)

    def test_counter_same_window_and_unavailable_sensor(self):
        rows = [{'monotonic_seconds':t, 'joules_counter':100+50*t} for t in (0,10,20,30,40)]
        result = ENERGY.counter_crosscheck(rows,5,35,1500)
        self.assertEqual(result['same_window_counter_gpu_joules'],1500)
        self.assertEqual(result['relative_difference_from_trapezoid'],0)
        rows[2]['joules_counter'] = None
        self.assertFalse(ENERGY.counter_crosscheck(rows,5,35,1500)['available'])
        rows[2]['joules_counter'] = 0
        self.assertFalse(ENERGY.counter_crosscheck(rows,5,35,1500)['available'])
        self.assertAlmostEqual(ENERGY.percentile([1,2,3,4],.95),3.85)

    def test_best_val_log_binding_excludes_val_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'validation.jsonl'
            raw = b'{"update":0,"mean_scene_epe":1}\n{"update":250,"mean_scene_epe":5}\n'
            path.write_bytes(raw+b'{"update":500,"mean_scene_epe":6}\n')
            receipt = {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
            state = {'update':250,'best_update':250,'best_val_epe':5}
            row = ENERGY.verify_validation_selection(path,receipt,state)
            self.assertEqual(row['update'],250)
            self.assertIn('Val0 excluded',row['selection'])
            with self.assertRaisesRegex(RuntimeError,'selection'):
                ENERGY.verify_validation_selection(path,receipt,{**state,'best_update':0})
            with self.assertRaisesRegex(RuntimeError,'schedule'):
                ENERGY.verify_validation_selection(path,receipt,state,required_updates=[250,500,1000])
            path.write_bytes(b'x'+raw[1:])
            with self.assertRaisesRegex(RuntimeError,'prefix'):
                ENERGY.verify_validation_selection(path,receipt,state)

    def test_cpu_checkpoint_loading_uses_bound_weights_and_only_diagnostic_override(self):
        # CPU one-parameter model + stub Val loader exercise integration, not
        # actual study inference. CUDA synchronization/allocation are mocked.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); sources=root/'sources'; runs=root/'runs'; data=root/'data'
            source=sources/'frozen'; run=runs/'trained'; dataset=data/'paired'
            for folder in (source,run,dataset): folder.mkdir(parents=True)
            member=ENERGY.PREFIX+'configs/trained.json'
            model_member=ENERGY.PREFIX+'core/study_models.py'
            training = {'schema':'stereo-study-training-v1','seed':9471,'scope':'engineering','pretraining':False,
                        'test_access':'sealed','total_updates':1000,
                        'training':{'evaluation_updates':[250,500,1000]},
                        'model':{'family':'study','config':{'architecture':'A','time_steps':50,'collect_diagnostics':True}},
                        'data':{'dataset_version':'paired','modality':'spike'}}
            (dataset/'manifest.json').write_text('{}')
            manifest_sha=ENERGY.file_sha(dataset/'manifest.json'); training['data']['manifest_sha256']=manifest_sha
            for name,content in ((member,json.dumps(training)),(model_member,'mock_model_source')):
                path=source/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(content)
            files={name:ENERGY.file_sha(source/name) for name in (member,model_member)}
            registry={'source_files':files,'git_commit':'commit','source_identity_sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}
            registry_path=source/'source-registry.json'; registry_path.write_text(json.dumps(registry))
            fresh=ENERGY.tensor_state_sha(TinyModel())
            case={'case_id':'tiny','training_run':'trained','training_source_snapshot':'frozen','training_config_member':member,
                  'seed':9471,'training_source_identity_sha256':registry['source_identity_sha256'],
                  'training_source_registry_sha256':ENERGY.file_sha(registry_path),'training_config_sha256':files[member],
                  'manifest_sha256':manifest_sha,'training_initial_weights_sha256':fresh}
            ids=['val'+str(i) for i in range(20)]
            binding={'source_identity_sha256':case['training_source_identity_sha256'],
                     'source_registry_sha256':case['training_source_registry_sha256'], 'config_sha256':files[member],
                     'manifest_sha256':manifest_sha,'git_commit':'commit','config_identity':ENERGY.digest(training),
                     'total_updates':1000,'effective_batch':8,'validation_samples':20,'val_sample_ids_sha256':ENERGY.digest(ids)}
            log=b'{"update":250,"mean_scene_epe":5}\n'
            final_log=log+b'{"update":500,"mean_scene_epe":6}\n{"update":1000,"mean_scene_epe":7}\n'
            (run/'validation.jsonl').write_bytes(final_log)
            case.update(selection_validation_log_sha256=hashlib.sha256(final_log).hexdigest(),selection_validation_log_bytes=len(final_log))
            payload={'binding':binding,'state':{'update':250,'presentations':2000,'best_update':250,'best_val_epe':5},
                     'logs':{'validation':{'bytes':len(log),'sha256':hashlib.sha256(log).hexdigest()}},
                     'model':{'weight':torch.tensor([7.])}}
            torch.save(payload,run/'best-val.pt'); case['checkpoint_sha256']=ENERGY.file_sha(run/'best-val.pt')
            metadata={'binding':binding,'fresh_seed_model_sha256':fresh,'torch':'mock'}
            (run/'metadata.json').write_text(json.dumps(metadata))
            class Val:
                rows=[{'sample_id':identifier} for identifier in ids]
                def __init__(self,*args,**kwargs): pass
                def __len__(self): return 20
                def __getitem__(self,index):
                    return {'left':torch.zeros(50,1,1,1),'right':torch.ones(50,1,1,1),'sample_id':ids[index]}
            captured=[]
            def builder(values): captured.append(values); return TinyModel()
            verifier=types.SimpleNamespace(verify_source=lambda path:registry)
            with mock.patch.dict('os.environ',{'STUDY_SOURCE_PARENT':str(sources),'STUDY_TRAINED_PARENT':str(runs),'STUDY_PAIRED_PARENT':str(data)}), \
                 mock.patch.dict(sys.modules,{'train_study':verifier}), \
                 mock.patch('core.study_models.build_model',side_effect=builder), \
                 mock.patch('core.study_data.PairedStereoDataset',Val), \
                 mock.patch.object(ENERGY,'load_sensor_only',return_value=Val()[0]) as sensor_loader, \
                 mock.patch('torch.cuda.manual_seed_all'),mock.patch('torch.cuda.synchronize'),mock.patch('torch.cuda.memory_allocated',return_value=0):
                model,samples,provenance=ENERGY.load_case(case,{'val_samples':1},registry,torch.device('cpu'))
                self.assertEqual(sensor_loader.call_count,1)
                sensor_loader.reset_mock()
                _,pipeline_samples,pipeline_provenance=ENERGY.load_case(case,{'val_samples':1},registry,torch.device('cpu'),boundary='pipeline')
                sensor_loader.assert_not_called()
                self.assertIn('sensor_root',pipeline_samples[0])
                self.assertEqual(pipeline_provenance['input_tensor_bytes'],0)
            self.assertEqual(model.weight.item(),7.)
            self.assertFalse(model.training)
            self.assertFalse(captured[0]['collect_diagnostics'])
            self.assertTrue(training['model']['config']['collect_diagnostics'])
            self.assertEqual(provenance['variant']['overrides'],{'collect_diagnostics':False})
            self.assertEqual(samples[0]['left'].shape,(1,50,1,1,1))
            self.assertEqual(provenance['input_tensor_bytes'],400)
            bad=copy.deepcopy(case); bad['seed']=2
            with self.assertRaisesRegex(RuntimeError,'seed'):
                ENERGY.check_training_binding(bad,training,metadata,payload,registry,case['training_source_registry_sha256'])

    def test_partial_interval_keeps_completed_negative_outputs_without_energy_claim(self):
        # Fake events and CPU tensors verify negative outputs are delivered,
        # partial telemetry persists, and no <30s result can be manufactured.
        class Event:
            def record(self): pass
            def elapsed_time(self,other): return 1.
        class Model:
            calls=0
            def __call__(self,left,right):
                self.calls+=1
                if self.calls==2: raise ENERGY.EnergyDeadline('stop')
                return {'disparity':torch.full((1,1,1,1),-5.)}
        trace=types.SimpleNamespace(rows=[{'monotonic_seconds':0.}],errors=[])
        sample={'left':torch.zeros(1,50,1,1,1),'right':torch.zeros(1,50,1,1,1),'sample_id':'val'}
        with tempfile.TemporaryDirectory() as temporary, \
             mock.patch('torch.cuda.Event',side_effect=lambda **kwargs:Event()), \
             mock.patch('torch.cuda.synchronize'),mock.patch('torch.cuda.reset_peak_memory_stats'):
            path=Path(temporary)/'latency.jsonl'
            with self.assertRaises(ENERGY.EnergyDeadline):
                ENERGY.measure_interval(Model(),[sample],trace,{'measurement_seconds':30},torch.device('cpu'),path)
            self.assertEqual(len(path.read_text().splitlines()),1)
            boundary=json.loads(path.with_name('measurement-boundary.json').read_text())
            self.assertEqual(boundary['complete_outputs'],1)
            self.assertEqual(boundary['status'],'INCOMPLETE_NO_ENERGY_CLAIM')
            self.assertNotIn('gpu_joules_per_map',boundary)

    def test_repeat_aggregation_does_not_promote_partial_results(self):
        measurement={'gpu_joules_per_map':2.,'throughput_maps_per_second':5.,'latency_wall_p50_seconds':.2,
                     'latency_wall_p95_seconds':.3,'peak_allocated_bytes':100}
        row={'case_id':'A','boundary':'pipeline','status':'COMPLETED_ENERGY_INTERVAL','measurement':measurement}
        result=ENERGY.aggregate_intervals([row,{'case_id':'A','boundary':'pipeline','status':'INCOMPLETE_NO_ENERGY_CLAIM'}])[0]
        self.assertEqual(result['completed_intervals'],1)
        self.assertIn('INCOMPLETE',result['status'])
        self.assertEqual(result['gpu_joules_per_map']['mean'],2.)
        self.assertEqual(ENERGY.aggregate_intervals([row,row,row])[0]['status'],'COMPLETE_THREE_REPEATS')
        secondary={**row,'boundary':'gpu_resident','measurement':{**measurement,'gpu_joules_per_map':1.}}
        grouped=ENERGY.aggregate_intervals([row,secondary])
        self.assertEqual(len(grouped),2)
        self.assertEqual([r['gpu_joules_per_map']['mean'] for r in grouped],[2.,1.])

    def test_sensor_only_exact_input_parity_without_ground_truth_access(self):
        from core.study_data import PairedStereoDataset
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            rows={split:[{'sample_id':split,'scene_id':split+'-scene','target_time':1.,
                 'bin_edges':np.linspace(0,1,51).tolist(),'source_support_end':1.,'event_channels':['ON','OFF'],
                 'spike':'spike.npz','event':'event.npz','disparity':'disp.npy'}] for split in ('train','val','test')}
            manifest={'schema':'stereo-paired-v1','T':50,'disparity':'positive_left_pixels','rectified':True,
                'time_unit':'source_frame_interval','source_identity':'tiny-fixture','license':'fixture',
                'simulators':{m:{'implementation':'fixture','version':'v1','configuration':{'tiny':True}} for m in ('spike','event')},**rows}
            path=root/'manifest.json';path.write_text(json.dumps(manifest))
            spike=np.zeros((50,1,2,4),dtype=np.uint8);spike[0,0,0,0]=1;spike[-1,0,1,3]=1
            event=np.zeros((50,2,2,4),dtype=np.uint16);event[0,0,0,0]=3;event[0,1,0,0]=2
            for name,array in [('spike',spike),('event',event)]:np.savez(root/(name+'.npz'),left=array,right=array[...,::-1])
            np.save(root/'disp.npy',np.ones((2,4),dtype=np.float32))
            expected={m:PairedStereoDataset(root,path,m,'val')[0] for m in ('spike','event')}
            (root/'disp.npy').unlink()
            for modality in ('spike','event'):
                actual=ENERGY.load_sensor_only(root,rows['val'][0],modality)
                for view in ('left','right'):
                    self.assertTrue(torch.equal(actual[view],expected[modality][view]))
                    self.assertEqual(actual[view].dtype,torch.float32)
                self.assertNotIn('target',actual)
            self.assertEqual(ENERGY.load_sensor_only(root,rows['val'][0],'event')['left'][0,:,0,0].tolist(),[3.,2.])
            np.savez(root/'spike.npz',left=spike*2,right=spike*2)
            with self.assertRaisesRegex(ValueError,'binary'):
                ENERGY.load_sensor_only(root,rows['val'][0],'spike')

    def test_pipeline_wall_includes_decode_and_cuda_spans_exclude_decode(self):
        class Event:
            def record(self):pass
            def elapsed_time(self,other):return 1.
        cpu={'left':torch.zeros(50,2,2,4),'right':torch.ones(50,2,2,4),'sample_id':'val'}
        sample={'sensor_root':'unused','row':{},'modality':'event','sample_id':'val'}
        def model(left,right):
            self.assertEqual(left.shape,(1,50,2,2,4))
            return {'disparity':torch.full((1,1,2,4),-1.)}
        with mock.patch.object(ENERGY,'load_sensor_only',return_value=cpu) as decoder, \
             mock.patch.object(ENERGY.time,'perf_counter',side_effect=[0,.1,.1,.2,.3]), \
             mock.patch('torch.cuda.Event',side_effect=lambda **kwargs:Event()),mock.patch('torch.cuda.synchronize'):
            row=ENERGY.run_output(model,sample,'pipeline',torch.device('cpu'))
        decoder.assert_called_once()
        self.assertAlmostEqual(row['cpu_sensor_decode_tensor_seconds'],.1)
        self.assertAlmostEqual(row['wall_seconds'],.3)
        self.assertAlmostEqual(row['h2d_cuda_span_seconds'],.001)
        self.assertAlmostEqual(row['model_cuda_span_seconds'],.001)

    def test_foreign_process_and_resource_receipt_controls_are_read_only(self):
        commands=[]
        def runner(command,**kwargs):
            commands.append(command)
            if '--query-gpu=' in ' '.join(command):return 'GPU-abc,RTX 5090,driver,575,2000,2000,12000,45,P0,3000,32000\n'
            return '7,GPU-abc,python\n99,GPU-abc,foreign\n888,GPU-other,other-card\n'
        row=ENERGY.gpu_snapshot('abc',own_pid=7,runner=runner)
        self.assertEqual([r['pid'] for r in row['foreign_compute_processes']],[99])
        self.assertTrue(all('--id=GPU-abc' in command for command in commands))
        self.assertTrue(all(any(flag.startswith('--query-') for flag in command) for command in commands))
        summary=ENERGY.context_summary([row])
        self.assertEqual(summary['max_observed_device_memory_used_bytes'],3000*1024**2)
        self.assertTrue(summary['foreign_compute_observed'])
        monitor=ENERGY.GPUContextTrace('abc','unused',575)
        monitor.foreign=row['foreign_compute_processes']
        with self.assertRaisesRegex(RuntimeError,'contaminated'):
            monitor.check()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path=root/'forecast.md';path.write_text('registered resource forecast')
            config=ENERGY.example_config();sha=ENERGY.file_sha(path)
            config['resource_forecast']={'source_member':'forecast.md','sha256':sha,'reserved_gpu_seconds':2700}
            verified=ENERGY.verify_resource_forecast(config,{'source_files':{'forecast.md':sha}},root)
            self.assertTrue(verified['verified_before_cuda'])
            path.write_text('changed')
            with self.assertRaisesRegex(RuntimeError,'forecast'):
                ENERGY.verify_resource_forecast(config,{'source_files':{'forecast.md':sha}},root)

    def test_full_val_cycles_and_negative_idle_adjustment_preserve_primary_total(self):
        class Trace:
            rows=[]
            def summary(self,start,end,delivered,idle_watts=None):
                return {'integrated_gpu_joules':10.,'gpu_joules_per_map':10./delivered,
                        'separately_idle_adjusted_gpu_joules_per_map':(10.-idle_watts*(end-start))/delivered}
        latency={'sample_id':'one','wall_seconds':.1,'h2d_cuda_span_seconds':.01,
                 'model_cuda_span_seconds':.05,'cpu_sensor_decode_tensor_seconds':.04}
        with tempfile.TemporaryDirectory() as temporary, \
             mock.patch.object(ENERGY,'wait_trace'),mock.patch.object(ENERGY,'counter_crosscheck',return_value={'available':False}), \
             mock.patch.object(ENERGY,'power_trace_health',return_value={'mock':'no NVML used'}), \
             mock.patch.object(ENERGY,'run_output',side_effect=[latency,{**latency,'sample_id':'two'}]), \
             mock.patch.object(ENERGY.time,'monotonic',side_effect=[0,1,40,41,42]), \
             mock.patch('torch.cuda.synchronize'),mock.patch('torch.cuda.reset_peak_memory_stats'), \
             mock.patch('torch.cuda.max_memory_allocated',return_value=0),mock.patch('torch.cuda.max_memory_reserved',return_value=0):
            result=ENERGY.measure_interval(None,[{'sample_id':'one'},{'sample_id':'two'}],Trace(),
                {'measurement_seconds':30,'complete_val_cycles':True},torch.device('cpu'),Path(temporary)/'latency.jsonl',
                boundary='pipeline',idle={'round':1,'mean_gpu_watts':2.})
        self.assertEqual(result['integrated_gpu_joules'],10.)
        self.assertLess(result['separately_idle_adjusted_gpu_joules'],0)
        self.assertEqual(result['per_sample_output_counts'],{'one':1,'two':1})
        self.assertTrue(result['complete_val_cycles'])

    def test_unbracketed_nonfinite_and_large_power_gaps_rejected(self):
        def trace(points):
            return types.SimpleNamespace(rows=[{'monotonic_seconds':t,'watts':w,'query_finished_monotonic_seconds':t+.01} for t,w in points])
        good=trace([(i*.5,100.) for i in range(65)])
        health=ENERGY.power_trace_health(good,.25,31.75)
        self.assertEqual(health['max_interval_sample_gap_seconds'],.5)
        self.assertAlmostEqual(health['max_query_duration_seconds'],.01)
        for bad in (trace([(1.,100.),(2.,100.)]),trace([(0.,100.),(32.,100.)]),
                    trace([(i*.5,None if i==20 else 100.) for i in range(65)])):
            with self.assertRaises(RuntimeError):
                ENERGY.power_trace_health(bad,.25,31.75)

    def test_cache_priming_reads_all_twenty_sensors_without_labels_or_model_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            values=np.zeros((50,2,1,1),dtype=np.uint16);values[0,0,0,0]=3
            np.savez(root/'sensor.npz',left=values,right=values)
            samples=[{'sensor_root':str(root),'modality':'event','sample_id':'val'+str(i),
                      'row':{'sample_id':'val'+str(i),'event':'sensor.npz','disparity':'ABSENT-GT.npy'}} for i in range(20)]
            with mock.patch.object(ENERGY,'load_sensor_only',wraps=ENERGY.load_sensor_only) as decoder, \
                 mock.patch.object(ENERGY,'run_output') as inference:
                record=ENERGY.prime_sensor_assets(samples,'pipeline')
                self.assertEqual(decoder.call_count,20)
                self.assertEqual([call.args[1]['sample_id'] for call in decoder.call_args_list],[s['sample_id'] for s in samples])
                inference.assert_not_called()
                decoder.reset_mock()
                resident=ENERGY.prime_sensor_assets(samples,'gpu_resident')
                decoder.assert_not_called()
                self.assertEqual(resident['additional_priming_seconds'],0.)
            self.assertEqual(record['primed_sensor_assets'],20)
            self.assertFalse(record['cpu_priming_tensors_retained'])
            self.assertEqual(record['total_cpu_tensor_bytes_processed'],20*50*2*4*2)

    def test_api_warmup_honors_time_and_output_minima_without_forcing_twenty(self):
        samples=[{'sample_id':'val'+str(i)} for i in range(20)]
        for ticks,expected in [([0,2.1,2.2],3),([0,1.,1.5,2.1,2.2],5)]:
            with mock.patch.object(ENERGY,'run_output') as inference, \
                 mock.patch.object(ENERGY.time,'monotonic',side_effect=ticks):
                record=ENERGY.warmup_model(None,samples,'pipeline','mock_device')
            self.assertEqual(record['warmup_outputs'],expected)
            self.assertEqual(inference.call_count,expected)
            self.assertLess(expected,20)
            self.assertGreaterEqual(record['warmup_seconds'],2.)
            self.assertFalse(record['warmup_all_samples'])
            self.assertTrue(all(call.args[2]=='pipeline' for call in inference.call_args_list))

    def test_full_prediction_parity_restores_config_flags_inputs_state_and_rng(self):
        model=ParityStudy(consume_rng=True);original=model.config
        left=torch.ones(1,50,1,1,1);right=torch.zeros_like(left)
        inputs=ENERGY.value_fingerprint((left,right));state=ENERGY.tensor_state_sha(model)
        rng=ENERGY.value_fingerprint(ENERGY.capture_rng())
        audit=ENERGY.run_study_diagnostics_parity(model,left,right)
        self.assertEqual(audit['status'],'PASSED_CHECKPOINT_DIAGNOSTICS_PARITY')
        self.assertEqual(model.calls,[(True,True),(False,False)])
        self.assertIs(model.config,original);self.assertFalse(model.neuron.collect_diagnostics)
        self.assertEqual(ENERGY.value_fingerprint((left,right)),inputs)
        self.assertEqual(ENERGY.tensor_state_sha(model),state)
        self.assertEqual(ENERGY.value_fingerprint(ENERGY.capture_rng()),rng)
        self.assertTrue(audit['rng_restored']);self.assertFalse(audit['forward_rng_unchanged'])
        self.assertTrue(audit['post_forward_rng_equal_between_modes'])
        self.assertTrue(audit['all_prediction_tensors_exact'])
        self.assertEqual(audit['compared_prediction_tensors'],8)
        self.assertTrue(any('match_confidence_native' in key for key in audit['prediction_tensors']))
        self.assertFalse(any('diagnostics' in key for key in audit['prediction_tensors']))

    def test_parity_mismatch_exception_and_input_mutation_restore_flags_and_rng(self):
        left=torch.ones(1,50,1,1,1);right=torch.zeros_like(left)
        for options in ({'mismatch':True},{'fail':True,'consume_rng':True},{'mutate_input':True}):
            model=ParityStudy(**options);original=model.config;audit={}
            rng=ENERGY.value_fingerprint(ENERGY.capture_rng())
            with self.assertRaises(RuntimeError):
                ENERGY.run_study_diagnostics_parity(model,left,right,audit=audit)
            self.assertEqual(audit['status'],'FAILED_CHECKPOINT_DIAGNOSTICS_PARITY')
            self.assertIs(model.config,original);self.assertFalse(model.neuron.collect_diagnostics)
            self.assertTrue(audit['config_object_restored']);self.assertTrue(audit['neuron_flags_restored'])
            self.assertTrue(audit['rng_restored']);self.assertTrue(audit['source_inputs_unchanged'])
            self.assertTrue(audit['trained_state_unchanged'])
            self.assertEqual(ENERGY.value_fingerprint(ENERGY.capture_rng()),rng)
            self.assertTrue(torch.equal(left,torch.ones_like(left)))
            if options.get('mismatch'):
                errors=[row for row in audit['prediction_tensors'].values() if not row['within_tolerance']]
                self.assertEqual(len(errors),1)
                self.assertGreater(errors[0]['max_scaled_tolerance_error'],1.)

    def test_parity_receipt_reuse_checks_checkpoint_input_and_trained_state(self):
        model=ParityStudy();sample={'left':torch.ones(1,50,1,1,1),'right':torch.zeros(1,50,1,1,1),'sample_id':'first-val'}
        case={'case_id':'A','checkpoint_sha256':'checkpoint','manifest_sha256':'manifest','seed':9471,'training_initial_weights_sha256':'initializer'}
        provenance={'model_family':'study','trained_state_sha256':ENERGY.tensor_state_sha(model),'training_core_file_map_sha256':'core'}
        cache={}
        first=ENERGY.ensure_diagnostics_parity(model,sample,'gpu_resident',torch.device('cpu'),case=case,provenance=provenance,cache=cache)
        self.assertEqual(len(model.calls),2);self.assertEqual(len(cache),1)
        self.assertEqual(first['status'],'PASSED_CHECKPOINT_DIAGNOSTICS_PARITY')
        reloaded=ParityStudy()
        reused=ENERGY.ensure_diagnostics_parity(reloaded,sample,'gpu_resident',torch.device('cpu'),case=case,provenance=provenance,cache=cache)
        self.assertEqual(reused['status'],'REUSED_VERIFIED_JOB_RECEIPT');self.assertEqual(reloaded.calls,[])
        altered={**sample,'left':sample['left']+1}
        with self.assertRaisesRegex(RuntimeError,'binding mismatch'):
            ENERGY.ensure_diagnostics_parity(reloaded,altered,'gpu_resident',torch.device('cpu'),case=case,provenance=provenance,cache=cache)
        with torch.no_grad():reloaded.weight.add_(1)
        with self.assertRaisesRegex(RuntimeError,'changed trained'):
            ENERGY.ensure_diagnostics_parity(reloaded,sample,'gpu_resident',torch.device('cpu'),case=case,provenance=provenance,cache=cache)
        self.assertEqual(reloaded.calls,[])

    def test_baseline_parity_is_explicit_no_override_no_forward_or_decode(self):
        with mock.patch.object(ENERGY,'load_sensor_only') as decoder,mock.patch.object(ENERGY,'run_study_diagnostics_parity') as forward:
            audit=ENERGY.ensure_diagnostics_parity(None,None,'pipeline',torch.device('cpu'),case={},provenance={'model_family':'baseline'},cache={})
        self.assertEqual(audit['status'],'NOT_APPLICABLE_BASELINE')
        decoder.assert_not_called();forward.assert_not_called()

    def test_tiny_complete_study_model_uses_all_actual_prediction_outputs(self):
        from core.study_models import build_model,TemporalNeuron
        previous_threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            model=build_model({'architecture':'A','widths':[8,8,8,8],'time_steps':3,'max_disparity':12,
                               'collect_diagnostics':False,'feature_norm_eps':.01}).eval()
            original=model.config
            left=torch.randint(0,2,(1,3,1,17,19)).float();right=torch.flip(left,[-1])
            audit=ENERGY.run_study_diagnostics_parity(model,left,right)
            self.assertEqual(audit['status'],'PASSED_CHECKPOINT_DIAGNOSTICS_PARITY')
            self.assertEqual(audit['compared_prediction_tensors'],22)
            self.assertTrue(audit['all_prediction_tensors_exact'])
            self.assertIs(model.config,original)
            self.assertTrue(all(not module.collect_diagnostics for module in model.modules() if isinstance(module,TemporalNeuron)))
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == '__main__':
    unittest.main()
