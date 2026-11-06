"""Tiny CPU/mocked interface controls; no NVML library, GPU or workload runs."""
import copy
import ctypes as C
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

STUDY = Path(__file__).resolve().parents[1]/'experiments/stereo-spiking-transformer-study'
SPEC = importlib.util.spec_from_file_location('energy_interface_controls', STUDY/'scripts/diagnose_energy_interfaces.py')
D = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(D)


class InterfaceControls(unittest.TestCase):
    def config(self):
        return json.loads((STUDY/'configs/energy-interface-cadence-v1.json').read_text())

    def test_registered_case_epoch_and_resource_guards(self):
        config = self.config(); self.assertTrue(D.validate_config(config))
        self.assertEqual(config['cases'][0]['checkpoint_sha256'], '8bb679229140bec3309e851c75bd459f34198225ca5bc4f786eba33e8aaa901c')
        for name, value in [('max_seconds',601), ('val_samples',1), ('require_zero_allocations',False),
                            ('field_ids',[185,186,84]), ('driver','energy_benchmark'), ('test_access','open')]:
            changed = copy.deepcopy(config); changed[name] = value
            with self.assertRaises(ValueError): D.validate_config(changed)
        changed = copy.deepcopy(config); changed['epochs'].reverse()
        with self.assertRaisesRegex(ValueError, '50ms'): D.validate_config(changed)
        changed = copy.deepcopy(config); changed['cases'][0]['checkpoint_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'Placeholder'): D.validate_config(changed)

    def test_registered_reverse_forward_block_preserves_exact_phases_and_legacy_order(self):
        c=json.loads((STUDY/'configs/energy-interface-order-block-v1.json').read_text())
        self.assertTrue(D.validate_config(c))
        self.assertEqual([e['cadence_seconds'] for e in c['epochs']],[.5,.05,.05,.5])
        for key,value in [('max_seconds',600),('order_protocol',{**c['order_protocol'],'cadences_seconds':[.5,.05]}),
                          ('epochs',c['epochs'][::-1][1:]),('require_zero_allocations',False)]:
            broken=copy.deepcopy(c);broken[key]=value
            with self.assertRaises(ValueError):D.validate_config(broken)
        for key in ('idle_seconds','load_seconds','cooldown_seconds'):
            broken=copy.deepcopy(c);broken['epochs'][0][key]+=1
            with self.assertRaises(ValueError):D.validate_config(broken)
        broken=copy.deepcopy(c);broken.pop('order_protocol')
        with self.assertRaises(ValueError):D.validate_config(broken)

    def test_official_field_layout_and_unavailable_values(self):
        self.assertEqual(C.sizeof(D.NVMLFieldValue),40)
        self.assertEqual([getattr(D.NVMLFieldValue,k).offset for k,_ in D.NVMLFieldValue._fields_], [0,4,8,16,24,28,32])
        value = D.NVMLFieldValue(fieldId=83,scopeId=0,timestamp=123456,latencyUsec=17,valueType=3,nvmlReturn=0)
        value.value.ullVal = 2**50+13
        decoded = D.decode_field(value,0)
        self.assertEqual(decoded['raw_value'],2**50+13)
        self.assertEqual(decoded['timestamp_epoch_microseconds'],123456)
        value.nvmlReturn = 3
        self.assertIsNone(D.decode_field(value,0)['raw_value'])
        self.assertIsNone(D.decode_field(value,3)['field_status'])
        value.nvmlReturn=0; value.valueType=999
        self.assertIsNone(D.decode_field(value,0)['raw_value'])
        self.assertEqual(D.decode_field(value,0)['value_decode_status'],'unknown_type_or_nonfinite')

    def test_typed_direct_pointer_width_status_and_timing(self):
        client = D.TypedNVML.__new__(D.TypedNVML); client.handle = C.c_void_p(7)
        def counter(handle, pointer):
            self.assertEqual(handle.value,7)
            C.cast(pointer,C.POINTER(C.c_ulonglong)).contents.value = 2**52+123
            return 0
        client.calls = {'nvmlDeviceGetTotalEnergyConsumption':counter}
        row = client.direct('nvmlDeviceGetTotalEnergyConsumption',C.c_ulonglong,'mJ')
        self.assertEqual(row['raw_value'],2**52+123)
        self.assertEqual(row['status'],0)
        self.assertLessEqual(row['pre']['monotonic_ns'], row['post']['monotonic_ns'])
        self.assertIn('epoch_ns',row['pre'])
        client.calls['nvmlDeviceGetTotalEnergyConsumption']=lambda *_:3
        self.assertIsNone(client.direct('nvmlDeviceGetTotalEnergyConsumption',C.c_ulonglong,'mJ')['raw_value'])
        client.calls['nvmlDeviceGetTotalEnergyConsumption']=None
        self.assertEqual(client.direct('nvmlDeviceGetTotalEnergyConsumption',C.c_ulonglong,'mJ')['status'],'symbol_missing')

    def test_field_batch_retains_per_field_failure_and_scope(self):
        client = D.TypedNVML.__new__(D.TypedNVML); client.handle=C.c_void_p(7)
        def fields(handle, count, values):
            self.assertEqual(count,3)
            self.assertEqual([(x.fieldId,x.scopeId) for x in values],[(185,0),(186,0),(83,0)])
            for i,value in enumerate(values):
                value.valueType=3;value.nvmlReturn=3 if i==1 else 0
                value.timestamp=1000+i;value.latencyUsec=12;value.value.ullVal=200000+i
            return 0
        client.get_fields=fields
        row=client.fields()
        self.assertEqual(row['fields'][0]['raw_value'],200000)
        self.assertIsNone(row['fields'][1]['raw_value'])
        self.assertEqual(row['fields'][1]['field_status'],3)
        self.assertEqual(row['fields'][2]['raw_unit'],'mJ')

    def test_actual_unchanged_wrapper_read_has_per_call_raw_receipts(self):
        def power(handle,pointer):
            C.cast(pointer,C.POINTER(C.c_uint)).contents.value=208077;return 0
        def counter(handle,pointer):
            C.cast(pointer,C.POINTER(C.c_ulonglong)).contents.value=2**45+3;return 0
        library=types.SimpleNamespace(nvmlDeviceGetPowerUsage=power,nvmlDeviceGetTotalEnergyConsumption=counter)
        current=D.CurrentWrapper.__new__(D.CurrentWrapper)
        current.proxy=D.WrapperLibraryProxy(library)
        current.wrapper=D.energy.GPUPowerTrace.__new__(D.energy.GPUPowerTrace)
        current.wrapper.lib=current.proxy;current.wrapper.handle=C.c_void_p(7);current.wrapper.uuid='GPU-mocked'
        row=current.read()
        self.assertEqual(row['output']['watts'],208.077)
        self.assertEqual([x['raw_value'] for x in row['calls']],[208077,2**45+3])
        self.assertEqual(len(row['calls']),2)
        for call in row['calls']:
            self.assertLessEqual(row['pre']['monotonic_ns'],call['pre']['monotonic_ns'])
            self.assertLessEqual(call['pre']['monotonic_ns'],call['post']['monotonic_ns'])
            self.assertLessEqual(call['post']['monotonic_ns'],row['post']['monotonic_ns'])

    def test_combined_probe_order_retains_phase_change_during_calls(self):
        order=[]
        class Typed:
            def direct(self,name,kind,unit): order.append(name);return {'api':name}
            def fields(self): order.append('fields');return {'status':3}
        class Wrapper:
            def read(self): order.append('wrapper');probe.phase='load';return {'calls':[]}
        probe=D.InterfaceTrace(Typed(),Wrapper(),Path('unused'),.5,2)
        probe.phase='setup_load';row=probe.read()
        self.assertEqual(order,['nvmlDeviceGetPowerUsage','nvmlDeviceGetTotalEnergyConsumption','wrapper','fields'])
        self.assertEqual((row['phase_at_start'],row['phase_at_finish']),('setup_load','load'))
        self.assertEqual(row['requested_cadence_seconds'],.5)

    def test_cleanup_receipt_zero_and_retained_allocation_failure_persists(self):
        cuda=types.SimpleNamespace(memory_allocated=mock.Mock(side_effect=[100,0,0]),
            memory_reserved=mock.Mock(side_effect=[200,0,0]),synchronize=mock.Mock(),empty_cache=mock.Mock())
        with mock.patch.object(D.gc,'collect'):
            row=D.clean_allocation_receipt(types.SimpleNamespace(cuda=cuda),'mock')
        self.assertTrue(row['zero_allocations_verified']);cuda.empty_cache.assert_called_once()
        cuda.memory_allocated=mock.Mock(return_value=8519680);cuda.memory_reserved=mock.Mock(return_value=10485760)
        with tempfile.TemporaryDirectory() as raw, mock.patch.object(D.gc,'collect'):
            path=Path(raw)/'allocation.json'
            with self.assertRaisesRegex(RuntimeError,'FAILED_TENSOR_FREE'):
                D.require_clean(types.SimpleNamespace(cuda=cuda),'mock',path)
            self.assertEqual(json.loads(path.read_text())['after_cleanup']['allocated_bytes'],8519680)

    def test_library_workspace_owner_release_is_staged_and_residual_guard_remains(self):
        # Simulate a C++-owned allocation that ordinary Python GC/cache release
        # cannot free. No CUDA library or tensor is created by this fixture.
        for residual in (0,1024):
            with self.subTest(residual=residual):
                state={'allocated':8519680+residual,'reserved':10485760};events=[]
                def empty():
                    events.append('empty_cache')
                    if not state['allocated']:state['reserved']=0
                def clear():
                    events.append('clear_workspaces');state['allocated']-=8519680
                cuda=types.SimpleNamespace(memory_allocated=lambda _:state['allocated'],memory_reserved=lambda _:state['reserved'],
                    synchronize=lambda _:events.append('synchronize'),empty_cache=empty)
                torch=types.SimpleNamespace(cuda=cuda,_C=types.SimpleNamespace(_cuda_clearCublasWorkspaces=clear),__version__='2.7.1+mock')
                with tempfile.TemporaryDirectory() as raw,mock.patch.object(D.gc,'collect'):
                    path=Path(raw)/'allocation.json'
                    if residual:
                        with self.assertRaisesRegex(RuntimeError,'FAILED_TENSOR_FREE'):D.require_clean(torch,'mock',path)
                        receipt=json.loads(path.read_text())
                    else:receipt=D.require_clean(torch,'mock',path)
                self.assertEqual(receipt['after_ordinary_cleanup']['allocated_bytes'],8519680+residual)
                self.assertEqual(receipt['cublas_workspace_release']['allocated_bytes_released'],8519680)
                self.assertEqual(receipt['after_workspace_release_before_empty_cache']['allocated_bytes'],residual)
                self.assertEqual(receipt['zero_allocations_verified'],not bool(residual))
                self.assertFalse(receipt['cublas_workspace_release']['configuration_changed'])
                self.assertEqual(events,['synchronize','empty_cache','synchronize','clear_workspaces','synchronize','empty_cache','synchronize'])

    def test_trace_boundary_requires_a_completed_row_starting_after_boundary(self):
        trace = D.InterfaceTrace.__new__(D.InterfaceTrace)
        trace.errors = []
        trace.last_pre_monotonic_ns = 99
        trace.last_post_monotonic_ns = 101
        def complete_later_row(_):
            trace.last_pre_monotonic_ns = 102
            trace.last_post_monotonic_ns = 103
        with mock.patch.object(D.time,'monotonic',return_value=0), mock.patch.object(D.time,'sleep',side_effect=complete_later_row) as sleep:
            trace.bracket_after(100)
        sleep.assert_called_once_with(.01)
        self.assertGreater(trace.last_pre_monotonic_ns,100)
        self.assertGreater(trace.last_post_monotonic_ns,100)

    def test_trace_boundary_timeout_does_not_accept_straddling_row(self):
        trace = D.InterfaceTrace.__new__(D.InterfaceTrace)
        trace.errors = []
        trace.last_pre_monotonic_ns = 99
        trace.last_post_monotonic_ns = 101
        with mock.patch.object(D.time,'monotonic',side_effect=[0,4]), mock.patch.object(D.time,'sleep'):
            with self.assertRaisesRegex(RuntimeError,'bracket'):
                trace.bracket_after(100)

    def test_library_workspace_hook_failure_is_persisted_not_ignored(self):
        cuda=types.SimpleNamespace(memory_allocated=lambda _:8519680,memory_reserved=lambda _:10485760,
            synchronize=mock.Mock(),empty_cache=mock.Mock())
        hook=mock.Mock(side_effect=RuntimeError('mock unavailable runtime binding'))
        torch=types.SimpleNamespace(cuda=cuda,_C=types.SimpleNamespace(_cuda_clearCublasWorkspaces=hook))
        with tempfile.TemporaryDirectory() as raw,mock.patch.object(D.gc,'collect'):
            path=Path(raw)/'allocation.json'
            with self.assertRaisesRegex(RuntimeError,'FAILED_TENSOR_FREE'):D.require_clean(torch,'mock',path)
            receipt=json.loads(path.read_text())
        self.assertEqual(receipt['cublas_workspace_release']['status'],'failed')
        self.assertIn('mock unavailable',receipt['cublas_workspace_release']['error'])
        self.assertFalse(receipt['zero_allocations_verified'])

    def test_scoped_load_whole20_cycles_and_final_neuron_reset_no_gpu_refs(self):
        # No model forward executes: only CPU control flow and fake API receipts.
        model=types.SimpleNamespace(model=object())
        provenance={'model_family':'baseline','variant':{'overrides':{},'training_model_config':{'input_channels':2,'architecture':'original_I'}},'trained_state_sha256':'fixed'}
        samples=[{'sample_id':str(i)} for i in range(20)]
        functional=types.SimpleNamespace(reset_net=mock.Mock())
        fake_package=types.ModuleType('spikingjelly.activation_based');fake_package.functional=functional
        with tempfile.TemporaryDirectory() as raw, mock.patch.dict(sys.modules,{'spikingjelly':types.ModuleType('spikingjelly'),'spikingjelly.activation_based':fake_package}), \
                mock.patch.object(D.energy,'load_case',return_value=(model,samples,provenance)), \
                mock.patch.object(D.energy,'run_output',side_effect=lambda m,s,*a:{'sample_id':s['sample_id'],'wall_seconds':0.}), \
                mock.patch.object(D.energy,'tensor_state_sha',return_value='fixed'), \
                mock.patch.object(D.time,'monotonic',side_effect=[0,0]+[2]*30):
            result=D.scoped_load({},self.config(),{},'unused',1,Path(raw),lambda _:None,lambda:None)
            self.assertEqual(result['delivered_outputs'],20)
            self.assertEqual(result['complete_cycles'],1)
            self.assertNotIn('model',result);self.assertNotIn('samples',result)
            self.assertEqual([json.loads(x)['sample_id'] for x in (Path(raw)/'outputs.jsonl').read_text().splitlines()],list(map(str,range(20))))
            self.assertTrue(json.loads((Path(raw)/'neuron-reset.json').read_text())['trained_state_unchanged'])
        functional.reset_net.assert_called_once_with(model.model)


if __name__=='__main__': unittest.main()
