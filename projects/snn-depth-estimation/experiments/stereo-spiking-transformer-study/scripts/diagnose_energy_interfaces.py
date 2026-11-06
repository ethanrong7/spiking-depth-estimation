"""Read-only NVML interface/cadence diagnostic; no submission or calibration.

CLI: --config REGISTERED.json --source-registry SOURCE/source-registry.json
--output FRESH. Uses the existing energy environment: STUDY_TRAINED_PARENT,
STUDY_SOURCE_PARENT, STUDY_PAIRED_PARENT, STUDY_BASELINE_CAPSULE. The controller
and Slurm dispatch need an explicit energy_interfaces branch; this file never
changes them. --validate-config is a CPU-only control.

Two fixed50ms/500ms epochs each trace60s zero-PyTorch-allocation idle, tagged
model/input setup, >=90s full20-target cycles, tagged neuron reset/cleanup,
and30s zero-allocation cooldown. Setup is traced but not called idle/load.
Every NVML call retains its own host pre/post timestamps. Typed direct calls
and the unchanged current wrapper are sequential checks of the same NVML
instrumentation, not independent sensors or simultaneous readings. Fields
185/186/83, GPU scope0, were verified against NVIDIA's official header below.
Only Get/init/shutdown APIs are used; unsupported fields remain unavailable.
Raw streams support later transition/middle60s/30s-window analysis; this driver
does not choose an energy interface, rank models, subtract idle or calibrate.
"""
import argparse
import copy
import ctypes as C
import datetime
import gc
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import threading
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_energy as energy

HEADER_COMMIT = '6b38eea7ed6854b19001f44231e9385661e90c46'
HEADER_SHA256 = 'c5de65a896402f4770b14e12128016ff974104b569ac08b4ba2aee8e66f5d9bf'
HEADER_URL = 'https://raw.githubusercontent.com/NVIDIA/go-nvml/'+HEADER_COMMIT+'/pkg/nvml/nvml.h'
DOCS = {'header_url': HEADER_URL, 'header_sha256': HEADER_SHA256,
        'header_version': '13.4.92', 'api_version': 13, 'checked_date': '2026-10-29',
        'fields': 'https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlFieldValueEnums.html',
        'direct_queries': 'https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html',
        'field_layout': 'https://docs.nvidia.com/deploy/nvml-api/latest/api/structnvmlFieldValue__t.html',
        'field_query': 'https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlFieldValueQueries.html'}
FIELDS = ((185, 'NVML_FI_DEV_POWER_AVERAGE', 'mW'),
          (186, 'NVML_FI_DEV_POWER_INSTANT', 'mW'),
          (83, 'NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION', 'mJ'))
DIRECT = (('nvmlDeviceGetPowerUsage', C.c_uint, 'mW'),
          ('nvmlDeviceGetTotalEnergyConsumption', C.c_ulonglong, 'mJ'))
ORDER_PROTOCOL = {
    'name': 'reverse_then_forward_same_allocation_v1',
    'cadences_seconds': [0.5, 0.05, 0.05, 0.5],
    'device_policy': 'one_recorded_allocated_UUID_all_four_epochs',
    'phase_policy': 'unchanged60_idle90_load30_cooldown_per_epoch',
    'prior_forward_run': 'energy-interface-cadence-v2',
    'prior_job_id': '2333',
    'prior_result_sha256': 'e1171f9d3c699485f91895f5af7dc34a509ccafda4f81d592ab8031d0f4c98ad',
    'interpretation': 'one_order_block_not_independent_replicates_or_calibration',
}


VALUE_MEMBERS = {0: 'dVal', 1: 'uiVal', 2: 'ulVal', 3: 'ullVal',
                 4: 'sllVal', 5: 'siVal', 6: 'usVal'}


class NVMLValue(C.Union):
    _fields_ = [('dVal', C.c_double), ('uiVal', C.c_uint), ('ulVal', C.c_ulong),
                ('ullVal', C.c_ulonglong), ('sllVal', C.c_longlong),
                ('siVal', C.c_int), ('usVal', C.c_ushort)]


class NVMLFieldValue(C.Structure):
    _fields_ = [('fieldId', C.c_uint), ('scopeId', C.c_uint),
                ('timestamp', C.c_longlong), ('latencyUsec', C.c_longlong),
                ('valueType', C.c_int), ('nvmlReturn', C.c_int), ('value', NVMLValue)]


def stamp():
    return {'monotonic_ns': time.monotonic_ns(), 'epoch_ns': time.time_ns()}


def timed_call(function, *args):
    before = stamp()
    status = int(function(*args)) if function is not None else 'symbol_missing'
    return {'pre': before, 'post': stamp(), 'status': status}


def decode_field(value, call_status):
    """Never interpret a field's undefined value on either failed status."""
    result = {'field_id': int(value.fieldId), 'scope_id': int(value.scopeId),
              'scope': 'GPU only (scope0); no module/board/chip equivalence inferred',
              'field_status': int(value.nvmlReturn) if call_status == 0 else None,
              'timestamp_epoch_microseconds': int(value.timestamp) if call_status == 0 else None,
              'latency_microseconds': int(value.latencyUsec) if call_status == 0 else None,
              'value_type': int(value.valueType) if call_status == 0 else None,
              'raw_value': None, 'value_union_bytes_hex': None}
    if call_status == 0 and value.nvmlReturn == 0:
        member = VALUE_MEMBERS.get(value.valueType)
        raw = getattr(value.value, member) if member else None
        result['raw_value'] = raw if not isinstance(raw, float) or math.isfinite(raw) else None
        result['value_union_bytes_hex'] = bytes(value.value).hex()
        result['value_decode_status'] = 'decoded' if member and result['raw_value'] is not None else 'unknown_type_or_nonfinite'
    else:
        result['value_decode_status'] = 'unavailable'
    return result


class TypedNVML:
    """Independent ctypes function objects with fully typed output pointers."""
    def __init__(self, uuid):
        self.lib = C.CDLL('libnvidia-ml.so.1')
        self.calls = {}
        self.init = self.bind('nvmlInit_v2', [])
        self.shutdown = self.bind('nvmlShutdown', [])
        status = self.init()
        if status: raise RuntimeError('Typed NVML init status '+str(status))
        self.handle = C.c_void_p()
        get = self.bind('nvmlDeviceGetHandleByUUID', [C.c_char_p, C.POINTER(C.c_void_p)])
        status = get(energy.normalized_uuid(uuid).encode(), C.byref(self.handle))
        if status:
            self.shutdown()
            raise RuntimeError('Typed NVML UUID handle status '+str(status))
        for name, kind, _ in DIRECT:
            self.calls[name] = self.bind(name, [C.c_void_p, C.POINTER(kind)], optional=True)
        self.get_fields = self.bind('nvmlDeviceGetFieldValues', [C.c_void_p, C.c_int, C.POINTER(NVMLFieldValue)], optional=True)

    def bind(self, name, args, optional=False):
        try: return C.CFUNCTYPE(C.c_int, *args)((name, self.lib))
        except AttributeError:
            if optional: return None
            raise

    def versions(self):
        result = {'definitions': DOCS, 'field_struct_size': C.sizeof(NVMLFieldValue),
                  'field_struct_offsets': {name: getattr(NVMLFieldValue, name).offset for name, _ in NVMLFieldValue._fields_},
                  'ctypes_sizes': {'uint': C.sizeof(C.c_uint), 'ulong': C.sizeof(C.c_ulong), 'ulonglong': C.sizeof(C.c_ulonglong)},
                  'library_argument': 'libnvidia-ml.so.1', 'library_maps': []}
        for name in ('nvmlSystemGetNVMLVersion', 'nvmlSystemGetDriverVersion'):
            output = C.create_string_buffer(256)
            call = timed_call(self.bind(name, [C.POINTER(C.c_char), C.c_uint], optional=True), output, len(output))
            call['value'] = output.value.decode(errors='replace') if call['status'] == 0 else None
            result[name] = call
        maps = Path('/proc/self/maps')
        if maps.exists():
            for line in maps.read_text().splitlines():
                if 'libnvidia-ml.so' in line:
                    path = Path(line.split()[-1])
                    if path.is_file() and not any(x['path'] == str(path) for x in result['library_maps']):
                        result['library_maps'].append({'path': str(path), 'sha256': energy.file_sha(path)})
        return result

    def direct(self, name, kind, unit):
        value = kind()
        row = timed_call(self.calls[name], self.handle, C.byref(value))
        row.update(api=name, implementation='typed_ctypes', raw_unit=unit,
                   raw_value=int(value.value) if row['status'] == 0 else None,
                   output_ctype=kind.__name__, scope='direct allocated GPU device API; physical domain not inferred')
        return row

    def fields(self):
        values = (NVMLFieldValue*len(FIELDS))()
        for value, (identifier, _, _) in zip(values, FIELDS):
            value.fieldId, value.scopeId = identifier, 0
        row = timed_call(self.get_fields, self.handle, len(values), values)
        row.update(api='nvmlDeviceGetFieldValues', implementation='typed_ctypes', fields=[])
        for value, (_, name, unit) in zip(values, FIELDS):
            row['fields'].append(dict(decode_field(value, row['status']), name=name, raw_unit=unit))
        return row

    def close(self):
        status = self.shutdown()
        if status: raise RuntimeError('Typed NVML shutdown status '+str(status))


class WrapperLibraryProxy:
    """Observe actual current-wrapper call sites without changing its code."""
    def __init__(self, library):
        self.library, self.receipts = library, []

    def __getattr__(self, name):
        function = getattr(self.library, name)
        types = {name: (kind, unit) for name, kind, unit in DIRECT}
        if name not in types: return function
        kind, unit = types[name]
        def observed(handle, pointer):
            row = timed_call(function, handle, pointer)
            row.update(api=name, implementation='unchanged_GPUPowerTrace_wrapper', raw_unit=unit,
                       raw_value=int(C.cast(pointer, C.POINTER(kind)).contents.value) if row['status'] == 0 else None,
                       output_ctype=kind.__name__, scope='direct allocated GPU device API; physical domain not inferred')
            self.receipts.append(row)
            return row['status']
        return observed


class CurrentWrapper:
    def __init__(self, uuid, unused_path):
        self.wrapper = energy.GPUPowerTrace(uuid, unused_path)
        self.proxy = WrapperLibraryProxy(self.wrapper.lib)
        self.wrapper.lib = self.proxy

    def read(self):
        self.proxy.receipts = []
        before = stamp()
        output = self.wrapper.read()
        return {'pre': before, 'post': stamp(), 'output': output, 'calls': self.proxy.receipts}

    def close(self):
        # Never start the wrapper thread: only its unchanged read() is probed.
        status = self.proxy.library.nvmlShutdown()
        if status: raise RuntimeError('Wrapper NVML shutdown status '+str(status))


class InterfaceTrace:
    def __init__(self, typed, wrapper, path, cadence, epoch):
        self.typed, self.wrapper, self.path = typed, wrapper, Path(path)
        self.cadence, self.epoch, self.phase = cadence, epoch, 'epoch_start'
        self.stop_event, self.errors = threading.Event(), []
        self.count = 0; self.last_post_monotonic_ns = 0; self.last_pre_monotonic_ns = 0
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def read(self):
        row = {'sample_index': self.count, 'epoch': self.epoch, 'requested_cadence_seconds': self.cadence,
               'pre': stamp(), 'phase_at_start': self.phase}
        row['typed_direct'] = [self.typed.direct(*args) for args in DIRECT]
        row['current_wrapper'] = self.wrapper.read()
        row['field_query'] = self.typed.fields()
        row.update(post=stamp(), phase_at_finish=self.phase)
        return row

    def loop(self):
        try:
            with self.path.open('x') as stream:
                while not self.stop_event.is_set():
                    start = time.monotonic()
                    row = self.read(); self.count += 1
                    stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
                    self.last_post_monotonic_ns = row['post']['monotonic_ns']
                    self.last_pre_monotonic_ns = row['pre']['monotonic_ns']
                    self.stop_event.wait(max(0., self.cadence-(time.monotonic()-start)))
        except BaseException:
            self.errors.append(traceback.format_exc())

    def start(self): self.thread.start()

    def check(self):
        if self.errors: raise RuntimeError('Interface trace failed: '+str(self.errors))

    def bracket_after(self, boundary):
        deadline = time.monotonic()+3
        # A row that merely finishes after a boundary can contain earlier
        # per-call midpoints. Require a fully recorded row that began after it.
        while self.last_pre_monotonic_ns <= boundary:
            self.check()
            if time.monotonic() >= deadline: raise RuntimeError('Trace did not bracket final epoch boundary')
            time.sleep(.01)

    def stop(self):
        self.stop_event.set(); self.thread.join(timeout=5)
        if self.thread.is_alive(): raise RuntimeError('NVML interface thread failed to stop')
        self.check()


def validate_config(config):
    if (config.get('schema') != 'stereo-study-energy-interfaces-v1' or config.get('registered') is not True
            or config.get('driver') != 'energy_interfaces' or config.get('stage') != 'engineering_diagnostic'
            or config.get('scope') != 'engineering' or config.get('test_access') != 'sealed'
            or config.get('allocated_gpu_hour_cap') != 24):
        raise ValueError('Registered bounded interface engineering config required')
    ordered_block = config.get('order_protocol') is not None
    if ordered_block and config['order_protocol'] != ORDER_PROTOCOL:
        raise ValueError('Exact prospective same-allocation reverse/forward order block required')
    cap, scheduler, cadences = (1200, '00:20:00', ORDER_PROTOCOL['cadences_seconds']) if ordered_block else (600, '00:10:00', [.05,.5])
    if config.get('max_seconds') != cap or config.get('scheduler_resources') != {'time': scheduler}:
        raise ValueError('Exact registered time reservation required')
    expected = [{'cadence_seconds': cadence, 'idle_seconds': 60, 'load_seconds': 90, 'cooldown_seconds': 30} for cadence in cadences]
    if config.get('epochs') != expected or config.get('val_samples') != 20 or config.get('boundary') != 'gpu_resident':
        raise ValueError('Fixed50ms/500ms order, continuous60/90/30 epochs and full20resident workload required')
    if (config.get('field_scope_id') != 0 or config.get('field_ids') != [x[0] for x in FIELDS]
            or config.get('context_sampling_seconds') != 1 or config.get('complete_val_cycles') is not True
            or config.get('require_zero_allocations') is not True):
        raise ValueError('Verified fields/scope,1s process checks, whole cycles and explicit zero-allocation guard required')
    cases = config.get('cases') or []
    if len(cases) != 1 or cases[0].get('case_id') != 'ALIF-I-event':
        raise ValueError('Only existing selected ALIF-I-event checkpoint is registered')
    # Reuse the already audited case SHA/name validation, without weakening its
    # energy protocol or pretending this diagnostic has that protocol schema.
    check = energy.example_config()
    check.update(registered=True, cases=copy.deepcopy(cases))
    check['resource_forecast']['sha256'] = '1'*64
    energy.validate_energy_config(check)
    forecast = config.get('resource_forecast', {})
    fixed = 180*len(expected)
    if (forecast.get('reserved_gpu_seconds') != cap or forecast.get('fixed_phase_seconds') != fixed
            or forecast.get('setup_cleanup_headroom_seconds') != cap-fixed):
        raise ValueError('Prospective full reservation forecast required')
    if config.get('expected_hostname') != 'cccl3': raise ValueError('Verified cluster hostname required')
    return True


def clean_allocation_receipt(torch, device):
    """Called only after the scoped workload has returned no GPU references."""
    def allocation():
        return {'allocated_bytes': torch.cuda.memory_allocated(device), 'reserved_bytes': torch.cuda.memory_reserved(device)}
    before = allocation()
    gc.collect(); torch.cuda.synchronize(device); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
    ordinary = allocation()
    # PyTorch2.7.1 owns a process-global cuBLAS(handle,stream)->DataPtr map;
    # its non-Hopper default is exactly8,519,680bytes. gc/empty_cache cannot
    # release a live map entry. The private binding below clears that map.
    # Source: pytorch/pytorch v2.7.1 aten/src/ATen/cuda/CublasHandlePool.cpp,
    # clearCublasWorkspaces/parseChosenWorkspaceSize/getNewWorkspace; binding:
    # torch/csrc/cuda/Module.cpp. Feature-detect rather than assume availability.
    # Call only after model release + synchronization, never during inference.
    clear_workspaces = getattr(getattr(torch, '_C', None), '_cuda_clearCublasWorkspaces', None)
    release = {'hook': 'torch._C._cuda_clearCublasWorkspaces', 'available': callable(clear_workspaces),
               'attempted': False, 'status': 'not_needed_zero_after_ordinary_cleanup',
               'torch_version': getattr(torch, '__version__', None),
               'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
               'configuration_changed': False,
               'scope': 'untimed process-owned library workspace release; model already out of scope; no GPU power/clock setter'}
    after_clear = None
    if ordinary != {'allocated_bytes': 0, 'reserved_bytes': 0}:
        if callable(clear_workspaces):
            release['attempted'] = True
            try:
                clear_workspaces(); torch.cuda.synchronize(device)
                after_clear = allocation()
                release.update(status='applied', allocated_bytes_released=ordinary['allocated_bytes']-after_clear['allocated_bytes'])
                torch.cuda.empty_cache(); torch.cuda.synchronize(device)
            except BaseException as error:
                release.update(status='failed', error_type=type(error).__name__, error=repr(error))
        else:
            release['status'] = 'unavailable_nonzero_allocation_preserved'
    after = allocation()
    return {'pre_cleanup': before, 'after_ordinary_cleanup': ordinary,
            'cublas_workspace_release': release, 'after_workspace_release_before_empty_cache': after_clear,
            'after_cleanup': after,
            'zero_allocations_verified': after == {'allocated_bytes': 0, 'reserved_bytes': 0} and release['status'] != 'failed',
            'scope': 'PyTorch allocated/reserved tensors zero; CUDA context and driver allocations may remain',
            'method': 'scoped workload returns CPU records only; baseline reset_net; gc+synchronize+empty_cache; feature-detected cuBLAS workspace release if residual; synchronize+empty_cache; strict final zero check'}


def require_clean(torch, device, path):
    receipt = clean_allocation_receipt(torch, device)
    energy.atomic_json(path, receipt)
    if not receipt['zero_allocations_verified']:
        raise RuntimeError('FAILED_TENSOR_FREE_ALLOCATION_GUARD: '+json.dumps(receipt))
    return receipt


def scoped_load(case, config, registry, device, seconds, folder, phase, check):
    """No model/input/output tensor escapes this scope, even through returns."""
    import torch
    model, samples = None, None
    try:
        phase('setup_load')
        model, samples, provenance = energy.load_case(case, config, registry, device, boundary='gpu_resident')
        if (provenance['model_family'] != 'baseline' or provenance['variant']['overrides']
                or provenance['variant']['training_model_config']['input_channels'] != 2
                or provenance['variant']['training_model_config']['architecture'] != 'original_I'):
            raise RuntimeError('Exact event baseline with no model override required')
        phase('load')
        start = time.monotonic(); delivered = 0
        with (folder/'outputs.jsonl').open('x') as stream, torch.inference_mode():
            while time.monotonic()-start < seconds or delivered % len(samples):
                check()
                before = stamp()
                row = energy.run_output(model, samples[delivered % len(samples)], 'gpu_resident', device)
                row.update(output_index=delivered, pre=before, post=stamp())
                stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
                delivered += 1
        end = time.monotonic()
        phase('cleanup')
        state_after = energy.tensor_state_sha(model)
        if state_after != provenance['trained_state_sha256']:
            raise RuntimeError('Inference changed trained state')
        return {'provenance': provenance, 'load_start_monotonic_seconds': start,
                'load_end_monotonic_seconds': end, 'delivered_outputs': delivered,
                'complete_cycles': delivered//len(samples), 'trained_state_unchanged': True}
    finally:
        if model is not None:
            # Legacy forward resets at entry, so the last forward can retain
            # transient SpikingJelly voltages. Reset only after inference.
            from spikingjelly.activation_based import functional
            before_reset = energy.tensor_state_sha(model)
            functional.reset_net(model.model)
            after_reset = energy.tensor_state_sha(model)
            energy.atomic_json(folder/'neuron-reset.json', {'functional_reset_net_applied': True,
                'trained_state_sha256_before_reset': before_reset, 'trained_state_sha256_after_reset': after_reset,
                'trained_state_unchanged': before_reset == after_reset,
                'scope': 'transient neuron memory reset after workload only; no weight/model-code change'})
            if before_reset != after_reset: raise RuntimeError('Cleanup neuron reset changed trained weights')
        model = samples = None


def wait_phase(seconds, check):
    began = time.monotonic()
    while time.monotonic()-began < seconds:
        check(); time.sleep(min(.05, max(0., seconds-(time.monotonic()-began))))
    return {'start_monotonic_seconds': began, 'end_monotonic_seconds': time.monotonic()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source-registry', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--validate-config', action='store_true')
    args = parser.parse_args(); config = json.loads(args.config.read_text()); validate_config(config)
    if args.validate_config:
        print(json.dumps({'status': 'VALID_CONFIG_ONLY', 'config_sha256': energy.file_sha(args.config),
                          'reservation_gpu_seconds': config['max_seconds'], 'fixed_phase_seconds': 180*len(config['epochs'])})); return
    if (not os.environ.get('SLURM_JOB_ID') or socket.gethostname() != config['expected_hostname']
            or args.source_registry is None or args.output is None):
        raise RuntimeError('Authorized cccl3 Slurm allocation, immutable source and fresh output required')
    from train_study import verify_source
    registry = verify_source(args.source_registry)
    source = args.source_registry.parent.resolve()
    for path in (Path(__file__).resolve(), args.config.resolve()):
        if registry['source_files'].get(str(path.relative_to(source))) != energy.file_sha(path):
            raise RuntimeError('Diagnostic script/config absent from exact verified source')
    if args.output.resolve().is_relative_to(source): raise ValueError('Output must be outside immutable source')
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'stereo-study-energy-interfaces-result-v1', 'status': 'INITIALIZING',
        'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'config_sha256': energy.file_sha(args.config),
        'source_identity_sha256': registry['source_identity_sha256'], 'source_registry_sha256': energy.file_sha(args.source_registry),
        'git_commit': registry['git_commit'], 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'protocol': config,
        'epochs': [], 'test_access': False, 'scope': 'read-only same-instrument interface/cadence capability; no calibration, correction, architecture ranking or energy-efficiency claim',
        'analysis_policy': 'retain raw traces for middle60s/all30s subwindows/full transitions; no interface chosen from these readings',
        'call_order': ['typed_power', 'typed_counter', 'unchanged_wrapper_power', 'unchanged_wrapper_counter', 'typed_field_batch'],
        'observer_policy': 'combined sequential calls at registered50ms/500ms cadence; host poll count is not physical sample resolution; registered epochs are ordered and thermally/history confounded',
        'definitions': DOCS, 'host_start': energy.host_snapshot()}
    save = lambda: energy.atomic_json(args.output/'result.json', result)
    save(); began = time.monotonic(); context = typed = wrapper = trace = None
    def deadline(number, frame): raise TimeoutError('Registered interface diagnostic deadline')
    previous = signal.signal(signal.SIGALRM, deadline); signal.setitimer(signal.ITIMER_REAL, config['max_seconds']-5)
    try:
        import torch
        if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
                or not any(x in torch.cuda.get_device_name(0) for x in ('4090', '5090'))):
            raise RuntimeError('Exactly one authorized RTX4090/5090 required')
        device = torch.device('cuda:0'); gpu = torch.cuda.get_device_properties(device)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.cuda.synchronize(device)
        initial = energy.gpu_snapshot(gpu.uuid)
        if initial['foreign_compute_processes']: raise RuntimeError('Foreign allocated-UUID compute process')
        context = energy.GPUContextTrace(gpu.uuid, args.output/'gpu-context.jsonl', initial['settings']['power.limit'])
        context.start()
        typed = TypedNVML(gpu.uuid); wrapper = CurrentWrapper(gpu.uuid, args.output/'unused-wrapper-thread.jsonl')
        result.update(status='RUNNING', gpu=gpu.name, gpu_uuid=energy.normalized_uuid(gpu.uuid),
                      torch=torch.__version__, cuda=torch.version.cuda, precision='FP32', tf32=False,
                      initial_gpu_snapshot=initial, nvml=typed.versions())
        save()
        for index, epoch in enumerate(config['epochs'], 1):
            folder = args.output/('epoch-'+str(index)); folder.mkdir()
            row = {'epoch': index, 'cadence_seconds': epoch['cadence_seconds'], 'status': 'RUNNING', 'phases': []}
            result['epochs'].append(row); save()
            trace = InterfaceTrace(typed, wrapper, folder/'interfaces.jsonl', epoch['cadence_seconds'], index)
            trace.start()
            def phase(name):
                trace.phase = name
                event = dict(stamp(), phase=name)
                row['phases'].append(event)
                with (folder/'phases.jsonl').open('a') as stream: stream.write(json.dumps(event)+'\n')
                save()
            def check(): context.check(); trace.check()
            phase('cleanup_before_idle')
            row['before_idle_allocation'] = require_clean(torch, device, folder/'before-idle-allocation.json')
            # All seven calls in this completed row precede the declared idle
            # start. Starting the thread alone did not guarantee that in v18.
            trace.bracket_after(stamp()['monotonic_ns'])
            row['idle_leading_trace_post_monotonic_ns'] = trace.last_post_monotonic_ns
            phase('idle'); row['idle'] = wait_phase(epoch['idle_seconds'], check)
            # Only CPU dictionaries return; the helper resets last-step neuron
            # memories and its complete frame exits before the zero guard.
            row['workload'] = scoped_load(config['cases'][0], config, registry, device,
                                           epoch['load_seconds'], folder, phase, check)
            row['before_cooldown_allocation'] = require_clean(torch, device, folder/'before-cooldown-allocation.json')
            phase('cooldown'); row['cooldown'] = wait_phase(epoch['cooldown_seconds'], check)
            phase('epoch_end'); trace.bracket_after(row['phases'][-1]['monotonic_ns'])
            trace.stop(); row.update(status='COMPLETED', trace_rows=trace.count)
            trace = None; save()
        result.update(status='COMPLETED_READ_ONLY_NVML_INTERFACE_DIAGNOSTIC', elapsed_seconds=time.monotonic()-began,
                      host_finish=energy.host_snapshot())
    except BaseException as error:
        if result['epochs'] and result['epochs'][-1]['status'] == 'RUNNING':
            result['epochs'][-1].update(status='INCOMPLETE', stopped=stamp())
        result.update(status='INCOMPLETE_DEADLINE' if isinstance(error, TimeoutError) else 'FAILED_INTERFACE_DIAGNOSTIC',
                      error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(), elapsed_seconds=time.monotonic()-began)
        save(); raise
    finally:
        # A stop error is recorded without erasing the original failure receipt.
        errors = []
        for obj in (trace, context, wrapper, typed):
            if obj is not None:
                try: obj.stop() if hasattr(obj, 'stop') else obj.close()
                except BaseException as error: errors.append(repr(error))
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)
        if errors:
            result['monitor_shutdown_errors'] = errors
            if result['status'].startswith('COMPLETED'): result['status'] = 'FAILED_MONITOR_SHUTDOWN'
        save()
        if errors and result['status'] == 'FAILED_MONITOR_SHUTDOWN':
            raise RuntimeError('Diagnostic monitoring shutdown failed: '+str(errors))


if __name__ == '__main__': main()
