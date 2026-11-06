"""Synthetic CPU telemetry only; no NVML, model, sensor or GPU execution."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
import tempfile
from unittest import mock

STUDY=Path(__file__).resolve().parents[1]/'experiments/stereo-spiking-transformer-study'
SPEC=importlib.util.spec_from_file_location('interface_analysis_controls',STUDY/'scripts/analyze_energy_interfaces.py')
D=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(D)


def clock(t,span=.02):
    return {'pre':{'monotonic_ns':round((t-span/2)*1e9),'epoch_ns':round((1700000000+t-span/2)*1e9)},
            'post':{'monotonic_ns':round((t+span/2)*1e9),'epoch_ns':round((1700000000+t+span/2)*1e9)}}


def points(raw_function,ts):return [D.point(clock(t),0,raw_function(t)) for t in ts]


def trace_rows(count=6):
    rows=[];base=2**50
    for i in range(count):
        t=float(i+1)
        def call(api,when):
            return dict(clock(when),api=api,status=0,raw_value=20000 if api==D.POWER_API else base+round(20000*when),raw_unit='mW' if api==D.POWER_API else 'mJ')
        typed=[call(D.POWER_API,t+.1),call(D.COUNTER_API,t+.25)]
        wrapper=[call(D.POWER_API,t+.45),call(D.COUNTER_API,t+.55)]
        query=dict(clock(t+.8),status=0,fields=[])
        for fid in (185,186,83):
            query['fields'].append({'field_id':fid,'scope_id':0,'scope':'GPU only','field_status':0,
                'raw_unit':'mJ' if fid==83 else 'mW','raw_value':base+round(20000*(t+.8)) if fid==83 else 20000,
                'timestamp_epoch_microseconds':round((1700000000+t+.8-.25)*1e6),'latency_microseconds':12,
                'value_type':3,'value_decode_status':'decoded','value_union_bytes_hex':'0000000000000000'})
        outer=clock(t+.475,span=.95)
        rows.append(dict(outer,sample_index=i,epoch=1,requested_cadence_seconds=.5,
            typed_direct=typed,field_query=query,phase_at_start='load',phase_at_finish='load',
            current_wrapper=dict(clock(t+.5,span=.2),calls=wrapper,
                output={'uuid':'GPU-mock','watts':20.,'joules_counter':wrapper[1]['raw_value']/1000,
                        'watts_status':0,'joules_counter_status':0})))
    return rows


def completed_result():
    ids=['scene%02d/%s'%(i,frame) for i in range(10) for frame in ('a','b')]
    result={'schema':'stereo-study-energy-interfaces-result-v1','status':'COMPLETED_READ_ONLY_NVML_INTERFACE_DIAGNOSTIC',
        'source_identity_sha256':D.SOURCE_ID,'source_registry_sha256':D.REGISTRY_SHA,'config_sha256':D.CONFIG_SHA,
        'test_access':False,'precision':'FP32','tf32':False,
        'protocol':{'max_seconds':600,'val_samples':20,'boundary':'gpu_resident','cases':[{'checkpoint_sha256':D.CHECKPOINT_SHA}]},'epochs':[]}
    for i,cadence in enumerate((.05,.5),1):
        t=(i-1)*200.
        receipt={'zero_allocations_verified':True,'after_cleanup':{'allocated_bytes':0,'reserved_bytes':0}}
        p={'checkpoint_sha256':D.CHECKPOINT_SHA,'training_source_identity_sha256':D.TRAIN_SOURCE,'training_config_sha256':D.TRAIN_CONFIG,
            'training_initial_weights_sha256':D.TRAIN_INITIAL,'manifest_sha256':D.MANIFEST,'training_seed':9471,'model_family':'baseline',
            'variant':{'overrides':{},'training_model_config':{'input_channels':2,'architecture':'original_I'}},
            'val_sample_ids':ids,'trained_state_sha256':'a'*64}
        result['epochs'].append({'epoch':i,'cadence_seconds':cadence,'status':'COMPLETED',
            'before_idle_allocation':copy.deepcopy(receipt),'before_cooldown_allocation':copy.deepcopy(receipt),
            'idle':{'start_monotonic_seconds':t,'end_monotonic_seconds':t+60},
            'workload':{'load_start_monotonic_seconds':t+62,'load_end_monotonic_seconds':t+152,'delivered_outputs':40,
                'complete_cycles':2,'trained_state_unchanged':True,'provenance':p},
            'cooldown':{'start_monotonic_seconds':t+154,'end_monotonic_seconds':t+184}})
    return result,ids


class InterfaceAnalysis(unittest.TestCase):
    def test_prospectively_registered_repaired_source_keeps_completion_and_stage_gates(self):
        result,ids=completed_result()
        result.update(source_identity_sha256=D.REPAIRED_SOURCE_ID,source_registry_sha256=D.REPAIRED_REGISTRY_SHA)
        for epoch in result['epochs']:
            for phase in ('idle','cooldown'):
                for key in epoch[phase]:epoch[phase][key]+=1
            for key in ('load_start_monotonic_seconds','load_end_monotonic_seconds'):epoch['workload'][key]+=1
            epoch['idle_leading_trace_post_monotonic_ns']=round((epoch['idle']['start_monotonic_seconds']-.01)*1e9)
            for key in ('before_idle_allocation','before_cooldown_allocation'):
                epoch[key].update(after_ordinary_cleanup={'allocated_bytes':0,'reserved_bytes':0},
                    cublas_workspace_release={'configuration_changed':False,'status':'not_needed_zero_after_ordinary_cleanup'})
        with mock.patch.object(D,'VAL_IDS_SHA',D.canonical_sha(ids)):
            self.assertTrue(D.validate_result(result))
            for change in ('failed','registry','stage','leading'):
                invalid=copy.deepcopy(result)
                if change=='failed':invalid['status']='FAILED_INTERFACE_DIAGNOSTIC'
                if change=='registry':invalid['source_registry_sha256']=D.REGISTRY_SHA
                if change=='stage':del invalid['epochs'][0]['before_idle_allocation']['after_ordinary_cleanup']
                if change=='leading':invalid['epochs'][0]['idle_leading_trace_post_monotonic_ns']=2*10**9
                with self.assertRaises(ValueError):D.validate_result(invalid)

    def test_exact_four_epoch_source_order_and_cleanup_gates(self):
        r,ids=completed_result();base=copy.deepcopy(r['epochs'][0])
        r.update(source_identity_sha256=D.ORDER_BLOCK_SOURCE,source_registry_sha256=D.ORDER_BLOCK_REGISTRY,config_sha256=D.ORDER_BLOCK_CONFIG)
        r['protocol'].update(max_seconds=1200,order_protocol=dict(name='reverse_then_forward_same_allocation_v1',cadences_seconds=[.5,.05,.05,.5]))
        r['epochs']=[]
        for i,cadence in enumerate([.5,.05,.05,.5]):
            epoch=copy.deepcopy(base);epoch.update(epoch=i+1,cadence_seconds=cadence)
            for phase in ('idle','cooldown'):
                for key in epoch[phase]:epoch[phase][key]+=i*200+1
            for key in ('load_start_monotonic_seconds','load_end_monotonic_seconds'):epoch['workload'][key]+=i*200+1
            epoch['idle_leading_trace_post_monotonic_ns']=round((epoch['idle']['start_monotonic_seconds']-.01)*1e9)
            for key in ('before_idle_allocation','before_cooldown_allocation'):
                epoch[key].update(after_ordinary_cleanup={'allocated_bytes':0,'reserved_bytes':0},cublas_workspace_release={'configuration_changed':False,'status':'not_needed_zero_after_ordinary_cleanup'})
            r['epochs'].append(epoch)
        with mock.patch.object(D,'VAL_IDS_SHA',D.canonical_sha(ids)):
            self.assertTrue(D.validate_result(r))
            for change in ('registry','config','order','count','cleanup','cross_epoch'):
                broken=copy.deepcopy(r)
                if change=='registry':broken['source_registry_sha256']=D.REPAIRED_REGISTRY_SHA
                if change=='config':broken['config_sha256']=D.CONFIG_SHA
                if change=='order':broken['epochs'][0]['cadence_seconds']=.05
                if change=='count':broken['epochs'].pop()
                if change=='cleanup':broken['epochs'][2]['before_cooldown_allocation']['after_cleanup']['allocated_bytes']=1
                if change=='cross_epoch':broken['epochs'][3]['idle']['start_monotonic_seconds']=1
                with self.assertRaises(ValueError):D.validate_result(broken)

    def test_analytic_piecewise_power_and_large_integer_counter_units(self):
        times=list(range(1,102));origin=2**60+17
        power=points(lambda t:1000*(10+2*t),times)
        counter=points(lambda t:origin+1000*(10*t+t*t),times)
        a=D.window_readout(power,10,40,'power');b=D.window_readout(counter,10,40,'counter')
        self.assertTrue(a['available']);self.assertTrue(b['available'])
        self.assertEqual(a['instrument_readout_joules'],1800.)
        self.assertEqual(b['instrument_readout_joules'],1800.)
        self.assertEqual(a['mean_instrument_readout_watts'],60.)
        # Linear interpolation within a segment is analytically trapezoidal.
        constant=points(lambda t:25000,[1,2,3,4])
        self.assertEqual(D.window_readout(constant,1.25,3.75,'power')['instrument_readout_joules'],62.5)

    def test_sequential_typed_wrapper_use_own_time_windows(self):
        streams,audit=D.extract_streams(trace_rows(),1,.5,'GPU-mock')
        self.assertEqual(len(streams),7)
        for key,stream in streams.items():
            row=D.window_readout(stream,2,5,D.STREAMS[key][0])
            self.assertTrue(row['available'],key)
            self.assertAlmostEqual(row['instrument_readout_joules'],60.,places=6)
        self.assertAlmostEqual(audit['sequential_typed_wrapper_alignment']['counter']['same_poll_midpoint_delta_seconds']['median'],.3)
        self.assertEqual(audit['sequential_typed_wrapper_alignment']['counter']['raw_wrapper_minus_typed']['median'],6000)
        self.assertEqual(audit['wrapper_raw_to_scaled_absolute_difference']['maximum'],0.)
        field=D.stream_summary(streams['field185_average_power'],'power')['native_field_metadata']
        self.assertAlmostEqual(field['apparent_source_age_seconds']['median'],.25,places=5)
        self.assertIn('NOT measurement latency',field['clock_domain_status'])

    def test_unsupported_field_status_and_failed_reading_not_bridged(self):
        rows=trace_rows()
        for row in rows:
            f=row['field_query']['fields'][1];f['field_status']=3;f['raw_value']=None;f['value_decode_status']='unavailable'
        streams,_=D.extract_streams(rows,1,.5,'GPU-mock')
        unavailable=D.window_readout(streams['field186_instant_power'],2,5,'power')
        self.assertFalse(unavailable['available']);self.assertEqual(unavailable['status_counts'],{'3':5})
        self.assertTrue(D.window_readout(streams['typed_power'],2,5,'power')['available'])
        failed=points(lambda t:20000,[1,2,3,4]);failed[1]['valid']=False;failed[1]['status']=3
        self.assertEqual(D.window_readout(failed,1.5,3.5,'power')['reason'],'failed_unsupported_or_nonfinite_reading')

    def test_brackets_gaps_nonfinite_and_counter_continuity(self):
        base=points(lambda t:1000*t,[1,2,3,4])
        self.assertEqual(D.window_readout(base,.5,3,'counter')['reason'],'unbracketed_window')
        bad=copy.deepcopy(base);bad[2]['raw']=1
        self.assertEqual(D.window_readout(bad,1,4,'counter')['reason'],'counter_decrease_reset_or_wrap')
        gap=points(lambda t:20000,[1,2,5,6])
        self.assertEqual(D.window_readout(gap,1,6,'power')['reason'],'host_midpoint_gap_above_2s')
        nonfinite=points(lambda t:float('nan') if t==2 else 20000,[1,2,3])
        self.assertFalse(D.window_readout(nonfinite,1,3,'power')['available'])
        backwards=copy.deepcopy(base);backwards[2]['t']=1.5
        self.assertEqual(D.window_readout(backwards,1,4,'counter')['reason'],'nonmonotonic_host_midpoints')

    def test_complete_epoch_checkpoint_and_zero_memory_receipt_guards(self):
        result,ids=completed_result()
        with mock.patch.object(D,'VAL_IDS_SHA',D.canonical_sha(ids)):
            self.assertTrue(D.validate_result(result))
            for mutate in (lambda x:x.update(status='RUNNING'),lambda x:x['epochs'].pop(),
                           lambda x:x['epochs'][0]['workload'].update(complete_cycles=1),
                           lambda x:x['epochs'][0]['before_idle_allocation']['after_cleanup'].update(allocated_bytes=1),
                           lambda x:x.update(source_identity_sha256='0'*64),
                           lambda x:x['epochs'][0]['workload'].update(load_end_monotonic_seconds=151)):
                bad=copy.deepcopy(result);mutate(bad)
                with self.assertRaises(ValueError):D.validate_result(bad)

    def test_prospective_windows_are_fixed_not_selected_from_values(self):
        result,_=completed_result();windows=dict((name,(start,end)) for name,start,end in D.prescribed_windows(result['epochs'][0]))
        self.assertEqual(windows['idle_first30'],(0.,30.));self.assertEqual(windows['idle_second30'],(30.,60.))
        self.assertEqual(windows['load_first30'],(62.,92.));self.assertEqual(windows['load_middle30'],(92.,122.))
        self.assertEqual(windows['load_last30'],(122.,152.));self.assertEqual(windows['load_middle60'],(77.,137.))
        self.assertEqual(windows['full_transition_including_setup_cleanup'],(0.,184.))

    def test_copied_folder_analysis_binds_files_and_retains_invalid_windows(self):
        result,ids=completed_result()
        result.update(git_commit='fixture',slurm_job_id='fixture',gpu_uuid='GPU-mock',gpu='RTX5090')
        with tempfile.TemporaryDirectory() as raw, mock.patch.object(D,'VAL_IDS_SHA',D.canonical_sha(ids)):
            folder=Path(raw)
            for index,epoch in enumerate(result['epochs'],1):
                rows=trace_rows(187)
                if index==2:
                    def shift(value):
                        if isinstance(value,list):return [shift(x) for x in value]
                        if not isinstance(value,dict):return value
                        return {k:v+200_000_000_000 if k in ('monotonic_ns','epoch_ns') else v+200_000_000 if k=='timestamp_epoch_microseconds'
                                else shift(v) for k,v in value.items()}
                    rows=shift(rows)
                for row in rows:row.update(epoch=index,requested_cadence_seconds=epoch['cadence_seconds'])
                epoch['trace_rows']=len(rows);sub=folder/('epoch-'+str(index));sub.mkdir()
                (sub/'interfaces.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
            (folder/'result.json').write_text(json.dumps(result))
            analyzed=D.analyze(folder)
            self.assertEqual(len(analyzed['epochs']),2);self.assertEqual(len(analyzed['streams']),7)
            self.assertEqual(len(analyzed['source_files']),3)
            # A deliberately absent initial bracket remains invalid, never 0J.
            first=analyzed['epochs'][0]['windows'][0]['streams']['typed_power']
            self.assertFalse(first['available']);self.assertIsNone(first['instrument_readout_joules'])
            middle=next(x for x in analyzed['epochs'][0]['windows'] if x['name']=='load_middle60')
            self.assertEqual(middle['streams']['typed_power']['instrument_readout_joules'],1200.)
            json.dumps(analyzed,allow_nan=False)
            self.assertIn('NOT_CALIBRATION',analyzed['status'])


if __name__=='__main__':unittest.main()
