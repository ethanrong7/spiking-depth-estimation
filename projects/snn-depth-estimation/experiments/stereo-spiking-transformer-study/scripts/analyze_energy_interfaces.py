"""Pure-CPU analysis of copied completed v18 NVML interface trace receipts.

--input COPIED_RUN --output-prefix FRESH writes compact JSON+CSV. No fetch,
GPU/NVML calls, model imports, tensors, checkpoints or sensor arrays. Raw traces
stay in their ignored input folder. Unsupported/invalid windows stay explicit.
"""
import argparse
import bisect
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import csv

SOURCE_ID='8e97d9b0c8e6da7857b6d72e15cfe40f7aafbf3ef949a117b12bea9dd827f592'
REGISTRY_SHA='d7dd393b8ae0a20a19b1adb92786e11f4d46aeb3d6ee459fe24c4b0cf6afbb84'
REPAIRED_SOURCE_ID='246efb5f4ee5a5ed0693c1fe0c5f70975e5fb8b9224b74cffd06615fb2cdb835'
REPAIRED_REGISTRY_SHA='627744e2f88dc23437fe5c6be2f3643d2f2da49f897316e717d78a700adace11'
REGISTERED_SOURCES={SOURCE_ID:REGISTRY_SHA,REPAIRED_SOURCE_ID:REPAIRED_REGISTRY_SHA}
CONFIG_SHA='6957b5ad1e7045337786eea509d71a75a8c46acc6aa3429333e589afc9eea72f'
ORDER_BLOCK_SOURCE='28cd5613efa68b6be1b807c6b17d1702ef05b50d645b5c52114c801b2c5c67af'
ORDER_BLOCK_REGISTRY='2cc284683336fd842b0f456151d294cbced7e1a846bcff7cd0bc6eeafa556737'
ORDER_BLOCK_CONFIG='3a9681232411c86d3e0b88f9220f138eebdcdfdc88f65ab20604b2938bdeaa5f'
REGISTERED_SOURCES[ORDER_BLOCK_SOURCE]=ORDER_BLOCK_REGISTRY
CHECKPOINT_SHA='8bb679229140bec3309e851c75bd459f34198225ca5bc4f786eba33e8aaa901c'
TRAIN_SOURCE='e84cbdafeeb332ac33d6ae509168cd418bae1958bcc8d47b4fe2cc4c7349149a'
TRAIN_CONFIG='a63f9db743777e3dbff3ce527c173e5a4cce5f4ae36948827c77cbd86e66e245'
TRAIN_INITIAL='5edf49c5a5ca0740fb9d96b059f9ec04b37d7ec2b7a46734cf783378f3d1848a'
MANIFEST='be80defb27c948b9766a6bb46381b56ce9dd1ee7062782f53dc6be2b518b99c5'
VAL_IDS_SHA='f4d15f5ed39ebf3a9f4accc0edb053163f63ec1fc9856d7da46d3650230b83ed'
MAX_GAP_SECONDS=2.
POWER_API='nvmlDeviceGetPowerUsage';COUNTER_API='nvmlDeviceGetTotalEnergyConsumption'
STREAMS={'typed_power':('power','mW'),'typed_counter':('counter','mJ'),
         'wrapper_power':('power','mW'),'wrapper_counter':('counter','mJ'),
         'field185_average_power':('power','mW'),'field186_instant_power':('power','mW'),
         'field83_counter':('counter','mJ')}
FIELD_IDS={185:'field185_average_power',186:'field186_instant_power',83:'field83_counter'}


def require(value,message):
    if not value:raise ValueError(message)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def finite(value):return type(value) in (int,float) and math.isfinite(value)


def stats(values):
    values=list(values)
    return {'count':len(values),'minimum':min(values) if values else None,
            'median':statistics.median(values) if values else None,'maximum':max(values) if values else None}


def call_clock(call):
    pre,post=call['pre'],call['post']
    for stamp in (pre,post):
        require(all(type(stamp.get(k)) is int and stamp[k] >= 0 for k in ('monotonic_ns','epoch_ns')), 'Invalid host clock receipt')
    require(post['monotonic_ns'] >= pre['monotonic_ns'], 'Call post precedes pre')
    return (pre['monotonic_ns']+post['monotonic_ns'])/2e9


def point(call,status,raw,*,field=None,no_call=False):
    timestamp=call_clock(call)
    decoded=status == 0 and finite(raw) and raw >= 0
    result={'t':timestamp,'raw':raw,'status':status,'valid':decoded,
            'pre':call['pre'],'post':call['post'],'metadata':field,
            'timestamp_assignment':'own_call_midpoint' if not no_call else 'unavailable_symbol_wrapper_envelope_only_no_call'}
    return result


def extract_streams(rows,epoch,cadence,uuid):
    streams={key:[] for key in STREAMS};envelopes=[];wrapper_scaling=[];alignments={'power':[],'counter':[]}
    for index,row in enumerate(rows):
        require(row['sample_index'] == index and row['epoch'] == epoch and row['requested_cadence_seconds'] == cadence,'Trace epoch/index/cadence mismatch')
        call_clock(row);envelopes.append(row)
        wrapper=row['current_wrapper'];call_clock(wrapper)
        require(wrapper['output']['uuid'] == uuid,'Wrapper trace escaped allocated UUID')
        for prefix,calls in (('typed',row['typed_direct']),('wrapper',wrapper['calls'])):
            for api,kind in ((POWER_API,'power'),(COUNTER_API,'counter')):
                matches=[x for x in calls if x['api'] == api]
                require(len(matches) <= 1,'Duplicate direct API call')
                label=prefix+'_'+kind
                if not matches:
                    require(prefix == 'wrapper','Missing typed API status receipt')
                    status=wrapper['output']['watts_status' if kind == 'power' else 'joules_counter_status']
                    require(status == 'symbol_missing','Wrapper call missing without unavailable symbol')
                    streams[label].append(point(wrapper,status,None,no_call=True));continue
                call=matches[0]
                require(call['raw_unit'] == STREAMS[label][1],'Direct API raw-unit mismatch')
                p=point(call,call['status'],call['raw_value'])
                require(row['pre']['monotonic_ns'] <= call['pre']['monotonic_ns'] <= call['post']['monotonic_ns'] <= row['post']['monotonic_ns'],'Own call escaped trace envelope')
                if call['status'] == 0:require(type(call['raw_value']) is int,'Direct raw output must retain integer width')
                streams[label].append(p)
                if prefix == 'wrapper' and p['valid']:
                    scaled=wrapper['output']['watts' if kind == 'power' else 'joules_counter']
                    require(finite(scaled),'Invalid wrapper scaled readout')
                    wrapper_scaling.append(abs(scaled-call['raw_value']/1000))
        query=row['field_query'];call_clock(query)
        require(row['pre']['monotonic_ns'] <= query['pre']['monotonic_ns'] <= query['post']['monotonic_ns'] <= row['post']['monotonic_ns'], 'Field call escaped trace envelope')
        require({x['field_id'] for x in query['fields']} == set(FIELD_IDS) and len(query['fields']) == 3,'Missing/duplicate/unknown field IDs')
        for field in query['fields']:
            label=FIELD_IDS[field['field_id']]
            require(field['scope_id'] == 0 and field['raw_unit'] == STREAMS[label][1],'Field scope/unit mismatch')
            status=field['field_status'] if query['status'] == 0 else 'batch_status:'+str(query['status'])
            streams[label].append(point(query,status,field['raw_value'],field=dict(field)))
        for kind in ('power','counter'):
            a,b=streams['typed_'+kind][-1],streams['wrapper_'+kind][-1]
            if a['valid'] and b['valid']:
                alignments[kind].append({'midpoint_delta_seconds':b['t']-a['t'],'raw_wrapper_minus_typed':b['raw']-a['raw']})
    require(len(rows) >= 2,'At least two trace rows required')
    envelope_times=[call_clock(x) for x in envelopes]
    require(all(b > a for a,b in zip(envelope_times,envelope_times[1:])),'Nonmonotonic trace envelopes')
    alignment={key:{'same_poll_midpoint_delta_seconds':stats(x['midpoint_delta_seconds'] for x in values),
                    'raw_wrapper_minus_typed':stats(x['raw_wrapper_minus_typed'] for x in values),
                    'same_poll_raw_equal_fraction':sum(x['raw_wrapper_minus_typed'] == 0 for x in values)/len(values) if values else None}
               for key,values in alignments.items()}
    return streams,{'trace_rows':len(rows),'poll_envelope_midpoint_gap_seconds':stats(b-a for a,b in zip(envelope_times,envelope_times[1:])),
                    'wrapper_raw_to_scaled_absolute_difference':stats(wrapper_scaling),'sequential_typed_wrapper_alignment':alignment}


def stream_summary(points,kind):
    times=[x['t'] for x in points];good=[x for x in points if x['valid']]
    changes=[b['t'] for a,b in zip(points,points[1:]) if a['valid'] and b['valid'] and a['raw'] != b['raw']]
    offsets=[((x['pre']['epoch_ns']+x['post']['epoch_ns'])-(x['pre']['monotonic_ns']+x['post']['monotonic_ns']))/2e9 for x in points]
    result={'readings':len(points),'successful_numeric_readings':len(good),
        'status_counts':dict(collections.Counter(str(x['status']) for x in points)),
        'max_midpoint_gap_seconds':max((b-a for a,b in zip(times,times[1:])),default=None),
        'host_midpoints_strictly_increasing':all(b > a for a,b in zip(times,times[1:])),
        'call_span_seconds':stats((x['post']['monotonic_ns']-x['pre']['monotonic_ns'])/1e9 for x in points),
        'epoch_minus_monotonic_seconds':stats(offsets),'raw_value':stats(x['raw'] for x in good),
        'reported_value_changes':len(changes),'between_reported_value_change_seconds':stats(b-a for a,b in zip(changes,changes[1:])),
        'first_call_receipt':points[0] if points else None,'last_call_receipt':points[-1] if points else None,
        'refresh_interpretation':'value-change intervals are host-observed refresh behavior, not independent sensor resolution'}
    if kind == 'counter':result['counter_decreases']=sum(b['raw'] < a['raw'] for a,b in zip(points,points[1:]) if a['valid'] and b['valid'])
    fields=[x for x in points if x['metadata'] is not None]
    if fields:
        ages=[];native=[]
        for x in fields:
            value=x['metadata'].get('timestamp_epoch_microseconds')
            if x['valid'] and type(value) is int and value > 0:
                ages.append((x['pre']['epoch_ns']+x['post']['epoch_ns'])/2e9-value/1e6);native.append(value)
        result['native_field_metadata']={'scope_id_counts':dict(collections.Counter(str(x['metadata'].get('scope_id')) for x in fields)),
            'value_type_counts':dict(collections.Counter(str(x['metadata'].get('value_type')) for x in fields)),
            'timestamp_epoch_microseconds':stats(native),'apparent_source_age_seconds':stats(ages),
            'negative_apparent_age_count':sum(x < 0 for x in ages),'native_timestamp_decreases':sum(b < a for a,b in zip(native,native[1:])),
            'native_timestamp_change_count':sum(a != b for a,b in zip(native,native[1:])),
            'clock_domain_status':'documented epoch microseconds; actual domain/source/averaging validity unverified; apparent age is NOT measurement latency',
            'integration_clock':'host own-call monotonic midpoint only; native timestamp never substituted'}
    return result


def interpolate(points,boundary,origin=0):
    times=[x['t'] for x in points];right=bisect.bisect_left(times,boundary)
    if right < len(points) and times[right] == boundary:return points[right]['raw']-origin
    require(0 < right < len(points),'Interpolation boundary unbracketed')
    a,b=points[right-1],points[right]
    # Integer differences occur before conversion, preserving large counters.
    return (a['raw']-origin)+(b['raw']-a['raw'])*(boundary-a['t'])/(b['t']-a['t'])


def window_readout(points,start,end,kind):
    result={'available':False,'start_monotonic_seconds':start,'end_monotonic_seconds':end,'seconds':end-start,
            'max_allowed_midpoint_gap_seconds':MAX_GAP_SECONDS,'instrument_readout_joules':None,'mean_instrument_readout_watts':None}
    require(finite(start) and finite(end) and start < end,'Invalid window')
    times=[x['t'] for x in points]
    if not points or not all(b > a for a,b in zip(times,times[1:])):
        result['reason']='nonmonotonic_host_midpoints';return result
    left=bisect.bisect_right(times,start)-1;right=bisect.bisect_left(times,end)
    if left < 0 or right >= len(points):result['reason']='unbracketed_window';return result
    selected=points[left:right+1];gaps=[b['t']-a['t'] for a,b in zip(selected,selected[1:])]
    result.update(bracketing_readings=len(selected),max_midpoint_gap_seconds=max(gaps,default=0.),
                  status_counts=dict(collections.Counter(str(x['status']) for x in selected)))
    if not all(x['valid'] for x in selected):result['reason']='failed_unsupported_or_nonfinite_reading';return result
    if any(x > MAX_GAP_SECONDS for x in gaps):result['reason']='host_midpoint_gap_above_2s';return result
    if kind == 'counter' and any(b['raw'] < a['raw'] for a,b in zip(selected,selected[1:])):
        result['reason']='counter_decrease_reset_or_wrap';return result
    origin=selected[0]['raw'] if kind == 'counter' else 0
    a,b=interpolate(selected,start,origin),interpolate(selected,end,origin)
    if kind == 'counter':
        joules=(b-a)/1000
        result.update(counter_integer_origin=origin,start_raw_minus_origin=a,end_raw_minus_origin=b,
                      method='counter endpoint interpolation on same host window; mJ delta /1000 once')
    else:
        knots=[(start,a)]+[(x['t'],x['raw']) for x in selected if start < x['t'] < end]+[(end,b)]
        joules=sum((y0+y1)/2000*(t1-t0) for (t0,y0),(t1,y1) in zip(knots,knots[1:]))
        result.update(start_raw_milliwatts=a,end_raw_milliwatts=b,method='endpoint interpolation and trapezoid of raw mW /1000 once')
    if not finite(joules):result['reason']='nonfinite_integral';return result
    result.update(available=True,instrument_readout_joules=joules,mean_instrument_readout_watts=joules/(end-start),
                  host_bracket_call_receipts={'first':selected[0],'last':selected[-1]})
    return result


def prescribed_windows(epoch):
    idle=epoch['idle'];load=epoch['workload'];cool=epoch['cooldown']
    a=idle['start_monotonic_seconds'];b=idle['end_monotonic_seconds']
    c=load['load_start_monotonic_seconds'];d=load['load_end_monotonic_seconds']
    e=cool['start_monotonic_seconds'];f=cool['end_monotonic_seconds']
    return [('idle_first30',a,a+30),('idle_second30',a+30,a+60),('full_idle',a,b),
            ('load_first30',c,c+30),('load_middle30',c+30,c+60),('load_last30',c+60,c+90),
            ('load_middle60',c+15,c+75),('full_load',c,d),('full_cooldown',e,f),
            ('idle_through_load_including_setup',a,d),('load_through_cooldown_including_cleanup',c,f),
            ('full_transition_including_setup_cleanup',a,f)]


def validate_result(result):
    require(result.get('schema') == 'stereo-study-energy-interfaces-result-v1'
            and result.get('status') == 'COMPLETED_READ_ONLY_NVML_INTERFACE_DIAGNOSTIC'
            and not result.get('monitor_shutdown_errors'),'Incomplete/failed diagnostic cannot enter completed analysis')
    source=result.get('source_identity_sha256')
    order_block=source==ORDER_BLOCK_SOURCE
    expected_config=ORDER_BLOCK_CONFIG if order_block else CONFIG_SHA
    require(source in REGISTERED_SOURCES and result.get('source_registry_sha256') == REGISTERED_SOURCES[source]
            and result.get('config_sha256') == expected_config,'Exact registered diagnostic identity mismatch')
    require(result.get('test_access') is False and result.get('precision') == 'FP32' and result.get('tf32') is False,'Diagnostic boundary mismatch')
    protocol=result['protocol'];cases=protocol['cases']
    cadences=[.5,.05,.05,.5] if order_block else [.05,.5]
    if order_block:
        require(protocol.get('order_protocol',{}).get('name')=='reverse_then_forward_same_allocation_v1'
                and protocol['order_protocol'].get('cadences_seconds')==cadences,'Exact registered order block required')
    require(protocol.get('max_seconds') == (1200 if order_block else 600) and protocol.get('val_samples') == 20 and protocol.get('boundary') == 'gpu_resident'
            and len(cases) == 1 and cases[0]['checkpoint_sha256'] == CHECKPOINT_SHA,'Registered case/workload mismatch')
    epochs=result.get('epochs',[]);require(len(epochs) == len(cadences),'Exact complete registered cadence epochs required')
    trained_state=None
    for i,(epoch,cadence) in enumerate(zip(epochs,cadences),1):
        require(epoch.get('epoch') == i and epoch.get('cadence_seconds') == cadence and epoch.get('status') == 'COMPLETED','Incomplete/misordered cadence epoch')
        for key in ('before_idle_allocation','before_cooldown_allocation'):
            receipt=epoch[key]
            require(receipt.get('zero_allocations_verified') is True and receipt.get('after_cleanup') == {'allocated_bytes':0,'reserved_bytes':0},'Zero PyTorch allocation receipt missing/failed')
            if source in (REPAIRED_SOURCE_ID,ORDER_BLOCK_SOURCE):
                release=receipt.get('cublas_workspace_release',{})
                require(isinstance(receipt.get('after_ordinary_cleanup'),dict)
                        and release.get('configuration_changed') is False
                        and release.get('status') in ('applied','not_needed_zero_after_ordinary_cleanup'),
                        'Repaired cleanup stage receipts missing/failed')
        idle,load,cool=epoch['idle'],epoch['workload'],epoch['cooldown']
        if source in (REPAIRED_SOURCE_ID,ORDER_BLOCK_SOURCE):
            leading=epoch.get('idle_leading_trace_post_monotonic_ns')
            require(type(leading) is int and 0<=leading<idle['start_monotonic_seconds']*1e9,
                    'Repaired idle requires a completed leading trace row')
        if order_block and i>1:
            require(epochs[i-2]['cooldown']['end_monotonic_seconds']<=idle['start_monotonic_seconds'],'Cross-epoch order overlap')
        require(idle['end_monotonic_seconds']-idle['start_monotonic_seconds'] >= 60
                and load['load_end_monotonic_seconds']-load['load_start_monotonic_seconds'] >= 90
                and cool['end_monotonic_seconds']-cool['start_monotonic_seconds'] >= 30,'Incomplete fixed phase duration')
        require(idle['end_monotonic_seconds'] <= load['load_start_monotonic_seconds'] < load['load_end_monotonic_seconds'] <= cool['start_monotonic_seconds'],'Phase ordering invalid')
        require(type(load.get('delivered_outputs')) is int and load['delivered_outputs'] > 0
                and load['delivered_outputs'] == 20*load.get('complete_cycles',0) and load.get('trained_state_unchanged') is True,'Incomplete20-target workload cycles/state receipt')
        p=load['provenance']
        require(p.get('checkpoint_sha256') == CHECKPOINT_SHA and p.get('training_source_identity_sha256') == TRAIN_SOURCE
                and p.get('training_config_sha256') == TRAIN_CONFIG and p.get('training_initial_weights_sha256') == TRAIN_INITIAL
                and p.get('manifest_sha256') == MANIFEST and p.get('training_seed') == 9471 and p.get('model_family') == 'baseline',
                'Checkpoint/training/data/seed provenance mismatch')
        require(p['variant']['overrides'] == {} and p['variant']['training_model_config']['input_channels'] == 2
                and p['variant']['training_model_config']['architecture'] == 'original_I','Baseline workload adaptation changed')
        ids=p['val_sample_ids'];require(len(ids) == len(set(ids)) == 20 and canonical_sha(ids) == VAL_IDS_SHA,'Full ordered20-target binding mismatch')
        if trained_state is None:trained_state=p['trained_state_sha256']
        require(p['trained_state_sha256'] == trained_state,'Epochs used different trained state')
    return True


def analyze(folder):
    folder=Path(folder);sources={}
    def read(member,lines=False):
        raw=(folder/member).read_bytes();sources[str(member)]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
        if lines:
            require(raw.endswith(b'\n'),'Incomplete final JSONL line: '+str(member))
            return [json.loads(line) for line in raw.splitlines() if line.strip()]
        return json.loads(raw)
    result=read(Path('result.json'));validate_result(result)
    output={'schema':'stereo-study-energy-interface-analysis-v1','status':'ANALYZED_COMPLETED_INSTRUMENT_READOUTS_NOT_CALIBRATION',
        'binding':{k:result[k] for k in ('source_identity_sha256','source_registry_sha256','config_sha256','git_commit','slurm_job_id','gpu_uuid','gpu')},
        'checkpoint_sha256':CHECKPOINT_SHA,'nvml':result.get('nvml'),'definitions':result.get('definitions'),
        'method':'each API own-call pre/post monotonic midpoint; field values share field-batch envelope; raw units converted once',
        'native_clock_policy':'native timestamps retained/age diagnosed; documented epoch domain unverified; no host-clock substitution or measurement-latency claim',
        'max_window_host_midpoint_gap_seconds':MAX_GAP_SECONDS,'streams':STREAMS,'epochs':[],
        'limitations':['all interfaces share NVML instrumentation; agreement is not calibration',
            'prospectively registered sequential epochs confound cadence with thermal/history/shared-host effects; no IID replicate claim',
            'power averaging/native source timestamp semantics may differ; polling is not IID physical sampling',
            'no calibration, correction factor, interface winner, architecture ranking, idle subtraction or poll-based confidence'],
        'raw_storage_policy':'raw traces remain in ignored copied run folder; hashes/own-call pointers retained, no raw sensor/GT/checkpoint tensor access'}
    for epoch in result['epochs']:
        index=epoch['epoch'];prefix=Path('epoch-'+str(index));rows=read(prefix/'interfaces.jsonl',True)
        require(len(rows) == epoch['trace_rows'],'Trace row count differs from completed receipt')
        streams,cadence=extract_streams(rows,index,epoch['cadence_seconds'],result['gpu_uuid'])
        summary={'epoch':index,'cadence_seconds':epoch['cadence_seconds'],'cadence_and_alignment':cadence,
            'allocation_receipts':{k:epoch[k] for k in ('before_idle_allocation','before_cooldown_allocation')},
            'load_seconds':epoch['workload']['load_end_monotonic_seconds']-epoch['workload']['load_start_monotonic_seconds'],
            'delivered_outputs':epoch['workload']['delivered_outputs'],
            'stream_summaries':{key:stream_summary(points,STREAMS[key][0]) for key,points in streams.items()},'windows':[],'optional_sidecar_validation':{}}
        for name,start,end in prescribed_windows(epoch):
            values={key:window_readout(points,start,end,STREAMS[key][0]) for key,points in streams.items()}
            # References only define subtraction signs, never a preferred sensor.
            differences=[]
            pairs=[('typed_power',key) for key in STREAMS if key != 'typed_power']
            pairs += [('typed_counter','wrapper_counter'),('wrapper_power','wrapper_counter'),
                      ('field185_average_power','field83_counter'),('field186_instant_power','field83_counter')]
            for left,right in pairs:
                a,b=values[left],values[right];available=a['available'] and b['available']
                delta=b['instrument_readout_joules']-a['instrument_readout_joules'] if available else None
                differences.append({'left':left,'right':right,'available':available,'right_minus_left_readout_joules':delta,
                    'relative_difference_to_left':delta/a['instrument_readout_joules'] if available and a['instrument_readout_joules'] != 0 else None})
            summary['windows'].append({'name':name,'start_monotonic_seconds':start,'end_monotonic_seconds':end,'streams':values,'differences':differences})
        for member,key in ((prefix/'before-idle-allocation.json','before_idle_allocation'),(prefix/'before-cooldown-allocation.json','before_cooldown_allocation')):
            if (folder/member).exists():require(read(member) == epoch[key],'Allocation sidecar differs from completed result')
        if (folder/prefix/'neuron-reset.json').exists():
            reset=read(prefix/'neuron-reset.json');require(reset['functional_reset_net_applied'] is True and reset['trained_state_unchanged'] is True
                and reset['trained_state_sha256_before_reset'] == reset['trained_state_sha256_after_reset'] == epoch['workload']['provenance']['trained_state_sha256'],'Neuron-reset sidecar state mismatch')
            summary['optional_sidecar_validation']['neuron_reset']='verified'
        if (folder/prefix/'outputs.jsonl').exists():
            delivered=read(prefix/'outputs.jsonl',True);ids=epoch['workload']['provenance']['val_sample_ids']
            require(len(delivered) == epoch['workload']['delivered_outputs'] and all(x['output_index'] == i and x['sample_id'] == ids[i%20] for i,x in enumerate(delivered)),'Delivered output order/count differs')
            summary['optional_sidecar_validation']['output_order']='verified complete ordered cycles'
        output['epochs'].append(summary)
    output['context_validation']='completed driver status/allocated UUID receipts only; raw context sidecar not supplied'
    if (folder/'gpu-context.jsonl').exists():
        context=read(Path('gpu-context.jsonl'),True);require(context,'Empty context sidecar')
        require(all(x['gpu_uuid'] == result['gpu_uuid'] and not x['foreign_compute_processes'] for x in context),'Foreign/mismatched GPU context')
        require(len({x['settings']['power.limit'] for x in context}) == 1,'Power limit changed')
        output['context_validation']={'raw_snapshots_verified':len(context),'settings_first':context[0]['settings'],'settings_last':context[-1]['settings'],
            'clock_policy':'read-only observed dynamic clocks; no locked-clock claim'}
    # Every ordered contrast is fixed before seeing the new trace.
    pairs=[(0,1,'forward_50_to500')] if len(output['epochs'])==2 else [
        (0,1,'reverse_500_to50'),(2,3,'forward_50_to500'),
        (0,3,'same500_epoch1_to4'),(1,2,'same50_epoch2_to3')]
    output['registered_epoch_cadences_seconds']=[v['cadence_seconds'] for v in result['epochs']]
    output['ordered_epoch_differences']=[]
    for a_index,b_index,label in pairs:
        for first,second in zip(output['epochs'][a_index]['windows'],output['epochs'][b_index]['windows']):
            require(first['name'] == second['name'],'Window order mismatch')
            for key in STREAMS:
                a,b=first['streams'][key],second['streams'][key];available=a['available'] and b['available']
                output['ordered_epoch_differences'].append({'contrast':label,'first_epoch':a_index+1,'second_epoch':b_index+1,
                    'window':first['name'],'stream':key,'available':available,
                    'later_minus_earlier_mean_readout_watts':b['mean_instrument_readout_watts']-a['mean_instrument_readout_watts'] if available else None,
                    'interpretation':'ordered cadence/thermal/history contrast, not identified causal polling effect'})
    output.update(source_files=sources,analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    readings=[reading for epoch in output['epochs'] for window in epoch['windows'] for reading in window['streams'].values()]
    output['window_availability_summary']={'available':sum(x['available'] for x in readings),
        'unavailable_reasons':dict(collections.Counter(x['reason'] for x in readings if not x['available'])),
        'interpretation':'availability validates arithmetic prerequisites only, never calibration'}
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output-prefix',type=Path,required=True);args=parser.parse_args()
    targets=[args.output_prefix.with_suffix('.'+x) for x in ('json','csv')]
    require(not any(x.exists() for x in targets),'Fresh analysis output prefix required')
    result=analyze(args.input);args.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    with targets[1].open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['epoch','cadence_seconds','window','stream','available','reason','seconds','instrument_readout_joules','mean_instrument_readout_watts','max_midpoint_gap_seconds'])
        writer.writeheader()
        for epoch in result['epochs']:
            for window in epoch['windows']:
                for name,row in window['streams'].items():
                    writer.writerow({'epoch':epoch['epoch'],'cadence_seconds':epoch['cadence_seconds'],'window':window['name'],'stream':name,
                        **{key:row.get(key) for key in ('available','reason','seconds','instrument_readout_joules','mean_instrument_readout_watts','max_midpoint_gap_seconds')}})
    result['csv_sha256']=hashlib.sha256(targets[1].read_bytes()).hexdigest()
    targets[0].write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'epochs':len(result['epochs']),'streams':len(STREAMS),'output_files':[str(x) for x in targets]},indent=2))


if __name__=='__main__':main()
