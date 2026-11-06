"""Freeze the single predetermined QC pilot after acquisition/scientific review."""
import argparse,datetime,json,subprocess
from pathlib import Path
from ema_adapter import sha
from source_io import validate_index


def main():
 p=argparse.ArgumentParser();p.add_argument('--source-index',required=True);p.add_argument('--protocol',required=True);p.add_argument('--output',required=True);p.add_argument('--review-record',required=True)
 a=p.parse_args();output=Path(a.output)
 if output.exists():raise RuntimeError('registration exists; never overwrite')
 index=json.loads(Path(a.source_index).read_text());sequences=validate_index(index);seq=next(s for s in sequences if s['split']=='train');protocol=json.loads(Path(a.protocol).read_text())
 review=json.loads(Path(a.review_record).read_text())
 if review['status']!='APPROVED_ONE_SEQUENCE_PILOT' or review['bulk_authorized'] is not False:raise RuntimeError('root/Astra one-sequence scope review missing')
 scripts=Path(__file__).parent
 names=['convert_pilot.py','ema_adapter.py','spike_codec.py','source_io.py','register_pilot.py']
 registration={'registration_id':'flyingthings3d-native-qc-pilot-v1','status':'registered','mode':'one-sequence-pilot','registered_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source_id':seq['source_id'],'source_index_sha256':sha(a.source_index),'protocol_sha256':sha(a.protocol),'review_record_sha256':sha(a.review_record),'converter_source_sha256':{n:sha(scripts/n) for n in names},'max_allocated_gpus':1,'time_limit_seconds':1800,'scientific_sequence_count':1,'train_windows':8,'dev_windows':0,'interpolated_view_images':882,'exact_native_endpoint_view_images':18,'bulk_authorized':False,'training_performed':False,'wait_for_original_job':1903,'source_cohort_rule':protocol['source']['selection_cohort'],'selection_seed_role':protocol['source']['selection_seed_role']}
 output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(registration,indent=2)+'\n');print(json.dumps(registration,indent=2))
if __name__=='__main__':main()
