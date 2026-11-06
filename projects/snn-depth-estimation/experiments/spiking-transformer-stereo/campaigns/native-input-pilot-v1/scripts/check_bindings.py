"""Tiny synthetic registration fixture; never a data or GPU PASS claim."""
import copy,hashlib,json,subprocess,sys,tempfile
from pathlib import Path

def dump(path,obj):path.write_text(json.dumps(obj))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 here=Path(__file__).parent;templates=here.parent/'configs';before={p.name:sha(p) for p in templates.glob('*.json')}
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp);manifest=root/'manifest.json';dump(manifest,{'synthetic_binding_fixture':True});source=root/'source.json';dump(source,{'source_identity_sha256':'fixture-code','archive_sha256':'fixture-archive'});audit=root/'audit.json';dump(audit,{'status':'AUDIT_PASS','manifest_sha256':sha(manifest),'train_samples':960,'dev_samples':60,'scope':'syntheticfixture_only'})
  common=[sys.executable,str(here/'bind_artifacts.py'),'--source-registry',str(source),'--manifest',str(manifest),'--data-audit',str(audit)]
  preflight=root/'preflight.json';subprocess.run(common+['--mode','preflight','--output',str(preflight)],check=True,capture_output=True,text=True);p=json.loads(preflight.read_text());assert p['optimizer_updates']==0 and p['resident_accumulated_examples']==2 and p['source_identity_sha256']=='fixture-code'
  geometry=root/'geometry.json';dump(geometry,{'status':'PASS','source_identity_sha256':'fixture-code','manifest_sha256':sha(manifest),'selected_training_geometry':'crop384x768','models_passed':['original','raft_stereo','selective_raft'],'registration_sha256':sha(preflight)})
  bound=root/'final';subprocess.run(common+['--mode','training','--geometry-receipt',str(geometry),'--template-root',str(templates),'--output',str(bound)],check=True,capture_output=True,text=True);registration=json.loads((bound/'registration.json').read_text())
  for r in registration['runs']:
   old=json.loads((templates/r['config_file']).read_text());new=json.loads((bound/r['config_file']).read_text());changed={k for k in old if old[k]!=new[k]};assert changed=={'training_geometry','manifest_sha256'} and new['training_geometry']=='crop384x768'
  assert {p.name:sha(p) for p in templates.glob('*.json')}==before
 print(json.dumps({'synthetic_fixture_only':True,'real_data_audit_or_GPU_preflight_performed':False,'external_preflight_registration_bound':True,'all_three_configs_only_resolve_manifest_and_common_geometry':True,'source_templates_unchanged':True,'no_optimizer_or_training':True},indent=2))
if __name__=='__main__':main()
