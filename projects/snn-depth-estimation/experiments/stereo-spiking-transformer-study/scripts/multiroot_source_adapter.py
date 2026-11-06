"""Train/Val native-source catalog and reader; metadata CLI, no sensor/model imports.

This source adapter does not generate sensors or make a final scientific freeze.
It never changes the existing v12 training loader, dataset or recipes.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path,PurePosixPath
import re

from prepare_train_fidelity_sources import (ALIASES,accepted_source_metadata,relative_path,require)


def _roots(roots):
    require(set(roots)==set(ALIASES),'Both explicit historical/new root aliases required')
    values={key:PurePosixPath(value) for key,value in roots.items()}
    require(all(v.is_absolute() and '..' not in v.parts and str(v)==roots[k] for k,v in values.items()),'Canonical absolute configured roots required')
    a,b=values.values()
    require(a!=b and a not in b.parents and b not in a.parents,'Distinct nonoverlapping source roots required')
    return {key:str(value) for key,value in values.items()}


def bind_catalog(index,roots,receipts):
    """Bind every declared RGB/selected-GT member; do not inspect source payloads."""
    require(index.get('schema')=='stereo-study-multiroot-source-index-v1'
            and index.get('test')==[] and index.get('test_values_opened') is False
            and index.get('sensor_generation_performed') is False and index.get('training_compatible') is False,
            'Unmodified source-only index and sealed Test required')
    require(index.get('legacy_split_mapping')=={'dev':'val'} and set(index.get('root_aliases',{}))==set(ALIASES),
            'Explicit legacy dev-to-val roles and both source aliases required')
    roots=_roots(roots)
    require(set(receipts)==set(ALIASES),'Attributed source receipts for both roots required')
    scenes=set();members=set();sequences=[];targets=[]
    for source in index['sequences']:
        sid=source.get('source_id','');split=source.get('split');alias=source.get('root_alias')
        require(re.fullmatch(r'TRAIN/C/[0-9]{4}',sid) and sid not in scenes,'Unique original TRAIN/C scene required')
        scenes.add(sid)
        require(split in ('train','val') and source.get('original_split')=='TRAIN' and alias in ALIASES,
                'Only declared Train/Val original-TRAIN sources allowed')
        expected_legacy='dev' if split=='val' else 'train'
        require(source.get('legacy_source_split')==expected_legacy,'Legacy role and current split disagree')
        require(source.get('historical_exposure') is (alias==ALIASES[0])
                and (alias==ALIASES[0] or split=='train'),'Historical exposure/root/Train-only acquisition changed')
        centers=list(range(1,9)) if split=='train' else [3,6]
        require(source.get('reference_centers')==centers,'Preserve all registered Train or sparse Val target centers')
        require(len(source['frames'])==10,'Exactly ten ordered native RGB frames required')
        files=[]
        for position,frame in enumerate(source['frames']):
            stem=f'{position+6:04d}'
            require(type(frame.get('index')) is int and frame['index']==position and frame.get('source_frame')==stem,
                    'Ordered native frame identity changed')
            roles=[('rgb_l','rgb','left','.png'),('rgb_r','rgb','right','.png')]
            if position in centers:roles.append(('disparity_l','gt','left','.pfm'))
            elif frame.get('disparity_l') is not None:
                # No non-target GT may be fabricated from an inventory row.
                raise ValueError('Non-target disparity must stay null in this accepted source index')
            for field,role,view,suffix in roles:
                name=frame.get(field);relative_path(name)
                expected=f'{"frames_cleanpass" if role=="rgb" else "disparity"}/{sid}/{view}/{stem}{suffix}'
                require(name==expected,'Frame/view/scene source path changed')
                receipt=receipts[alias].get(name)
                require(receipt and receipt.get('source_id')==sid and receipt.get('role')==role
                        and type(receipt.get('size')) is int and receipt['size']>0
                        and re.fullmatch(r'[0-9a-f]{64}',receipt.get('sha256','')),
                        'Required source member absent or unattributed in its exact root')
                key=(alias,name);require(key not in members,'Duplicate source member');members.add(key)
                files.append(dict(path=name,role=role,view=view,frame_index=position,size=receipt['size'],sha256=receipt['sha256']))
        sequence={**copy.deepcopy(source),'files':files};sequences.append(sequence)
        for center in centers:
            stem=source['frames'][center]['source_frame']
            targets.append(dict(sample_id=sid+'/'+stem,source_id=sid,split=split,root_alias=alias,center=center,
                                historical_exposure=source['historical_exposure'],source_frame=stem))
    require({s['split'] for s in sequences}=={'train','val'},'Both existing Train/Val roles required; no fake Val')
    return dict(schema='stereo-study-train-val-source-catalog-v1',scope='engineering_native_source_adapter_only',
        source_roots=roots,sequences=sequences,targets=targets,test=[],test_values_opened=False,
        registered_for_generation=False,source_arrays_opened=False,sensor_generation_performed=False,
        training_compatible=False,scientific_freeze=False,
        source_steps_per_view=450,window_bins=50,
        counts=dict(train_scenes=sum(s['split']=='train' for s in sequences),val_scenes=sum(s['split']=='val' for s in sequences),
                    train_targets=sum(t['split']=='train' for t in targets),val_targets=sum(t['split']=='val' for t in targets),
                    files=len(members),expected_source_bytes=sum(f['size'] for s in sequences for f in s['files'])),
        remaining='Separate registered generation/controller path, source-byte verification at use, generated-asset audit/loader acceptance, quarter fidelity, physical time/radiometry/provider provenance and final scientific freeze')


class NativeSourceReader:
    """Resolve only catalogued native inputs through their exact acquisition root.

    No decoding/fallback/download. read_bytes verifies only the requested member;
    the metadata CLI never calls it. Future pixel use belongs in a registered Slurm job.
    """
    def __init__(self,catalog):
        require(catalog.get('schema')=='stereo-study-train-val-source-catalog-v1'
                and catalog.get('test')==[] and catalog.get('test_values_opened') is False
                and catalog.get('training_compatible') is False,'Sealed engineering source catalog required')
        self.catalog=copy.deepcopy(catalog);self.roots=_roots(catalog['source_roots'])
        self.sequences={row['source_id']:row for row in self.catalog['sequences']}
        require(len(self.sequences)==len(catalog['sequences']),'Duplicate catalog source')
        self.members={};expected_targets=[]
        for row in self.sequences.values():
            sid,split,alias=row['source_id'],row['split'],row['root_alias']
            require(re.fullmatch(r'TRAIN/C/[0-9]{4}',sid) and split in ('train','val') and alias in ALIASES
                    and row.get('historical_exposure') is (alias==ALIASES[0])
                    and row.get('original_split')=='TRAIN'
                    and row.get('legacy_source_split')==('dev' if split=='val' else 'train')
                    and (alias==ALIASES[0] or split=='train'),'Catalog source identity changed')
            centers=list(range(1,9)) if split=='train' else [3,6]
            require(row.get('reference_centers')==centers and len(row['frames'])==10,'Catalog target centers/frames changed')
            for i,frame in enumerate(row['frames']):
                stem=f'{i+6:04d}'
                require(frame['index']==i and frame['source_frame']==stem
                        and frame['rgb_l']==f'frames_cleanpass/{sid}/left/{stem}.png'
                        and frame['rgb_r']==f'frames_cleanpass/{sid}/right/{stem}.png'
                        and frame['disparity_l']==(f'disparity/{sid}/left/{stem}.pfm' if i in centers else None),
                        'Catalog frame/view identity changed')
            expected_keys={(sid,'rgb',view,i) for view in ('left','right') for i in range(10)}|{(sid,'gt','left',i) for i in centers}
            found=set()
            for record in row['files']:
                key=(sid,record['role'],record['view'],record['frame_index'])
                require(key in expected_keys and key not in self.members,'Undeclared or duplicate catalog source role/frame')
                prefix='frames_cleanpass' if record['role']=='rgb' else 'disparity'
                suffix='.png' if record['role']=='rgb' else '.pfm'
                expected=f"{prefix}/{sid}/{record['view']}/{record['frame_index']+6:04d}{suffix}"
                require(record['path']==expected and type(record['size']) is int and record['size']>0
                        and re.fullmatch(r'[0-9a-f]{64}',record['sha256']),'Catalog source path/hash binding changed')
                self.members[key]=(alias,copy.deepcopy(record));found.add(key)
            require(found==expected_keys,'Incomplete catalog RGB/GT members')
            for center in centers:
                stem=f'{center+6:04d}'
                expected_targets.append(dict(sample_id=sid+'/'+stem,source_id=sid,split=split,root_alias=alias,center=center,
                    historical_exposure=row['historical_exposure'],source_frame=stem))
        require(self.catalog['targets']==expected_targets,'Catalog target role/identity/order changed')

    def targets(self,split):
        require(split in ('train','val'),'Only Train/Val targets can be iterated')
        return [copy.deepcopy(t) for t in self.catalog['targets'] if t['split']==split]

    def member(self,source_id,role,view,frame_index):
        key=(source_id,role,view,frame_index)
        require(key in self.members,'Unacquired/undeclared source role, frame or view')
        alias,record=self.members[key]
        return dict(root_alias=alias,configured_root=self.roots[alias],**record)

    def path(self,source_id,role,view,frame_index):
        record=self.member(source_id,role,view,frame_index)
        root=Path(record['configured_root'])
        require(root.is_dir() and not root.is_symlink(),'Exact configured source root missing or symlinked; no fallback')
        current=root
        for part in relative_path(record['path']).parts:
            current=current/part
            require(not current.is_symlink(),'Symlinked source member/parent not allowed')
        require(current.is_file() and current.resolve().is_relative_to(root.resolve()),'Exact source member missing or outside root')
        return current

    def read_bytes(self,source_id,role,view,frame_index):
        record=self.member(source_id,role,view,frame_index);path=self.path(source_id,role,view,frame_index)
        raw=path.read_bytes()
        require(len(raw)==record['size'] and hashlib.sha256(raw).hexdigest()==record['sha256'],'Requested source bytes changed')
        return raw


def build_from_receipts(source_state,historical_state,deployment_path):
    index,roots,records,binding=accepted_source_metadata(source_state,historical_state,deployment_path)
    catalog=bind_catalog(index,roots,records)
    require({k:catalog['counts'][k] for k in ('train_scenes','val_scenes','train_targets','val_targets','files')}
            ==dict(train_scenes=150,val_scenes=30,train_targets=1200,val_targets=60,files=4860),
            'Complete existing180-scene/1260-target/4860-source cohort required')
    catalog['input_sha256']=binding
    catalog['verification']='Accepted acquisition metadata only; historical payloads not newly rehashed, no source arrays opened or sensors generated'
    return catalog


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-state',required=True,type=Path);p.add_argument('--historical-state',required=True,type=Path)
    p.add_argument('--cluster-config',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();repository=Path(__file__).resolve().parents[5]
    require(any(a.output.resolve().is_relative_to(repository/k) for k in ('storage','scratch')),'Fresh ignored metadata output required')
    catalog=build_from_receipts(a.source_state,a.historical_state,a.cluster_config)
    with a.output.open('x') as f:json.dump(catalog,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='BOUND_COMPLETE_TRAIN_VAL_NATIVE_SOURCE_METADATA',counts=catalog['counts'],
        source_arrays_opened=False,training_compatible=False,GPU_generation_registered=False,
        output_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())))

if __name__=='__main__':main()
