"""Bounded Slurm setup of two pinned decoding wheels in a private output prefix."""
import argparse
import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def validate_config(config):
    if (config.get('schema')!='stereo-study-dsec-runtime-v1' or config.get('registered') is not True
            or config.get('stage')!='engineering_diagnostic' or config.get('driver')!='dsec_runtime_setup'
            or config.get('test_access')!='sealed' or config.get('allocated_gpu_hour_cap')!=24
            or config.get('packages')!=['h5py==3.14.0','hdf5plugin==6.0.0']
            or config.get('max_seconds')!=300 or config.get('scheduler_resources',{}).get('time')!='00:05:00'):
        raise ValueError('Exact bounded private DSEC runtime registration required')


def verify_installed_runtime(path,expected_sha):
    from dsec_adapter import file_sha,safe
    path=Path(path)
    if file_sha(path)!=expected_sha:
        raise ValueError('Bound private runtime receipt changed')
    receipt=json.loads(path.read_text());prefix=path.parent/'packages'
    if (receipt.get('status')!='COMPLETED_PRIVATE_DSEC_RUNTIME'
            or receipt.get('package_prefix')!=str(prefix)
            or not receipt.get('runtime',{}).get('Blosc_filter_32001')):
        raise ValueError('Private DSEC runtime not ready')
    manifest=path.parent/'package-files.json'
    if file_sha(manifest)!=receipt['package_files_manifest_sha256']:
        raise ValueError('Private runtime file manifest changed')
    files=json.loads(manifest.read_text())
    for name,sha in files.items():
        if file_sha(safe(prefix,name))!=sha:
            raise ValueError('Private runtime package file changed: '+name)
    return dict(status='VERIFIED_PRIVATE_DSEC_RUNTIME',package_files=len(files),package_prefix=str(prefix))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--source-registry',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--validate-config',action='store_true')
    parser.add_argument('--verify-existing',type=Path)
    parser.add_argument('--receipt-sha256')
    args=parser.parse_args()
    if args.verify_existing:
        print(json.dumps(verify_installed_runtime(args.verify_existing,args.receipt_sha256)));return
    if args.config is None:
        parser.error('--config required for setup')
    config=json.loads(args.config.read_text());validate_config(config)
    if args.validate_config:
        print('VALID_CONFIG_ONLY');return
    if args.source_registry is None or args.output is None:
        raise ValueError('Frozen source and fresh private output required')
    from train_study import verify_source
    from dsec_adapter import require_allocation,file_sha
    require_allocation(config['expected_hostname'])
    registry=verify_source(args.source_registry)
    root=args.source_registry.parent.resolve()
    for path in (Path(__file__).resolve(),args.config.resolve()):
        member=str(path.relative_to(root))
        if registry['source_files'].get(member)!=file_sha(path):
            raise ValueError('Executing script/config not bound to frozen source')
    if args.output.resolve().is_relative_to(root):
        raise ValueError('Runtime packages cannot enter immutable source')
    args.output.mkdir(parents=True,exist_ok=False)
    began=time.monotonic();packages=args.output/'packages'
    result=dict(schema='stereo-study-dsec-runtime-result-v1',status='INSTALLING_PRIVATE_WHEELS',
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        slurm_job_id=os.environ['SLURM_JOB_ID'],hostname=socket.gethostname(),
        source_identity_sha256=registry['source_identity_sha256'],git_commit=registry['git_commit'],
        config_sha256=file_sha(args.config),packages=config['packages'],
        changes_shared_environment=False,test_access=False,model_inference=False,event_conversion=False)
    def save():
        path=args.output/'result.json';temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(path)
    save()
    try:
        # Isolated pip ignores user configuration; no dependencies or base
        # packages are replaced. Wheel identities are retained by pip's report.
        command=[sys.executable,'-m','pip','--isolated','install','--index-url','https://pypi.org/simple',
            '--only-binary=:all:','--no-deps','--target',str(packages),'--report',str(args.output/'pip-report.json'),
            *config['packages']]
        with (args.output/'install.log').open('w') as stream:
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True,
                           timeout=max(1,config['max_seconds']-30-(time.monotonic()-began)))
        report=json.loads((args.output/'pip-report.json').read_text())
        installed={row['metadata']['name'].lower():row['metadata']['version'] for row in report['install']}
        if installed!={'h5py':'3.14.0','hdf5plugin':'6.0.0'}:
            raise RuntimeError('Unexpected private wheel set')
        probe='''import hdf5plugin,h5py,json,numpy,yaml,PIL,platform,importlib.metadata
print(json.dumps(dict(python=platform.python_version(),numpy=numpy.__version__,h5py=h5py.__version__,hdf5plugin=importlib.metadata.version('hdf5plugin'),PyYAML=yaml.__version__,Pillow=PIL.__version__,Blosc_filter_32001=bool(h5py.h5z.filter_avail(32001)),h5py_file=h5py.__file__,hdf5plugin_file=hdf5plugin.__file__)))
'''
        env=dict(os.environ);env['PYTHONPATH']=str(packages)+os.pathsep+env.get('PYTHONPATH','')
        runtime=json.loads(subprocess.check_output([sys.executable,'-c',probe],text=True,env=env,
            timeout=max(1,config['max_seconds']-10-(time.monotonic()-began))))
        if not runtime['Blosc_filter_32001']:
            raise RuntimeError('Required Blosc filter unavailable in private runtime')
        for key in ('h5py_file','hdf5plugin_file'):
            if not Path(runtime[key]).resolve().is_relative_to(packages.resolve()):
                raise RuntimeError('Decoder resolved outside private package prefix')
        files={str(path.relative_to(packages)):file_sha(path) for path in sorted(packages.rglob('*'))
               if path.is_file() and '__pycache__' not in path.parts}
        (args.output/'package-files.json').write_text(json.dumps(files,indent=2)+'\n')
        result.update(status='COMPLETED_PRIVATE_DSEC_RUNTIME',runtime=runtime,
            package_prefix=str(packages),package_files=len(files),
            package_files_manifest_sha256=file_sha(args.output/'package-files.json'),
            pip_report_sha256=file_sha(args.output/'pip-report.json'),
            wheel_sources=[dict(name=row['metadata']['name'],version=row['metadata']['version'],
                download_info=row['download_info']) for row in report['install']])
    except BaseException as error:
        result.update(status='FAILED_PRIVATE_DSEC_RUNTIME',error=repr(error))
        raise
    finally:
        result['elapsed_seconds']=time.monotonic()-began;save()
    print(json.dumps({'status':result['status'],'runtime':result['runtime']}))


if __name__=='__main__':main()
