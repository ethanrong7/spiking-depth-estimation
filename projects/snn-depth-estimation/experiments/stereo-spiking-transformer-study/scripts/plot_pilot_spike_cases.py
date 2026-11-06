"""Render only nine existing hash-pinned selected-case NPZs; no model/data runner.

Run with the existing plotting Python and --raw-root pointing to ignored
selected-cases/<functionality-run> directories. Only correct/target/valid arrays
are opened. Absolute error is a display transform, not a new metric evaluation.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
import numpy as np

STUDY=Path(__file__).resolve().parents[1]
ROOT=Path(__file__).resolve().parents[5]
METHODS=(('original-I','ALIF+I'),('A','Spiking A'),('B','Spiking B'))
DISPARITY_LIMITS=(0.,32.)
ERROR_LIMITS=(0.,16.)
BAD='#e6e6e6'
UNDER='#CC79A7'
DISP_OVER='#E69F00'
ERROR_OVER='#56B4E9'


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(array):
    answer=hashlib.sha256()
    answer.update((str(array.dtype)+repr(array.shape)).encode())
    answer.update(np.ascontiguousarray(array).tobytes())
    return answer.hexdigest()


def relative(path):
    try:return str(path.resolve().relative_to(ROOT))
    except ValueError:return path.name


def plane(array):
    if array.shape!=(1,135,240):
        raise ValueError('Expected saved native quarter-resolution [1,135,240] field')
    return array[0]


def load_sources(raw_root):
    jobs=json.loads((STUDY/'results/jobs.json').read_text())
    entries={}
    ids=None
    source={'schema':'pilot-spike-cases-visual-sources-v1','selection':'existing lexicographic first/middle/last Val IDs, fixed before inference',
            'condition':'correct-right','seed':9471,'cases':[],'methods':[],'new_inference':False,'test_access':False,
            'aggregate_metric_evaluation':False,'original_dataset_assets_opened':False}
    for method,label in METHODS:
        run='functionality-pilot-'+method+'-spike-seed9471-v1'
        report_path=STUDY/'results'/run/'result.json'
        report=json.loads(report_path.read_text())
        if (report.get('mode')!='checkpoint' or report.get('test_access') is not False
                or report['config']['seed']!=9471 or report.get('confidence_filter') is not False
                or report['selection_rule']!='lexicographic_Val_ids_first_middle_last_before_inference'):
            raise ValueError('Source must be the completed registered fixed-case checkpoint diagnostic')
        wanted=report['preselected_val_case_ids']
        if len(wanted)!=3 or len(set(wanted))!=3:raise ValueError('Exactly three distinct preselected IDs required')
        if ids is None:ids=list(wanted)
        elif ids!=wanted:raise ValueError('Preselected case IDs/order differ between methods')
        selected={row['sample_id']:row for row in report['selected_cases']}
        if set(selected)!=set(ids):raise ValueError('Saved cases differ from preselected IDs')
        matching=[job for job in jobs if job['run']==run and str(job['job_id'])==str(report['slurm_job_id'])]
        if len(matching)!=1 or matching[0].get('state')!='COMPLETED':raise ValueError('Completed diagnostic ledger binding missing')
        checkpoint=report['checkpoint']
        if checkpoint['selection']!='best-val.pt' or checkpoint['state']['update']!=1000:raise ValueError('Unexpected checkpoint selection/update')
        source['methods'].append({'method':method,'label':label,'run':run,'diagnostic_job_id':str(report['slurm_job_id']),
            'report':relative(report_path),'report_sha256':file_sha(report_path),
            'training_run':report['config']['training_run'],'checkpoint_sha256':checkpoint['checkpoint_sha256'],
            'checkpoint_update':checkpoint['state']['update'],'training_source_identity_sha256':checkpoint['binding']['source_identity_sha256'],
            'training_config_sha256':checkpoint['binding']['config_sha256'],
            'manifest_sha256':report['binding']['manifest_sha256'],
            'historical_training_protocol':{'Train_scenes':report['config']['data']['train_scenes'],
                'Val_scenes':report['config']['data']['val_scenes'],'Train_frames':checkpoint['binding']['training_samples'],
                'Val_frames':checkpoint['binding']['validation_samples']}})
        entries[method]={}
        for index,sample_id in enumerate(ids):
            record=selected[sample_id]
            if record['artifact']!='selected-case-'+str(index)+'.npz':raise ValueError('Artifact index is not fixed preselected-ID index')
            path=raw_root/run/record['artifact']
            if file_sha(path)!=record['sha256']:raise ValueError('Selected NPZ SHA mismatch: '+relative(path))
            with np.load(path,allow_pickle=False) as saved:
                arrays={key:plane(saved[key]).copy() for key in ('correct','target','valid')}
            pred,gt,valid=arrays['correct'],arrays['target'],arrays['valid']
            if valid.dtype!=np.bool_ or not np.isfinite(pred).all() or not np.isfinite(gt[valid]).all() or not (gt[valid]>0).all():
                raise ValueError('Invalid saved finite prediction/trusted-GT array contract')
            entries[method][sample_id]=arrays
            source['cases'].append({'method':method,'sample_id':sample_id,'scene_id':record['scene_id'],
                'artifact':relative(path),'artifact_sha256':record['sha256'],
                'remote_locator':'registered diagnostic ledger output for job '+str(report['slurm_job_id'])+'/'+record['artifact'],
                'sample_id_binding':'report selected_cases record and exact NPZ SHA; archives do not embed sample IDs',
                'array_shape':[135,240],'array_sha256':{key:array_sha(value) for key,value in arrays.items()},
                'display_only_qa':{'prediction_range_on_saved_valid_GT':[float(pred[valid].min()),float(pred[valid].max())],
                    'has_under_range_prediction':bool((pred[valid]<DISPARITY_LIMITS[0]).any()),
                    'has_over_range_prediction':bool((pred[valid]>DISPARITY_LIMITS[1]).any()),
                    'absolute_error_range_on_saved_valid_GT':[float(np.abs(pred-gt)[valid].min()),float(np.abs(pred-gt)[valid].max())],
                    'has_over_range_error':bool((np.abs(pred-gt)[valid]>ERROR_LIMITS[1]).any())}})
    for sample_id in ids:
        reference=entries['original-I'][sample_id]
        for method,_ in METHODS:
            value=entries[method][sample_id]
            for key in ('target','valid'):
                if not np.array_equal(reference[key],value[key],equal_nan=True):
                    raise ValueError('Cross-method saved '+key+' mismatch for '+sample_id)
    source.update(preselected_case_ids=ids,shared_GT_and_valid_masks_verified=True,
        display={'disparity_limits_pixels':list(DISPARITY_LIMITS),'error_limits_pixels':list(ERROR_LIMITS),
            'units':'quarter-image pixels','disparity_colormap':'viridis','error_colormap':'magma',
            'invalid_GT_color':BAD,'negative_disparity_color':UNDER,'over_disparity_color':DISP_OVER,'over_error_color':ERROR_OVER,
            'support':'exact saved valid GT mask, identical for all methods; no confidence or prediction-value filtering',
            'transform':'absolute value of saved correct prediction minus saved target; no array clipping, resizing, or crop',
            'interpolation':'nearest','origin':'upper'})
    return entries,source


def render(entries,source,prefix):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none',
                         'svg.hashsalt':'pilot-spike-cases-v1','axes.unicode_minus':True})
    fig=plt.figure(figsize=(17.6,7.4),facecolor='white')
    width=.1235;gap=.010;group_gap=.020
    x=[.030+j*(width+gap)+(group_gap if j>=4 else 0) for j in range(7)]
    height=width*17.6*(135/240)/7.4
    bottoms=[.640,.430,.220]
    title='Saved spike-input disparity on three fixed Val cases'
    fig.text(.030,.978,title,fontsize=18,fontweight='bold',ha='left',va='top',color='#222222')
    fig.text(.030,.932,'Seed 9471 · 135 × 240 · best checkpoint by Val at update 1000 · historical pilot: 40 Train / 10 Val scenes',
             fontsize=11.5,ha='left',va='top',color='#444444')
    fig.text((x[0]+x[3]+width)/2,.881,'Disparity fields',fontsize=12.5,fontweight='bold',ha='center')
    fig.text((x[4]+x[6]+width)/2,.881,'Absolute error |prediction − GT|',fontsize=12.5,fontweight='bold',ha='center')
    names=['GT','ALIF+I','Spiking A','Spiking B','ALIF+I','Spiking A','Spiking B']
    for index,name in enumerate(names):fig.text(x[index]+width/2,.847,name,fontsize=11.5,ha='center',fontweight='normal')
    disparity_cmap=matplotlib.colormaps['viridis'].copy();disparity_cmap.set_bad(BAD);disparity_cmap.set_under(UNDER);disparity_cmap.set_over(DISP_OVER)
    error_cmap=matplotlib.colormaps['magma'].copy();error_cmap.set_bad(BAD);error_cmap.set_over(ERROR_OVER)
    dnorm=Normalize(*DISPARITY_LIMITS,clip=False);enorm=Normalize(*ERROR_LIMITS,clip=False)
    for row,sample_id in enumerate(source['preselected_case_ids']):
        reference=entries['original-I'][sample_id];gt=reference['target'];valid=reference['valid']
        fig.text(x[0],bottoms[row]+height+.012,'('+chr(ord('a')+row)+') '+sample_id,fontsize=11.5,ha='left',va='bottom',fontweight='normal')
        fields=[gt]+[entries[method][sample_id]['correct'] for method,_ in METHODS]
        fields += [np.abs(entries[method][sample_id]['correct']-gt) for method,_ in METHODS]
        for col,field in enumerate(fields):
            ax=fig.add_axes([x[col],bottoms[row],width,height])
            ax.imshow(np.ma.array(field,mask=~valid),cmap=disparity_cmap if col<4 else error_cmap,
                      norm=dnorm if col<4 else enorm,interpolation='nearest',origin='upper',aspect='equal')
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_linewidth(.45);spine.set_color('#bbbbbb')
    for first,last,cmap,norm,label,ticks,extend in [(0,3,disparity_cmap,dnorm,'Disparity (quarter-image pixels)',[0,8,16,24,32],'both'),
                                                   (4,6,error_cmap,enorm,'Absolute error (quarter-image pixels)',[0,4,8,12,16],'max')]:
        axis=fig.add_axes([x[first]+.006,.145,x[last]+width-x[first]-.012,.017])
        colorbar=fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),cax=axis,orientation='horizontal',ticks=ticks,extend=extend,extendfrac=.025)
        colorbar.solids.set_rasterized(False)
        colorbar.ax.tick_params(labelsize=10,length=2.5,pad=2)
        colorbar.set_label(label,fontsize=10.5,labelpad=4)
        colorbar.outline.set_linewidth(.5)
    legend=[Patch(facecolor=UNDER,label='Disparity < 0'),Patch(facecolor=DISP_OVER,label='Disparity > 32'),
            Patch(facecolor=ERROR_OVER,label='Error > 16'),Patch(facecolor=BAD,edgecolor='#bbbbbb',label='Saved invalid GT')]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.078),ncol=4,frameon=False,
               fontsize=10.5,handlelength=1.4,columnspacing=2.0,handletextpad=.6)
    fig.text(.030,.020,'Fixed first/middle/last Val IDs, not error-selected. Same saved GT support; zero/negative predictions retained. No confidence filtering, crop or per-panel rescaling.',
             fontsize=9.5,ha='left',va='bottom',color='#444444')
    prefix.parent.mkdir(parents=True,exist_ok=True)
    description='Fixed correct-right spike cases; seed9471; common saved valid-GT support; native quarter135x240; disparity0–32px and error0–16px with explicit under/over colors.'
    fig.savefig(prefix.with_suffix('.png'),dpi=240,facecolor='white',metadata={'Description':description})
    fig.savefig(prefix.with_suffix('.svg'),facecolor='white',metadata={'Date':None,'Description':description})
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root',type=Path,default=ROOT/'storage/experiment-runs/snn-depth-estimation/stereo-spiking-transformer-study/selected-cases')
    parser.add_argument('--output-prefix',type=Path,default=STUDY/'results/pilot-spike-cases-v1')
    args=parser.parse_args()
    entries,source=load_sources(args.raw_root)
    render(entries,source,args.output_prefix)
    source.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        script=relative(Path(__file__)),script_sha256=file_sha(Path(__file__)),
        runtime={'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__},
        output_files={suffix:{'path':relative(args.output_prefix.with_suffix('.'+suffix)),
                              'sha256':file_sha(args.output_prefix.with_suffix('.'+suffix))} for suffix in ('png','svg')},
        SVG_editability='live text, vector layout/colorbars; native raster array images embedded, not auto-traced')
    args.output_prefix.with_suffix('.sources.json').write_text(json.dumps(source,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'RENDERED_SAVED_RESULTS_ONLY','sample_ids':source['preselected_case_ids'],
                      'artifacts_verified':len(source['cases']),'shared_GT_and_masks':True,
                      'output_prefix':relative(args.output_prefix)},indent=2))


if __name__=='__main__':main()
