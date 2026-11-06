"""Render actual preliminary receipts; no inference, smoothing or fabricated points."""
import hashlib
import json
import math
from pathlib import Path


def main():
    study = Path(__file__).resolve().parents[1]
    results = study/'results'
    sources = {}
    def read(path):
        raw = path.read_bytes()
        sources[str(path.relative_to(study))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    queue = read(study/'configs/pilot-first-seed-queue-v1.json')['runs']
    priors = read(results/'pilot-train-gt-priors-v1/result.json')['scene_equal_summary']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, ScalarFormatter, NullFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    colors = {'original_I':'#333333','A':'#0072B2','B':'#D55E00'}
    fig, axes = plt.subplots(1,2,figsize=(11.5,4.5),sharey=True)
    losses, loss_axes = plt.subplots(5,2,figsize=(11.5,12),sharex=True)
    seen = set()
    for item in queue:
        column = ('spike','event').index(item['sensor'])
        row = 0 if item['family']=='baseline' else (1 if item['architecture']=='A' else 2)+(0 if item['spiking'] else 2)
        label = 'ALIF+I' if item['family']=='baseline' else item['architecture']+(' spiking' if item['spiking'] else ' continuous')
        loss_ax = loss_axes[row,column]
        loss_ax.set_title(label+' / '+item['sensor'],loc='left',fontsize=10)
        loss_ax.grid(axis='y',color='#DDDDDD',linewidth=.6)
        loss_ax.set_xlim(0,1000)
        loss_ax.set_ylabel('Training objective')
        path = results/item['run']/'training-summary.json'
        if not path.exists():
            loss_ax.text(.5,.5,'Pending',ha='center',transform=loss_ax.transAxes,color='#777777')
            loss_ax.set_yticks([])
            continue
        training = read(path)['rows']
        if any(not math.isfinite(r['loss']) for r in training):
            raise ValueError('Nonfinite training receipt requires review')
        loss_ax.plot([r['update'] for r in training],[r['loss'] for r in training],color=colors[item['architecture']],linewidth=.8)
        val = read(results/item['run']/'validation-summary.json')['rows']
        if any(r['update'] not in (0,250,500,1000) or not math.isfinite(r['mean_scene_epe']) or r['mean_scene_epe']<=0 for r in val):
            raise ValueError('Unexpected validation visit or nonpositive EPE requires review')
        ax = axes[column]
        style = '-' if item['spiking'] else '--'
        initial = [r for r in val if r['update']==0]
        trained = [r for r in val if r['update']>0]
        if initial:
            ax.scatter([0],[initial[0]['mean_scene_epe']],facecolors='none',edgecolors=colors[item['architecture']],s=30)
        if trained:
            ax.plot([r['update'] for r in trained],[r['mean_scene_epe'] for r in trained],style,
                    color=colors[item['architecture']],marker='o',markersize=4,label=label,linewidth=1.6)
            seen.add(label)
    for column, ax in enumerate(axes):
        ax.set_title(('Spike input','Event input')[column],loc='left')
        for prior, color, style in zip(priors,('#777777','#999999','#AAAAAA'),(':','--','-.')):
            label = {'train_constant_median':'Train constant','train_pixel_median':'Train pixel median','zero_disparity':'Zero disparity'}[prior['method']]
            ax.axhline(prior['mean_scene_epe'],color=color,linestyle=style,linewidth=1,label=label)
        ax.set_yscale('log'); ax.set_xlim(-25,1025); ax.set_xticks([0,250,500,1000])
        ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel('Optimizer update'); ax.grid(axis='y',which='both',color='#DDDDDD',linewidth=.6)
        ax.legend(frameon=False,fontsize=8,loc='upper right')
    axes[0].set_ylabel('Val mean-scene EPE (quarter pixels, log scale)')
    fig.text(.5,.025,'Seed 9471 · 10 Val scenes · open marker: untrained (excluded from selection) · no seed uncertainty estimate',ha='center',fontsize=9)
    fig.subplots_adjust(left=.09,right=.985,bottom=.18,top=.90,wspace=.10)
    for ax in loss_axes[-1]: ax.set_xlabel('Optimizer update')
    losses.suptitle('Observed training trajectories — individual objectives and independent y scales',fontsize=13)
    losses.text(.5,.015,'Full 1,000-update schedule · seed 9471 · unsmoothed observations · objectives differ between ALIF+I and new models',ha='center',fontsize=9)
    losses.tight_layout(rect=(0,.035,1,.965))
    for figure,name in ((fig,'pilot-validation-curves-v1'),(losses,'pilot-training-curves-v1')):
        for suffix in ('svg','png'): figure.savefig(results/(name+'.'+suffix),dpi=150,facecolor='white')
        plt.close(figure)
    (results/'pilot-curves-v1.sources.json').write_text(json.dumps({'script':'scripts/plot_pilot_curves.py',
        'sources_sha256':sources,'matplotlib':matplotlib.__version__,'trained_series':sorted(seen),
        'semantics':'Actual individual-seed observations; Val0 descriptive; no smoothing or seed averaging; independent training-objective axes'},indent=2)+'\n')
    print(results/'pilot-validation-curves-v1.png')


if __name__=='__main__':
    main()
