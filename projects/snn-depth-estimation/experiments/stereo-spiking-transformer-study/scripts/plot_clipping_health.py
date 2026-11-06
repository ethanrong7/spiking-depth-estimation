"""Plot supplied compact health receipts; no model/data inference or new metrics."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    results=args.study/'results'
    output=args.output or results/'clipping-health-curves-v1'
    data=[];sources={}
    for seed in (9471,9472):
        for clip in (1,1000):
            path=results/f'health-A-spike-clip{clip}-seed{seed}-v1/training-summary.json'
            raw=path.read_bytes();receipt=json.loads(raw);rows=receipt['rows']
            if [r['update'] for r in rows]!=list(range(1,33)):
                raise ValueError('Plot requires all32 registered updates: '+str(path))
            targets=[r['readout_optimizer_health']['targets']['decoder.residual.2.weight'] for r in rows]
            ratio=[t['raw_energy_weighted_denominator_attenuation'] for t in targets]
            if any(x is None or not math.isfinite(x) for x in ratio):
                raise ValueError('Unavailable/nonfinite health values must be reviewed before plotting')
            data.append(dict(seed=seed,clip=clip,x=[r['update'] for r in rows],loss=[r['loss'] for r in rows],ratio=ratio))
            sources[str(path.relative_to(args.study))]=hashlib.sha256(raw).hexdigest()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.2))
    colors={1:'#0072B2',1000:'#D55E00'}
    for series in data:
        style='-' if series['seed']==9471 else '--'
        label=f"Clip {series['clip']}, seed {series['seed']}"
        for ax,key in zip(axes,('loss','ratio')):
            ax.plot(series['x'],series[key],color=colors[series['clip']],linestyle=style,
                    linewidth=1.8,label=label)
    axes[0].set_title('Repeated-panel training loss',loc='left',fontsize=12)
    axes[0].set_ylabel('Total training loss')
    axes[1].set_title('Final readout numerical health',loc='left',fontsize=12)
    axes[1].set_ylabel('Raw-gradient-energy-weighted ratio')
    axes[1].set_ylim(0,1.06)
    axes[1].axhline(.1,color='#666666',linestyle=':',linewidth=1.2)
    axes[1].text(31,.12,'Registered threshold 0.1',ha='right',va='bottom',color='#555555',fontsize=9)
    for ax in axes:
        ax.set_xlabel('Optimizer update');ax.set_xlim(1,32);ax.set_xticks([1,8,16,24,32])
        ax.grid(axis='y',color='#D9D9D9',linewidth=.6)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.0),ncol=4,frameon=False,fontsize=9)
    fig.text(.5,.015,'Eight repeated targets · full T=50 · first 32/1,000 scheduled updates · numerical screen, no convergence verdict',
             ha='center',fontsize=9,color='#555555')
    fig.subplots_adjust(left=.075,right=.985,bottom=.18,top=.80,wspace=.28)
    output.parent.mkdir(parents=True,exist_ok=True)
    for extension in ('svg','png'):
        fig.savefig(str(output)+'.'+extension,dpi=160)
    plt.close(fig)
    (Path(str(output)+'.sources.json')).write_text(json.dumps({'sources_sha256':sources,
        'script':'scripts/plot_clipping_health.py','matplotlib':matplotlib.__version__,
        'threshold_source':'analysis/clipping-health-gate-v1.md','no_error_bars':'Individual fixed-seed trajectories; no population uncertainty estimate'},indent=2)+'\n')
    print(str(output)+'.png')


if __name__=='__main__':main()
