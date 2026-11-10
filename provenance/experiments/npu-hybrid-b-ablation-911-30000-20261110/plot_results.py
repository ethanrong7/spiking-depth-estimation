"""Regenerate the publication figures with matplotlib and numpy."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parent
curves=list(csv.DictReader((root/'summaries/curves.csv').open()))
endpoints=list(csv.DictReader((root/'summaries/endpoints.csv').open()))
variants=('linear-identity','linear-surrogate','signed-identity','signed-surrogate')
colors=dict(zip(variants,('#24658a','#a365a8','#359477','#d66a38')))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
out=root/'figures';out.mkdir(exist_ok=True)
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for col,seed in enumerate((1234,2345)):
    for row,metric in enumerate(('mean_scene_epe','bad3')):
        ax=axes[row,col]
        for v in variants:
            data=[r for r in curves if int(r['seed'])==seed and r['variant']==v and r['metric']==metric]
            ax.plot([int(r['update']) for r in data],[float(r['value']) for r in data],'-o',ms=3,color=colors[v],label=v)
        ax.set(xlabel='Optimizer update',ylabel='Mean-scene EPE (px, log scale)' if row==0 else 'Pooled bad3 (%)',title=f'Seed {seed}')
        if row==0:ax.set_yscale('log')
        ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8)
fig.suptitle('All saved Dev evaluations | four B encoder controls | fixed 30k endpoints')
for ext in ('png','svg'):fig.savefig(out/f'dev-curves.{ext}',dpi=160)
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
x=np.arange(4)
for ax,metric,label in zip(axes,('mean_scene_epe','bad3_percent'),('Mean-scene EPE (px)','Pooled bad3 (%)')):
    for seed,offset,color in ((1234,-.18,'#24658a'),(2345,.18,'#d66a38')):
        values=[float(next(r for r in endpoints if int(r['seed'])==seed and r['variant']==v)[metric]) for v in variants]
        bars=ax.bar(x+offset,values,.36,color=color,label=f'Seed {seed}')
        ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=8)
    ax.set(xticks=x,xticklabels=[v.replace('-','\n') for v in variants],ylabel=label)
    ax.set_ylim(0,ax.get_ylim()[1]*1.13);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].legend(fontsize=8);fig.suptitle('Final 30,000-update Dev results | paired seeds reported separately')
for ext in ('png','svg'):fig.savefig(out/f'final-endpoints.{ext}',dpi=160)
plt.close(fig)

# Matplotlib emits trailing spaces in SVG path data; normalize them for Git.
for p in out.glob("*.svg"):
    p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines())+"\n")
