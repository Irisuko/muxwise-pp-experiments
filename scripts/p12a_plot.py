#!/usr/bin/env python3
"""Read-only development figures, never sends inference requests."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
OUT=Path('/root/workspace/muxpp_work/logs/p12a')

def main():
    rows=json.loads((OUT/'comparison.json').read_text())
    colors={'v4':'#6b7280','v5':'#d97706','r1':'#2563eb','r3':'#059669','r4':'#8b5cf6'}
    names={'r1':'R1 (crop only)','r3':'v5.1 (R3)','r4':'R4 (layers)'}
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for r in rows:
        if r['rate']!='full' or not r['run'].endswith(('llama_pp2tp2','big_pp2tp2')):continue
        row=int(r['run'].endswith('big_pp2tp2'))
        label=r['run'].rsplit('-',1)[0].replace('ab-','').replace('-full','')
        raw=[json.loads(l) for l in (OUT/r['run']/'full/ttft-requests.jsonl').open()]
        values=sorted(v['ttft'] for v in raw)
        color=colors.get(label)
        style=dict(label=names.get(label,label),color=color,linewidth=2.5 if label=='r3' else 1.5,
                   linestyle='--' if label in ['r1','r4'] else '-')
        axes[row,0].plot(values,np.arange(1,len(values)+1)/len(values),**style)
        bins=r['arrival_windows'];axes[row,1].plot([v['start']+30 for v in bins],[v['ttft']['p50'] for v in bins],marker='o',**style)
    for row,model in enumerate(['8B','24B']):
        axes[row,0].axvline(5,color='black',linestyle=':',linewidth=1)
        axes[row,0].set(xlabel='TTFT (s)',ylabel='CDF',title=f'{model} PP2 x TP2, full development trace')
        axes[row,1].set(xlabel='Client send window (s)',ylabel='TTFT p50 (s)',title=f'{model}: requests grouped by send time')
    for ax in axes.flat:ax.legend();ax.grid(alpha=.2)
    fig.savefig(OUT/'dev-ttft.png',dpi=160);fig.savefig(OUT/'dev-ttft.pdf')
if __name__=='__main__':main()
