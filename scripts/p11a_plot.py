#!/usr/bin/env python3
"""Export a development-only two-objective comparison; no serving/GPU work."""
import argparse
import json
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from p11a_analyze import OUT, MAIN


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all-rounds',action='store_true')
    args=parser.parse_args()
    rows = json.loads((OUT/'comparison.json').read_text())
    grouped = {}
    for row in rows:
        if row['rate'] in ['standard','saturated']:
            key = row['run'].replace('baseline-resume-', 'baseline-')
            grouped.setdefault(key, {})[row['rate']] = row['summary']
    selected_path = OUT.parent.parent/'configs/p11a_selected.json'
    selected = json.loads(selected_path.read_text())['performance_rounds'] if selected_path.exists() else {}
    fig, axes = plt.subplots(1, 2, figsize=(12.8,5.5) if args.all_rounds else (10.8,4.5), layout='constrained')
    round_offsets={
        ('llama','pp4','baseline'):(7,-20),
        ('llama','pp4','r2'):(-28,12),
        ('llama','pp4','r6'):(-28,-13),
        ('llama','pp4','r3'):(8,-7),
        ('llama','pp2tp2','r2'):(-40,-22),
        ('llama','pp2tp2','r6'):(-25,10),
        ('big','pp4','r1'):(12,0),
        ('big','pp4','r4'):(-28,-19),
        ('big','pp4','r5'):(10,11),
        ('big','pp2tp2','r2'):(-30,5),
        ('big','pp2tp2','r6'):(10,8),
        ('big','pp2tp2','r5'):(10,-8),
        ('big','pp2tp2','r4'):(-25,-16),
    }
    for ax, family, title in zip(axes, ['llama','big'], ['Llama 8B', 'Mistral 24B']):
        for topo, color, name in [('pp4','#2878b5','PP4'),('pp2tp2','#df7a20','PP2 x TP2')]:
            points=[]
            for run, values in grouped.items():
                m = re.fullmatch(r'(baseline|r[1-6])-'+family+'_'+topo, run)
                if m is None or set(values) != {'standard','saturated'}:
                    continue
                rnd=m.group(1)
                if selected and not args.all_rounds and rnd not in ['baseline', selected.get(family+'_'+topo)]:
                    continue
                order=0 if rnd=='baseline' else int(rnd[1:])
                x=100*values['standard']['slo'][MAIN]['rate']
                y=values['saturated']['throughput']['output_tokens_per_sec']
                points.append((order,x,y,rnd))
            points.sort()
            ax.plot([p[1] for p in points],[p[2] for p in points],color=color,alpha=.45,lw=1)
            for order,x,y,rnd in points:
                chosen=selected.get(family+'_'+topo)==rnd
                marker='s' if order==0 else ('x' if order==1 else 'o')
                ax.scatter(x,y,color=color,marker=marker,s=65 if chosen else 40,edgecolors='black' if chosen else None,
                           label=name if order==0 else None,zorder=3)
                offset=(5,5 if topo=='pp4' else -12)
                if args.all_rounds:
                    offset=round_offsets.get((family,topo,rnd),offset)
                if family=='llama' and not args.all_rounds:
                    if topo=='pp4' and order==0:
                        offset=(7,-18)
                    elif topo=='pp2tp2' and chosen:
                        offset=(7,5)
                ax.annotate('v4' if order==0 else ('v5 ('+rnd.upper()+')' if chosen else rnd.upper()),(x,y),
                            xytext=offset,textcoords='offset points',fontsize=8,color=color,
                            arrowprops=dict(arrowstyle='-',color=color,alpha=.35,lw=.6) if args.all_rounds else None)
        mux='llama_muxwise_tp2dp2' if family=='llama' else 'big_muxwise_tp4'
        if 'baseline-'+mux in grouped:
            v=grouped['baseline-'+mux]
            x=100*v['standard']['slo'][MAIN]['rate'];y=v['saturated']['throughput']['output_tokens_per_sec']
            ax.scatter(x,y,color='#20916b',marker='*',s=130,label='MuxWise reference',zorder=4)
        ax.set_title(title)
        ax.set_xlabel('Main SLO, standard arrival rate (%)')
        ax.set_ylabel('Output tok/s, saturated arrivals')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8,loc='best')
        ax.margins(x=.18,y=.2)
    fig.suptitle('P11a development traces only: SLO and capacity'+
                 ('\nR1 is exploratory and failed numerical prechecks' if args.all_rounds else ''),fontsize=13)
    name='slo-throughput-all-rounds' if args.all_rounds else 'slo-throughput'
    for suffix in ['png','pdf']:
        fig.savefig(OUT/(name+'.'+suffix),dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
