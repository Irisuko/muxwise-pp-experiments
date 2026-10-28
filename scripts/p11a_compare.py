#!/usr/bin/env python3
"""Compare completed development measurements to their same-trace v4 baseline."""
import json
from pathlib import Path
from p11a_analyze import OUT, MAIN


def main():
    data=json.loads((OUT/'comparison.json').read_text())
    labels=['llama_pp4','llama_pp2tp2','big_pp4','big_pp2tp2','llama_muxwise_tp2dp2','big_muxwise_tp4']
    indexed={}
    for row in data:
        label=next((label for label in labels if row['run'].endswith('-'+label)),None)
        if label is None:continue
        rnd=row['run'][:-(len(label)+1)]
        if rnd.startswith('baseline'):rnd='baseline'
        key=(rnd,label,row['rate'])
        assert key not in indexed,key
        indexed[key]=row
    rows=[]
    for (rnd,label,rate),row in indexed.items():
        if rnd=='baseline' or ('baseline',label,rate) not in indexed:continue
        before=indexed['baseline',label,rate]
        a,b=before['summary'],row['summary']
        slo_delta=100*(b['slo'][MAIN]['rate']-a['slo'][MAIN]['rate'])
        throughput_delta=100*(b['throughput']['output_tokens_per_sec']/a['throughput']['output_tokens_per_sec']-1)
        rows.append(dict(round=rnd,label=label,rate=rate,
            main_slo_before=100*a['slo'][MAIN]['rate'],main_slo_after=100*b['slo'][MAIN]['rate'],main_slo_delta_pp=slo_delta,
            output_before=a['throughput']['output_tokens_per_sec'],output_after=b['throughput']['output_tokens_per_sec'],output_delta_pct=throughput_delta,
            goodput_before=before['goodput'],goodput_after=row['goodput'],
            ttft_p50_before=a['latencies_sec']['ttft']['p50'],ttft_p50_after=b['latencies_sec']['ttft']['p50'],
            both_non_decreasing=slo_delta>=0 and throughput_delta>=0))
    (OUT/'deltas.json').write_text(json.dumps(rows,indent=2)+'\n')
    text=['| 轮次 | 拓扑 | 速率 | 主SLO v4→本轮 % | Δ百分点 | 输出 v4→本轮 tok/s | Δ% | goodput v4→本轮 req/s |',
          '| --- | --- | --- | --- | ---: | --- | ---: | --- |']
    for x in rows:
        text.append(f"| {x['round']} | {x['label']} | {x['rate']} | {x['main_slo_before']:.3f}→{x['main_slo_after']:.3f} | {x['main_slo_delta_pp']:+.3f} | {x['output_before']:.2f}→{x['output_after']:.2f} | {x['output_delta_pct']:+.2f} | {x['goodput_before']:.4f}→{x['goodput_after']:.4f} |")
    (OUT/'deltas.md').write_text('\n'.join(text)+'\n')
    print('\n'.join(text))


if __name__=='__main__':main()
