#!/usr/bin/env python3
"""Audit each selected topology against v4 and same-model MuxWise at all rates."""
import json
from pathlib import Path
from p11a_analyze import OUT, MAIN, LOOSE


def main():
    selection = json.loads((OUT.parent.parent/'configs/p11a_selected.json').read_text())
    data = json.loads((OUT/'comparison.json').read_text())
    def row(rnd, label, rate):
        matches = [r for r in data if r['rate'] == rate and r['run'].endswith('-'+label)
                   and (r['run'].startswith('baseline-') if rnd == 'baseline' else r['run'] == rnd+'-'+label)]
        assert len(matches) == 1, (rnd, label, rate, len(matches))
        return matches[0]
    records = []
    lines = ['| 拓扑 | 所选轮次 | 速率 | v4→v5 主SLO % | v4→v5 输出tok/s | v5 goodput req/s | 相对MuxWise主SLO百分点 / 输出% |',
             '| --- | --- | --- | --- | --- | ---: | --- |']
    for label, rnd in selection['performance_rounds'].items():
        mux = 'big_muxwise_tp4' if label.startswith('big_') else 'llama_muxwise_tp2dp2'
        for rate in ['standard','double','saturated']:
            before, chosen, reference = [row(r,l,rate) for r,l in [('baseline',label),(rnd,label),('baseline',mux)]]
            a,b,c = [v['summary'] for v in [before,chosen,reference]]
            rate_slo = lambda x:100*x['slo'][MAIN]['rate']
            out = lambda x:x['throughput']['output_tokens_per_sec']
            slo_delta, out_delta = rate_slo(b)-rate_slo(a),100*(out(b)/out(a)-1)
            mux_slo, mux_out = rate_slo(b)-rate_slo(c),100*(out(b)/out(c)-1)
            rec = dict(label=label,round=rnd,rate=rate,baseline_run=before['run'],selected_run=chosen['run'],muxwise_run=reference['run'],
                       main_slo_v4=rate_slo(a),main_slo_v5=rate_slo(b),main_slo_delta_pp=slo_delta,
                       output_v4=out(a),output_v5=out(b),output_delta_pct=out_delta,
                       input_v5=b['throughput']['input_tokens_per_sec'],goodput_v4=before['goodput'],goodput_v5=chosen['goodput'],
                       loose_slo_v5=100*b['slo'][LOOSE]['rate'],duration_v5=b['duration_sec'],
                       both_non_decreasing=slo_delta>=0 and out_delta>=0,muxwise_main_gap_pp=mux_slo,muxwise_output_gap_pct=mux_out)
            records.append(rec)
            lines.append(f'| {label} | {rnd} | {rate} | {rate_slo(a):.3f}→{rate_slo(b):.3f} | {out(a):.2f}→{out(b):.2f} | {chosen["goodput"]:.4f} | {mux_slo:+.3f} / {mux_out:+.2f} |')
    report = dict(selected=selection,measurements=records,
                  all_rates_non_decreasing=all(r['both_non_decreasing'] for r in records))
    (OUT/'selected-comparison.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    (OUT/'selected-comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':main()
