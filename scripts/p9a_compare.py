#!/usr/bin/env python3
import json,sys
from pathlib import Path
root=Path('/root/workspace/muxpp_work/logs/p9a')
rows=[]
for p in sorted(root.glob('r*-*/analysis.json')):
 a=json.loads(p.read_text());s=a['summary'];l=s['latencies_sec']
 rows.append(dict(run=p.parent.name,completed=s['completed'],failed=s['failed'],unfinished=s['unfinished'],
  ttft={k:l['ttft'][k] for k in ('p50','p90','p95','p99')},
  tpot_ms={k:1000*l['tpot'][k] for k in ('p50','p90','p95','p99')},
  main_slo=100*s['slo']['ttft_le_5s_and_tpot_le_100ms']['rate'],
  ttft2=100*s['slo']['ttft_le_2s']['rate'],throughput=s['throughput'],
  gpu_mean=sum(g['util_mean'] for g in a['gpu'].values())/len(a['gpu']),
  matched=a['matched'],cache=100*s['cached_tokens']['token_weighted_ratio'],
  inflight_wall=a.get('arrival',{}).get('admission',{}).get('wall',{}).get('inflight',0)))
(root/'comparison.json').write_text(json.dumps(rows,indent=2))
lines=['| run | done/fail | TTFT p50/p90/p95/p99 s | TPOT p50/p90/p95/p99 ms | main SLO % | TTFT≤2s % | in/out tok/s | GPU % | cache % | inflight wall s |',
 '| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |']
for r in rows:
 lines.append('| '+r['run']+' | '+str(r['completed'])+'/'+str(r['failed'])+' | '+' / '.join(f'{v:.3f}' for v in r['ttft'].values())+' | '+' / '.join(f'{v:.3f}' for v in r['tpot_ms'].values())+f" | {r['main_slo']:.3f} | {r['ttft2']:.3f} | {r['throughput']['input_tokens_per_sec']:.2f} / {r['throughput']['output_tokens_per_sec']:.2f} | {r['gpu_mean']:.2f} | {r['cache']:.2f} | {r['inflight_wall']:.3f} |")
(root/'comparison.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
