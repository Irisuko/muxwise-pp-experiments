import json
from pathlib import Path
root=Path('/root/workspace/muxpp_work');rows=[]
for topo in ['pp4','pp2tp2']:
 path=root/'logs/p10a'/f'sanity-big_muxpp_{topo}'/'sanity'/f'big_muxpp_{topo}_sanity_summary.json'
 d=json.loads(path.read_text());rows.append(('P10A',topo,d))
for f in sorted((root/'logs/p10b').glob('r*-dev-*/sanity/*_summary.json')):
 d=json.loads(f.read_text());tag=f.parents[1].name;rows.append((tag.split('-')[0],tag.split('-')[-1],d))
result=[]
for round,topo,d in rows:
 v=dict(round=round,topology=topo,completed=d['completed'],failed=d['failed'],unfinished=d['unfinished'],duration=d['duration_sec'],ttft=d['latencies_sec']['ttft'],tpot=d['latencies_sec']['tpot'],slo=d['slo']['ttft_le_5s_and_tpot_le_100ms'],cache=d['cached_tokens'],throughput=d['throughput']);result.append(v)
 print(round,topo,'done',v['completed'],'TTFT',*[round_val for round_val in [v['ttft'][k] for k in ['p50','p90','p95','p99']]],'TPOT ms',v['tpot']['p50']*1000,'SLO',v['slo']['passed'],'cache',v['cache']['token_weighted_ratio'])
(root/'logs/p10b/results.json').write_text(json.dumps(result,indent=2)+'\n')
