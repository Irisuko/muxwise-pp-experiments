"""Render auditable development tables from per-run analysis.json."""
import json
from pathlib import Path
R=Path('/root/workspace/muxpp_work/logs/p4b')
names=['v1','v1-profile','chunked','v2-r1','v2-r2','v2-r3','v2-r4','v2-static']
out=['| 运行 | 完成/失败 | TTFT p50/p90/p95/p99 (s) | TPOT p50/p90/p95/p99 (ms) | 联合SLO | 输入/输出 tok/s | GPU均值 | 峰值MiB |', '|---|---:|---|---|---:|---:|---:|---:|']
parts=['| 运行 | stage | 分区驻留百分比（按配置分区顺序） |', '|---|---:|---|']
lanes=['| 运行 | stage | prefill CUDA | 槽内间隙 | upstream | READY | send | empty | decode CUDA | decode proxy/token/send/control_recv |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
for name in names:
 p=R/name/'analysis.json'
 if not p.exists():continue
 a=json.loads(p.read_text()); s=a['summary'];failed=s['failed_requests']
 lat=s['latencies_sec']; gpu=a['gpu']
 mean=sum(v['util_mean'] for v in gpu.values())/len(gpu) if gpu else 0
 peak=max((v['memory_max_MiB'] for v in gpu.values()),default=0)
 q=lambda key,mul: '/'.join(f'{lat[key][p]*mul:.2f}' for p in ['p50','p90','p95','p99'])
 if failed:
  out.append(f'| {name}（失败；延迟不作比较） | {s["completed_requests"]}/{failed} | — | — | {s["slo"]["ttft_le_5s_and_tpot_le_100ms"]["rate"]*100:.2f}% | — | — | {peak} |')
 else:
  out.append(f'| {name} | {s["completed_requests"]}/0 | {q("ttft",1)} | {q("tpot",1000)} | {s["slo"]["ttft_le_5s_and_tpot_le_100ms"]["rate"]*100:.2f}% | {s["throughput"]["input_tokens_per_sec"]:.1f}/{s["throughput"]["output_tokens_per_sec"]:.1f} | {mean:.2f}% | {peak} |')
 for i,v in a.get('timing_strict',a.get('timing',{})).items():
  pct=v['partition_percent']; length=5 if name.startswith('v1') else 8
  parts.append(f'| {name} | {i} | '+ '/'.join(f'{pct.get(str(j),0):.1f}' for j in range(length))+' |')
 if name=='v1':
  for i,v in a['partition_summary_counts'].items():
   total=sum(v.values()); parts.append(f'| v1（summary采样频率） | {i} | '+ '/'.join(f'{100*v.get(str(j),0)/total:.1f}' for j in range(5))+' |')
 for i,v in a.get('timing_arrival_strict',a.get('timing_arrival',{})).items():
  w=v['wall_percent']['prefill'];d=v['wall_percent']['decode'];c=v['cuda_percent']; values=[c.get('prefill',0),w.get('compute_slot',0)-c.get('prefill',0),*[w.get(k,0) for k in ['upstream','ready','send','empty']],c.get('decode',0)]
  lanes.append(f'| {name} | {i} | '+' | '.join(f'{x:.2f}%' for x in values)+' | '+' / '.join(f'{d.get(k,0):.2f}%' for k in ['proxy','token','send','control_recv'])+' |')
(R/'tables.md').write_text('\n'.join(out)+'\n\n'+'\n'.join(parts)+'\n\n'+'\n'.join(lanes)+'\n')
print('\n'.join(out))
