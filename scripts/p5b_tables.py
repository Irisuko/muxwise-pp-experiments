"""Rebuild development-only comparisons and allocator figure."""
import json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=pathlib.Path('/root/workspace/muxpp_work');log=root/'logs/p5b'
tags=['r1-v2-instrumented','r2-async','r3-capacity','r4-inflight8']
runs={tag:json.loads((log/tag/'analysis.json').read_text()) for tag in tags}
lines=['| 运行 | TTFT p50/p90/p95/p99 (s) | TPOT p50/p90/p95/p99 (ms) | 联合SLO | 输入/输出 tok/s | GPU均值 | NVML峰值 MiB |','|---|---|---|---:|---|---:|---:|']
for tag,a in runs.items():
 s=a['summary'];lat=s['latencies_sec'];q=lambda key,scale:'/'.join(f'{lat[key][p]*scale:.2f}' for p in ['p50','p90','p95','p99']);gpu=sum(v['util_mean'] for v in a['gpu'].values())/4;peak=max(v['memory_max_MiB'] for v in a['gpu'].values())
 lines.append(f"| {tag} | {q('ttft',1)} | {q('tpot',1000)} | {s['slo']['ttft_le_5s_and_tpot_le_100ms']['rate']*100:.2f}% | {s['throughput']['input_tokens_per_sec']:.1f}/{s['throughput']['output_tokens_per_sec']:.1f} | {gpu:.2f}% | {peak} |")
lines += ['','| 运行 | 无准入计数 inflight/KV/预留/预算/请求槽/其它 | 等待墙钟 inflight/KV/预留 (s) | waiting时 protected p50/p90 (k) | 可用+可驱逐 p50 (k) |','|---|---|---|---|---:|']
for tag,a in runs.items():
 d=a['admission_arrival'];counts='/'.join(str(d['blocked'].get(k,0)) for k in ['inflight','kv','decode_reserve','budget','max_running_requests','other']);wall='/'.join(f"{d['wall'].get(k,0):.2f}" for k in ['inflight','kv','decode_reserve']);kv=d['waiting_allocator']['protected'];lines.append(f"| {tag} | {counts} | {wall} | {kv['p50']/1000:.1f}/{kv['p90']/1000:.1f} | {d['waiting_available']['p50']/1000:.1f} |")
lines += ['','| 运行/stage | prefill CUDA % | prefill empty % | decode CUDA % | decode send/token/control_recv/control % |','|---|---:|---:|---:|---|']
for tag,a in runs.items():
 for stage,d in a['timing_arrival_strict'].items():
  w=d['wall_percent'];dec=w['decode'];lines.append(f"| {tag}/{stage} | {d['cuda_percent'].get('prefill',0):.2f} | {w['prefill'].get('empty',0):.2f} | {d['cuda_percent'].get('decode',0):.2f} | {'/'.join(f'{dec.get(k,0):.2f}' for k in ['send','token','control_recv','control'])} |")
(log/'tables.md').write_text('\n'.join(lines)+'\n')
fig,axes=plt.subplots(4,2,figsize=(13,11),sharex=True)
for row,(tag,a) in enumerate(runs.items()):
 s=a['summary'];series=a['allocator_series'];x=[v['at']-s['run_started_unix'] for v in series]
 ax=axes[row,0];ax.stackplot(x,*[[v[k]/1000 for v in series] for k in ['protected','evictable','free']],labels=['Protected','Evictable','Free'],colors=['#bf4a47','#dfb35b','#6ba89e'],alpha=.85);ax.set_title(tag);ax.set_ylabel('KV tokens (k)');ax.set_ylim(0,380);ax.grid(alpha=.2)
 ax=axes[row,1];ax.plot(x,[v['waiting'] for v in series],label='Waiting requests',color='#364b80');ax.set_title('Admission queue');ax.set_ylabel('Requests');ax.grid(alpha=.2)
 for ax in axes[row]:
  ax.axvline(s['scheduled_duration_sec'],color='#555555',linestyle='--',linewidth=.8)
  ax.set_xlim(0,155)
max_wait=max(v['waiting'] for a in runs.values() for v in a['allocator_series'])
for ax in axes[:,1]:ax.set_ylim(0,max_wait*1.08)
axes[0,0].legend(loc='upper left',ncols=3,fontsize=8);axes[-1,0].set_xlabel('Seconds from client start');axes[-1,1].set_xlabel('Seconds from client start')
fig.tight_layout();fig.savefig(log/'allocator-comparison.png',dpi=160);fig.savefig(log/'allocator-comparison.svg')
print('\n'.join(lines[:6]))
