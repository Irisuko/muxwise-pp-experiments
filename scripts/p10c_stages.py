#!/usr/bin/env python3
"""Read existing ten-second PP summaries; never add runtime instrumentation."""
import ast,datetime,json,re
from pathlib import Path
import numpy as np
ROOT=Path('/root/workspace/muxpp_work')

def delta(a,b):return {k:b.get(k,0)-a.get(k,0) for k in a.keys()|b.keys()}

def analyze(label):
    s=json.loads((ROOT/'results'/f'{label}_summary.json').read_text())
    start=s['run_started_unix'];end=start+s['duration_sec'];snapshots=[]
    for line in (ROOT/'logs/p10c'/label/'server.log').open():
        if 'MuxWise-PP summary' not in line:continue
        stamp=datetime.datetime.strptime(line[1:20],'%Y-%m-%d %H:%M:%S').replace(tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()
        if start<=stamp<=end:
            snapshots.append((stamp,ast.literal_eval(re.search(r'stages=(.*?) mbs=',line).group(1))))
    out=dict(label=label,measurement_start_unix=start,measurement_end_unix=end,snapshot_count=len(snapshots),windows={})
    for name,stop in [('arrival',start+s['scheduled_duration_sec']),('full',end)]:
        rows=[v for v in snapshots if v[0]<=stop]
        assert len(rows)>=2
        first,last=rows[0][1],rows[-1][1];stages={}
        for stage in first:
            a,b=first[stage]['timing'],last[stage]['timing'];duration=b['elapsed']-a['elapsed']
            stages[stage]=dict(seconds=duration,wall_percent={lane:{k:100*v/duration for k,v in delta(a['wall'][lane],b['wall'][lane]).items()} for lane in a['wall']},
                cuda_percent={k:100*v/duration for k,v in delta(a['cuda'],b['cuda']).items()},
                partition_percent={k:100*v/duration for k,v in delta(a['partitions'],b['partitions']).items()})
        admission={field:delta(first[0]['admission'][field],last[0]['admission'][field]) for field in ['wall','blocked','stops','decisions']}
        unique={v['at']:v for _,states in rows for v in states[0]['admission'].get('series',[]) if start<=v['at']<=stop}
        pressure={}
        for reason in ['inflight','kv','decode_reserve']:
            values=[v for v in unique.values() if v.get('reason')==reason]
            pressure[reason]=dict(samples=len(values),protected_fraction_p50=float(np.median([v['protected']/v['capacity'] for v in values])) if values else None,
                                  waiting_p50=float(np.median([v['waiting'] for v in values])) if values else None)
        out['windows'][name]=dict(first_snapshot_unix=rows[0][0],last_snapshot_unix=rows[-1][0],summary_window_sec=rows[-1][0]-rows[0][0],
            admission=admission,stages=stages,pressure_samples=pressure)
    return out

def report():
    result={label:analyze(label) for label in ['big_muxpp_pp4','big_muxpp_pp2tp2']}
    (ROOT/'logs/p10c/formal-stage-analysis.json').write_text(json.dumps(result,indent=2))
    lines=['| MuxWise-PP配置 | 正式summary窗口(s) | inflight/KV/decode reserve阻塞(s) | 末级prefill等上游(%) | KV阻塞样本protected/capacity中位数 |',
           '| --- | ---: | --- | ---: | ---: |']
    for label,x in result.items():
        w=x['windows']['full'];block=w['admission']['wall'];last=w['stages'][max(w['stages'])]
        protected=w['pressure_samples']['kv']['protected_fraction_p50']
        lines.append('| '+label+' | '+f'{w["summary_window_sec"]:.0f}'+' | '+' / '.join(f'{block.get(k,0):.3f}' for k in ['inflight','kv','decode_reserve'])+
                     ' | '+f'{last["wall_percent"]["prefill"].get("upstream",0):.3f}'+' | '+(f'{100*protected:.3f}%' if protected is not None else '无样本')+' |')
    lines+=['','这些是正式客户端窗口内首末可匹配10秒summary的累计计数差，排除预热但会缺少窗口两端不足10秒；包含排空。'
            '阻塞时间是调度状态驻留，不能与各stage/请求延迟相加；GPU/双stream统计可能重叠。'
            'protected占比取KV阻塞时已有离散日志样本，不能解读为所有时刻的占用。'
            '在途上限、KV/未来decode预留与下游等上游均参与限制流水线；本次没有正式逐请求插桩或消融，不能独立量化通信与计算因果贡献。'
            '[原始差分及到达窗口/全窗口数据](logs/p10c/formal-stage-analysis.json)。']
    text='\n'.join(lines);(ROOT/'logs/p10c/formal-stage-analysis.md').write_text(text+'\n');return text

if __name__=='__main__':print(report())
