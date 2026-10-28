#!/usr/bin/env python3
"""Read-only P11a reports, including full measurement-window stage counters."""
import ast
from collections import Counter
import csv
import datetime
import json
import re
from pathlib import Path
import numpy as np

ROOT = Path('/root/workspace/muxpp_work')
OUT = ROOT/'logs/p11a'
MAIN = 'ttft_le_5s_and_tpot_le_100ms'
LOOSE = 'ttft_le_10s_and_tpot_le_100ms'


def delta(a, b):
    return {k:b.get(k,0)-a.get(k,0) for k in a.keys()|b.keys()}


def distribution(values):
    return dict(n=len(values), mean=float(np.mean(values)),
                **dict(zip(['p50','p90','p95','p99'], map(float,np.percentile(values,[50,90,95,99]))))) if values else {}


def analyze(run, rate):
    summary = json.loads((run/rate/f'{rate}_summary.json').read_text())
    start = summary['run_started_unix']
    end = start + summary['duration_sec']
    result = dict(run=run.name, rate=rate, summary=summary,
                  goodput=summary['slo'][MAIN]['passed']/summary['duration_sec'])
    gpu = {}
    for row in csv.reader((run/'gpu.csv').open()):
        try:
            at = datetime.datetime.strptime(row[0].strip(),'%Y/%m/%d %H:%M:%S.%f').timestamp()
        except ValueError:
            continue
        if start <= at <= end:
            gpu.setdefault(row[1].strip(),[]).append((int(row[4].split()[0]), int(row[2].split()[0])))
    result['gpu'] = {i:dict(samples=len(v),util_mean=float(np.mean([x[0] for x in v])),
                          memory_max_mib=max(x[1] for x in v)) for i,v in gpu.items()}
    rounds=[]
    for path in run.glob('decode-rounds-*.jsonl'):
        for line in path.open():
            row=json.loads(line)
            if start<=row['at']<=end:rounds.append(row)
    result['decode_rounds']=dict(samples=len(rounds),bs_histogram=dict(Counter(v['bs'] for v in rounds)),
        bs=distribution([v['bs'] for v in rounds]),period_seconds=distribution([v['seconds'] for v in rounds]))
    snapshots = {}
    for log in run.glob('*.log'):
        if not log.name.startswith(('server','worker')):
            continue
        for line in log.open():
            if 'MuxWise-PP summary' not in line:
                continue
            at = datetime.datetime.strptime(line[1:20],'%Y-%m-%d %H:%M:%S').timestamp()
            if start <= at <= end:
                snapshots.setdefault(log.stem,[]).append((at,ast.literal_eval(re.search(r'stages=(.*?) mbs=',line).group(1))))
    result['stage_windows'] = {}
    result['worker_stage_windows'] = {}
    for name, stop in [('arrival',start+summary['scheduled_duration_sec']), ('full',end)]:
        workers = {}
        for worker, samples in snapshots.items():
            rows = sorted((at,s) for at,s in samples if at <= stop)
            if len(rows)<2:
                continue
            first,last = rows[0][1],rows[-1][1]
            stages = {}
            for stage in first:
                a,b = first[stage]['timing'],last[stage]['timing']
                elapsed = b['elapsed']-a['elapsed']
                stages[stage] = dict(seconds=elapsed,
                    wall_seconds={lane:delta(a['wall'][lane],b['wall'][lane]) for lane in a['wall']},
                    cuda_seconds=delta(a['cuda'],b['cuda']),
                    partition_seconds=delta(a['partitions'],b['partitions']))
            admission = {key:delta(first[0]['admission'][key],last[0]['admission'][key])
                         for key in ['wall','blocked','stops','decisions']}
            workers[worker] = dict(first_unix=rows[0][0],last_unix=rows[-1][0],
                                  stages=stages,admission=admission)
        result['worker_stage_windows'][name] = workers
        # Preserve the original single-instance schema for the four core
        # topologies. Independent DP workers must never share counter deltas.
        if len(workers) == 1:
            result['stage_windows'][name] = next(iter(workers.values()))
    events = {}
    for path in run.glob('requests-timeline*.jsonl'):
        for line in path.open():
            e = json.loads(line)
            if not start <= e['at'] <= end:
                continue
            for rid in e.get('rids',[]):
                events.setdefault(rid,{}).setdefault(e['stage'],{}).setdefault(e['event'],[]).append(e)
    waits = {}
    matched = 0
    for line in (run/rate/f'{rate}_requests.jsonl').open():
        req = json.loads(line)
        ev = events.get(req.get('rid'),{})
        if 0 not in ev or 'admit' not in ev[0]:
            continue
        matched += 1
        def interval(name, left, right):
            waits.setdefault(name,[]).append(right-left)
        first = ev[0]
        if 'arrival' in first:
            interval('admission_queue', first['arrival'][0]['at'], first['admit'][0]['at'])
        if 'prefill_start' in first:
            interval('stage0_queue',first['admit'][0]['at'],first['prefill_start'][0]['at'])
        for stage,se in ev.items():
            if 'prefill_start' in se and 'prefill_done' in se:
                starts = {v['batch_key']:v['at'] for v in se['prefill_start']}
                durations = [v['at']-starts[v['batch_key']] for v in se['prefill_done'] if v['batch_key'] in starts]
                waits.setdefault(f'stage{stage}_prefill_service_sum',[]).append(sum(durations))
    result.update(matched_requests=matched, request_distributions={k:distribution(v) for k,v in waits.items()})
    (run/rate/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    results = []
    for path in sorted(OUT.glob('*/*/*_summary.json')):
        run,rate = path.parent.parent,path.parent.name
        if rate not in ['standard','double','saturated']:
            continue
        results.append(analyze(run,rate))
    (OUT/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
    lines = ['| run | 速率 | done/fail/unfinished | TTFT p50/p90/p95/p99 s | TPOT p50/p90/p95/p99 ms | 主/宽松 SLO % | 时长 s | in/out tok/s | goodput req/s | GPU % |',
             '| --- | --- | --- | --- | --- | --- | ---: | --- | ---: | ---: |']
    detail=[]
    for x in results:
        s=x['summary']; lat=s['latencies_sec']; t=s['throughput']
        def quant(kind,mul):
            return ' / '.join(f'{lat[kind][p]*mul:.3f}' if isinstance(lat[kind][p],(int,float)) else str(lat[kind][p]) for p in ['p50','p90','p95','p99'])
        gpu=np.mean([g['util_mean'] for g in x['gpu'].values()]) if x['gpu'] else float('nan')
        lines.append(f"| {x['run']} | {x['rate']} | {s['completed']}/{s['failed']}/{s['unfinished']} | {quant('ttft',1)} | {quant('tpot',1000)} | {100*s['slo'][MAIN]['rate']:.3f} / {100*s['slo'][LOOSE]['rate']:.3f} | {s['duration_sec']:.3f} | {t['input_tokens_per_sec']:.2f} / {t['output_tokens_per_sec']:.2f} | {x['goodput']:.4f} | {gpu:.2f} |")
        detail.extend([f"## {x['run']} / {x['rate']}", '',
            '完整计数见本档 analysis.json；以下取测量全窗口内首末summary差分，包含排空，窗口两端不足10秒不计。CUDA/stream可重叠。', ''])
        for worker, window in x['worker_stage_windows'].get('full',{}).items():
            detail.extend([f'Worker: `{worker}`',''])
            detail.extend(['| stage | 窗口 s | prefill/decode CUDA % | prefill compute/upstream/READY/send/switch % | decode control/control_recv/token % |',
                           '| --- | ---: | --- | --- | --- |'])
            for stage,v in window['stages'].items():
                elapsed=v['seconds']
                def perc(values,keys):return ' / '.join(f'{100*values.get(key,0)/elapsed:.2f}' for key in keys)
                detail.append(f"| {stage} | {elapsed:.2f} | {perc(v['cuda_seconds'],['prefill','decode'])} | {perc(v['wall_seconds']['prefill'],['compute_slot','upstream','ready','send','switch'])} | {perc(v['wall_seconds']['decode'],['control','control_recv','token'])} |")
            adm=window['admission']
            detail.extend(['','| 准入原因 | blocked 次数 | stop 次数 | decision 次数 | 驻留 s |','| --- | ---: | ---: | ---: | ---: |'])
            for reason in sorted(set().union(*(v.keys() for v in adm.values()))):
                detail.append(f"| {reason} | {adm['blocked'].get(reason,0)} | {adm['stops'].get(reason,0)} | {adm['decisions'].get(reason,0)} | {adm['wall'].get(reason,0):.3f} |")
            detail.append('')
        if x['decode_rounds']['samples']:
            detail.extend(['decode轮转微批大小直方图：`'+json.dumps(x['decode_rounds']['bs_histogram'],sort_keys=True)+'`。',''])
    (OUT/'comparison.md').write_text('\n'.join(lines)+'\n')
    (OUT/'stage-details.md').write_text('\n'.join(detail)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
