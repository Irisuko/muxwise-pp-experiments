#!/usr/bin/env python3
"""P12a CPU-only per-request parent completion decomposition, supports CPP."""
from collections import defaultdict, Counter
import json
from pathlib import Path
import sys
import numpy as np
import p11a_analyze as base

OUT=Path('/root/workspace/muxpp_work/logs/p12a')

def dist(v):return base.distribution(list(v))

def analyze(run,rate):
    result=base.analyze(run,rate)
    summary=result['summary'];start=summary['run_started_unix'];end=start+summary['duration_sec']
    batches=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    arrivals={};admissions={};emits={};raw=[]
    for p in run.glob('requests-timeline*.jsonl'):
        worker=p.name.split('requests-timeline-')[1].rsplit('.',2)[0]
        for line in p.open():
            e=json.loads(line)
            if not start<=e['at']<=end:continue
            e['worker']=worker;raw.append(e)
            if e['event']=='arrival':
                for rid in e['rids']:arrivals[rid]=e
            if e['event']=='emit':
                for rid in e['rids']:emits[rid]=e
            if e['event']=='admit' and e['stage']==0:
                for rid in e['rids']:admissions[rid]=(worker,e['batch'])
            bid=e.get('parent_batch',e.get('batch'))
            if bid is not None:batches[(worker,bid)][e['stage']][e['event']].append(e)
    rows=[];errors=[]
    for line in (run/rate/(rate+'_requests.jsonl')).open():
        q=json.loads(line);rid=q.get('rid')
        try:
            key=admissions[rid];stages=batches[key];e0=stages[0];components={};prev=q['sent_unix']
            def point(name,t):
                nonlocal prev
                components[name]=t-prev;prev=t
            point('client_to_stage0',arrivals[rid]['at'])
            point('admission_queue',e0['admit'][0]['at'])
            point('stage0_queue',e0['prefill_start'][0]['at'])
            point('stage0_parent_prefill_span',max(e['at'] for e in e0['prefill_done']))
            for s in sorted(stages):
                if s==0:continue
                # Only exposed tails are additive; earlier chunks overlap stages.
                first=min(e['at'] for e in stages[s]['prefill_start'])
                last=max(e['at'] for e in stages[s]['prefill_done'])
                point(f'stage{s}_exposed_queue',max(prev,first))
                point(f'stage{s}_exposed_prefill_tail',last)
            point('last_result_dispatch',stages[max(stages)]['result_send'][-1]['at'])
            point('result_return',e0['result_received'][0]['at'])
            point('merge_wait',e0['merge'][0]['at'])
            point('merge_to_emit',emits[rid]['at'])
            point('emit_to_client',q['sent_unix']+q['ttft'])
            assert min(components.values())>=-.003,components
            assert abs(sum(components.values())-q['ttft'])<1e-5
            rows.append(dict(idx=q['idx'],rid=rid,ttft=q['ttft'],components=components,parent=key,
                             at=q['sent_unix']-start,input_len=q['input_len'],cached_tokens=q['cached_tokens']))
        except (KeyError,AssertionError) as e:errors.append(dict(idx=q['idx'],error=str(e)))
    ordered=sorted(rows,key=lambda r:r['ttft']);quantiles={}
    for name,q in [('p50',.5),('p90',.9),('p95',.95),('p99',.99)]:
        if not ordered:break
        pos=(len(ordered)-1)*q;lo=int(pos);hi=int(np.ceil(pos));w=pos-lo
        quantiles[name]=dict(indices=[ordered[lo]['idx'],ordered[hi]['idx']],ttft=ordered[lo]['ttft']*(1-w)+ordered[hi]['ttft']*w,
                            components={k:ordered[lo]['components'][k]*(1-w)+ordered[hi]['components'][k]*w for k in ordered[lo]['components']})
    timelines={}
    for stage in sorted({e['stage'] for e in raw}):
        es=[e for e in raw if e['stage']==stage]
        starts={(e['worker'],e['batch']):e for e in es if e['event']=='prefill_start'}
        waits=defaultdict(list)
        for name,left,right in [('ready','hidden_offer','send_launched'),('send','send_launched','send_done'),('receive_slot','receive_slot_wait','receive_posted'),('receive_and_fifo','receive_posted','hidden_received')]:
            a={(e['worker'],e['batch']):e['at'] for e in es if e['event']==left}
            waits[name]=[e['at']-a[(e['worker'],e['batch'])] for e in es if e['event']==right and (e['worker'],e['batch']) in a]
            assert all(v>=0 for v in waits[name]), (stage,name)
        timelines[stage]=dict(events=dict(Counter(e['event'] for e in es)),waits={k:dist(v) for k,v in waits.items()},
            service=dist(e['at']-starts[(e['worker'],e['batch'])]['at'] for e in es if e['event']=='prefill_done' and (e['worker'],e['batch']) in starts),
            cuda_slices=dist(e['seconds'] for e in es if e['event']=='cuda_slice'),
            slice_partitions=dict(Counter(e.get('partition') for e in es if e['event']=='slice_launch')))
    windows=[]
    for lo in range(0,int(summary['scheduled_duration_sec'])+1,60):
        select=[r for r in rows if lo<=r['at']<lo+60]
        if select:windows.append(dict(start=lo,n=len(select),ttft=dist(r['ttft'] for r in select),means={k:float(np.mean([r['components'][k] for r in select])) for k in select[0]['components']}))
    result.update(parent_matched=len(rows),decomposition_errors=errors,quantile_decomposition=quantiles,
                  component_distributions={k:dist(r['components'][k] for r in rows) for k in rows[0]['components']} if rows else {},
                  timeline_diagnostics=timelines,arrival_windows=windows)
    (run/rate/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    (run/rate/'ttft-requests.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in rows))
    return result

def report():
    results=[]
    for p in sorted(OUT.glob('*/*/*_summary.json')):
        if p.parent.name not in ['partial','full']:continue
        results.append(analyze(p.parent.parent,p.parent.name))
    (OUT/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
    lines=['| run | trace | n/fail/unfinished | TTFT p50/p90/p95/p99 s | TPOT p50/p90/p95/p99 ms | main/loose % | in/out tok/s | GPU % |','| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in results:
        s=r['summary'];lat=s['latencies_sec'];t=s['throughput']
        def q(k,m):return '/'.join(f'{lat[k][p]*m:.3f}' for p in ['p50','p90','p95','p99'])
        gpu=np.mean([g['util_mean'] for g in r['gpu'].values()])
        lines.append(f"| {r['run']} | {r['rate']} | {s['completed']}/{s['failed']}/{s['unfinished']} | {q('ttft',1)} | {q('tpot',1000)} | {s['slo'][base.MAIN]['rate']*100:.3f}/{s['slo'][base.LOOSE]['rate']*100:.3f} | {t['input_tokens_per_sec']:.2f}/{t['output_tokens_per_sec']:.2f} | {gpu:.2f} |")
    (OUT/'comparison.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

if __name__=='__main__':report()
