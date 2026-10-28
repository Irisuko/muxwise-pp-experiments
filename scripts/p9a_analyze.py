#!/usr/bin/env python3
"""Audit request-ID timelines and development metrics; never generates traffic."""
import ast, collections, csv, datetime, json, re, sys
from pathlib import Path
import numpy as np


def dist(v):
    v=list(v)
    return dict(n=len(v),mean=float(np.mean(v)),**dict(zip(('p50','p90','p95','p99'),map(float,np.percentile(v,[50,90,95,99]))))) if v else {}

def diff(a,b):
    return {k:b.get(k,0)-a.get(k,0) for k in a.keys()|b.keys()}

def analyze(path):
    summary=json.loads(next(path.glob('*_summary.json')).read_text())
    requests=[json.loads(x) for x in next(path.glob('*_requests.jsonl')).read_text().splitlines()]
    start=summary['run_started_unix'];end=start+summary['duration_sec'];arrival_end=start+summary['scheduled_duration_sec']
    choices=collections.defaultdict(lambda:dict(planned=0,notice=0,shortened=0,deferred_layers=0))
    events=collections.defaultdict(lambda:collections.defaultdict(dict));cuda=collections.defaultdict(lambda:collections.defaultdict(float))
    for f in sorted(path.glob('requests-timeline.*.jsonl')):
        keys={}
        for line in f.open():
            e=json.loads(line);stage=e['stage']
            if e['event']=='slice_launch' and 'base_layers' in e and start<=e['at']<=end:
                c=choices[stage];c['planned']+=1;c['notice']+=bool(e.get('upstream_decode'))
                c['shortened']+=e['layers']<e['base_layers'];c['deferred_layers']+=e['base_layers']-e['layers']
            if e['event']=='admit':keys[e['batch_key']]=e['rids']
            if e['event']=='cuda_slice':
                for rid in keys.get(e['batch_key'],[]):cuda[rid][stage]+=e['seconds']
            for rid in e.get('rids',[]):events[rid][stage].setdefault(e['event'],e)
    rows=[];incomplete=[]
    for r in requests:
        rid=r.get('rid'); ev=events[rid];stages=sorted(ev)
        try:
            assert stages and 'prefill_start' in ev[0]
            e0=ev[0];first=r['sent_unix']+r['ttft'];components={}
            prev=r['sent_unix']
            def point(name,at):
                nonlocal prev
                components[name]=at-prev;prev=at
            point('client_to_stage0',e0['arrival']['at'])
            point('admission_queue',e0['admit']['at'])
            point('stage0_local_queue',e0['prefill_start']['at'])
            for stage in stages:
                es=ev[stage]
                if stage:
                    point(f'link{stage-1}_and_stage{stage}_queue',es['prefill_start']['at'])
                point(f'stage{stage}_prefill_span',es['prefill_done']['at'])
            point('last_result_dispatch',ev[stages[-1]]['result_send']['at'])
            point('result_return',e0['result_received']['at'])
            point('merge_wait',e0['merge']['at'])
            point('merge_to_emit',e0['emit']['at'])
            point('emit_to_client',first)
            assert min(components.values())>=-0.002,components
            assert abs(sum(components.values())-r['ttft'])<1e-5
            waits={}
            for stage,es in ev.items():
                def duration(key,a,b):
                    if a in es and b in es:waits[f'stage{stage}_{key}']=es[b]['at']-es[a]['at']
                duration('ready_observed','hidden_offer','send_launched')
                duration('send_observed','send_launched','send_done')
                duration('upstream_metadata','admit','hidden_metadata')
                duration('receive_slot','receive_slot_wait','receive_posted')
                duration('receive_and_fifo','receive_posted','hidden_received')
            rows.append(dict(idx=r['idx'],rid=rid,ttft=r['ttft'],components=components,
                waits=waits,cuda_seconds=dict(cuda[rid]),batch=e0['admit']['batch'],
                input_len=r['input_len'],cached_tokens=r['cached_tokens']))
        except (KeyError,AssertionError) as ex:incomplete.append(dict(idx=r['idx'],rid=rid,error=str(ex)))
    ordered=sorted(rows,key=lambda r:r['ttft']);quantiles={}
    for name,q in (('p50',.5),('p90',.9)):
        if not ordered:continue
        pos=(len(ordered)-1)*q;lo=int(pos);hi=int(np.ceil(pos));w=pos-lo
        quantiles[name]=dict(ttft=ordered[lo]['ttft']*(1-w)+ordered[hi]['ttft']*w,
            indices=[ordered[lo]['idx'],ordered[hi]['idx']],
            components={k:ordered[lo]['components'][k]*(1-w)+ordered[hi]['components'][k]*w for k in ordered[lo]['components']})
    result=dict(summary=summary,matched=len(rows),incomplete=incomplete,quantile_decomposition=quantiles,
        component_distributions={k:dist(r['components'][k] for r in rows) for k in rows[0]['components']} if rows else {},
        wait_distributions={k:dist(r['waits'][k] for r in rows if k in r['waits']) for k in set().union(*(r['waits'].keys() for r in rows))},
        cuda_per_request={stage:dist(r['cuda_seconds'].get(stage,0) for r in rows) for stage in stages})
    gpu=collections.defaultdict(list)
    for row in csv.reader((path/'gpu.csv').open()):
        try:at=datetime.datetime.strptime(row[0].strip(),'%Y/%m/%d %H:%M:%S.%f').timestamp()
        except ValueError:continue
        if start<=at<=end:gpu[int(row[1])].append((int(row[4].split()[0]),int(row[2].split()[0])))
    result['gpu']={i:dict(util_mean=float(np.mean([x[0] for x in v])),memory_max_MiB=max(x[1] for x in v),n=len(v)) for i,v in gpu.items()}
    samples=[]
    for line in (path/'server.log').open():
        if 'MuxWise-PP summary' not in line:continue
        at=datetime.datetime.strptime(line[1:20],'%Y-%m-%d %H:%M:%S').timestamp()
        data=ast.literal_eval(re.search(r'stages=(.*?) mbs=',line).group(1))
        samples.append((at,data))
    for key,stop in [('arrival',arrival_end),('full',end)]:
        selected=[(t,d) for t,d in samples if start+1<=t<=stop]
        if len(selected)<2:continue
        first,last=selected[0][1],selected[-1][1];d={}
        for stage in first:
            a,b=first[stage]['timing'],last[stage]['timing'];elapsed=b['elapsed']-a['elapsed']
            d[stage]=dict(seconds=elapsed,
                wall_percent={lane:{k:100*v/elapsed for k,v in diff(a['wall'][lane],b['wall'][lane]).items()} for lane in a['wall']},
                cuda_percent={k:100*v/elapsed for k,v in diff(a['cuda'],b['cuda']).items()},
                partition_percent={k:100*v/elapsed for k,v in diff(a['partitions'],b['partitions']).items()})
        for stage in d:
            if 'admission_budget_eval' in first[stage]:
                d[stage]['admission_budget_eval'] = diff(first[stage]['admission_budget_eval'],last[stage]['admission_budget_eval'])
        a,b=first[0]['admission'],last[0]['admission']
        result[key]=dict(window=[selected[0][0],selected[-1][0]],stages=d,
            admission={field:diff(a[field],b[field]) for field in ('blocked','stops','decisions','wall')})
    result['slice_planning']=dict(choices)
    result['source']=str(path)
    (path/'analysis.json').write_text(json.dumps(result,indent=2))
    (path/'ttft-requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    print(json.dumps(dict(run=path.name,matched=len(rows),ttft=summary['latencies_sec']['ttft'],tpot=summary['latencies_sec']['tpot'],slo=summary['slo']['ttft_le_5s_and_tpot_le_100ms'],decomposition=quantiles),indent=2))
    return result

if __name__=='__main__':
    for p in sys.argv[1:]:analyze(Path(p))
