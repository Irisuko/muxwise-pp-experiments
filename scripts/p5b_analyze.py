#!/usr/bin/env python3
"""Development admission/allocator and matched decode-cycle audit."""
import ast, collections, csv, datetime, json, pathlib, re, sys
import numpy as np
from p4b_analyze import analyze


def percentiles(values):
    if not values: return {}
    return dict(zip(['min','p10','p50','p90','p95','p99','max'], map(float,np.percentile(values,[0,10,50,90,95,99,100]))))


def audit(path):
    result=analyze(path)
    summary=result['summary'];start=summary['run_started_unix'];end=start+summary['duration_sec'];arrival_end=start+summary['scheduled_duration_sec']
    entries=[];series=[];memory=collections.defaultdict(dict)
    for line in (path/'pp4.log').read_text().splitlines():
        if 'MuxWise-PP summary' not in line:continue
        at=datetime.datetime.strptime(line[1:20],'%Y-%m-%d %H:%M:%S').timestamp()
        stages=ast.literal_eval(re.search(r'stages=(.*?) mbs=',line).group(1))
        for stage,v in stages.items():
            for key,value in v.get('cuda_memory',{}).items(): memory[stage][key]=max(memory[stage].get(key,0),value)
        a=stages[0]['admission']
        entries.append((at,a))
        series.extend(a['series'])
    series=list({s['at']:s for s in series}.values())
    def section(stop):
        during=[v for v in entries if start<=v[0]<=stop]
        if len(during)<2:return {}
        first,last=during[0][1],during[-1][1]
        d={field:{key:last[field].get(key,0)-first[field].get(key,0) for key in last[field].keys()|first[field].keys()} for field in ['blocked','stops','decisions','wall']}
        d['strict_window']=[during[0][0],during[-1][0]]
        d['blocked_pct']={k:100*v/max(1,sum(d['blocked'].values())) for k,v in d['blocked'].items()}
        d['wall_pct']={k:100*v/max(1e-6,sum(d['wall'].values())) for k,v in d['wall'].items()}
        rows=[s for s in series if start<=s['at']<=stop]
        d['allocator']={key:percentiles([s[key] for s in rows]) for key in ['capacity','used','free','evictable','protected','inflight_new_tokens','waiting']}
        rows=[s for s in rows if s['waiting']>0]
        d['waiting_allocator']={key:percentiles([s[key] for s in rows]) for key in ['free','evictable','protected','inflight_new_tokens']}
        d['waiting_available']=percentiles([s['free']+s['evictable'] for s in rows])
        return d
    result['cuda_memory_process_peak_MiB']={i:{k:v/2**20 for k,v in d.items()} for i,d in memory.items()}
    result['admission']=section(end);result['admission_arrival']=section(arrival_end)
    result['allocator_series']=[s for s in series if start<=s['at']<=end]
    if (path/'decode-rounds.jsonl').exists():
        rows=[json.loads(line) for line in (path/'decode-rounds.jsonl').read_text().splitlines()]
        rows=[r for r in rows if start<=r['at']<=arrival_end]
        result['decode_round_ms']=percentiles([1000*r['seconds'] for r in rows])
        bins=collections.defaultdict(list)
        for r in rows: bins[(r['bs'],int(r['ctx'])//8192,r['partition'])].append(r['seconds']*1000)
        result['decode_round_bins']={str(k):dict(n=len(v),**percentiles(v)) for k,v in bins.items()}
    (path/'analysis.json').write_text(json.dumps(result,indent=2))
    keys=sorted(set().union(*(s.keys() for s in result['allocator_series'])))
    with (path/'allocator.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(result['allocator_series'])
    return result


if __name__=='__main__':
    for path in sys.argv[1:]:
        r=audit(pathlib.Path(path));print(json.dumps({k:r[k] for k in ['admission_arrival','decode_round_ms'] if k in r},indent=2))
