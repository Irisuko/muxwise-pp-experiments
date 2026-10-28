#!/usr/bin/env python3
"""Summarize P10a evidence without submitting any requests."""
import hashlib, json, re, shlex
from pathlib import Path
from p10a_runner import ROOT, LOG, MATRIX, commands


def load(p):return json.loads(p.read_text())

def main():
    selected=load(ROOT/'configs/p10a_selected.json') if (ROOT/'configs/p10a_selected.json').exists() else {}
    result={'selected':{},'runs':{},'correctness':{}}
    for label,selection in selected.items():
        server=selection['server_info']['server']
        result['selected'][label]={'mem_fraction':selection['mem_fraction'],
            'preflight_status':selection.get('preflight_status','PASS'),
            'qualification':selection.get('qualification'),
            'kv_capacity':server.get('max_total_num_tokens'),
            'memory_usage':server.get('internal_states',[{}])[0].get('memory_usage'),
            'preflight':selection['preflight']}
    for p in sorted(LOG.glob('*/run.json')):
        meta=load(p);v={'status':meta.get('status'),'error':meta.get('error'),'label':meta['label'],
            'commit':meta.get('commit'),'startup_s':meta.get('startup_s')}
        for phase in ('smoke','sanity'):
            if phase in meta:
                s=meta[phase];v[phase]={k:s.get(k) for k in ('completed','failed','unfinished','duration_sec','throughput','latencies_sec','cached_tokens')}
                v[phase]['main_slo']=s['slo']['ttft_le_5s_and_tpot_le_100ms']
        if (p.parent/'long-warmup.json').exists():v['long_warmup_s']=load(p.parent/'long-warmup.json')['elapsed_s']
        log=(p.parent/'server.log').read_text(errors='replace') if (p.parent/'server.log').exists() else ''
        v['errors']=[line for line in log.splitlines() if re.search(r'#new-token: -|Scheduler hit an exception|CUDA error:|OutOfMemoryError|AssertionError',line)]
        result['runs'][p.parent.name]=v
    refpath=LOG/'reference-tp4-serial20/serial20.json'
    if refpath.exists():
        ref=load(refpath)
        prompts=load(refpath.parent/'serial20-prompts.json')
        for p in sorted(LOG.glob('*/serial20.json')):
            if p==refpath:continue
            actual=load(p)
            if len(actual)!=20:continue
            assert load(p.parent/'serial20-prompts.json')==prompts,'Prompt mismatch: '+str(p)
            rows=[]
            for i,(a,b) in enumerate(zip(ref,actual)):
                x,y=a['output_ids'],b['output_ids']
                assert len(x)==len(y)==64
                divergence=next((j for j,(m,n) in enumerate(zip(x,y)) if m!=n),None)
                rows.append({'request':i,'input_len':len(prompts[i]),'first_divergence_0based':divergence,
                    'matching_prefix_tokens':64 if divergence is None else divergence,
                    'reference_token':None if divergence is None else x[divergence],
                    'actual_token':None if divergence is None else y[divergence],
                    'equal_positions':sum(m==n for m,n in zip(x,y))})
            result['correctness'][p.parent.name]={'exact_requests':sum(r['first_divergence_0based'] is None for r in rows),
                'total_requests':20,'equal_positions':sum(r['equal_positions'] for r in rows),'total_positions':1280,
                'first_divergence':next((r for r in rows if r['first_divergence_0based'] is not None),None),'requests':rows}
    (LOG/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (LOG/'final-commands.sh').open('w') as f:
        f.write('source /root/workspace/muxpp_work/env.sh\n')
        for label in selected:
            for _,cmd,env,_ in commands(label):f.write('\n# '+label+'\n'+shlex.join(['env']+[k+'='+v for k,v in env.items()]+cmd)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))

if __name__=='__main__':main()
