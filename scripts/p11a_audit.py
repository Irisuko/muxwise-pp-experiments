#!/usr/bin/env python3
"""Audit recorded P11a metrics, per-rank resource conservation and trace scope."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT=Path('/root/workspace/muxpp_work')
OUT=ROOT/'logs/p11a'
sys.path.insert(0,str(ROOT/'bench'))
from metrics import summarize_records


def equal(a,b):
    if isinstance(a,dict):
        return all(k in b and equal(v,b[k]) for k,v in a.items())
    if isinstance(a,float):
        return isinstance(b,(int,float)) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)
    return a==b


def run_audit(path):
    meta=json.loads((path/'run.json').read_text())
    results=[]
    for command in path.glob('*command.json'):
        cmd=json.loads(command.read_text())
        if '--trace' in cmd:
            trace=Path(cmd[cmd.index('--trace')+1])
            assert trace.name in ['dev_toolagent_1200_240_x2.5.jsonl','dev_big_toolagent_1200_80_x7.5.jsonl']
            assert all(key in cmd for key in ['--limit','--warmup','--force-close','--drain-timeout'])
            assert cmd[cmd.index('--warmup')+1]=='2' and cmd[cmd.index('--drain-timeout')+1]=='300'
    for p in path.glob('*/*_summary.json'):
        s=json.loads(p.read_text())
        reqs=[json.loads(line) for line in Path(s['request_details_file']).open()]
        rebuilt=summarize_records(reqs,s['planned_requests'],s['cutoff_time'])
        assert equal(rebuilt,s),str(p)
        results.append(dict(file=str(p),recomputed=True,completed=s['completed'],failed=s['failed'],unfinished=s['unfinished']))
    ranks={};errors=[];expected=0
    for worker,info in meta.get('server_info',{}).items():
        log=path/(worker+'.log')
        if not log.exists():continue
        text=log.read_text()
        errors += [line for line in text.splitlines() if any(marker in line for marker in
            ['Scheduler hit an exception','torch.OutOfMemoryError','illegal memory access','AssertionError'])]
        if not info.get('enable_pdmux') or info.get('pp_size',1)==1:continue
        expected+=info['pp_size']*info['tp_size']
        capacity=info['max_total_num_tokens']
        pattern=r'([^\n]*?)idle self-check PASS stage=(\d+) free_kv=(\d+) evictable=(\d+) free_reqs=(\d+).*?free_receive_slots=(\d+)'
        for line in text.splitlines():
            if 'idle self-check PASS' not in line:continue
            m=re.search(pattern,line)
            assert m,line
            prefix,pp,free,evict,slots,receive=m.groups()
            tp=re.search(r'TP(\d+)',prefix)
            key=f'{worker}/pp{pp}/tp{tp.group(1) if tp else 0}'
            free,evict,slots,receive=map(int,[free,evict,slots,receive])
            ranks[key]=dict(free_kv=free,evictable=evict,capacity=capacity,free_reqs=slots,receive_slots=receive,
                            conserved=free+evict==capacity and slots==128 and receive==(0 if pp=='0' else 2))
    result=dict(run=path.name,status=meta.get('status'),measurements=results,expected_ranks=expected,
                ranks=ranks,all_conserved=(len(ranks)==expected and all(v['conserved'] for v in ranks.values())) if expected else None,
                errors=errors)
    (path/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');x=p.parse_args()
    manifest=json.loads((OUT/'protected-manifest.json').read_text())
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h for name,h in manifest.items())
    runs=[run_audit(p.parent) for p in OUT.glob('*/run.json') if 'finished_utc' in json.loads(p.read_text())]
    result=dict(protected_files_verified=len(manifest),runs=runs)
    if x.final:
        import pynvml
        import p11a_runner as r
        pynvml.nvmlInit();r.monitoring.OUT=OUT
        result['final_idle']=r.monitoring.idle('final')
        result['head']=r.monitoring.git(r.MAIN,'rev-parse','HEAD')
        result['git_status']=r.monitoring.git(r.MAIN,'status','--porcelain')
    (OUT/('final-audit.json' if x.final else 'audit-progress.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(protected=result['protected_files_verified'],runs=len(runs),
                         failed=[r['run'] for r in runs if r['errors'] or r['all_conserved'] is False]),indent=2))


if __name__=='__main__':main()
