#!/usr/bin/env python3
import json,sys
from pathlib import Path
root=Path('/root/workspace/muxpp_work')
out=root/'logs/p11b'
labels=sorted(out.glob('*.launch-started.json'),key=lambda p:p.stat().st_mtime)
done=[]
for p in sorted((root/'results').glob('*_summary.json'),key=lambda p:p.stat().st_mtime):
    if 'v5' in p.name or p.name.startswith('sat_'):
        s=json.loads(p.read_text())
        if isinstance(s,dict):
            done.append(dict(label=s['label'],completed=s['completed'],unfinished=s['unfinished'],failed=s['failed'],
                main_pct=round(100*s['slo']['ttft_le_5s_and_tpot_le_100ms']['rate'],3),
                out_tps=round(s['throughput']['output_tokens_per_sec'],3)))
x=dict(completed_labels=len(done),results=done if '--all' in sys.argv else done[-1:])
if labels:
    label=labels[-1].name.removesuffix('.launch-started.json');folder=out/label
    x['current']=label
    meta=folder/'run.json'
    if meta.exists():x['status']=json.loads(meta.read_text()).get('status','running')
    for stage in ['formal','smoke']:
        log=folder/(stage+'-client.log')
        if log.exists():
            lines=log.read_text().splitlines()
            x['phase']=stage;x['last_client_line']=lines[-1] if lines else 'starting';break
    else:
        x['phase']='startup/warmup'
        for p in folder.glob('*.log'):
            lines=p.read_text().splitlines()
            if lines:x['last_server_line']=lines[-1][-240:]
stop=out/'STOP-external-gpu.json'
if stop.exists():x['STOP']=json.loads(stop.read_text()).get('reason')
x['batch_finished']=(out/'batch-finished.json').exists()
print(json.dumps(x,ensure_ascii=False))
