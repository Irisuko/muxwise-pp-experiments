"""Read-only evidence checks; does not run workloads or edit RESULTS.md."""
import json
from pathlib import Path
import re
ROOT = Path('/root/workspace/muxpp_work')
LOG = ROOT / 'logs/p8a'
audit = {'correctness': {}, 'runs': {}}
for ref, actual in [('tp2-reference', 'tp2pp2-serial'), ('tp2-reference-cold', 'tp2pp2-cold'),
                    ('tp2-reference', 'tp2-reference-cold')]:
    paths = [LOG / f'{name}-result.json' for name in (ref, actual)]
    if not all(p.exists() for p in paths): continue
    a, b = [json.loads(p.read_text()) for p in paths]
    rows = []
    for i, (x,y) in enumerate(zip(a['outputs'], b['outputs'])):
        x,y = x['output_ids'],y['output_ids']
        first = next((j for j,(u,v) in enumerate(zip(x,y)) if u != v),None)
        rows.append({'prompt':i, 'first_divergence_0based':first, 'equal_tokens':sum(u==v for u,v in zip(x,y)),
                     'reference_token': x[first] if first is not None else None,
                     'actual_token': y[first] if first is not None else None})
    audit['correctness'][actual+'_vs_'+ref] = {'exact_requests':sum(r['first_divergence_0based'] is None for r in rows),
                                             'equal_tokens':sum(r['equal_tokens'] for r in rows), 'rows':rows}
for p in LOG.glob('*-result.json'):
    data=json.loads(p.read_text())
    text=p.with_name(p.name.replace('-result.json','-server.log')).read_text()
    latest = {}
    for line in text.splitlines():
        if 'idle self-check PASS' not in line: continue
        match = re.search(r'TP(\d+) PP(\d+)', line)
        key = f'tp{match[1]}pp{match[2]}' if match else line.split(']')[0]
        latest[key] = {k:int(v) for k,v in re.findall(r'(free_kv|evictable|free_reqs|prefill_slices|decode_steps|cached_tokens|free_receive_slots)=(\d+)',line)}
    control_sequences = {}
    for line in text.splitlines():
        match=re.search(r'TP(\d+) PP(\d+)\] MuxWise-PP (admit .*|partition=.*|migrate .*|retract .*)',line)
        if match: control_sequences.setdefault(f'tp{match[1]}pp{match[2]}',[]).append(match[3])
    tp_consistent = all(control_sequences.get(f'tp0pp{stage}') == control_sequences.get(f'tp1pp{stage}') for stage in [0,1]) if data.get('CUDA_VISIBLE_DEVICES')=='0,1,2,3' and '--tp' in data['command'] and data['command'][data['command'].index('--tp')+1]=='2' else None
    audit['runs'][p.stem] = {'status':data['status'], 'error':data.get('error'), 'idle':latest,
                           'tp_control_sequences_equal':tp_consistent,
                           'aborts':len(data.get('aborts',[])),
                           'long_warmup':data.get('long_warmup',{}).get('meta_info'),
                           'scheduler_error':'Scheduler hit an exception' in text}
(LOG/'audit.json').write_text(json.dumps(audit, indent=2))
print(json.dumps({k:{'exact_requests':v['exact_requests'], 'equal_tokens':v['equal_tokens']} for k,v in audit['correctness'].items()},indent=2))
print(json.dumps({k: {'status':v['status'],'tp_equal':v['tp_control_sequences_equal']} for k,v in audit['runs'].items()},indent=2))
