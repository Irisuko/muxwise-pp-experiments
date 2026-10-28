#!/usr/bin/env python3
"""Audit P9a saved evidence, rank agreement, resource conservation and token equality."""
import hashlib,json,re
from pathlib import Path
ROOT=Path('/root/workspace/muxpp_work');LOG=ROOT/'logs/p9a'

def check_log(p):
 text=p.read_text();latest={};capacity={};seq={}
 for line in text.splitlines():
  pp=re.search(r'PP(\d+)\]',line);tp=re.search(r'TP(\d+)',line)
  if not pp:continue
  key=f'tp{tp[1] if tp else 0}pp{pp[1]}'
  if 'MuxWise-PP stage=' in line:
   capacity[key]=int(re.search(r'capacity=(\d+)',line)[1])
  if 'idle self-check PASS' in line:
   latest[key]={k:int(v) for k,v in re.findall(r'(free_kv|evictable|free_reqs|prefill_slices|decode_steps|cached_tokens|free_receive_slots)=(\d+)',line)}
  m=re.search(r'MuxWise-PP (admit .*|partition=.*|migrate .*|retract .*)',line)
  if m:seq.setdefault(key,[]).append(m[1])
 errors=[line for line in text.splitlines() if any(x in line for x in ['Scheduler hit an exception','torch.OutOfMemoryError','CUDA error:','PP radix divergence','PP decode batch divergence'])]
 resource=bool(capacity) and set(capacity)==set(latest) and all(v['free_kv']+v['evictable']==capacity[k] and v['free_reqs']==128 and v['free_receive_slots']==(0 if k.endswith('pp0') else 2) for k,v in latest.items())
 tp_equal={k:seq.get(k)==seq.get(k.replace('tp0','tp1')) for k in seq if k.startswith('tp0') and k.replace('tp0','tp1') in seq}
 return dict(capacity=capacity,idle=latest,resource_conservation=resource,tp_control_equal=tp_equal,errors=errors)

audit={'runs':{},'regressions':{},'correctness':{}}
for p in sorted(LOG.glob('r*-*/run.json')):
 r=json.loads(p.read_text());v=check_log(p.parent/'server.log');s=r.get('summary',{})
 v.update(driver_status=r.get('status'),driver_error=r.get('error'),completed=s.get('completed'),failed=s.get('failed'),unfinished=s.get('unfinished'))
 v['workload_pass']=s.get('completed')==300 and s.get('failed')==0 and s.get('unfinished')==0 and v['resource_conservation'] and not v['errors'] and all(v['tp_control_equal'].values())
 audit['runs'][p.parent.name]=v
reg=LOG/'regressions'
for p in sorted(reg.glob('*-result.json')):
 r=json.loads(p.read_text());v=check_log(p.with_name(p.name.replace('-result.json','-server.log')))
 v.update(status=r['status'],aborts=len(r.get('aborts',[])),error=r.get('error'))
 audit['regressions'][p.stem.removesuffix('-result')]=v
for tp,actual in [(2,'pp2tp2'),(1,'pp4')]:
 for suffix in ['serial','cold']:
  ref=reg/f'tp{tp}-reference-{suffix}-result.json';out=reg/f'{actual}-{suffix}-result.json'
  if not ref.exists() or not out.exists():continue
  a,b=[json.loads(p.read_text()) for p in (ref,out)];rows=[]
  assert len(a['outputs'])==len(b['outputs'])==20
  for i,(x,y) in enumerate(zip(a['outputs'],b['outputs'])):
   x,y=x['output_ids'],y['output_ids'];first=next((j for j,(u,v) in enumerate(zip(x,y)) if u!=v),None)
   rows.append(dict(prompt=i,exact=x==y,equal_tokens=sum(u==v for u,v in zip(x,y)),first_divergence=first))
  audit['correctness'][f'{actual}-{suffix}']=dict(exact_requests=sum(r['exact'] for r in rows),equal_tokens=sum(r['equal_tokens'] for r in rows),rows=rows)
protected=json.loads((LOG/'protected-inputs.json').read_text())
audit['protected_unchanged']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items()}
(LOG/'audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps({k:{'workload_pass':v['workload_pass'],'driver':v['driver_status']} for k,v in audit['runs'].items()},indent=2))
print(json.dumps({k:{'exact':v['exact_requests'],'tokens':v['equal_tokens']} for k,v in audit['correctness'].items()},indent=2))
