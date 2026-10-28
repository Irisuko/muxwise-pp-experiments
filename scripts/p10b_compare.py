#!/usr/bin/env python3
import json,sys,hashlib
from pathlib import Path
root=Path('/root/workspace/muxpp_work/logs/p10b')
def compare(ref,actual):
 a=json.loads(ref.read_text());b=json.loads(actual.read_text());assert len(a)==len(b)==20
 rows=[]
 for i,(x,y) in enumerate(zip(a,b)):
  u,v=x['output_ids'],y['output_ids'];assert len(u)==len(v)==64
  j=next((j for j,(m,n) in enumerate(zip(u,v)) if m!=n),None)
  rows.append(dict(index=i,first_diff=j,ref_token=u[j] if j is not None else None,actual_token=v[j] if j is not None else None,equal_tokens=sum(m==n for m,n in zip(u,v)),ref_cached=x['meta_info'].get('cached_tokens'),actual_cached=y['meta_info'].get('cached_tokens')))
 return dict(reference=str(ref),actual=str(actual),equal_requests=sum(r['first_diff'] is None for r in rows),equal_tokens=sum(r['equal_tokens'] for r in rows),early=[r for r in rows if r['first_diff'] is not None and r['first_diff']<=2],rows=rows)
if __name__=='__main__':
 result={}
 for topo in ['pp4','pp2tp2']:
  for kind in ['serial20','cold20']:
   ref=root/f'r0-plain-{topo}'/(kind+'.json');actual=root/f'{sys.argv[1] if len(sys.argv)>1 else "r0"}-mux-{topo}'/(kind+'.json')
   if ref.exists() and actual.exists() and len(json.loads(actual.read_text()))==20:result[topo+'-'+kind]=compare(ref,actual)
 out=root/f'correctness-{sys.argv[1] if len(sys.argv)>1 else "r0"}.json';out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:{f:v[f] for f in ['equal_requests','equal_tokens','early']} for k,v in result.items()},indent=2))
