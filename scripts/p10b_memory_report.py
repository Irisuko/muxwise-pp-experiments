#!/usr/bin/env python3
import json,re,sys
from pathlib import Path

def report(path):
 stages=[]
 for file in sorted(path.glob('memory-*.jsonl')):
  rows=[json.loads(x) for x in file.read_text().splitlines()]
  init={r['event']:r for r in rows if 'pp' in r}
  if 'ready' not in init:continue
  ready=init['ready'];w={v['ptr']:v['bytes'] for k,v in ready['backend_tensors'].items() if 'workspace' in k}
  d=dict(stage=ready['pp'],tp=ready['tp'],device=ready['device'],params=ready['params'],workspace_bytes=sum(w.values()),
    init={k:{f:r[f] for f in ['allocated','reserved','free','inactive_split']} for k,r in init.items()},
    peak_allocated=max(r['peak'] for r in rows if r['event'].startswith(('before_','after_'))),
    peak_reserved=max(r['reserved'] for r in rows),peak_inactive_split=max(r['inactive_split'] for r in rows))
  for kind in ['attention','mlp']:
   pairs=[];before=None
   for r in rows:
    if r['event']=='before_'+kind:before=r
    if r['event']=='after_'+kind and before:
     pairs.append(dict(layer=r['layer'],temporary_peak=r['peak']-before['allocated'],peak=r['peak'],after=r['allocated']));before=None
   d[kind]=pairs
  stages.append(d)
 result=dict(run=str(path),stages=sorted(stages,key=lambda d:(d['stage'],d['tp'])))
 (path/'memory-report.json').write_text(json.dumps(result,indent=2)+'\n')
 for d in result['stages']:
  print(d['stage'],d['tp'], 'weight/KV/backend/graphs/slots GiB',*[round((d['init'][k]['allocated']-(d['init'][p]['allocated'] if p else 0))/2**30,3) for k,p in [('weights',None),('kv','weights'),('backend','kv'),('graphs','backend'),('slots','calibration')]],'peak',round(d['peak_allocated']/2**30,3),'reserved',round(d['peak_reserved']/2**30,3),'fragment',round(d['peak_inactive_split']/2**30,3))
 return result
if __name__=='__main__':
 for p in sys.argv[1:]:report(Path(p))
