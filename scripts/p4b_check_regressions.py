import json,re
from pathlib import Path
ROOT=Path('/root/workspace/muxpp_work')
R=ROOT/'logs/p4b'
report={}
for tag in ['tp1-correctness','v2-correctness','v2-control','v2-stress','original-pdmux','original-pp']:
 p=R/f'{tag}-result.json'
 if not p.exists():
  report[tag]={'status':'PENDING'}
  continue
 data=json.loads(p.read_text());assert data['status']=='PASS',data.get('error')
 report[tag]={'status':data['status'],'startup_s':data['startup_s']}
 if tag.startswith('v2-'):
  text=(R/f'{tag}-server.log').read_text()
  caps={int(i):int(c) for i,c in re.findall(r'MuxWise-PP stage=(\d+) layers=\[.*?capacity=(\d+)',text)}
  checks={}
  for line in text.splitlines():
   if 'idle self-check PASS' not in line:continue
   vals={k:int(v) for k,v in re.findall(r'(stage|free_kv|evictable|free_reqs|free_receive_slots)=(\d+)',line)}
   checks[vals['stage']]=vals
  assert set(checks)==set(range(4)),checks
  for stage,v in checks.items():
   assert v['free_kv']+v['evictable']==caps[stage],(v,caps)
   assert v['free_reqs']==128,v
   assert v['free_receive_slots']==(2 if stage else 0),v
  report[tag]['last_idle_checks']=checks
  assert not any(x in text for x in ['CUDA out of memory','CUDA error','AssertionError','Scheduler hit an exception']),tag
 if tag=='v2-control':
  report[tag].update(aborted=len(data['aborts']),unsupported_status=data['unsupported_status'],flush_status=data['flush_status'])
 if tag=='v2-stress':
  summary=json.loads((R/'v2-stress_summary.json').read_text());assert summary['completed_requests']==200 and summary['failed_requests']==0
  assert summary['time_scale']==0.02 and summary['force_close']
  assert summary['trace'].endswith('dev_toolagent_1200_240_x2.5.jsonl')
  report[tag].update(completed=200,failed=0,duration_sec=summary['duration_sec'])
 if tag.startswith('original-'):
  report[tag]['requests']=len(data['outputs'])
refs=[ROOT/'logs/p1/normal-correctness-result.json',R/'tp1-correctness-result.json',R/'v2-correctness-result.json']
if all(p.exists() for p in refs):
 outputs=[[v['output_ids'] for v in json.loads(p.read_text())['outputs']] for p in refs]
 assert all(len(v)==20 and all(len(ids)==64 for ids in v) for v in outputs)
 mismatch=[[i for i,(x,y) in enumerate(zip(outputs[0],other)) if x!=y] for other in outputs[1:]]
 assert mismatch==[[],[]],mismatch
 report['token_equality']={'requests':20,'tokens_each':64,'mismatches':0,'references':[str(p) for p in refs]}
for p in R.glob('*/*_summary.json'):
 summary=json.loads(p.read_text());assert summary['trace'].endswith('dev_toolagent_1200_240_x2.5.jsonl'),p
report['formal_trace_replayed']=False
(R/'regression-audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
