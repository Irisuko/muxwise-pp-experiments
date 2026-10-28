import json,subprocess,time
from pathlib import Path
from p10b_compare import compare
root=Path('/root/workspace/muxpp_work');log=root/'logs/p10b';py='/root/venvs/sgl/bin/python';cfg=json.loads((root/'configs/p10b_candidate_r2.json').read_text())
while not (log/'r1-dev-pp2tp2/gpu-cleanup.txt').exists():time.sleep(2)
assert json.loads((log/'r1-dev-pp2tp2/run.json').read_text())['status']=='PASS'
for mode in ['preflight','correctness','dev']:
 for topo in ['pp4','pp2tp2']:
  tag='r2d-'+('mux' if mode=='correctness' else mode)+'-'+topo;label='big_muxpp_'+topo;c=cfg[label]
  cmd=[py,str(root/'scripts/p10b_runner.py'),mode,label,'--mem',str(c['mem_fraction']),'--tag',tag,'--smoke-seconds','120']
  for k,v in c['env'].items():cmd+=['--env',k+'='+v]
  cmd+=['--extra']+c['flags']
  with (log/f'{tag}-driver.log').open('x') as f:rc=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=root)
  print(tag,rc,flush=True);assert rc==0
  if mode=='correctness':
   for kind in ['serial20','cold20']:
    result=compare(log/f'r0-plain-{topo}'/(kind+'.json'),log/tag/(kind+'.json'))
    (log/tag/(kind+'-comparison.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(kind,result['equal_requests'],result['equal_tokens'],result['early'],flush=True);assert not result['early']
  if mode=='dev':
   with (log/f'{tag}-analysis.log').open('x') as f:subprocess.run([py,str(root/'scripts/p9a_analyze.py'),str(log/tag)],stdout=f,stderr=subprocess.STDOUT,check=True)
