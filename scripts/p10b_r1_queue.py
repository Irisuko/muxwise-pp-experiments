import json,subprocess
from pathlib import Path
from p10b_compare import compare
root=Path('/root/workspace/muxpp_work');log=root/'logs/p10b';python='/root/venvs/sgl/bin/python';cfg=json.loads((root/'configs/p10b_candidate_r1.json').read_text())
for mode in ['correctness','preflight']:
 for topo in ['pp4','pp2tp2']:
  label='big_muxpp_'+topo;tag='r1b-'+('mux' if mode=='correctness' else 'preflight')+'-'+topo
  cmd=[python,str(root/'scripts/p10b_runner.py'),mode,label,'--mem',str(cfg[label]['mem_fraction']),'--tag',tag,'--smoke-seconds','120']
  for k,v in cfg['env'].items():cmd+=['--env',k+'='+v]
  cmd+=['--extra']+cfg['flags']
  with (log/f'{tag}-driver.log').open('x') as f:rc=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=root)
  print(tag,rc,flush=True);assert rc==0
  if mode=='correctness':
   for kind in ['serial20','cold20']:
    result=compare(log/f'r0-plain-{topo}'/(kind+'.json'),log/tag/(kind+'.json'))
    (log/tag/(kind+'-comparison.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(kind,result['equal_requests'],result['equal_tokens'],result['early'],flush=True)
    assert not result['early']
