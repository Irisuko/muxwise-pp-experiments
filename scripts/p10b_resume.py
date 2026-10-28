#!/usr/bin/env python3
"""Resume P10B R2 only after the four GPUs are idle; unique evidence tags."""
import argparse,json,subprocess
from pathlib import Path
from p10b_compare import compare
from p10b_audit_run import audit
root=Path('/root/workspace/muxpp_work');log=root/'logs/p10b';py='/root/venvs/sgl/bin/python'
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prefix',required=True);a=p.parse_args()
assert a.prefix.startswith('r2') and '/' not in a.prefix
cfg=json.loads((root/'configs/p10b_candidate_r2.json').read_text())
for mode in ['preflight','correctness','dev','profile']:
 for topo in ['pp4','pp2tp2']:
  tag=a.prefix+'-'+('mux' if mode=='correctness' else mode)+'-'+topo;label='big_muxpp_'+topo;c=cfg[label]
  cmd=[py,str(root/'scripts/p10b_runner.py'),mode,label,'--mem',str(c['mem_fraction']),'--tag',tag,'--smoke-seconds','120']
  env=dict(c['env'])
  if mode=='profile':env['SGLANG_PDMUX_PP_MEMORY_TRACE']=str(log/tag/'memory')
  for k,v in env.items():cmd+=['--env',k+'='+v]
  cmd+=['--extra']+c['flags']
  with (log/f'{tag}-driver.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=root,check=True)
  if mode=='correctness':
   for kind in ['serial20','cold20']:
    result=compare(log/f'r0-plain-{topo}'/(kind+'.json'),log/tag/(kind+'.json'))
    (log/tag/(kind+'-comparison.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(tag,kind,result['equal_requests'],result['equal_tokens'],result['early'],flush=True);assert not result['early']
  if mode in ['dev','profile']:
   tool='p9a_analyze.py' if mode=='dev' else 'p10b_memory_report.py'
   with (log/f'{tag}-analysis.log').open('x') as f:subprocess.run([py,str(root/'scripts'/tool),str(log/tag)],stdout=f,stderr=subprocess.STDOUT,check=True)
  resource=audit(log/tag)
  assert resource['status']=='PASS' and resource['all_conserved'] and not resource['errors']
  assert resource['capacity'] >= (180000 if topo=='pp4' else 190000)
print('R2 complete; inspect performance before selecting final configuration. Llama regression and six-launch selection remain.',flush=True)
