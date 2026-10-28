import json,subprocess,time
from pathlib import Path
from p10b_compare import compare
root=Path('/root/workspace/muxpp_work');log=root/'logs/p10b';py='/root/venvs/sgl/bin/python';cfg=json.loads((root/'configs/p10b_candidate_r1.json').read_text())
while not (log/'r1b-preflight-pp2tp2/gpu-cleanup.txt').exists():time.sleep(2)
for topo in ['pp4','pp2tp2']:
 assert json.loads((log/f'r1b-preflight-{topo}/run.json').read_text())['status']=='PASS'
 for kind in ['serial20','cold20']:
  result=compare(log/f'r0-plain-{topo}'/(kind+'.json'),log/f'r1b-mux-{topo}'/(kind+'.json'))
  assert not result['early']
for mode in ['profile','dev']:
 for topo in ['pp4','pp2tp2']:
  tag='r1-'+mode+'-'+topo;label='big_muxpp_'+topo
  cmd=[py,str(root/'scripts/p10b_runner.py'),mode,label,'--mem',str(cfg[label]['mem_fraction']),'--tag',tag]
  env=dict(cfg['env'])
  if mode=='profile':env['SGLANG_PDMUX_PP_MEMORY_TRACE']=str(log/tag/'memory')
  for k,v in env.items():cmd+=['--env',k+'='+v]
  cmd+=['--extra']+cfg['flags']
  with (log/f'{tag}-driver.log').open('x') as f:rc=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=root)
  print(tag,rc,flush=True);assert rc==0
  tool='p10b_memory_report.py' if mode=='profile' else 'p9a_analyze.py'
  with (log/f'{tag}-analysis.log').open('x') as f:subprocess.run([py,str(root/'scripts'/tool),str(log/tag)],stdout=f,stderr=subprocess.STDOUT,check=True)
