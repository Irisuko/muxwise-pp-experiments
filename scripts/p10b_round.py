#!/usr/bin/env python3
import argparse,json,os,subprocess
from pathlib import Path
from p10b_audit_run import audit
from p10b_compare import compare
root=Path('/root/workspace/muxpp_work');log=root/'logs/p10b';py='/root/venvs/sgl/bin/python'
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--prefix',required=True);p.add_argument('--topology',choices=['pp4','pp2tp2'],required=True);p.add_argument('--modes',nargs='+',default=['preflight','dev']);a=p.parse_args()
assert set(a.modes)<=set(['preflight','dev','profile','correctness']);assert '/' not in a.prefix
cfg=json.loads(a.config.read_text());label='big_muxpp_'+a.topology;c=cfg[label]
with (log/(a.prefix+'-gpu-guard.log')).open('x') as f:
 guard=subprocess.Popen([py,str(root/'scripts/p10b_gpu_guard.py'),'--root',str(os.getpid()),'--evidence',str(log/(a.prefix+'-external-gpu.json'))],stdout=f,stderr=subprocess.STDOUT)
for mode in a.modes:
 tag=a.prefix+'-'+('mux' if mode=='correctness' else mode)+'-'+a.topology
 cmd=[py,str(root/'scripts/p10b_runner.py'),mode,label,'--mem',str(c['mem_fraction']),'--tag',tag,'--smoke-seconds','120'];env=dict(c['env'])
 if mode=='profile':env['SGLANG_PDMUX_PP_MEMORY_TRACE']=str(log/tag/'memory')
 for k,v in env.items():cmd+=['--env',k+'='+v]
 cmd+=['--extra']+c['flags']
 with (log/(tag+'-driver.log')).open('x') as f:subprocess.run(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
 r=audit(log/tag);assert r['status']=='PASS' and r['all_conserved'] and not r['errors']
 if mode=='correctness':
  for kind in ['serial20','cold20']:
   result=compare(log/f'r0-plain-{a.topology}'/(kind+'.json'),log/tag/(kind+'.json'))
   (log/tag/(kind+'-comparison.json')).write_text(json.dumps(result,indent=2)+'\n');assert not result['early']
 if mode in ['dev','profile']:
  tool='p9a_analyze.py' if mode=='dev' else 'p10b_memory_report.py'
  with (log/(tag+'-analysis.log')).open('x') as f:subprocess.run([py,str(root/'scripts'/tool),str(log/tag)],stdout=f,stderr=subprocess.STDOUT,check=True)
 if mode=='dev':subprocess.run([py,str(root/'scripts/p10b_diagnose.py'),str(log/tag)],check=True)
 print(tag,'PASS',flush=True)
