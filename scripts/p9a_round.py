#!/usr/bin/env python3
"""Serial development evaluations, with an optional completed-run dependency."""
import argparse,json,os,subprocess,time
from pathlib import Path
root=Path('/root/workspace/muxpp_work');py='/root/venvs/sgl/bin/python'
p=argparse.ArgumentParser();p.add_argument('--round',required=True);p.add_argument('--source',required=True);p.add_argument('--after');p.add_argument('--pp4',action='store_true');p.add_argument('--extra',nargs=argparse.REMAINDER,default=[]);a=p.parse_args()
if a.after:
 deadline=time.monotonic()+1800;last=root/'logs/p9a'/a.after/'run.json'
 while True:
  try:d=json.loads(last.read_text())
  except (FileNotFoundError,json.JSONDecodeError):d={}
  if 'finished_utc' in d:
   assert d['status']=='PASS',d.get('error')
   if not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():break
  if time.monotonic()>deadline:raise TimeoutError(a.after)
  time.sleep(2)
env=dict(os.environ,P9A_SOURCE=a.source)
for topology in (['pp2tp2','pp4'] if a.pp4 else ['pp2tp2']):
 for trace in ['toolagent','conversation']:
  tag=f'{a.round}-{topology}-{trace}'
  with (root/f'logs/p9a/{tag}-driver.log').open('x') as f:
   subprocess.run([py,str(root/'scripts/p9a_runner.py'),'--topology',topology,'--trace',trace,'--tag',tag,'--extra']+a.extra,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  subprocess.run([py,str(root/'scripts/p9a_analyze.py'),str(root/'logs/p9a'/tag)],check=True)
subprocess.run([py,str(root/'scripts/p9a_compare.py')],check=True)
subprocess.run([py,str(root/'scripts/p9a_audit.py')],check=True)
