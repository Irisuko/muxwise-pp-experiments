import json,os,subprocess,time
from pathlib import Path
root=Path('/root/workspace/muxpp_work')
last=root/'logs/p9a/r0-pp4-conversation/run.json'
started=time.monotonic()
while True:
 if last.exists():
  try:d=json.loads(last.read_text())
  except json.JSONDecodeError:d={}
  if 'finished_utc' in d:
   assert d['status']=='PASS',d.get('error');break
 if time.monotonic()-started>1800:raise TimeoutError('baseline queue')
 time.sleep(2)
env=dict(os.environ,P9A_SOURCE='/root/workspace/sglang-p9a-r1')
for trace in ['toolagent','conversation']:
 tag=f'r1-pp2tp2-{trace}'
 with (root/f'logs/p9a/{tag}-driver.log').open('x') as f:
  subprocess.run(['/root/venvs/sgl/bin/python',str(root/'scripts/p9a_runner.py'),'--trace',trace,'--tag',tag],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 subprocess.run(['/root/venvs/sgl/bin/python',str(root/'scripts/p9a_analyze.py'),str(root/'logs/p9a'/tag)],check=True)
