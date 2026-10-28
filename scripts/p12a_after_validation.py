#!/usr/bin/env python3
"""Paired unchanged DP2 regression repeat; no additional optimization round."""
import json,subprocess,time
from pathlib import Path
import psutil
ROOT=Path('/root/workspace/muxpp_work');OUT=ROOT/'logs/p12a';PY='/root/venvs/sgl/bin/python'
row=json.loads((OUT/'validation-orchestrator-identity.json').read_text())
while psutil.pid_exists(row['pid']):
    try:
        p=psutil.Process(row['pid'])
        if p.create_time()!=row['created'] or p.status()==psutil.STATUS_ZOMBIE:break
    except psutil.NoSuchProcess:break
    time.sleep(1)
assert not list(OUT.glob('STOP-*.json'))
selected=json.loads((ROOT/'configs/p12a_selected.json').read_text())
for tag in list(selected['verification_runs'].values())+[v['tag'] for v in json.loads((OUT/'legacy-queue.json').read_text())]:
    assert json.loads((OUT/tag/'run.json').read_text())['status']=='PASS',tag
print('NEXT paired unchanged DP2 verification',flush=True)
with (OUT/'dp2-repeat-driver.log').open('x') as log:
    subprocess.run([PY,'-u','scripts/p12a_runner.py','--queue','logs/p12a/dp2-repeat-queue.json','--config','configs/p12a_r3.json'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
print('PAIRED DP2 VERIFICATION COMPLETE',flush=True)
