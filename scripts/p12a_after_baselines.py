#!/usr/bin/env python3
"""Frozen R1 validation sequence after baseline jobs have fully cleaned GPUs."""
import json
from pathlib import Path
import subprocess
import time
import psutil
ROOT=Path('/root/workspace/muxpp_work');OUT=ROOT/'logs/p12a';PY='/root/venvs/sgl/bin/python'
identity=json.loads((OUT/'big-runner-identity.json').read_text())
while psutil.pid_exists(identity['pid']):
    try:
        if psutil.Process(identity['pid']).create_time()!=identity['create_time']:break
    except psutil.NoSuchProcess:break
    time.sleep(2)
assert not list(OUT.glob('STOP-*.json'))
for item in json.loads((OUT/'big-ab-queue.json').read_text()):
    assert json.loads((OUT/item['tag']/'run.json').read_text())['status']=='PASS',item['tag']
commands=[
 ('r1-driver.log',[PY,'-u','scripts/p12a_runner.py','--queue','logs/p12a/r1-queue.json','--config','configs/p12a_r1.json']),
 ('references-driver.log',[PY,'-u','scripts/p12a_checks.py','--queue','logs/p12a/reference-queue.json']),
 ('r1-verification-driver.log',[PY,'-u','scripts/p12a_checks.py','--queue','logs/p12a/r1-verify-queue.json','--config','configs/p12a_r1.json']),
 ('legacy-driver.log',[PY,'-u','scripts/p12a_checks.py','--queue','logs/p12a/legacy-queue.json']),
]
for name,cmd in commands:
    print('NEXT',name,cmd,flush=True)
    with (OUT/name).open('x') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
print('R1 PERFORMANCE AND REGRESSIONS COMPLETE',flush=True)
