#!/usr/bin/env python3
"""Continue authorized diagnostics after the sequential A/B queue releases GPUs."""
import json
from pathlib import Path
import subprocess
import time
import psutil
ROOT=Path('/root/workspace/muxpp_work');OUT=ROOT/'logs/p12a';PY='/root/venvs/sgl/bin/python'
pid=33339;created=1793150091.33
while psutil.pid_exists(pid):
    try:
        if abs(psutil.Process(pid).create_time()-created)>.01:break
    except psutil.NoSuchProcess:break
    time.sleep(2)
assert not list(OUT.glob('STOP-*.json'))
for tag in ['ab-v4-full-llama_pp2tp2','ab-v5-llama_pp2tp2','ab-v5-llama_pp4']:
    assert json.loads((OUT/tag/'run.json').read_text())['status']=='PASS',tag
commands=[
 [PY,'-u','scripts/p12a_tp_probe.py','--tag','diag-shm-direct','--memcpy','0'],
 [PY,'-u','scripts/p12a_tp_probe.py','--tag','diag-shm-ce','--memcpy','1'],
 [PY,'-u','scripts/p12a_runner.py','--queue','logs/p12a/big-ab-queue.json'],
]
for i,cmd in enumerate(commands):
    print('NEXT',cmd,flush=True)
    subprocess.run(cmd,cwd=ROOT,check=True)
print('DIAGNOSTICS AND BIG BASELINES COMPLETE',flush=True)
