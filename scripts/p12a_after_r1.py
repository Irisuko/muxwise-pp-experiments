#!/usr/bin/env python3
"""Finish the R1 8B measurement, then hand off at an idle boundary to R2."""
import json
import signal
import subprocess
import time
from pathlib import Path
import psutil
ROOT=Path('/root/workspace/muxpp_work');OUT=ROOT/'logs/p12a';PY='/root/venvs/sgl/bin/python'
ids=json.loads((OUT/'r1-handoff-identities.json').read_text())
def alive(which):
    row=ids[which]
    try:
        p=psutil.Process(row['pid'])
        return p.create_time()==row['created'] and p.status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:return False
while alive('runner'):
    assert not list(OUT.glob('STOP-*.json'))
    if 'END r1-llama_pp2tp2 ' in (OUT/'r1-driver.log').read_text():
        meta=json.loads((OUT/'r1-llama_pp2tp2/run.json').read_text())
        assert meta['status']=='PASS' and 'finished_utc' in meta
        psutil.Process(ids['runner']['pid']).send_signal(signal.SIGTERM)
        (OUT/'r1-handoff.json').write_text(json.dumps(dict(reason='R1 short main SLO fell 78.667 to 70.333%; both 8B rates finished. Continue with R2 transport ablation.',run_finished=meta['finished_utc'],signal_pid=ids['runner']['pid']),indent=2)+'\n')
        break
    time.sleep(.25)
else:raise RuntimeError('R1 ended before the expected clean measurement boundary')
while alive('runner') or alive('orchestrator'):time.sleep(.5)
assert not list(OUT.glob('STOP-*.json'))
commands=[
 ('r2-driver.log',[PY,'-u','scripts/p12a_runner.py','--queue','logs/p12a/r2-queue.json','--config','configs/p12a_r2.json']),
 ('references-driver.log',[PY,'-u','scripts/p12a_checks.py','--queue','logs/p12a/reference-queue.json']),
 ('r2-verification-driver.log',[PY,'-u','scripts/p12a_checks.py','--queue','logs/p12a/r2-verify-queue.json','--config','configs/p12a_r2.json']),
 ('legacy-driver.log',[PY,'-u','scripts/p12a_checks.py','--queue','logs/p12a/legacy-queue.json']),
]
for name,cmd in commands:
    print('NEXT',name,cmd,flush=True)
    with (OUT/name).open('x') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
print('R2 PERFORMANCE AND REGRESSIONS COMPLETE',flush=True)
