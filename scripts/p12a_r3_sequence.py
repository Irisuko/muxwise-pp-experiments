#!/usr/bin/env python3
"""R3 numeric preflight before performance, then the remaining regressions."""
import json,subprocess
from pathlib import Path
ROOT=Path('/root/workspace/muxpp_work');OUT=ROOT/'logs/p12a';PY='/root/venvs/sgl/bin/python'
assert not list(OUT.glob('STOP-*.json'))
assert json.loads((OUT/'r2-llama_pp2tp2/run.json').read_text())['status']=='FAIL'
commands=[
 ('reference-first-driver.log',[PY,'-u','scripts/p12a_checks.py','--queue','logs/p12a/reference-first-queue.json']),
 ('r3-numeric-driver.log',[PY,'-u','scripts/p12a_checks.py','--queue','logs/p12a/r3-verify-first-queue.json','--config','configs/p12a_r3.json']),
 ('r3-driver.log',[PY,'-u','scripts/p12a_runner.py','--queue','logs/p12a/r3-queue.json','--config','configs/p12a_r3.json']),
 ('references-remaining-driver.log',[PY,'-u','scripts/p12a_checks.py','--queue','logs/p12a/reference-remaining-queue.json']),
 ('r3-verification-driver.log',[PY,'-u','scripts/p12a_checks.py','--queue','logs/p12a/r3-verify-remaining-queue.json','--config','configs/p12a_r3.json']),
 ('legacy-driver.log',[PY,'-u','scripts/p12a_checks.py','--queue','logs/p12a/legacy-queue.json']),
]
for name,cmd in commands:
    print('NEXT',name,cmd,flush=True)
    with (OUT/name).open('x') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
print('R3 PERFORMANCE AND REGRESSIONS COMPLETE',flush=True)
