#!/usr/bin/env python3
"""Complete R3 TP2 comparisons, then run the fourth and final candidate."""
import json,signal,subprocess,time
from pathlib import Path
import psutil
ROOT=Path('/root/workspace/muxpp_work');OUT=ROOT/'logs/p12a';PY='/root/venvs/sgl/bin/python'
ids=json.loads((OUT/'r3-handoff-identities.json').read_text())
def alive(which):
    row=ids[which]
    try:
        p=psutil.Process(row['pid']);return p.create_time()==row['created'] and p.status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:return False
while alive('runner'):
    assert not list(OUT.glob('STOP-*.json'))
    if 'END r3-big_pp2tp2 ' in (OUT/'r3-driver.log').read_text():
        for name in ['r3-llama_pp2tp2','r3-big_pp2tp2']:
            meta=json.loads((OUT/name/'run.json').read_text());assert meta['status']=='PASS' and 'finished_utc' in meta
        psutil.Process(ids['runner']['pid']).send_signal(signal.SIGTERM)
        (OUT/'r3-handoff.json').write_text(json.dumps(dict(reason='Both R3 TP2 model comparisons finished; last optimization round tests existing layer-partition control based on observed stage imbalance.',signal_pid=ids['runner']['pid']),indent=2)+'\n')
        break
    time.sleep(.25)
else:raise RuntimeError('R3 ended before clean TP2 comparison boundary')
while alive('runner') or alive('orchestrator'):time.sleep(.5)
assert not list(OUT.glob('STOP-*.json'))
commands=[
 ('r4-driver.log',[PY,'-u','scripts/p12a_runner.py','--queue','logs/p12a/r4-queue.json','--config','configs/p12a_r4.json']),
 ('r4-references-driver.log',[PY,'-u','scripts/p12a_checks.py','--queue','logs/p12a/r4-reference-queue.json']),
 ('r4-verification-driver.log',[PY,'-u','scripts/p12a_checks.py','--queue','logs/p12a/r4-verify-queue.json','--config','configs/p12a_r4.json']),
 ('legacy-driver.log',[PY,'-u','scripts/p12a_checks.py','--queue','logs/p12a/legacy-queue.json']),
]
for name,cmd in commands:
    print('NEXT',name,cmd,flush=True)
    with (OUT/name).open('x') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
print('R4 PERFORMANCE AND REGRESSIONS COMPLETE; NO FURTHER OPTIMIZATION',flush=True)
