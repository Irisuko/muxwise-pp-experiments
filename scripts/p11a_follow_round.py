#!/usr/bin/env python3
"""Advance an already chosen development round only after the prior matrix passes."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import p11a_runner as r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--after',required=True)
    p.add_argument('--round',required=True,choices=['r2','r3','r4','r5','r6'])
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    x=p.parse_args()
    frozen=r.monitoring.sha(x.config)
    while True:
        assert not list(r.OUT.glob('STOP-*.json'))
        try: meta=json.loads((r.OUT/(x.after+'-big_pp4')/'run.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):meta={}
        if 'finished_utc' in meta:
            assert meta['status']=='PASS',meta
            break
        time.sleep(5)
    time.sleep(15)
    for label in r.LABELS[:4]:
        m=json.loads((r.OUT/(x.after+'-'+label)/'run.json').read_text())
        assert m['status']=='PASS' and all(k in m for k in ['standard','double','saturated'])
    assert r.monitoring.sha(x.config)==frozen
    config=json.loads(x.config.read_text())
    assert r.monitoring.git(x.source,'rev-parse','HEAD')==config['commit']
    assert not r.monitoring.git(x.source,'status','--porcelain')
    for script in ['p11a_analyze.py','p11a_compare.py']:
        with (r.OUT/(x.after+'-'+script+'-driver.log')).open('x') as log:
            subprocess.run([r.b.PYTHON,'scripts/'+script],cwd=r.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('START',x.round,'config_sha256',frozen,flush=True)
    with (r.OUT/(x.round+'-performance-driver.log')).open('x') as log:
        subprocess.run([r.b.PYTHON,'scripts/p11a_runner.py','--round',x.round,
                        '--source',str(x.source),'--config',str(x.config)],
                       cwd=r.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('END',x.round,flush=True)


if __name__=='__main__':main()
