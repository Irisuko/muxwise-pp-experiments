#!/usr/bin/env python3
"""A serial CPU/GPU lifecycle queue, waiting only on the named predecessor."""
import argparse
import json
import subprocess
import time
from pathlib import Path
import p11a_runner as r


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--after',required=True)
    p.add_argument('--kind',choices=['references','r1'],required=True)
    x=p.parse_args()
    marker=r.OUT/x.after/'run.json'
    while True:
        try:meta=json.loads(marker.read_text())
        except (FileNotFoundError,json.JSONDecodeError):meta={}
        if list(r.OUT.glob('STOP-*.json')):
            raise RuntimeError('External GPU occupancy; stop sequence')
        if 'finished_utc' in meta:
            assert meta['status']=='PASS',meta
            break
        time.sleep(5)
    # Do not acquire GPUs until predecessor's final cleanup check and guard end.
    time.sleep(15)
    if x.kind=='references':
        jobs=[['scripts/p11a_verify.py','reference',label] for label in r.LABELS[:4]]
    else:
        jobs=[['scripts/p11a_verify.py','candidate-smoke','llama_pp4','--tag','r1-smoke-llama_pp4',
               '--extra','--pdmux-pp-prefill-chunk-size','4096'],
              ['scripts/p11a_verify.py','candidate-smoke','big_pp2tp2','--tag','r1-smoke-big_pp2tp2',
               '--extra','--pdmux-pp-prefill-chunk-size','4096'],
              ['scripts/p11a_runner.py','--round','r1','--extra','--pdmux-pp-prefill-chunk-size','4096']]
    for i,job in enumerate(jobs):
        print('START',job,flush=True)
        with (r.OUT/f'{x.kind}-queue-{i}.log').open('x') as log:
            subprocess.run([r.b.PYTHON,*job],cwd=r.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('END',job,flush=True)


if __name__=='__main__':main()
