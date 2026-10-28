#!/usr/bin/env python3
"""Run the last allowed full matrix after R5 and frozen CPU-metadata validation."""
import json
from pathlib import Path
import subprocess
import time
import p11a_runner as r


def main():
    prep=r.OUT/'r6-preparation.json'
    driver=r.ROOT/'logs/p11a-r6-prepare-v2-driver.log'
    print('WAIT R5 and R6 CPU metadata validation',flush=True)
    while True:
        assert not list(r.OUT.glob('STOP-*.json'))
        text=driver.read_text()
        assert 'Traceback (most recent call last)' not in text,text[-4000:]
        if prep.exists() and 'READY R6' in text:break
        time.sleep(5)
    state=json.loads(prep.read_text());assert state['status']=='CPU_PASS_COMMITTED'
    source=Path(state['source'])
    assert r.monitoring.git(source,'rev-parse','HEAD')==state['commit']
    assert not r.monitoring.git(source,'status','--porcelain')
    for rnd in ['r1','r2','r3','r4','r5']:
        for label in r.LABELS[:4]:
            m=json.loads((r.OUT/(rnd+'-'+label)/'run.json').read_text())
            assert m['status']=='PASS' and all(k in m for k in ['standard','double','saturated'])
    bases={'llama_pp4':'r5','llama_pp2tp2':'r2e','big_pp2tp2':'r2e','big_pp4':'r4'}
    def scores(rnd,rate):
        s=json.loads((r.OUT/(rnd+'-big_pp4')/rate/(rate+'_summary.json')).read_text())
        return s['slo']['ttft_le_5s_and_tpot_le_100ms']['rate'],s['throughput']['output_tokens_per_sec']
    if all(all(a>=b for a,b in zip(scores('r5',rate),scores('r4',rate)))
           for rate in ['standard','double','saturated']):bases['big_pp4']='r5'
    cfg=r.ROOT/'configs/p11a_r6.json';assert not cfg.exists()
    config=dict(commit=state['commit'],base_configs=bases,flags={},
        rationale='R6最后一轮：CPU seq/prefix元数据及FlashInfer host indptr消除读回。PP2两组用R2批数准入/8192 CPP；8BPP4用R5；24BPP4仅当R5三档主SLO和输出均不低于R4时采用R5，否则用R4。所有KV/output预留不变。')
    for label,base in bases.items():
        old=json.loads((r.ROOT/('configs/p11a_'+base+'.json')).read_text())
        config['flags'][label]=old['flags'][label]+['--pdmux-pp-cpu-prefill-metadata']
    r.b.save(cfg,config)
    print('START R6',state['commit'],bases,flush=True)
    with (r.OUT/'r6-performance-driver.log').open('x') as log:
        subprocess.run([r.b.PYTHON,'scripts/p11a_runner.py','--round','r6','--source',str(source),
                        '--config',str(cfg)],cwd=r.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('END R6',flush=True)


if __name__=='__main__':main()
