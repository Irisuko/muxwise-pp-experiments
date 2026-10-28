#!/usr/bin/env python3
"""Run selected-path regressions and explicitly audit output differences."""
import argparse
import json
from pathlib import Path
import subprocess
import p11a_runner as r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--config',type=Path,default=r.ROOT/'configs/p11a_selected.json')
    x=p.parse_args()
    selected=json.loads(x.config.read_text())
    assert r.monitoring.git(x.source,'rev-parse','HEAD')==selected['commit']
    assert not r.monitoring.git(x.source,'status','--porcelain')
    with (r.OUT/'final-legacy-driver.log').open('x') as log:
        subprocess.run([r.b.PYTHON,'scripts/p11a_legacy.py','--source',str(x.source),'--config',str(x.config)],
                       cwd=r.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    gate={}
    for label in r.LABELS[:4]:
        tag='final-verify-'+label
        print('START',tag,flush=True)
        with (r.OUT/(tag+'-driver.log')).open('x') as log:
            subprocess.run([r.b.PYTHON,'scripts/p11a_verify.py','verify',label,'--tag',tag,
                            '--source',str(x.source),'--extra',*selected['flags'][label]],
                           cwd=r.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        checked={}
        for kind in ['serial20','cold20']:
            actual=json.loads((r.OUT/tag/(kind+'.json')).read_text())
            ref=json.loads((r.OUT/('reference-'+label)/(kind+'.json')).read_text())
            old=r.ROOT/'logs/p10b/r4-mux-pp4'/(kind+'.json')
            previous=json.loads(old.read_text()) if label=='big_pp4' else ref
            checked[kind]=dict(exact_plain=sum(a['output_ids']==b['output_ids'] for a,b in zip(actual,ref)),
                               unexpected_cases=[i for i,(a,b,c) in enumerate(zip(actual,ref,previous))
                                                 if a['output_ids'] not in [b['output_ids'],c['output_ids']]],
                               previous_sha256=r.monitoring.sha(old) if label=='big_pp4' else None)
        checked['cold12288']=json.loads((r.OUT/tag/'cold12288-comparison.json').read_text())
        meta=json.loads((r.OUT/tag/'run.json').read_text())
        checked['stress']={k:meta['stress'][k] for k in ['completed','failed','unfinished','duration_sec']}
        checked['cancellation_requests']=len(json.loads((r.OUT/tag/'cancellation.json').read_text()))
        gate[label]=checked
        r.b.save(r.OUT/'final-correctness-gate.json',gate)
        print('END',tag,checked,flush=True)
    assert not any(v[k]['unexpected_cases'] for v in gate.values() for k in ['serial20','cold20']),gate
    assert all(v['cold12288']['exact'] for v in gate.values()),'Review actual CPP long-request numerical differences before claiming correctness'


if __name__=='__main__':main()
