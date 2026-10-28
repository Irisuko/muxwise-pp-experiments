#!/usr/bin/env python3
"""Validate a frozen final selection, then generate reports and an idle audit."""
import argparse
import json
from pathlib import Path
import subprocess
import p11a_runner as r


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--config',type=Path,default=r.ROOT/'configs/p11a_selected.json')
    p.add_argument('--skip-dp2',action='store_true')
    x=p.parse_args();digest=r.monitoring.sha(x.config)
    selected=json.loads(x.config.read_text())
    assert selected['commit']==r.monitoring.git(x.source,'rev-parse','HEAD')
    assert not r.monitoring.git(x.source,'status','--porcelain')
    jobs=[]
    if not x.skip_dp2:
        jobs.append(('selected-dp2-driver',['scripts/p11a_dp2.py','--source',str(x.source),'--config',str(x.config)]))
    jobs += [('final-checks-driver',['scripts/p11a_final_checks.py','--source',str(x.source),'--config',str(x.config)]),
             ('final-analysis',['scripts/p11a_analyze.py']),
             ('final-compare',['scripts/p11a_compare.py']),
             ('final-selection',['scripts/p11a_selection_report.py']),
             ('final-plot',['scripts/p11a_plot.py']),
             ('final-plot-all-rounds',['scripts/p11a_plot.py','--all-rounds']),
             ('final-audit-driver',['scripts/p11a_audit.py','--final']),
             ('final-notes',['scripts/p11a_write_notes.py'])]
    for tag,argv in jobs:
        assert not list(r.OUT.glob('STOP-*.json'))
        assert r.monitoring.sha(x.config)==digest
        print('START',tag,flush=True)
        with (r.OUT/(tag+'.log')).open('x') as log:
            subprocess.run([r.b.PYTHON,*argv],cwd=r.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('END',tag,flush=True)
    print('COMPLETE: inspect final figures, notes and audit before reporting.',flush=True)


if __name__=='__main__':main()
