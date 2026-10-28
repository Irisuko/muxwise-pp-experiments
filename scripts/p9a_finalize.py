#!/usr/bin/env python3
"""Require all P9a evidence and an idle host before sealing the report."""
import contextlib
import datetime
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import psutil

R=Path('/root/workspace/muxpp_work'); L=R/'logs/p9a'
assert 'ALL FINAL GATES COMPLETE' in (L/'final-gates.log').read_text()
for script in ['p9a_compare.py','p9a_stage_report.py','p9a_audit.py']:
    subprocess.run([sys.executable,str(R/'scripts'/script)],check=True,stdout=subprocess.DEVNULL)
audit=json.loads((L/'audit.json').read_text())
assert len(audit['runs'])==16 and all(x['workload_pass'] for x in audit['runs'].values())
for name in audit['runs']:
    analysis=json.loads((L/name/'analysis.json').read_text())
    assert analysis['matched']==300 and not analysis['incomplete'], name
assert len(audit['regressions'])==15 and all(x['status']=='PASS' and not x['errors'] for x in audit['regressions'].values())
assert len(audit['correctness'])==4 and all(x['exact_requests']==20 and x['equal_tokens']==1280 for x in audit['correctness'].values())
assert all(audit['protected_unchanged'].values())
for name,d in audit['regressions'].items():
    if name.startswith(('pp2tp2','pp4')):
        assert d['resource_conservation'] and all(d['tp_control_equal'].values()), (name,d)
smoke=json.loads((L/'v4-pp2dp2-smoke/run.json').read_text())
assert smoke['status']=='PASS'
with contextlib.redirect_stdout(io.StringIO()):
    from p9a_audit import check_log
smoke_resources={p.name:check_log(p) for p in (L/'v4-pp2dp2-smoke').glob('worker*.log')}
assert len(smoke_resources)==2 and all(v['resource_conservation'] and not v['errors'] for v in smoke_resources.values())
traces=[]
for p in L.glob('*/client-command.json'):
    c=json.loads(p.read_text());t=Path(c[c.index('--trace')+1]).name
    assert t in ['dev_toolagent_1200_240_x2.5.jsonl','dev_conversation_1200_240_x2.5.jsonl']
    assert c[c.index('--limit')+1]=='300' and c[c.index('--time-scale')+1]=='1'
    assert c[c.index('--warmup')+1]=='2' and '--force-close' in c and c[c.index('--drain-timeout')+1]=='300'
    traces.append(dict(run=p.parent.name,trace=t))
assert len(traces)==16
for p in (L/'regressions').glob('*stress-result.json'):
    r=json.loads(p.read_text());c=r['bench_command']
    assert Path(c[c.index('--trace')+1]).name=='dev_toolagent_1200_240_x2.5.jsonl'
    s=json.loads(p.with_name(p.name.replace('-result.json','_summary.json')).read_text())
    assert s['completed']==200 and s['failed']==s['unfinished']==0
assert '52 passed' in (L/'v4-unit.log').read_text()
assert '10 passed' in (L/'bench-unit.log').read_text()
commit=subprocess.check_output(['git','-C','/root/workspace/sglang','rev-parse','HEAD'],text=True).strip()
assert not subprocess.check_output(['git','-C','/root/workspace/sglang','status','--porcelain'],text=True).strip()
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
assert not gpu,gpu
left=[]
for p in psutil.process_iter(['pid','name','cmdline']):
    c=' '.join(p.info['cmdline'] or [])
    if (p.info['name'] or '').startswith('sglang') or c.startswith('sglang::') or any(x in c for x in ['-m sglang.launch_server','-m sglang_router.launch_router','bench/bench_client.py','scripts/p9a_runner.py','scripts/p9a_verify.py','scripts/p9a_round.py','scripts/p9a_final_gates.py','scripts/p9a_regression_queue.py']) or ('nvidia-smi' in c and '--loop' in c):
        left.append(p.info)
assert not left,left
ports=[c.laddr.port for c in psutil.net_connections(kind='tcp') if c.status==psutil.CONN_LISTEN and c.laddr.port in [30000,31000,31001,31002,31003]]
assert not ports,ports
cleanup=dict(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),commit=commit,
    compute_processes=gpu,related_processes=left,listening_ports=ports,
    protected_unchanged=audit['protected_unchanged'],development_commands=traces,
    pp2dp2_resource_audit=smoke_resources,
    generated_formal_conversation_sha256=hashlib.sha256((R/'bench/conversation_600_240_x2.5.jsonl').read_bytes()).hexdigest(),
    formal_start_markers=len(list((R/'results').glob('*.formal-started.json'))),
    performance_runs=16,regression_runs=15,correctness_requests=80,stress_requests=400,cancel_requests=32)
assert cleanup['formal_start_markers']==16
(L/'final-cleanup.json').write_text(json.dumps(cleanup,indent=2)+'\n')
(L/'nvidia-smi-final.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
p=R/'configs/p9a_selected.json';s=json.loads(p.read_text());assert s['commit']==commit
s.update(validation_status='PASS',validation='logs/p9a/audit.json',cleanup='logs/p9a/final-cleanup.json',
    pp4_development_gates=['logs/p9a/r1v4-pp4-toolagent/run.json','logs/p9a/r1v4-pp4-conversation/run.json'],
    note='PP2 performance measured at R1; final defaults choose identical policy. Final HEAD passes correctness/stress/PP4 development and PP2DP2 smoke.')
p.write_text(json.dumps(s,indent=2)+'\n')
subprocess.run([sys.executable,str(R/'scripts/p9a_write_notes.py')],check=True)
print(json.dumps({k:cleanup[k] for k in ['commit','performance_runs','regression_runs','correctness_requests','stress_requests','cancel_requests','formal_start_markers']},indent=2))
