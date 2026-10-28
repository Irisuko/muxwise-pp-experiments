"""Validate completed evidence and cleanup without launching any workload."""
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

import psutil
import p8a_runner as runner

root=runner.ROOT
log=root/'logs/p8a'
p=log/'dev300-mem071'
run=json.loads((p/'run.json').read_text())
assert run['status']=='PASS' and not run['dirty']
assert run['dev300']['completed']==300 and run['dev300']['failed']==run['dev300']['unfinished']==0
selected=json.loads(runner.SELECTED.read_text())
assert selected['signature']==json.loads(json.dumps(runner.signature()))
assert selected['commit']==run['commit']
runner.assert_unused_formal()
text=(p/'pp2tp2.log').read_text()
idle={}; seq={}
for line in text.splitlines():
    m=re.search(r'TP(\d+) PP(\d+)',line)
    if not m:continue
    key=f'tp{m[1]}pp{m[2]}'
    if 'idle self-check PASS' in line:
        idle[key]={k:int(v) for k,v in re.findall(r'(free_kv|evictable|free_reqs|prefill_slices|decode_steps|cached_tokens|free_receive_slots)=(\d+)',line)}
    event=re.search(r'MuxWise-PP (admit .*|partition=.*|migrate .*|retract .*)',line)
    if event:seq.setdefault(key,[]).append(event[1])
assert len(idle)==4
for key,data in idle.items():
    assert data['free_kv']+data['evictable']==390622 and data['free_reqs']==128
    assert data['free_receive_slots']==(2 if key.endswith('pp1') else 0)
assert all(seq[f'tp0pp{s}']==seq[f'tp1pp{s}'] for s in [0,1])
with (p/'gpu.csv').open() as f:
    peak=max(float(r[' memory.used [MiB]'].split()[0]) for r in csv.DictReader(f))
results={}
for f in log.glob('*-result.json'):
    d=json.loads(f.read_text());results[f.stem]=d['status'];assert d['status']=='PASS',f
assert len(results)==11
processes=[]
for proc in psutil.process_iter(['pid','cmdline']):
    argv=proc.info['cmdline'] or []
    if any(v=='sglang.launch_server' or v.endswith(('/bench_client.py','/p8a_verify.py','/p8a_mlp_gpu.py','/p8a_runner.py')) for v in argv):
        processes.append(proc.info)
listeners=[c.laddr.port for c in psutil.net_connections(kind='tcp') if c.status==psutil.CONN_LISTEN and c.laddr.port in (30000,31000,31001,31002,31003)]
cleanup=dict(gpu_compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip(),
             git_status=subprocess.check_output(['git','-C',str(runner.SOURCE),'status','--porcelain'],text=True),
             commit=run['commit'], idle=idle, tp_control_sequences_equal=True, regression_statuses=results,
             processes=processes, listeners=listeners,
             new_formal_marker_exists=(root/'results'/f'{runner.LABEL}.formal-started.json').exists(),
             dev_marker=json.loads((log/f'{runner.LABEL}.dev300-started.json').read_text()),
             nvml_peak_mib=peak,
             allocator_peak_mib=max(map(int,re.findall(r"'max_allocated': (\d+)",text)))/2**20,
             dev_trace_sha256=hashlib.sha256(runner.DEV.read_bytes()).hexdigest())
assert not cleanup['gpu_compute'] and not cleanup['git_status'] and not cleanup['new_formal_marker_exists']
assert not processes and not listeners, (processes,listeners)
(log/'final-audit.json').write_text(json.dumps(cleanup,indent=2))
print(json.dumps(cleanup,indent=2))
