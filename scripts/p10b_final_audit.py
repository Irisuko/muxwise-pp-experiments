#!/usr/bin/env python3
"""Audit completed P10B evidence and idle GPUs; never launch or replay work."""
import contextlib
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time

import psutil
import p10a_runner as old
import p10b_serve as selected
from p10b_audit_run import audit

root = Path('/root/workspace/muxpp_work')
log = root / 'logs/p10b'
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
before = read(log / 'before-audit.json')
protected = {p: digest for p, digest in before.items() if '/launch_big_' not in p}
assert all(sha(p) == digest for p, digest in protected.items())
markers = {str(p) for p in (root / 'results').glob('*.formal-started.json')}
assert markers == {p for p in before if p.endswith('.formal-started.json')}
assert sha(root / 'configs/pdmux_pp_4090_v4.yml') == '3bd2a392dcaae12cac690aa48de8f671de5b8cbff76769a5a388236d3aa784ae'

labels = [label for label in old.MATRIX if label != 'big_plain_tp4']
unchanged = [label for label in labels if not label.startswith('big_muxpp_')]
assert all(old.commands(label) == selected.commands(label) for label in unchanged)
for label in labels:
    subprocess.run(['bash', '-n', str(root / 'scripts' / ('launch_' + label + '.sh'))], check=True)
cfg = read(root / 'configs/p10b_selected.json')
evidence = []
for topo in ['pp4', 'pp2tp2']:
    item = cfg['big_muxpp_' + topo]
    for key in ['preflight', 'development']:
        path = Path(item[key])
        with contextlib.redirect_stdout(io.StringIO()):
            resource = audit(path)
        assert resource['all_conserved'] and not resource['errors']
        assert set(resource['idle']) == set(range(4 if topo == 'pp4' else 2))
        d = read(path / 'run.json')
        assert d['status'] == 'PASS'
        assert d['dev_trace_sha256'] == sha(root / 'bench/dev_big_toolagent_1200_80_x7.5.jsonl')
        evidence.append({'path': str(path), 'commit': d['commit'], 'capacity': resource['capacity'], 'resource_conserved': True})
    pre = next(iter(read(Path(item['preflight']) / 'run.json')['server_info'].values()))
    _, argv, env, _ = selected.commands('big_muxpp_' + topo)[0]
    # The final wrapper removes duplicate overrides; effective settings match
    # the validated candidate, with no diagnostic env or trace arguments.
    for arg in ['mem-fraction-static', 'pdmux-config-path', 'pdmux-pp-tpot-coeff', 'pdmux-pp-max-inflight-prefill']:
        assert argv[argv.index('--' + arg) + 1] == str(pre[arg.replace('-', '_')])
    assert not any('TRACE' in key for key in env)
    assert '--pdmux-pp-request-trace' not in argv

llama = log / 'llama-dev300'
with contextlib.redirect_stdout(io.StringIO()):
    resource = audit(llama)
assert resource['all_conserved'] and not resource['errors']
assert set(resource['idle']) == {0, 1}
assert read(log / 'llama-serial20-comparison.json')['equal_requests'] == 20
summary = read(next(llama.glob('*_summary.json')))
assert (summary['completed'], summary['failed'], summary['unfinished']) == (300, 0, 0)

head = subprocess.check_output(['git', '-C', str(old.SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
assert not subprocess.check_output(['git', '-C', str(old.SOURCE), 'status', '--porcelain'], text=True).strip()
subprocess.run(['git', '-C', str(old.SOURCE), 'diff', '--check'], check=True)

runtime_names = {'p10b_runner.py', 'p10b_round.py', 'p10b_resume.py', 'p10b_llama_regression.py', 'p10b_gpu_guard.py', 'p9a_verify.py', 'p9a_runner.py', 'trace_replay.py'}
live = []
for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
    if proc.pid == os.getpid():
        continue
    args = proc.info['cmdline'] or []
    names = {Path(x).name for x in args[:3]}
    if names & runtime_names or 'sglang.launch_server' in args or proc.info['name'].startswith('sglang'):
        live.append(proc.info)
ports = [c._asdict() for c in psutil.net_connections('tcp') if c.status == 'LISTEN' and c.laddr.port in (30000, 31000, 31001)]
samples = []
for i in range(3):
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    rows = [[v.strip() for v in line.split(',')] for line in raw.splitlines()]
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'], text=True).strip()
    samples.append({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'gpus': rows, 'compute_apps': apps})
    if i < 2:
        time.sleep(2)
cleanup = {'live_runtime_processes': live, 'listeners': ports, 'samples': samples, 'clean': not live and not ports and all(not s['compute_apps'] and len(s['gpus']) == 4 and all(int(r[2]) < 100 and int(r[3]) < 5 for r in s['gpus']) for s in samples)}
(log / 'final-cleanup.json').write_text(json.dumps(cleanup, indent=2) + '\n')
assert cleanup['clean'], cleanup

result = {'status': 'PASS', 'scope': 'Evidence integrity, resource safety and launch consistency; performance gates reported separately', 'llama_nonregression_status': cfg['_meta'].get('llama_nonregression_status'), 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'head': head, 'source_clean': True, 'formal_protected_hashes': protected, 'formal_marker_count': len(markers), 'formal_markers_unchanged': True, 'big_dev_trace_sha256': sha(root / 'bench/dev_big_toolagent_1200_80_x7.5.jsonl'), 'unchanged_launch_families': unchanged, 'selected_evidence': evidence, 'llama_complete': True, 'gpu_cleanup': 'final-cleanup.json', 'artifact_hashes': {str(p.relative_to(root)): sha(p) for p in [root / 'P10B_NOTES.md', root / 'configs/p10b_selected.json', log / 'final-launch-commands.json', *sorted((root / 'scripts').glob('launch_big_*.sh'))]}}
(log / 'final-audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': 'PASS', 'head': head, 'formal_marker_count': len(markers), 'clean': cleanup['clean']}))
