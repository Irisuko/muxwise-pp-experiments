#!/usr/bin/env python3
"""Audit P8c saved artifacts and cleanup; no server starts or GPU workloads."""
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

import psutil
from p8c_summarize import ROOT, OUT, NEW, summary


def load(path):
    return json.loads(path.read_text())


def main():
    audit = ROOT / 'logs/p8c'
    before = load(audit / 'input-sha256-before.json')
    for filename, sha in before.items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == sha, filename
    selected = load(ROOT / 'configs/p8a_selected.json')
    source = '/root/workspace/sglang'
    commit = subprocess.check_output(['git', '-C', source, 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', source, 'status', '--porcelain'], text=True)
    assert commit == selected['commit'] == 'd8dfbfaba84ba1257cc42a519c24fc6c71987ef7' and not dirty
    from p8a_runner import signature
    assert json.loads(json.dumps(signature())) == selected['signature']
    marker = load(OUT / f'{NEW}.formal-started.json')
    folder = Path(marker['log_dir'])
    run = load(folder / 'run.json')
    assert run['status'] == 'PASS' and run['commit'] == commit and not run['dirty']
    assert run['signature'] == selected['signature']
    assert (audit / 'runner.rc').read_text().strip() == '0'
    command = load(folder / 'formal-command.json')
    assert command == marker['command']
    assert command[command.index('--warmup') + 1] == '2'
    assert command[command.index('--drain-timeout') + 1] == '300'
    assert '--force-close' in command and '--limit' not in command and '--time-scale' not in command
    assert Path(command[command.index('--trace') + 1]) == ROOT / 'bench/toolagent_600_240_x2.5.jsonl'
    launches = [p for p in (ROOT / 'logs/p8a').glob('*/formal-command.json') if NEW in load(p)]
    assert launches == [folder / 'formal-command.json'], launches
    warmups = [load(folder / name) for name in ['long-warmup.json', 'long-warmup-after-smoke.json']]
    assert all(w['input_len'] == 31663 and w['response']['meta_info']['completion_tokens'] == 8 for w in warmups)
    assert run['smoke']['completed'] == 20 and run['smoke']['failed'] == run['smoke']['unfinished'] == 0
    assert run['smoke']['duration_sec'] <= 60

    summary.MATRIX[NEW] = ('MuxWise-PP', 'TP2PP2')
    paths = [OUT / f'{l}{"_cut300" if l in summary.HISTORICAL else ""}_summary.json' for l in summary.MATRIX]
    runs = summary.prepare_matrix(summary.load_runs([str(p) for p in paths]), OUT)
    for s, records in runs:
        raw = load(OUT / f'{s["label"]}{"_cut300" if s["label"] in summary.HISTORICAL else ""}_summary.json')
        for key in ['completed', 'unfinished', 'failed', 'latencies_sec', 'slo', 'throughput', 'cached_tokens']:
            assert s[key] == raw[key], (s['label'], key)
        assert len(records) == len({r['idx'] for r in records}) == s['planned_requests'] == 1551
        assert sum(s[k] for k in ['completed', 'unfinished', 'failed']) == 1551
        assert all(v['total'] == 1551 for v in s['slo'].values())
        assert abs(s['cutoff_time'] - max(r['sent_at'] for r in records) - 300) < 1e-7
    assert {p.name.removesuffix('.formal-started.json') for p in OUT.glob('*.formal-started.json')} == set(summary.MATRIX)
    with (OUT / 'matrix_summary.csv').open() as f:
        matrix = list(csv.DictReader(f))
    with (OUT / 'best_summary.csv').open() as f:
        best = list(csv.DictReader(f))
    assert len(matrix) == 16 and len(best) == 4
    assert [r['label'] for r in matrix] == [s['label'] for s, _ in runs]
    assert [r['label'] for r in best] == [s['label'] for s, _ in summary.select_best(runs)]
    report = (ROOT / 'RESULTS.md').read_text()
    for row in summary.matrix_table(runs).splitlines()[2:]:
        assert row.replace('| MuxWise-PP | TP2PP2 |', '| MuxWise-PP | PP2×TP2 |') in report
    assert '与矩阵其它配置运行于不同日期，GPU 物理映射可能不同' in report
    assert '当前MuxWise-PP仅支持TP1' not in report

    evidence = load(OUT / 'p8c_prediction_metrics.json')['runs'][NEW]
    assert evidence['conserved'] and not evidence['errors']
    idle = evidence['idle']
    assert set(idle) == {'TP0 PP0', 'TP1 PP0', 'TP0 PP1', 'TP1 PP1'}
    state = []
    for rank, item in idle.items():
        vals = dict(re.findall(r'(free_kv|evictable|free_reqs|prefill_slices|decode_steps|cached_tokens|free_receive_slots)=(\d+)', item['text']))
        vals = {k: int(v) for k, v in vals.items()}
        assert vals['free_kv'] + vals['evictable'] == run['server_info']['pp2tp2']['max_total_num_tokens']
        assert vals['free_reqs'] == 128
        if 'PP1' in rank:
            assert vals['free_receive_slots'] == 2
        state.append(tuple(vals[k] for k in ['free_kv', 'evictable', 'prefill_slices', 'decode_steps', 'cached_tokens']))
    assert len(set(state)) == 1
    # NVML process names confirm the CUDA rank to physical UUID mapping during the run.
    with (audit / 'gpu-rank-processes.csv').open() as f:
        mapping_processes = list(csv.DictReader(f, skipinitialspace=True))
    for gpu in load(audit / 'runtime-gpu-mapping.json')['gpus']:
        expected = f'sglang::scheduler_TP{gpu["tp_rank"]}_PP{gpu["pp_rank"]}'
        assert any(p['gpu_uuid'] == gpu['uuid'] and p['process_name'] == expected for p in mapping_processes)

    residual = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            argv, name = proc.info['cmdline'] or [], proc.info['name'] or ''
            if proc.pid != os.getpid() and (name.startswith('sglang::') or
                any(a in ['sglang.launch_server', 'sglang_router.launch_router'] for a in argv) or
                any(a.endswith('/bench_client.py') and not a.startswith('-') for a in argv)):
                residual.append(proc.info)
        except psutil.NoSuchProcess:
            pass
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True).strip()
    ports = []
    for port in [30000, 31000, 31001, 31002, 31003, 8998, 8999, 9000, 9001]:
        with socket.socket() as sock:
            sock.settimeout(.2)
            if sock.connect_ex(('127.0.0.1', port)) == 0:
                ports.append(port)
    assert not gpu and not residual and not ports, (gpu, residual, ports)
    s = load(OUT / f'{NEW}_summary.json')
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [OUT / f'{NEW}{suffix}' for suffix in ['_requests.jsonl', '_summary.json', '.formal-started.json']]}
    tests = (audit / 'unit-tests.log').read_text()
    assert tests.rstrip().endswith('OK'), tests[-1000:]
    tests_count = int(re.search(r'Ran (\d+) tests', tests)[1])
    assert '2 passed' in (audit / 'runner-tests.log').read_text()
    result = dict(completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  source_commit=commit, source_clean=True, frozen_inputs_unchanged=True,
                  historical_artifacts_unchanged=True, new_formal_runs=1, restarts=0,
                  run_status=run['status'], startup_seconds=run['startup_s'],
                  long_warmup_seconds=[w['elapsed_s'] for w in warmups],
                  smoke_seconds=run['smoke']['duration_sec'],
                  matrix_rows=16, best_rows=4, summaries_recomputed_equal=16,
                  best_labels=[r['label'] for r in best], unit_tests=tests_count, runner_tests=2,
                  request_counts={k: s[k] for k in ['completed', 'unfinished', 'failed']},
                  rank_idle_checks=idle, gpu_compute_processes=gpu, residual_processes=residual,
                  occupied_ports=ports, formal_sha256=hashes)
    (audit / 'final-audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (audit / 'final-gpu.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    (audit / 'final-processes.json').write_text(json.dumps({'residual': residual, 'occupied_ports': ports}, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
