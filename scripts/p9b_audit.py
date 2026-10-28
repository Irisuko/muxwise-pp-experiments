#!/usr/bin/env python3
"""Verify saved P9b evidence, frozen inputs, one-shot locks, metrics and cleanup."""
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
import p9b_report as report
import p9b_runner as runner

ROOT,OUT,AUDIT=report.ROOT,report.OUT,report.AUDIT


def load(path):
    return report.load(path)


def sha(path):
    return hashlib.sha256(report.resolve(path).read_bytes()).hexdigest()


def cleanup():
    residual=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            argv,name=p.info['cmdline'] or [],p.info['name'] or ''
            if p.pid!=os.getpid() and (name.startswith('sglang::') or
                any(a in ['sglang.launch_server','sglang_router.launch_router'] for a in argv) or
                any(a.endswith('/bench_client.py') and not a.startswith('-') for a in argv) or
                any('torch/_inductor/compile_worker/' in a for a in argv) or
                (any(a.endswith('/p9b_runner.py') for a in argv) and any(a in ['batch','experiment'] for a in argv))):
                residual.append(p.info)
        except psutil.NoSuchProcess:
            pass
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
                                 '--format=csv,noheader'],text=True).strip()
    ports=[]
    for port in [30000,31000,31001,31002,31003,8998,8999,9000,9001]:
        with socket.socket() as sock:
            sock.settimeout(.2)
            if sock.connect_ex(('127.0.0.1',port))==0:
                ports.append(port)
    return dict(gpu_compute_processes=gpu,residual_processes=residual,occupied_ports=ports)


def main():
    runner.source_gate()
    frozen=load(runner.FROZEN)
    for path,digest in frozen['protected_sha256'].items():
        assert sha(ROOT/path)==digest,path
    selected=load(ROOT/'configs/p9a_selected.json')
    assert sha(ROOT/'configs/pdmux_pp_4090_v4.yml')==selected['config_sha256']
    assert (ROOT/'configs/pdmux_pp_4090_v4.yml').read_bytes()==(ROOT/'configs/pdmux_pp_4090_v3.yml').read_bytes()
    test_log=(AUDIT/'bench-unit-tests.log').read_text()
    assert 'Ran 14 tests' in test_log and test_log.rstrip().endswith('OK')
    before=load(AUDIT/'historical-result-sha256.json')
    # Only generated aggregate artifacts may be refreshed; all original per-run data remain immutable.
    original=[p for p in before if p.endswith('_requests.jsonl') or p.endswith('.formal-started.json') or
              (p.endswith('_summary.json') and p.split('/')[-1].removesuffix('_summary.json').removesuffix('_cut300')
               in report.BASE_MATRIX)]
    for path in original:
        assert sha(ROOT/path)==before[path],path
    batch=load(AUDIT/'batch-started.json')
    assert batch['labels']==runner.LABELS
    assert (AUDIT/'batch-finished.json').exists()
    runs={}
    for label in runner.LABELS:
        path=AUDIT/label
        run=load(path/'run.json')
        marker=load(OUT/f'{label}.formal-started.json')
        cmd=load(path/'formal-command.json')
        assert cmd==marker['command']
        assert run['commit']==frozen['commit']==runner.HEAD
        assert load(path/'commands.json')==frozen['commands'][label]
        assert run['restart_count']==0
        assert cmd[cmd.index('--warmup')+1]=='2' and cmd[cmd.index('--drain-timeout')+1]=='300'
        assert '--force-close' in cmd and '--limit' not in cmd and '--time-scale' not in cmd
        workload='conversation' if label.startswith('conv_') else 'toolagent'
        assert Path(cmd[cmd.index('--trace')+1])==ROOT/'bench'/f'{workload}_600_240_x2.5.jsonl'
        smoke=load(path/'smoke-command.json')
        assert Path(smoke[smoke.index('--trace')+1])==ROOT/'bench'/f'dev_{workload}_1200_240_x2.5.jsonl'
        assert smoke[smoke.index('--limit')+1]=='20' and smoke[smoke.index('--time-scale')+1]=='0.2'
        assert smoke[smoke.index('--drain-timeout')+1]=='300' and '--force-close' in smoke
        assert run['smoke_wall_sec']<=60 and run['smoke_summary']['completed']==20
        warmups=[load(path/'long-warmup.json')]
        if 'muxwise_' in label or 'muxpp_' in label:
            warmups.append(load(path/'long-warmup-after-smoke.json'))
        for phase in warmups:
            assert len(phase)==len(run['server_info'])
            assert all(w['input_len']==31663 and w['response']['meta_info']['completion_tokens']==8 for w in phase)
        if len(run['server_info'])>1:
            assert run['router_formal_pid'] and (path/'router-formal.log').exists()
        s,records=report.sm.load_runs([str(OUT/f'{label}_summary.json')])[0]
        raw=load(OUT/f'{label}_summary.json')
        for key in ['completed','unfinished','failed','latencies_sec','slo','throughput','cached_tokens']:
            assert s[key]==raw[key],(label,key)
        expected=735 if workload=='conversation' else 1551
        assert len(records)==len({r['idx'] for r in records})==s['sent_requests']==expected
        assert sum(s[k] for k in ['completed','unfinished','failed'])==expected
        assert all(v['total']==expected for v in s['slo'].values())
        assert abs(s['cutoff_time']-max(r['sent_at'] for r in records)-300)<1e-7
        errors=[]
        idle={}
        for name in run['server_info']:
            for lineno,line in enumerate((path/f'{name}.log').read_text(errors='replace').splitlines(),1):
                if re.search(r'Traceback|RuntimeError|AssertionError|OutOfMemoryError|CUDA out of memory|Scheduler.*exception',line):
                    errors.append(dict(worker=name,line=lineno,text=line))
                if 'idle self-check PASS' in line:
                    match=re.match(r'\[[^\]]*?((?:TP\d+ )?PP\d+)\]',line)
                    rank=match[1] if match else 'unknown'
                    idle[name+'/'+rank]=dict(line=lineno,text=line)
        for key,item in idle.items():
            vals={k:int(v) for k,v in re.findall(r'(free_kv|evictable|free_reqs|free_receive_slots)=(\d+)',item['text'])}
            worker=key.split('/')[0]
            assert vals['free_kv']+vals['evictable']==run['server_info'][worker]['max_total_num_tokens'],(label,key,vals)
            assert vals['free_reqs']==128
            if not key.endswith('PP0'):
                assert vals['free_receive_slots']==2
        if 'muxpp_' in label:
            assert len(idle)==4,(label,idle)
        with (path/'gpu-rank-processes.csv').open() as f:
            gpu_processes=list(csv.DictReader(f,skipinitialspace=True))
        assert {r['gpu_uuid'] for r in gpu_processes}=={r['uuid'] for r in frozen['gpu_mapping']}
        for job in frozen['commands'][label]:
            if job['name']=='router':continue
            cmd=job['argv'];tp=int(cmd[cmd.index('--tp')+1]);pp=int(cmd[cmd.index('--pp-size')+1])
            for i,uuid in enumerate(job['env']['CUDA_VISIBLE_DEVICES'].split(',')):
                expected_name='sglang::scheduler'+(f'_TP{i%tp}' if tp>1 else '')+(f'_PP{i//tp}' if pp>1 else '')
                assert any(g['gpu_uuid']==uuid and g['process_name']==expected_name for g in gpu_processes),(label,uuid,expected_name)
        runs[label]=dict(status=run['status'],counts={k:s[k] for k in ['completed','unfinished','failed']},
                         main_slo=s['slo'][report.sm.MAIN_SLO],startup_s=run['startup_s'],
                         smoke_wall_sec=run['smoke_wall_sec'],errors=errors,idle_checks=idle,
                         gpu=report.gpu_stats(path/'gpu.csv',s),
                         formal_sha256={suffix:sha(OUT/f'{label}{suffix}') for suffix in
                             ['_requests.jsonl','_summary.json','.formal-started.json']})
    index=load(AUDIT/'report-index.json')
    for prefix,labels in [('matrix_',index['tool_labels']),('best_',index['tool_best']),
                          ('conv_',index['conv_labels']),('conv_best_',index['conv_best'])]:
        with (OUT/f'{prefix}summary.csv').open() as f:
            rows=list(csv.DictReader(f))
        assert [r['label'] for r in rows]==labels
    report_text=(ROOT/'RESULTS.md').read_text()
    for prefix in ['matrix_','conv_']:
        summary=load(OUT/f'{prefix}summary.json')
        for row in report.sm.matrix_table([(s,[]) for s in summary]).splitlines()[2:]:
            assert row in report_text,row
    clean=cleanup()
    assert not any(clean.values()),clean
    result=dict(completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=runner.HEAD,
                source_clean=True,frozen_inputs_unchanged=True,original_results_hashes_unchanged=len(original),
                new_formal_runs=len(runner.LABELS),restarts=0,runs=runs,report_index=index,**clean)
    report.save(AUDIT/'final-audit.json',result)
    (AUDIT/'final-gpu.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    report.save(AUDIT/'final-processes.json',clean)
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
