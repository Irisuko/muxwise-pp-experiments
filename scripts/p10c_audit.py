#!/usr/bin/env python3
"""Audit frozen inputs, saved metrics, one-shot execution and final cleanup."""
import csv,json,os,re,subprocess,datetime
from pathlib import Path
import pynvml
import p10c_runner as run
import p10c_report as rep
from p9b_audit import cleanup

ROOT,LOG,OUT=rep.ROOT,rep.LOG,rep.OUT
def load(p):return json.loads(p.read_text())

def main():
    frozen=load(LOG/'frozen.json');run.gate(frozen)
    testlog=(LOG/'bench-unit-tests.log').read_text()
    assert 'Ran 14 tests' in testlog and testlog.rstrip().endswith('OK')
    assert load(LOG/'llama-command-audit.json')['server_argv_equal_except_run_trace']
    assert run.sha(ROOT/'scripts/p10c_resume.py')==load(LOG/'cleanup-guard-false-alarm-review.json')['resume_script_sha256']
    original=load(LOG/'historical-result-sha256.json')
    for rel,digest in original.items():assert run.sha(ROOT/rel)==digest,rel
    runs={}
    for label in run.LABELS:
        folder=LOG/label;m=load(folder/'run.json');marker=load(OUT/(label+'.formal-started.json'))
        assert load(folder/'commands.json')==frozen['commands'][label]
        assert m['commit']==run.HEAD and m['restart_count']==0 and m['formal_launch_count']==1
        assert not (folder/'source.patch').read_text()
        cmd=load(folder/'formal-command.json');assert cmd==marker['command']
        assert cmd[cmd.index('--trace')+1]==str(run.FORMAL)
        assert '--limit' not in cmd and '--time-scale' not in cmd and '--force-close' in cmd
        assert cmd[cmd.index('--drain-timeout')+1]=='300' and cmd[cmd.index('--warmup')+1]=='2'
        smoke=load(folder/'smoke-command.json')
        assert smoke[smoke.index('--limit')+1]=='20' and smoke[smoke.index('--time-scale')+1]=='0.2'
        assert smoke[smoke.index('--trace')+1]==str(run.a.DEV) and '--force-close' in smoke
        assert m['smoke_summary']['completed']==20 and m['smoke_returncode']==0
        warmups=list(folder.glob('long-warmup*.json'))
        assert len(warmups)==(2 if 'mux' in label else 1)
        for p in warmups:
            w=load(p);assert w['input_len']==31663 and w['response']['meta_info']['completion_tokens']==8
        s,records=rep.r.sm.load_runs([str(OUT/(label+'_summary.json'))])[0]
        raw=load(OUT/(label+'_summary.json'))
        for key in ['completed','unfinished','failed','latencies_sec','slo','throughput','cached_tokens']:assert raw[key]==s[key],(label,key)
        assert len(records)==len({r['idx'] for r in records})==s['sent_requests']==555
        assert sum(s[k] for k in ['completed','unfinished','failed'])==555
        assert all(v['total']==555 for v in s['slo'].values())
        assert abs(s['cutoff_time']-max(r['sent_at'] for r in records)-300)<1e-6
        errors=[];idle={}
        for i,line in enumerate((folder/'server.log').read_text(errors='replace').splitlines(),1):
            if re.search(r'Traceback|RuntimeError|AssertionError|OutOfMemoryError|CUDA out of memory|Scheduler.*exception|#new-token: -',line):errors.append(dict(line=i,text=line))
            if 'idle self-check PASS' in line:
                match=re.match(r'\[[^\]]*?((?:TP\d+ )?PP\d+)\]',line)
                idle[match[1] if match else 'unknown']=dict(line=i,text=line)
        kv=m['server_info']['server']['max_total_num_tokens']
        for rank,item in idle.items():
            vals={k:int(v) for k,v in re.findall(r'(free_kv|evictable|free_reqs|free_receive_slots)=(\d+)',item['text'])}
            assert vals['free_kv']+vals['evictable']==kv and vals['free_reqs']==128
            if not rank.endswith('PP0'):assert vals['free_receive_slots']==2
        if 'muxpp_' in label:assert len(idle)==4,(label,idle)
        with (folder/'gpu-rank-processes.csv').open() as f:gpu=list(csv.DictReader(f,skipinitialspace=True))
        assert {g['gpu_uuid'] for g in gpu}=={g['uuid'] for g in frozen['gpu_mapping']}
        job=frozen['commands'][label][0];cmd=job['argv'];tp=int(cmd[cmd.index('--tp')+1]);pp=int(cmd[cmd.index('--pp-size')+1])
        for index,g in enumerate(frozen['gpu_mapping']):
            expected='sglang::scheduler'+(f'_TP{index%tp}' if tp>1 else '')+(f'_PP{index//tp}' if pp>1 else '')
            assert any(v['gpu_uuid']==g['uuid'] and v['process_name']==expected for v in gpu),(label,expected)
        runs[label]=dict(status=m['status'],completed=s['completed'],unfinished=s['unfinished'],failed=s['failed'],main_slo=s['slo'][rep.MAIN],
            kv_capacity=kv,smoke_duration_sec=m['smoke_summary']['duration_sec'],startup_sec=m['startup_s'],errors=errors,idle_checks=idle,
            started_utc=m['started_utc'],finished_utc=m['finished_utc'],gpu=rep.r.gpu_stats(folder/'gpu.csv',s),
            artifact_sha256={suffix:run.sha(OUT/(label+suffix)) for suffix in ['_requests.jsonl','_summary.json','.formal-started.json']})
    ab=load(LOG/'llama-ab.json')
    for row in ab['rows']:
        path=LOG/('llama-'+row['which'])
        assert not (path/'source.patch').read_text()
        assert row['meta']['commit'].startswith(row['head'])
        assert load(LOG/('llama-'+row['which']+'-launch.json'))['gpu_mapping']==frozen['gpu_mapping']
    for label in run.LABELS:
        for phase in ['before','after']:
            s=load(LOG/(label+'-'+phase+'-idle.json'))
            assert not s['residual_processes'] and all(g['memory_mib']<=4 and g['utilization']==0 and not g['processes'] for g in s['gpu'])
    assert not list(LOG.glob('STOP*.json'))
    assert load(LOG/'batch-finished.json')['labels']==run.LABELS
    pynvml.nvmlInit();gpu=run.idle('audit-final');clean=cleanup()
    assert not any(clean.values()),clean
    (LOG/'final-gpu.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    rep.r.save(LOG/'final-processes.json',clean)
    result=dict(completed_utc=run.now(),source_commit=run.HEAD,source_clean=True,protected_inputs_unchanged=True,
        historical_files_unchanged=len(original),formal_runs=6,restarts=0,external_gpu_interference=False,existing_cpu_tests_passed=14,
        llama_ab=ab['meets_regression_threshold'],runs=runs,final_gpu=gpu,cleanup=clean,
        note='Initial NVML v1 reserved-memory false alarm audited separately; v2 agrees with nvidia-smi.')
    rep.r.save(LOG/'final-audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['runs','final_gpu']},indent=2))

if __name__=='__main__':main()
