#!/usr/bin/env python3
"""Read-only final evidence verification; writes only P11b audit outputs."""
import json
from pathlib import Path
import subprocess
import sys
import time
import pynvml as nv
import psutil
import p11b_runner as run

ROOT=run.ROOT
LOG=run.OUT


def main():
    frozen=json.loads((LOG/'frozen.json').read_text())
    run.gate(frozen)
    batch=json.loads((LOG/'batch-finished.json').read_text())
    labels=batch['labels'];assert len(labels)==13 and len(set(labels))==13
    resume=json.loads((LOG/'resume-started.json').read_text())
    assert run.sha(ROOT/'scripts/p11b_resume.py')==resume['resume_source_sha256']
    assert not (LOG/'STOP-external-gpu.json').exists()
    sys.path.insert(0,str(ROOT/'bench'))
    import summarize
    rows=[]
    for label in labels:
        path=LOG/label
        m=json.loads((path/'run.json').read_text())
        expected_restarts=batch.get('startup_retries',{}).get(label,0)
        assert m['formal_launch_count']==1 and m['restart_count']==expected_restarts,label
        assert m['status'] in ['PASS','FORMAL_WITH_FAILURES'], (label,m['status'])
        assert m['gpu_mapping']==frozen['gpu_mapping']
        assert run.p9.serial(run.commands(label,frozen['gpu_mapping']))==frozen['commands'][label]
        marker=json.loads((ROOT/'results'/(label+'.formal-started.json')).read_text())
        assert marker['log_dir']==str(path)
        command=marker['command']
        for flag,value in [('--warmup','2'),('--drain-timeout','300'),('--time-scale','0.05' if label.startswith('sat_') else '1')]:
            assert command[command.index(flag)+1]==value,(label,flag)
        assert '--force-close' in command and '--limit' not in command
        assert m['smoke_summary']['completed']==20 and m['smoke_summary']['failed']==0
        raw=json.loads((ROOT/'results'/(label+'_summary.json')).read_text())
        s,records=summarize.load_runs([str(ROOT/'results'/(label+'_summary.json'))])[0]
        assert m['status']==('FORMAL_WITH_FAILURES' if s['failed'] else 'PASS'),label
        for k in ['completed','unfinished','failed','slo','latencies_sec','throughput','duration_sec']:
            assert s[k]==raw[k],(label,k)
        baseline=json.loads((LOG/(label+'-before-baseline.json')).read_text())
        assert len(baseline)==35
        for v in baseline:
            assert not run.check(v,{}, {})
            assert all(g['utilization']==0 for g in v['gpu'])
        ext=[];process_samples=[];unaccounted=[[] for _ in range(4)]
        with (LOG/(label+'-guard.jsonl')).open() as f:
            for line in f:
                v=json.loads(line)
                assert not run.check(v,{}, {},runtime=True),label
                for g in v['gpu']:unaccounted[g['index']].append(g['external_estimate_mib'])
                ext.append(v['gpu'][3]['external_estimate_mib'])
                process_samples.extend(u for g in v['gpu'] for u in g['process_utilization'] if not u['owned'])
        per_gpu=[dict(index=i,uuid=frozen['gpu_mapping'][i]['uuid'],numa=frozen['gpu_mapping'][i]['numa'],
            baseline_memory_range_mib=[min(v['gpu'][i]['memory_mib'] for v in baseline),max(v['gpu'][i]['memory_mib'] for v in baseline)],
            baseline_util_range_pct=[min(v['gpu'][i]['utilization'] for v in baseline),max(v['gpu'][i]['utilization'] for v in baseline)],
            runtime_unaccounted_memory_peak_mib=max(unaccounted[i])) for i in range(4)]
        rows.append(dict(label=label,started=marker['started'],completed=s['completed'],unfinished=s['unfinished'],failed=s['failed'],
                         formal_launch_count=1,restarts=m['restart_count'],summary_recomputed=True,status=m['status'],
                         gpu_baselines=per_gpu,
                         baseline_gpu3_memory_range_mib=[min(x['gpu'][3]['memory_mib'] for x in baseline),max(x['gpu'][3]['memory_mib'] for x in baseline)],
                         baseline_gpu3_util_range_pct=[min(x['gpu'][3]['utilization'] for x in baseline),max(x['gpu'][3]['utilization'] for x in baseline)],
                         baseline_gpu3_util_mean_pct=sum(x['gpu'][3]['utilization'] for x in baseline)/len(baseline),
                         runtime_gpu3_external_memory_peak_mib=max(ext),
                         external_process_sm_peak_pct=max((x['sm'] for x in process_samples),default=None)))
    old=json.loads((LOG/'historical-result-sha256.json').read_text());raw_checked=[]
    for rel,digest in old.items():
        p=ROOT/rel
        if p.name.endswith(('_requests.jsonl','.formal-started.json')) or (p.name.endswith('_summary.json') and
                p.name not in ['matrix_summary.json','conv_summary.json','big_summary.json','best_summary.json','conv_best_summary.json','big_best_summary.json','summary.json','final_summary.json']):
            if not p.exists():p=ROOT/'archive/superseded/p11b'/rel
            assert run.sha(p)==digest,rel
            raw_checked.append(rel)
    manifest=ROOT/'archive/superseded/p11b/manifest.json'
    for row in json.loads(manifest.read_text())['files']:
        assert run.sha(ROOT/row['destination'])==row['sha256']
    recovery=json.loads((LOG/'launch-recovery/manifest.json').read_text())
    assert recovery['formal_requests_sent']==0 and recovery['authorized_retry_count']==1
    for row in recovery['files']:assert run.sha(ROOT/row['destination'])==row['sha256']
    aborted=ROOT/'archive/aborted_p11b'
    for rel,digest in json.loads((aborted/'log-manifest.json').read_text())['files'].items():
        assert run.sha(aborted/rel)==digest
    nv.nvmlInit();gpu=run.snapshot()
    assert not run.residual()
    assert not run.check(gpu,{}, {})
    assert all(g['utilization']==0 and g['memory_mib']<=4 for g in gpu['gpu'])
    ports=[]
    for c in psutil.net_connections(kind='inet'):
        if c.status=='LISTEN' and c.laddr.port in [30000,31000,31001,31002,31003]:
            ports.append(dict(pid=c.pid,port=c.laddr.port))
    assert not ports,ports
    restored=json.loads((LOG/'preparation-lock-restored.json').read_text())
    assert restored['matches_original']
    value=dict(utc=run.now(),head=run.git('rev-parse','HEAD'),source_clean=not run.git('status','--porcelain'),
               frozen_inputs_verified=len(frozen['protected_sha256']),historical_raw_verified=len(raw_checked),
               archive_manifest_verified=True,aborted_archive_verified=True,startup_recovery_verified=True,
               preparation_lock_restored=restored,labels=rows,final_gpu=gpu,residual_processes=run.residual(),ports=ports,
               note='No inference/client processes remain; all four GPUs idle. Prior externally interrupted attempt archived separately.',
               report_script_sha256=run.sha(ROOT/'scripts/p11b_report.py'),runner_sha256=run.sha(ROOT/'scripts/p11b_runner.py'),
               resume_script_sha256=resume['resume_source_sha256'])
    run.save(LOG/'final-audit.json',value)
    (LOG/'final-gpu.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    print(json.dumps({k:value[k] for k in ['utc','head','source_clean','frozen_inputs_verified','historical_raw_verified','residual_processes','ports']},indent=2))


if __name__=='__main__':main()
