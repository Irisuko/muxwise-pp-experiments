#!/usr/bin/env python3
"""P11b one-shot orchestration; never changes inference code or selected flags."""
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import psutil
import pynvml as nv
import p3_runner as b
import p7a_runner as p7
import p9b_runner as p9
import p10a_runner as big
import p10b_serve as selected_big
import p11a_serve as v5

ROOT = b.ROOT
OUT = ROOT / 'logs/p11b'
HEAD = '1dcbb285e958936113e373d4b0ff1b30f63aaed2'
A = ['muxpp_v5_pp4', 'muxpp_v5_pp2tp2', 'muxpp_v5_pp2dp2',
     'conv_muxpp_v5_pp4', 'conv_muxpp_v5_pp2tp2',
     'big_muxpp_v5_pp2tp2', 'big_muxpp_v5_pp4']
TRACES = {'tool': 'toolagent_600_240_x2.5.jsonl',
          'conv': 'conversation_600_240_x2.5.jsonl',
          'big': 'big_toolagent_600_80_x7.5.jsonl'}
DEV = {'tool': 'dev_toolagent_1200_240_x2.5.jsonl',
       'conv': 'dev_conversation_1200_240_x2.5.jsonl',
       'big': 'dev_big_toolagent_1200_80_x7.5.jsonl'}
COUNTS = {'tool': 1551, 'conv': 735, 'big': 555}
MAIN = 'ttft_le_5s_and_tpot_le_100ms'
POLICY = dict(authority='User update 2026-10-28', allowed_external_gpus=[],
              sampling_seconds=1, idle_memory_limit_mib=4,
              runtime_unaccounted_memory_limit_mib=128,
              note='All four GPUs must be idle before each label. Stop immediately on foreign PID, '
                   'foreign process utilization, or unaccounted memory above the NVML driver/accounting allowance. '
                   'All four GPUs now NUMA0; prior interrupted attempt archived, not part of formal comparison.')


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, value):
    b.save(path, value)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(p7.MAIN), *args], text=True).strip()


def family(label):
    return 'big' if 'big_' in label else 'conv' if label.startswith('conv_') else 'tool'


def mapping():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,name','--format=csv,noheader'],text=True)
    rows=[]
    for line in raw.strip().splitlines():
        index,uuid,bus,name=[x.strip() for x in line.split(',')]
        pci='0000:'+bus.split(':',1)[1].lower()
        numa=int(Path('/sys/bus/pci/devices',pci,'numa_node').read_text())
        rows.append(dict(index=int(index),uuid=uuid,pci_bus_id=bus,numa=numa,name=name))
    assert len(rows)==4 and [r['numa'] for r in rows]==[0,0,0,0],rows
    return rows


def residual():
    rows = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        cmd = p.info['cmdline'] or []
        if any(x in ('sglang.launch_server', 'sglang_router.launch_router') or
               Path(x).name == 'bench_client.py' for x in cmd) or (p.info['name'] or '').startswith(('sglang::', 'SGLang')):
            rows.append(p.info)
    return rows


def snapshot(known=None, since=None):
    known = known or {}
    rows = []
    for i in range(nv.nvmlDeviceGetCount()):
        h = nv.nvmlDeviceGetHandleByIndex(i)
        m = nv.nvmlDeviceGetMemoryInfo(h, version=nv.nvmlMemory_v2)
        apps = []
        for a in nv.nvmlDeviceGetComputeRunningProcesses(h):
            memory = a.usedGpuMemory / 2**20 if a.usedGpuMemory < 2**60 else 0
            try:
                owned = a.pid in known and psutil.Process(a.pid).create_time() == known[a.pid]
            except psutil.NoSuchProcess:
                owned = a.pid in known
            apps.append(dict(pid=a.pid, memory_mib=memory, owned=owned, visible=psutil.pid_exists(a.pid)))
        samples, error = [], None
        try:
            for u in nv.nvmlDeviceGetProcessUtilization(h, (since or {}).get(i, int((time.time()-2)*1e6))):
                samples.append(dict(pid=u.pid, sm=u.smUtil, memory=u.memUtil,
                                    timestamp_us=u.timeStamp, owned=u.pid in known))
        except nv.NVMLError as exc:
            error = str(exc)
        owned_memory = sum(a['memory_mib'] for a in apps if a['owned'])
        rows.append(dict(index=i, uuid=nv.nvmlDeviceGetUUID(h), memory_mib=m.used/2**20,
                         total_mib=m.total/2**20, free_mib=m.free/2**20,
                         reserved_mib=m.reserved/2**20, utilization=nv.nvmlDeviceGetUtilizationRates(h).gpu,
                         processes=apps, external_estimate_mib=max(0, m.used/2**20-owned_memory),
                         process_utilization=samples, process_utilization_error=error))
    return dict(utc=now(), unix=time.time(), gpu=rows)


def check(s, streaks, last, runtime=False):
    if len(s['gpu']) != 4:
        return 'Expected four visible GPUs'
    for g in s['gpu']:
        i = g['index']
        # Outside runtime the whole footprint is external. During runtime compute
        # process memory accounts for our workload; driver baseline is <1MiB.
        # NVML device/process counters are sampled independently. The archived
        # owned-only startup has a 75MiB transient discrepancy; 128MiB is the
        # accounting allowance, not permission for an identified foreign process.
        limit = 128 if runtime else 4
        if g['external_estimate_mib'] > limit:
            return f'GPU{i} outside footprint {g["external_estimate_mib"]:.2f}MiB > {limit}'
        if any(not a['owned'] for a in g['processes']):
            return f'Unexpected external compute process on GPU{i}'
        for u in g['process_utilization']:
            if u['owned']:
                continue
            key = (i, u['pid']); stamp = u['timestamp_us']/1e6
            if stamp <= last.get(key, 0):
                continue
            if u['sm'] > 0 or u['memory'] > 0:
                return f'Unexpected external process utilization on GPU{i}: {u}'
            last[key] = stamp
    return None


def stop_external(reason, s, tag):
    save(OUT / 'STOP-external-gpu.json', dict(utc=now(), tag=tag, reason=reason, snapshot=s, policy=POLICY))
    raise RuntimeError(reason)


def idle(tag, seconds=35):
    if residual():
        stop_external('Residual inference/client process before label', {'processes':residual()}, tag)
    rows, streaks, last, since = [], {}, {}, {}
    for _ in range(seconds):
        s = snapshot(since=since); rows.append(s)
        for g in s['gpu']:
            if g['process_utilization']:
                since[g['index']] = max(u['timestamp_us'] for u in g['process_utilization'])
        reason = check(s, streaks, last)
        save(OUT / (tag+'-baseline.json'), rows)
        if reason:
            stop_external(reason, s, tag)
        if any(g['utilization'] > 0 for g in s['gpu']):
            stop_external('Unexpected GPU activity during preflight', s, tag)
        time.sleep(1)
    return rows


class Guard:
    def __init__(self, label):
        self.label=label; self.done=threading.Event(); self.error=None; self.known={}
        self.thread=threading.Thread(target=self.run, daemon=True)

    def track(self):
        root=psutil.Process(os.getpid())
        for p in [root, *root.children(recursive=True)]:
            try:self.known[p.pid]=p.create_time()
            except psutil.NoSuchProcess:pass

    def run(self):
        streaks, last, since = {}, {}, {}
        with (OUT/(self.label+'-guard.jsonl')).open('x', buffering=1) as f:
            while not self.done.wait(1):
                try:
                    self.track(); s=snapshot(self.known, since)
                    for g in s['gpu']:
                        if g['process_utilization']:
                            since[g['index']]=max(u['timestamp_us'] for u in g['process_utilization'])
                    f.write(json.dumps(s)+'\n')
                    reason=check(s,streaks,last,runtime=True)
                    if reason:
                        self.error=dict(reason=reason,snapshot=s,known=self.known)
                        save(OUT/'STOP-external-gpu.json',dict(tag=self.label,policy=POLICY,**self.error))
                        os.kill(os.getpid(),signal.SIGINT);return
                except Exception as exc:
                    self.error=dict(reason='Guard failed: '+repr(exc))
                    save(OUT/'STOP-external-gpu.json',self.error)
                    os.kill(os.getpid(),signal.SIGINT);return
        save(OUT/(self.label+'-owned-processes.json'),self.known)

    def close(self):
        self.done.set();self.thread.join(timeout=5)


def commands(label, mapping):
    plain=label.removeprefix('sat_')
    if 'muxpp_v5_' in plain:
        cfg=('big_' if plain.startswith('big_') else 'llama_')+plain.split('muxpp_v5_')[1]
        jobs=v5.commands(cfg,ROOT/'configs/p11a_selected.json')
        for name,cmd,env,port in jobs:
            if name!='router':
                env['CUDA_VISIBLE_DEVICES']=','.join(mapping[int(i)]['uuid'] for i in env['CUDA_VISIBLE_DEVICES'].split(','))
    elif plain.startswith('big_'):
        jobs=selected_big.commands(plain)
        for name,cmd,env,port in jobs:
            env['CUDA_VISIBLE_DEVICES']=','.join(mapping[int(i)]['uuid'] for i in env['CUDA_VISIBLE_DEVICES'].split(','))
    else:
        jobs=p9.commands(plain,mapping)
    return jobs


def freeze():
    assert git('rev-parse','HEAD')==HEAD and not git('status','--porcelain')
    assert not [k for k in os.environ if k.startswith(('P3_','SGLANG_PDMUX_PP_','P9A_SOURCE','P10A_STACKS'))]
    gpu_map=mapping()
    candidates=A+['sat_muxwise_tp2dp2','sat_chunked_pp2dp2','sat_big_muxwise_tp4','sat_big_chunked_pp2tp2']+['sat_'+l for l in A if not l.startswith('conv_')]
    jobs={l:[dict(name=n,argv=c,env=e,port=p) for n,c,e,p in commands(l,gpu_map)] for l in candidates}
    paths=[ROOT/'env.sh',ROOT/'P11A_NOTES.md',ROOT/'P7A_NOTES.md',ROOT/'P10B_NOTES.md',
           *sorted((ROOT/'configs').glob('*.json')),*sorted((ROOT/'configs').glob('*.yml')),
           *sorted((ROOT/'scripts').glob('*.py')),*sorted((ROOT/'scripts').glob('*.sh')),
           *sorted((ROOT/'bench').glob('*.py')),*sorted((ROOT/'bench').glob('*.jsonl'))]
    paths=[p for p in paths if p.name not in ['p11b_report.py','p11b_audit.py']]
    frozen=dict(utc=now(),head=HEAD,gpu_mapping=gpu_map,external_policy=POLICY,commands=jobs,
                prior_external_interruption='archive/aborted_p11b/README.md',
                reporting_initial_sha256={str(p.relative_to(ROOT)):sha(p) for p in
                    [ROOT/'scripts/p11b_report.py',ROOT/'scripts/p11b_audit.py']},
                protected_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths})
    with (OUT/'frozen.json').open('x') as f:json.dump(frozen,f,indent=2)
    (OUT/'RESULTS-before-P11b.md').write_bytes((ROOT/'RESULTS.md').read_bytes())
    (OUT/'topology.txt').write_text(subprocess.check_output(['nvidia-smi','topo','-m'],text=True))
    save(OUT/'historical-result-sha256.json',{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'results').glob('*') if p.is_file()})
    return frozen


def gate(frozen):
    assert git('rev-parse','HEAD')==HEAD and not git('status','--porcelain')
    assert mapping()==frozen['gpu_mapping']
    for rel,digest in frozen['protected_sha256'].items():
        assert sha(ROOT/rel)==digest, rel
    assert not (OUT/'STOP-external-gpu.json').exists()


def warmup(c,tag):
    b.flush(c)
    if family(c.label)=='big':
        big.warmup(c)
    else:
        import random
        rng=random.Random(730031)
        ids=[128000]+[rng.randrange(128000) for _ in range(31662)]
        values=[]
        for name,_,_,port in b.commands(c.label):
            if name!='router':
                result=b.generate(f'http://127.0.0.1:{port}',ids)
                result['worker']=name;values.append(result)
                save(c.path/'long-warmup.json',values)
        b.flush(c)
    (c.path/'long-warmup.json').rename(c.path/('long-warmup-'+tag+'.json'))


def replay(c,formal):
    fam=family(c.label)
    label=c.label if formal else c.label+'_smoke'
    out=ROOT/'results' if formal else c.path/'smoke'
    trace=ROOT/'bench'/(TRACES if formal else DEV)[fam]
    scale=.05 if c.label.startswith('sat_') else 1
    cmd=[b.PYTHON,str(ROOT/'bench/bench_client.py'),'--trace',str(trace),'--base-url',b.BASE,
         '--label',label,'--out-dir',str(out),'--warmup','2','--force-close','--drain-timeout','300',
         '--time-scale',str(scale if formal else .2),'--progress-interval','30']
    if not formal:cmd+=['--limit','20']
    stage='formal' if formal else 'smoke'
    if formal:
        with (out/(label+'.formal-started.json')).open('x') as f:
            json.dump(dict(started=now(),command=cmd,log_dir=str(c.path),commit=HEAD),f,indent=2)
    save(c.path/(stage+'-command.json'),cmd)
    print('REPLAY',label,flush=True)
    with (c.path/(stage+'-client.log')).open('x') as f:
        proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
        try:rc=proc.wait(timeout=1100 if formal else 400)
        except BaseException:
            proc.send_signal(signal.SIGINT)
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            raise
    s=json.loads((out/(label+'_summary.json')).read_text())
    c.meta[stage+'_summary']=s;c.meta[stage+'_returncode']=rc
    save(c.path/'run.json',c.meta)
    assert s['planned_requests']==(COUNTS[fam] if formal else 20)
    if not formal:assert rc==0 and (s['completed'],s['failed'],s['unfinished'])==(20,0,0)
    print('RESULT',label,json.dumps({k:s[k] for k in ('completed','unfinished','failed','duration_sec','throughput','slo')}),flush=True)
    return s


def experiment(label,frozen):
    gate(frozen)
    assert not list((ROOT/'results').glob(label+'*')), 'Label already has formal artifact'
    assert not (OUT/(label+'.launch-started.json')).exists(), 'Label already launched'
    before=idle(label+'-before')
    jobs=commands(label,frozen['gpu_mapping'])
    assert p9.serial(jobs)==frozen['commands'][label]
    save(OUT/(label+'.launch-started.json'),dict(utc=now(),label=label))
    b.commands=lambda _:jobs
    # Lifecycle hook verifies the current exclusive GPU assignment.
    def clean():
        s=snapshot()
        if residual():raise RuntimeError('Residual inference processes at lifecycle boundary')
        reason=check(s,{}, {})
        if reason:stop_external(reason,s,label)
    b.clean_gpu=clean
    c=p7.Cluster(label,OUT/label)
    c.meta.update(commit=HEAD,formal_launch_count=0,restart_count=0,
                  gpu_mapping=frozen['gpu_mapping'],external_policy=POLICY,
                  preflight_baseline=str(OUT/(label+'-before-baseline.json')),
                  gpu3_baseline_memory_mib=[s['gpu'][3]['memory_mib'] for s in before],
                  gpu3_baseline_utilization=[s['gpu'][3]['utilization'] for s in before],
                  trace_sha256=sha(ROOT/'bench'/TRACES[family(label)]))
    guard=Guard(label);guard.thread.start()
    print('START',label,now(),flush=True)
    success=False
    try:
        c.start();warmup(c,'before-smoke');replay(c,False)
        if 'mux' in label:warmup(c,'after-smoke')
        b.flush(c);c.reset_router()
        guard.track()
        save(c.path/'rank-processes.json',snapshot(guard.known))
        c.meta['formal_launch_count']=1
        s=replay(c,True)
        c.meta['status']='PASS' if s['failed']==0 else 'FORMAL_WITH_FAILURES'
        success=True
    except BaseException as exc:
        c.meta.update(status='FAIL',error=repr(exc))
        print('FAILED',label,repr(exc),flush=True)
        if isinstance(exc,KeyboardInterrupt) or guard.error:raise
    finally:
        try:c.close()
        finally:guard.close()
    if guard.error:raise RuntimeError(str(guard.error))
    time.sleep(3)
    idle(label+'-after',seconds=5)
    with (OUT/'progress.jsonl').open('a') as f:
        f.write(json.dumps(dict(label=label,ended=now(),status=c.meta['status'],formal_data=success))+'\n')
    print('END',label,c.meta['status'],now(),flush=True)


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','run'])
    x=parser.parse_args();OUT.mkdir(exist_ok=True);nv.nvmlInit()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    with (ROOT/'logs/p7a/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if x.mode=='freeze':freeze();return
        frozen=json.loads((OUT/'frozen.json').read_text());gate(frozen)
        with (OUT/'batch-started.json').open('x') as f:json.dump(dict(utc=now(),labels=A),f)
        for label in A:experiment(label,frozen)
        chosen={}
        for fam in ['tool','big']:
            summaries=[json.loads((ROOT/'results'/(l+'_summary.json')).read_text()) for l in A
                       if family(l)==fam and (ROOT/'results'/(l+'_summary.json')).exists()]
            assert summaries, 'No formal v5 data to select saturation topology'
            winner=min(summaries,key=lambda s:(-s['slo'][MAIN]['rate'],s['latencies_sec']['ttft']['p50']
                if isinstance(s['latencies_sec']['ttft']['p50'],(int,float)) else float('inf')))
            chosen[fam]=winner['label']
        save(OUT/'saturation-selection.json',dict(utc=now(),chosen=chosen,rule='highest A main SLO, tie lowest TTFT p50'))
        labels=['sat_muxwise_tp2dp2','sat_chunked_pp2dp2','sat_'+chosen['tool'],
                'sat_big_muxwise_tp4','sat_big_chunked_pp2tp2','sat_'+chosen['big']]
        for label in labels:experiment(label,frozen)
        gate(frozen);idle('final',seconds=35)
        save(OUT/'batch-finished.json',dict(utc=now(),labels=A+labels,restarts=0))


if __name__=='__main__':main()
