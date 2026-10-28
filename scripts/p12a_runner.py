#!/usr/bin/env python3
"""P12a development-only runner with strict four-GPU ownership monitoring."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import pynvml
import p11a_runner as r
import p12a_analyze as analysis
import p11b_runner as monitor
import p12a_gpu
monitor.snapshot = p12a_gpu.snapshot

ROOT = r.ROOT
OUT = ROOT/'logs/p12a'
SELECTED = ROOT/'configs/p11a_selected.json'

def git(source, *args):
    return subprocess.check_output(['git','-C',str(source),*args],text=True).strip()

def clean():
    s=monitor.snapshot()
    reason=monitor.check(s,{}, {})
    if reason or monitor.residual():
        raise RuntimeError('GPU is not clean: '+str(reason or monitor.residual()))

def replay(c,label,rate):
    trace=r.TRACES[label.split('_')[0]]
    assert trace.resolve() in [p.resolve() for p in r.TRACES.values()]
    count=sum(1 for line in trace.open() if line.strip())
    limit=(150 if label.startswith('big_') else 300) if rate=='partial' else count
    cmd=[r.b.PYTHON,str(ROOT/'bench/bench_client.py'),'--trace',str(trace),
         '--base-url',r.b.BASE,'--label',rate,'--out-dir',str(c.path/rate),
         '--time-scale','1','--warmup','2','--force-close','--drain-timeout','300',
         '--progress-interval','30']
    if rate=='partial':cmd+=['--limit',str(limit)]
    r.b.save(c.path/(rate+'-command.json'),cmd)
    print('REPLAY',c.path.name,rate,limit,monitor.now(),flush=True)
    with (c.path/(rate+'-client.log')).open('x') as log:
        p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        try:rc=p.wait(timeout=1200)
        except BaseException:
            p.send_signal(signal.SIGINT)
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            raise
    s=json.loads((c.path/rate/(rate+'_summary.json')).read_text())
    c.meta[rate]=s;r.b.save(c.path/'run.json',c.meta)
    print('RESULT',c.path.name,rate,json.dumps({k:s[k] for k in ('completed','failed','unfinished','duration_sec','latencies_sec','throughput','slo')}),flush=True)
    assert rc==0 and (s['planned_requests'],s['completed'],s['failed'],s['unfinished'])==(limit,limit,0,0)
    assert all(p.poll() is None for p in c.procs)
    analysis.analyze(c.path,rate)

def experiment(tag,label,source,flags,rates,environment=None):
    assert not list(OUT.glob('STOP-*.json'))
    assert not git(source,'status','--porcelain'), 'Freeze source before measurement'
    monitor.idle(tag+'-before',seconds=10)
    path=OUT/tag
    jobs=r.commands(label,source,path,flags)
    mapping=monitor.mapping()
    for name,cmd,env,port in jobs:
        if name=='router':continue
        env.update(NCCL_DEBUG='INFO',NCCL_DEBUG_FILE=str(path/'nccl-%h-%p.log'))
        env.update(environment or {})
        env['CUDA_VISIBLE_DEVICES']=','.join(mapping[int(i)]['uuid'] for i in env['CUDA_VISIBLE_DEVICES'].split(','))
    r.b.commands=lambda _:jobs
    r.b.clean_gpu=clean
    c=r.p7.Cluster(label,path)
    c.meta.update(tag=tag,source=str(source),commit=git(source,'rev-parse','HEAD'),flags=flags,
                  gpu_mapping=mapping,trace_sha256=monitor.sha(r.TRACES[label.split('_')[0]]),
                  runner_sha256=monitor.sha(Path(__file__)),rates=rates,environment=environment or {})
    guard=monitor.Guard(tag);guard.thread.start()
    print('START',tag,monitor.now(),flush=True)
    try:
        c.start()
        (path/'source.patch').write_text(git(source,'diff','HEAD','--binary'))
        (path/'runner.py').write_text(Path(__file__).read_text())
        for rate in rates:
            guard.track()
            reason=monitor.check(monitor.snapshot(guard.known),{}, {},runtime=True)
            if reason:monitor.stop_external(reason,monitor.snapshot(guard.known),tag)
            r.warmup(c,label,rate)
            replay(c,label,rate)
            r.b.flush(c)
            time.sleep(4)
        c.meta['status']='PASS'
    except BaseException as exc:
        c.meta.update(status='FAIL',error=repr(exc));raise
    finally:
        try:c.close()
        finally:guard.close()
    assert not guard.error
    time.sleep(3)
    monitor.idle(tag+'-after',seconds=5)
    print('END',tag,monitor.now(),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue',type=Path)
    p.add_argument('--tag')
    p.add_argument('--label',choices=r.LABELS[:4]+['llama_pp2dp2'])
    p.add_argument('--source',type=Path,default=r.MAIN)
    p.add_argument('--version',choices=['v4','v5'],default='v5')
    p.add_argument('--config',type=Path,default=SELECTED)
    p.add_argument('--rates',nargs='+',choices=['partial','full'],default=['partial','full'])
    p.add_argument('--env',action='append',default=[])
    p.add_argument('--extra',nargs=argparse.REMAINDER,default=[])
    x=p.parse_args()
    OUT.mkdir(exist_ok=True);monitor.OUT=OUT;pynvml.nvmlInit()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    with (ROOT/'logs/p7a/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if x.queue:
            for item in json.loads(x.queue.read_text()):
                flags=json.loads(x.config.read_text())['flags'][item['label']] if item['version']=='v5' else []
                environment=json.loads(x.config.read_text()).get('environment',{}).get(item['label'],{}) | item.get('environment',{})
                experiment(item['tag'],item['label'],Path(item['source']),flags+item.get('extra',[]),item['rates'],environment)
        else:
            assert x.tag and x.label
            flags=json.loads(x.config.read_text())['flags'][x.label] if x.version=='v5' else []
            environment=json.loads(x.config.read_text()).get('environment',{}).get(x.label,{}) | dict(v.split('=',1) for v in x.env)
            experiment(x.tag,x.label,x.source,flags+x.extra,x.rates,environment)

if __name__=='__main__':main()
