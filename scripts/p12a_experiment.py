#!/usr/bin/env python3
"""Prepared v5.1 formal entrypoint. P12a never invokes experiment mode."""
import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import time
import pynvml
import p12a_runner as r
import p12a_serve as serve

ROOT=r.ROOT
OUT=ROOT/'logs/p12b-v51'
LABELS={'muxpp_v51_pp2tp2':('llama_pp2tp2','toolagent_600_240_x2.5.jsonl',1551),
        'conv_muxpp_v51_pp2tp2':('llama_pp2tp2','conversation_600_240_x2.5.jsonl',735),
        'big_muxpp_v51_pp2tp2':('big_pp2tp2','big_toolagent_600_80_x7.5.jsonl',555)}

@contextmanager
def preparation_lock(c):
    """Hold the obsolete generation-only lock only for an explicit formal run."""
    lock=ROOT/'configs/p10a_formal.lock.json'
    held=c.path/'p10a_formal.lock.held.json';digest=None
    if lock.exists():
        digest=r.monitor.sha(lock);lock.rename(held)
    try:yield
    finally:
        if digest is not None:
            assert not lock.exists() and r.monitor.sha(held)==digest
            held.rename(lock)
            r.r.b.save(c.path/'preparation-lock-restored.json',dict(sha256=r.monitor.sha(lock),restored=True))

def client(c,label,trace,count,smoke=False):
    assert trace.name.startswith('dev_')==smoke
    name='smoke' if smoke else label
    output=c.path/'smoke' if smoke else ROOT/'results'
    cmd=[r.r.b.PYTHON,str(ROOT/'bench/bench_client.py'),'--trace',str(trace),
         '--base-url',r.r.b.BASE,'--label',name,'--out-dir',str(output),
         '--time-scale','0.2' if smoke else '1','--warmup','2','--force-close',
         '--drain-timeout','300','--progress-interval','30']
    if smoke:cmd+=['--limit','20']
    else:
        with (output/(label+'.formal-started.json')).open('x') as f:
            json.dump(dict(utc=r.monitor.now(),command=cmd,log_dir=str(c.path),commit=c.meta['commit']),f,indent=2)
    r.r.b.save(c.path/(name+'-command.json'),cmd)
    with (c.path/(name+'-client.log')).open('x') as log:
        p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        try:rc=p.wait(timeout=1100)
        except BaseException:
            p.send_signal(signal.SIGINT)
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            raise
    summary=json.loads((output/(name+'_summary.json')).read_text())
    c.meta[name]=summary;c.meta[name+'_returncode']=rc
    r.r.b.save(c.path/'run.json',c.meta)
    assert summary['planned_requests']==count
    if smoke:assert rc==0 and (summary['completed'],summary['failed'],summary['unfinished'])==(20,0,0)
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['experiment','commands']);p.add_argument('label',choices=LABELS)
    p.add_argument('--tag',default='formal-once');x=p.parse_args()
    selected=json.loads(serve.CONFIG.read_text());cfg,filename,count=LABELS[x.label]
    jobs=serve.commands(cfg)
    if x.mode=='commands':
        import shlex
        for name,cmd,env,port in jobs:print(shlex.join(['env',*[k+'='+v for k,v in env.items()],*cmd]))
        return
    assert selected['validated'], 'v5.1 development validation has not passed'
    assert r.git(r.r.MAIN,'rev-parse','HEAD')==selected['commit'] and not r.git(r.r.MAIN,'status','--porcelain')
    for rel,digest in selected['validation_sha256'].items():assert r.monitor.sha(ROOT/rel)==digest,rel
    # This function is reached only by an explicit future experiment command.
    assert not list((ROOT/'results').glob(x.label+'*')), 'Formal label already exists'
    assert not list((ROOT/'archive').rglob(x.label+'.formal-started.json')), 'Archived formal label already exists'
    OUT.mkdir(exist_ok=True);r.monitor.OUT=OUT;pynvml.nvmlInit()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    with (ROOT/'logs/p7a/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not list(OUT.glob('STOP-*.json')) and not list(r.OUT.glob('STOP-*.json'))
        r.monitor.idle(x.label+'-before',seconds=35)
        mapping=r.monitor.mapping()
        for name,cmd,env,port in jobs:
            if name!='router':
                env['CUDA_VISIBLE_DEVICES']=','.join(mapping[int(i)]['uuid'] for i in env['CUDA_VISIBLE_DEVICES'].split(','))
        r.r.b.commands=lambda _:jobs;r.r.b.clean_gpu=r.clean
        c=r.r.p7.Cluster(x.label,OUT/x.label)
        c.meta.update(commit=selected['commit'],tag=x.tag,gpu_mapping=mapping,formal_launch_count=0,
                      selected_config=str(serve.CONFIG),selected_sha256=r.monitor.sha(serve.CONFIG))
        guard=r.monitor.Guard(x.label);guard.thread.start()
        try:
            c.start();r.r.warmup(c,cfg,'before-smoke')
            dev=ROOT/'bench'/('dev_big_toolagent_1200_80_x7.5.jsonl' if cfg.startswith('big_') else
                'dev_conversation_1200_240_x2.5.jsonl' if x.label.startswith('conv_') else 'dev_toolagent_1200_240_x2.5.jsonl')
            client(c,x.label,dev,20,smoke=True)
            r.r.warmup(c,cfg,'after-smoke');r.r.b.flush(c);c.reset_router()
            c.meta['formal_launch_count']=1
            with preparation_lock(c):summary=client(c,x.label,ROOT/'bench'/filename,count)
            complete=(summary['completed'],summary['failed'],summary['unfinished'])==(count,0,0)
            c.meta['status']='PASS' if complete and c.meta[x.label+'_returncode']==0 else 'FORMAL_WITH_FAILURES'
        except BaseException as exc:c.meta.update(status='FAIL',error=repr(exc));raise
        finally:
            try:c.close()
            finally:guard.close()
        assert not guard.error
        time.sleep(3);r.monitor.idle(x.label+'-after',seconds=35)

if __name__=='__main__':main()
