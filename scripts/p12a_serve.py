#!/usr/bin/env python3
"""v5.1 command rendering and model lifecycle; never replays a trace."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import shlex
import signal
import time
import pynvml
import p12a_runner as r

CONFIG=r.ROOT/'configs/p12a_selected.json'
LABELS=['llama_pp2tp2','big_pp2tp2','llama_pp4','big_pp4','llama_pp2dp2']

def commands(label,config=CONFIG):
    selected=json.loads(config.read_text())
    jobs=r.r.commands(label,r.r.MAIN,r.OUT/'serve',selected['flags'][label])
    for name,cmd,env,port in jobs:
        if name=='router':continue
        at=cmd.index('--pdmux-pp-request-trace');del cmd[at:at+2]
        env.pop('SGLANG_PDMUX_PP_ROUND_TRACE',None)
        env.update(selected.get('environment',{}).get(label,{}))
    return jobs

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['commands','launch']);p.add_argument('label',choices=LABELS)
    p.add_argument('--config',type=Path,default=CONFIG);p.add_argument('--tag')
    x=p.parse_args();selected=json.loads(x.config.read_text());jobs=commands(x.label,x.config)
    if x.mode=='commands':
        print('source /root/workspace/muxpp_work/env.sh')
        for name,cmd,env,port in jobs:print(shlex.join(['env',*[k+'='+v for k,v in env.items()],*cmd]))
        return
    assert r.git(r.r.MAIN,'rev-parse','HEAD')==selected['commit'] and not r.git(r.r.MAIN,'status','--porcelain')
    tag=x.tag or 'serve-v51-'+x.label+'-'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    pynvml.nvmlInit();r.monitor.OUT=r.OUT
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    with (r.ROOT/'logs/p7a/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not list(r.OUT.glob('STOP-*.json'))
        r.monitor.idle(tag+'-before',seconds=35)
        r.r.b.commands=lambda _:jobs;r.r.b.clean_gpu=r.clean
        c=r.r.p7.Cluster(x.label,r.OUT/tag);c.meta.update(commit=selected['commit'],config=str(x.config))
        guard=r.monitor.Guard(tag);guard.thread.start()
        try:
            c.start()
            while True:
                time.sleep(1);assert all(p.poll() is None for p in c.procs)
        except KeyboardInterrupt:c.meta['status']='STOPPED'
        except BaseException as exc:c.meta.update(status='FAIL',error=repr(exc));raise
        finally:
            try:c.close()
            finally:guard.close()
        assert not guard.error
        time.sleep(3);r.monitor.idle(tag+'-after',seconds=5)

if __name__=='__main__':main()
