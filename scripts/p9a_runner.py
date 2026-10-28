#!/usr/bin/env python3
"""P9a development only: a hard whitelist prevents any formal trace replay."""
import argparse, datetime, fcntl, hashlib, json, os, random, signal, subprocess, time
from pathlib import Path
import p3_runner as base
from p8a_runner import commands as baseline_commands

ROOT=base.ROOT
SOURCE=Path(os.getenv('P9A_SOURCE','/root/workspace/sglang'))
DEV={name: ROOT/f'bench/dev_{name}_1200_240_x2.5.jsonl' for name in ('toolagent','conversation')}

def commands(topology, path, extra):
    _,cmd,env,port=baseline_commands()[0]
    if topology=='pp4':
        for key,value in (('--tp',1),('--pp-size',4),('--pdmux-pp-max-inflight-prefill',6)):
            cmd[cmd.index(key)+1]=str(value)
    cmd += ['--pdmux-pp-request-trace',str(path/'requests-timeline')]+extra
    env['PYTHONPATH']=str(SOURCE/'python')
    env.update(SGLANG_PDMUX_PP_ROUND_TRACE=str(path/'decode-rounds.jsonl'))
    return [('server',cmd,env,port)]

def run(a):
    path=ROOT/'logs/p9a'/a.tag
    lock=(ROOT/'logs/p9a/gpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    base.commands=lambda _:commands(a.topology,path,a.extra)
    c=base.Cluster(a.topology,path)
    c.meta.update(topology=a.topology,trace=str(DEV[a.trace]),tag=a.tag,
        commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip())
    source_patch=subprocess.check_output(['git','-C',str(SOURCE),'diff','HEAD','--binary'])
    c.meta['source_files']={str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in list((SOURCE/'python/sglang/srt/multiplex').glob('*.py'))+
        [SOURCE/'python/sglang/srt/server_args.py',SOURCE/'python/sglang/srt/managers/scheduler_output_processor_mixin.py']}
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        c.start()
        (path/'source.patch').write_bytes(source_patch)
        rng=random.Random(730031)
        base.save(path/'long-warmup.json',base.generate(base.BASE,[128000]+[rng.randrange(128000) for _ in range(31662)]))
        base.flush(c)
        cmd=[base.PYTHON,str(ROOT/'bench/bench_client.py'),'--trace',str(DEV[a.trace]),
            '--base-url',base.BASE,'--label',a.tag,'--out-dir',str(path),
            '--limit',str(a.limit),'--time-scale',str(a.time_scale),
            '--warmup','2','--force-close','--drain-timeout','300','--progress-interval','15']
        base.save(path/'client-command.json',cmd)
        with (path/'client.log').open('w') as f:
            p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
            try:
                rc=p.wait(timeout=1100)
            except BaseException:
                p.send_signal(signal.SIGINT)
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:p.kill();p.wait()
                raise
        summary=json.loads((path/f'{a.tag}_summary.json').read_text())
        c.meta['summary']=summary
        assert rc==0 and summary['completed']==a.limit and summary['failed']==0 and summary['unfinished']==0,summary
        time.sleep(4)
        assert all(p.poll() is None for p in c.procs)
        c.meta['status']='PASS'
        print(json.dumps({k:summary[k] for k in ('completed','failed','duration_sec','latencies_sec','slo')}),flush=True)
    except BaseException as e:
        c.meta.update(status='FAIL',error=repr(e));raise
    finally:
        c.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--topology',choices=['pp2tp2','pp4'],default='pp2tp2')
    p.add_argument('--trace',choices=list(DEV),required=True)
    p.add_argument('--tag',required=True)
    p.add_argument('--limit',type=int,default=300)
    p.add_argument('--time-scale',type=float,default=1)
    p.add_argument('--extra',nargs=argparse.REMAINDER,default=[])
    run(p.parse_args())
