#!/usr/bin/env python3
"""P10B development-only runs, with exact P10A prompt replay."""
import argparse, fcntl, hashlib, json, os, signal, subprocess, time
from pathlib import Path
import p10a_runner as a
from p7a_runner import Cluster
b=a.base
ROOT=a.ROOT
LOG=ROOT/'logs/p10b'
a.MATRIX.update(big_plain_pp4=('plain',1,4), big_plain_pp2tp2=('plain',2,2))

def serial(c, cold):
    prompts=json.loads((ROOT/'logs/p10a/reference-tp4-serial20/serial20-prompts.json').read_text())
    b.save(c.path/'serial20-prompts.json',prompts)
    name='cold20' if cold else 'serial20'
    b.flush(c)
    outputs=[]
    for i,ids in enumerate(prompts):
        if cold:b.flush(c)
        body=b.generate(b.BASE,ids,64)['response']
        assert len(body['output_ids'])==64
        if cold:assert body['meta_info']['cached_tokens']==0
        outputs.append(body)
        b.save(c.path/(name+'.json'),outputs)
        print(name,i,'cached',body['meta_info'].get('cached_tokens'),flush=True)

def smoke(c, seconds):
    assert seconds >= 60
    out=c.path/'smoke';label=c.label+'_smoke'
    cmd=[b.PYTHON,str(ROOT/'bench/bench_client.py'),'--trace',str(a.DEV),
         '--base-url',b.BASE,'--label',label,'--out-dir',str(out),'--limit','20',
         '--time-scale','0.2','--warmup','2','--force-close','--timeout',str(seconds-5),
         '--drain-timeout',str(seconds-5),'--progress-interval','15']
    b.save(c.path/'smoke-command.json',cmd)
    with (c.path/'smoke-client.log').open('x') as f:
        proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            rc=proc.wait(timeout=seconds)
        except BaseException:
            proc.send_signal(signal.SIGINT)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            raise
    result=json.loads((out/(label+'_summary.json')).read_text())
    c.meta['smoke']=result
    print('smoke',result['completed'],result['failed'],result['unfinished'],result['duration_sec'],flush=True)
    assert rc==0 and result['completed']==20 and result['failed']==0 and result['unfinished']==0

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['correctness','numeric','preflight','profile','dev','launch'])
    p.add_argument('label',choices=a.MATRIX)
    p.add_argument('--tag',required=True)
    p.add_argument('--mem',type=float)
    p.add_argument('--prompt-index',type=int,default=17)
    p.add_argument('--smoke-seconds',type=int,default=60)
    p.add_argument('--env',action='append',default=[])
    p.add_argument('--extra',nargs=argparse.REMAINDER,default=[])
    x=p.parse_args()
    LOG.mkdir(exist_ok=True)
    lock=(LOG/'gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Container NVML process lists can be empty while host jobs occupy devices.
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    values=[tuple(int(v.strip()) for v in row.split(',')) for row in raw.splitlines()]
    if not all(mem < 100 and util < 5 for mem,util in values):
        b.save(LOG/(x.tag+'-external-gpu-block.json'),{'time':time.time(),'gpu_memory_mib_and_util':values,'compute_apps':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],text=True)})
        raise RuntimeError('GPU is occupied; stop immediately rather than wait or replay')
    a.ensure()
    jobs=a.commands(x.label,x.mem,x.mode in ('correctness','numeric'))
    for _,cmd,env,_ in jobs:
        cmd+=x.extra
        env.update(dict(v.split('=',1) for v in x.env))
        if a.MATRIX[x.label][0]=='muxpp':
            cmd+=['--pdmux-pp-request-trace',str(LOG/x.tag/'requests-timeline')]
            env['SGLANG_PDMUX_PP_ROUND_TRACE']=str(LOG/x.tag/'decode-rounds.jsonl')
    b.commands=lambda _:jobs
    c=Cluster(x.label,LOG/x.tag)
    c.meta.update(mode=x.mode,commit=subprocess.check_output(['git','-C',str(a.SOURCE),'rev-parse','HEAD'],text=True).strip(),dev_trace_sha256=hashlib.sha256(a.DEV.read_bytes()).hexdigest())
    c.meta['pdmux_configs']={str(Path(cmd[i+1])):Path(cmd[i+1]).read_text() for _,cmd,_,_ in jobs for i,key in enumerate(cmd) if key=='--pdmux-config-path'}
    patch=subprocess.check_output(['git','-C',str(a.SOURCE),'diff','HEAD','--binary'])
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        c.start();(c.path/'source.patch').write_bytes(patch)
        if x.mode=='numeric':
            prompts=json.loads((ROOT/'logs/p10a/reference-tp4-serial20/serial20-prompts.json').read_text())
            b.flush(c)
            b.save(c.path/'numeric-output.json',b.generate(b.BASE,prompts[x.prompt_index],4))
        elif x.mode=='correctness':
            serial(c,False);serial(c,True)
        elif x.mode=='launch':signal.pause()
        else:
            a.warmup(c)
            if x.mode=='profile':
                pass
            elif x.mode=='preflight' and x.smoke_seconds!=60:
                smoke(c,x.smoke_seconds)
            else:
                a.replay(c,'sanity' if x.mode=='dev' else 'smoke')
            if x.mode=='dev':
                for file in (c.path/'sanity').glob('*'):
                    if file.name.endswith(('_summary.json','_requests.jsonl')):
                        (c.path/file.name).symlink_to(file.relative_to(c.path))
        time.sleep(3)
        assert all(v.poll() is None for v in c.procs)
        c.meta['status']='PASS' if c.meta.get('sanity_healthy',True) else 'INCOMPLETE'
    except BaseException as e:
        c.meta.update(status='FAIL',error=repr(e));raise
    finally:
        c.close()
        # Let the driver utilization sampling window retire this process.
        time.sleep(2)
if __name__=='__main__':main()
