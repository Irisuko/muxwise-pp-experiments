#!/usr/bin/env python3
"""P10a four-GPU preparation. Formal replay is unconditionally locked."""
import argparse, datetime, fcntl, hashlib, json, os, shlex, signal, subprocess, sys, time
from pathlib import Path
import numpy as np
import p3_runner as base
from p7a_runner import Cluster
from p9a_serve import commands as v4_commands
from p10a_model import RAM, ensure

ROOT = base.ROOT
SOURCE = Path('/root/workspace/sglang')
LOG = ROOT/'logs/p10a'
DEV = ROOT/'bench/dev_big_toolagent_1200_80_x7.5.jsonl'
SELECTED = ROOT/'configs/p10a_selected.json'
MATRIX = {
    'big_muxwise_tp4': ('muxwise',4,1),
    'big_chunked_tp4': ('chunked',4,1),
    'big_chunked_pp4': ('chunked',1,4),
    'big_chunked_pp2tp2': ('chunked',2,2),
    'big_muxpp_pp4': ('muxpp',1,4),
    'big_muxpp_pp2tp2': ('muxpp',2,2),
    'big_plain_tp4': ('plain',4,1),
}
DEFAULT_MEM = {'chunked':.88, 'muxwise':.82, 'muxpp':.80, 'plain':.82}


def commands(label, mem=None, correctness=False):
    family,tp,pp = MATRIX[label]
    if mem is None:
        selected = json.loads(SELECTED.read_text()) if SELECTED.exists() else {}
        mem = selected.get(label,{}).get('mem_fraction',DEFAULT_MEM[family])
    if family == 'muxpp':
        _,cmd,env,port = v4_commands('pp4' if pp==4 else 'pp2tp2')[0]
        for k,v in [('--model-path',RAM),('--context-length',32768),('--mem-fraction-static',mem)]:
            cmd[cmd.index(k)+1] = str(v)
    else:
        cmd = [base.PYTHON,'-m','sglang.launch_server','--model-path',str(RAM),
            '--tp',str(tp),'--pp-size',str(pp),'--dtype','bfloat16','--port','30000',
            '--disable-overlap-schedule','--context-length','32768','--schedule-policy','fcfs',
            '--max-running-requests','128','--watchdog-timeout','180','--log-level','debug',
            '--enable-cache-report','--attention-backend','flashinfer','--mem-fraction-static',str(mem),
            '--cuda-graph-max-bs','32','--cuda-graph-bs','1','4','8','16','32']
        env = {'CUDA_VISIBLE_DEVICES':'0,1,2,3','PYTHONPATH':str(SOURCE/'python')}
        port = 30000
        if tp>1: cmd += ['--disable-custom-all-reduce']
        if family=='chunked':
            cmd += ['--chunked-prefill-size','2048']
            if pp==1: cmd += ['--enable-mixed-chunk']
        elif family=='muxwise':
            cmd += ['--enable-pdmux','--chunked-prefill-size','-1','--pdmux-config-path',str(ROOT/'configs/pdmux_4090.yml')]
            env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        else: cmd += ['--chunked-prefill-size','-1']
    # Detokenization returns only decode-window IDs in this revision. Exact
    # token comparisons require bypassing it; trace sanity keeps normal serving.
    if correctness:
        cmd += ['--skip-tokenizer-init']
    if os.getenv('P10A_STACKS') == '1':
        env['PYTHONPATH'] = str(ROOT/'diagnostics/p10a') + ':' + str(SOURCE/'python')
    return [('server',cmd,env,port)]


def warmup(c):
    sys.path.insert(0,str(ROOT/'bench'))
    from build_trace import load_vocabulary
    vocab,bos,_ = load_vocabulary(RAM)
    ids = [bos]+np.random.default_rng(730031).choice(vocab,size=31662).tolist()
    result = base.generate(base.BASE,ids,8)
    base.save(c.path/'long-warmup.json',result)
    print('long warmup:',result['elapsed_s'],flush=True)
    base.flush(c)


def replay(c, stage):
    assert DEV.name.startswith('dev_') and DEV.resolve()==ROOT/'bench/dev_big_toolagent_1200_80_x7.5.jsonl'
    smoke = stage=='smoke'
    out=c.path/stage
    label=c.label+'_'+stage
    cmd=[base.PYTHON,str(ROOT/'bench/bench_client.py'),'--trace',str(DEV),'--base-url',base.BASE,
         '--label',label,'--out-dir',str(out),'--limit','20' if smoke else '150',
         '--time-scale','0.2' if smoke else '1','--warmup','2','--force-close',
         '--drain-timeout','55' if smoke else '300','--progress-interval','15']
    if smoke: cmd += ['--timeout','55']
    base.save(c.path/(stage+'-command.json'),cmd)
    with (c.path/(stage+'-client.log')).open('x') as f:
        proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            rc=proc.wait(timeout=60 if smoke else 1100)
        except subprocess.TimeoutExpired:
            proc.send_signal(signal.SIGINT)
            try:rc=proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait();raise
        except BaseException:
            proc.send_signal(signal.SIGINT)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            raise
    summary=json.loads((out/(label+'_summary.json')).read_text())
    c.meta[stage]=summary
    base.save(c.path/'run.json',c.meta)
    print(stage,json.dumps({k:summary[k] for k in ('completed','failed','unfinished','duration_sec')}),flush=True)
    if smoke:
        assert rc==0 and summary['completed']==20 and summary['failed']==0 and summary['unfinished']==0 and summary['duration_sec']<=60, 'Smoke failed'
    else:
        # Development data are recorded, never used to choose performance parameters.
        c.meta['sanity_healthy']=rc==0 and summary['completed']==150 and summary['failed']==0 and summary['unfinished']==0
    return summary


def serial(c):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(RAM,local_files_only=True)
    lengths=[100,128,256,512,768,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000]
    prompts=[]
    for i,n in enumerate(lengths):
        text=('The garden contains apple and pear trees. Water the plants daily. ' if i%4==0 else f'Report family {i%4}: the robot records temperature and rainfall. ')*1200
        prefix=tok.encode(text,add_special_tokens=False)
        suffix=tok.encode(f'\nSummarize report number {i} in two sentences.\nAnswer:',add_special_tokens=False)
        prompts.append([tok.bos_token_id]+prefix[:n-len(suffix)-1]+suffix)
    base.save(c.path/'serial20-prompts.json',prompts)
    outputs=[]
    for i,ids in enumerate(prompts):
        body=base.generate(base.BASE,ids,64)['response']
        base.save(c.path/f'serial20-response-{i:02d}.json',body)
        assert len(body['output_ids'])==64
        outputs.append(body)
        base.save(c.path/'serial20.json',outputs)
        print('serial',i,'cached',body['meta_info'].get('cached_tokens'),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['commands','launch','preflight','sanity','correctness','validated','experiment'])
    p.add_argument('label',choices=MATRIX)
    p.add_argument('--mem',type=float)
    p.add_argument('--tag')
    p.add_argument('--serial', action='store_true', help='Run serial20 before the development sanity replay')
    a=p.parse_args()
    if a.mode=='experiment':
        p.error('P10a FORMAL LOCK: preparation only; formal trace replay is forbidden.')
    jobs=commands(a.label,a.mem,a.mode=='correctness')
    if a.mode=='commands':
        print('\n'.join(shlex.join(['env']+[k+'='+v for k,v in e.items()]+cmd) for _,cmd,e,_ in jobs));return
    LOG.mkdir(exist_ok=True)
    lock=(LOG/'gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ensure()
    base.commands=lambda _:jobs
    path=LOG/(a.tag or a.mode+'-'+a.label+'-'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
    c=Cluster(a.label,path)
    c.meta.update(mode=a.mode,commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),
                  dev_trace_sha256=hashlib.sha256(DEV.read_bytes()).hexdigest())
    patch=subprocess.check_output(['git','-C',str(SOURCE),'diff','HEAD','--binary'])
    def stop(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,stop)
    try:
        c.start()
        (path/'source.patch').write_bytes(patch)
        if a.mode=='launch':signal.pause()
        elif a.mode=='correctness': serial(c)
        elif a.mode=='validated':
            warmup(c)
            replay(c,'smoke')
            base.flush(c)
            if MATRIX[a.label][0] in ('muxwise','muxpp'): serial(c)
            base.flush(c)
            replay(c,'sanity')
        else:
            warmup(c)
            if a.serial:
                serial(c)
                base.flush(c)
            replay(c,'smoke' if a.mode=='preflight' else 'sanity')
        time.sleep(3)
        assert all(proc.poll() is None for proc in c.procs),'server exited'
        c.meta['status']='PASS' if c.meta.get('sanity_healthy', True) else 'INCOMPLETE'
    except KeyboardInterrupt:c.meta['status']='STOPPED'
    except BaseException as exc:
        c.meta.update(status='FAIL',error=repr(exc));raise
    finally:c.close()

if __name__=='__main__':main()
