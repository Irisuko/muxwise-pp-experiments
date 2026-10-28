#!/usr/bin/env python3
"""P3 lifecycle: no retries of formal data; every launch has its own audit trail."""
import argparse, concurrent.futures, contextlib, datetime, json, os, pathlib, shlex, signal, subprocess, sys, threading, time
import psutil, requests
ROOT = pathlib.Path('/root/workspace/muxpp_work')
MODEL = '/root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct'
TRACE = ROOT/'bench/toolagent_600_240_x2.5.jsonl'
PYTHON = '/root/venvs/sgl/bin/python'
BASE = 'http://127.0.0.1:30000'
DEFAULT_MEM = {'chunked': '0.88', 'muxwise': '0.826', 'sglang_pd': '0.88', 'muxwise_pp': '0.65'}

def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str)+'\n')

def get(url, **kw):
    return requests.get(url, headers={'Connection':'close'}, **kw)

def post(url, **kw):
    return requests.post(url, headers={'Connection':'close'}, **kw)

def gpu():
    return subprocess.check_output(['nvidia-smi'], text=True)

def clean_gpu():
    value=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
    if value: raise RuntimeError('GPU is not clean: '+value)

def commands(label):
    common=[PYTHON,'-m','sglang.launch_server','--model-path',MODEL,'--tp','1','--dtype','bfloat16','--disable-overlap-schedule','--context-length','33792','--schedule-policy','fcfs','--cuda-graph-max-bs',os.getenv('P3_GRAPH_BS','8' if label=='muxwise' else '32'),'--max-running-requests','128','--watchdog-timeout','180','--log-level','debug','--enable-cache-report']
    attention=os.getenv('P3_ATTENTION','triton' if label=='muxwise' else '')
    if attention:common+=['--attention-backend',attention]
    if os.getenv('P3_DISABLE_GRAPH','1' if label=='muxwise' else '0')=='1':common+=['--disable-cuda-graph']
    mem=os.getenv('P3_MEM',DEFAULT_MEM[label])
    jobs=[]
    if label=='muxwise_pp':
        cmd=common+['--port','30000','--mem-fraction-static',mem,'--pp-size','4','--enable-pdmux','--chunked-prefill-size','-1','--pdmux-config-path',str(ROOT/'configs/pdmux_pp_4090.yml'),'--pdmux-pp-max-inflight-prefill','4','--pdmux-pp-radix-eviction','lru','--pdmux-pp-policy','alp','--pdmux-pp-slo-ttft','5.0','--pdmux-pp-slo-tpot','0.1','--pdmux-pp-reserve-decode','--pdmux-pp-balance-merge','--pdmux-pp-rebalance','--pdmux-pp-alp-admission','--pdmux-pp-rebalance-unit','8','--pdmux-pp-summary-interval','10']
        jobs.append(('pp4',cmd,{'CUDA_VISIBLE_DEVICES':'0,1,2,3'},30000))
    else:
        for i in range(4):
            m=os.getenv('P3_DECODE_MEM',mem) if label=='sglang_pd' and i>=2 else mem
            cmd=common+['--port',str(31000+i),'--mem-fraction-static',m]
            env={'CUDA_VISIBLE_DEVICES':str(i)}
            if label=='chunked': cmd+=['--chunked-prefill-size','2048','--enable-mixed-chunk']
            elif label=='muxwise': cmd+=['--enable-pdmux','--chunked-prefill-size','-1','--pdmux-config-path',str(ROOT/'configs/pdmux_4090.yml')]
            else:
                cmd+=['--disaggregation-mode','prefill' if i<2 else 'decode','--disaggregation-transfer-backend',os.getenv('P3_PD_BACKEND','mooncake'),'--disaggregation-bootstrap-port',str(8998+i)]
                if i<2: cmd+=['--chunked-prefill-size','4096']
                if os.getenv('P3_PD_BACKEND','mooncake')=='mooncake': env['MC_FORCE_TCP']='1'
                else: env.update(UCX_TLS='cuda_ipc,cuda_copy,tcp,sm,self',UCX_NET_DEVICES='lo')
            jobs.append((f'worker{i}',cmd,env,31000+i))
        router=[PYTHON,'-m','sglang_router.launch_router','--host','127.0.0.1','--port','30000','--policy','cache_aware','--request-timeout-secs','3600','--max-concurrent-requests','4096','--log-level','info']
        if label=='sglang_pd':
            router+=['--pd-disaggregation','--prefill','http://127.0.0.1:31000','8998','--prefill','http://127.0.0.1:31001','8999','--decode','http://127.0.0.1:31002','--decode','http://127.0.0.1:31003']
        else: router+=['--worker-urls']+[f'http://127.0.0.1:{31000+i}' for i in range(4)]
        router_env={}
        if label=='sglang_pd' and os.getenv('P3_PD_ROUTER','mini')=='mini':
            router+=['--mini-lb']
            router[router.index('--policy')+1]='random'
            router_env['PYTHONPATH']='/root/workspace/sglang/sgl-router/py_src:'+os.environ.get('PYTHONPATH','')
        jobs.append(('router',router,router_env,30000))
    for name,cmd,env,port in jobs:
        if name != "router" and os.getenv("P3_WORKSPACE"):
            env["FLASHINFER_WORKSPACE_SIZE"] = os.environ["P3_WORKSPACE"]
        alloc=os.getenv("P3_ALLOC_CONF","expandable_segments:True" if label=="muxwise" else "")
        if name != "router" and alloc:
            env["PYTORCH_CUDA_ALLOC_CONF"] = alloc
    return jobs

class Cluster:
    def __init__(self,label,path):
        self.label,self.path=label,path
        self.procs=[]; self.logs=[]; self.stop=threading.Event()
        self.meta={'label':label,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commands':[]}
    def sampling(self):
        with (self.path/'gpu.csv').open('w',buffering=1) as f:
            f.write('timestamp, index, memory.used [MiB], memory.total [MiB], utilization.gpu [%], utilization.memory [%]\n')
            while not self.stop.is_set():
                subprocess.run(['nvidia-smi','--query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,utilization.memory','--format=csv,noheader'],stdout=f,stderr=subprocess.STDOUT)
                self.stop.wait(5)
    def wait_health(self,port):
        start=time.monotonic()
        while time.monotonic()-start<600:
            for p in self.procs:
                if p.poll() is not None: raise RuntimeError(f'server {p.pid} exited: {p.returncode}')
            try:
                if get(f'http://127.0.0.1:{port}/health',timeout=2).status_code==200: return
            except requests.RequestException: pass
            time.sleep(1)
        raise TimeoutError(f'health timeout: {port}')
    def start(self):
        clean_gpu(); self.path.mkdir(parents=True,exist_ok=False)
        (self.path/'gpu-before.txt').write_text(gpu())
        self.thread=threading.Thread(target=self.sampling,daemon=True);self.thread.start()
        jobs=commands(self.label)
        save(self.path/'commands.json',[{'name':n,'argv':c,'env':e,'port':p} for n,c,e,p in jobs])
        (self.path/'commands.sh').write_text('source '+str(ROOT/'env.sh')+'\n'+'\n'.join(shlex.join(['env']+[k+'='+v for k,v in e.items()]+c) for n,c,e,p in jobs)+'\n')
        start=time.monotonic()
        for name,cmd,env,port in jobs:
            if name=='router':
                for _,_,_,workerport in jobs[:-1]: self.wait_health(workerport)
            log=(self.path/f'{name}.log').open('w',buffering=1);self.logs.append(log)
            proc=subprocess.Popen(cmd,env=dict(os.environ,**env),stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd='/root/workspace/sglang')
            self.procs.append(proc); self.meta['commands'].append({'name':name,'pid':proc.pid,'argv':cmd,'env':env})
            save(self.path/'run.json',self.meta)
        for _,_,_,port in jobs: self.wait_health(port)
        self.meta['startup_s']=time.monotonic()-start
        self.meta['server_info']={}
        for name,_,_,port in jobs:
            if name!='router':
                r=get(f'http://127.0.0.1:{port}/get_server_info',timeout=20);r.raise_for_status()
                self.meta['server_info'][name]=r.json()
        save(self.path/'run.json',self.meta)
        print(f'{self.label} READY {self.meta["startup_s"]:.1f}s',flush=True)
    def reset_router(self):
        if self.label=='muxwise_pp':return
        p=self.procs.pop()
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        name,cmd,env,port=commands(self.label)[-1]
        log=(self.path/'router-formal.log').open('w',buffering=1);self.logs.append(log)
        p=subprocess.Popen(cmd,env=dict(os.environ,**env),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        self.procs.append(p)
        self.meta['router_formal_pid']=p.pid
        self.wait_health(port)
        print('Router reset for next replay',flush=True)
    def close(self):
        for p in reversed(self.procs):
            try: children=psutil.Process(p.pid).children(recursive=True)
            except psutil.NoSuchProcess: children=[]
            try: os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: p.wait(timeout=12)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError): os.killpg(p.pid,signal.SIGKILL)
                p.wait(timeout=10)
            for child in children:
                with contextlib.suppress(psutil.NoSuchProcess): child.kill()
            psutil.wait_procs(children,timeout=5)
        self.stop.set()
        if hasattr(self,'thread'): self.thread.join(timeout=6)
        for f in self.logs:f.close()
        if self.path.exists():
            (self.path/'gpu-cleanup.txt').write_text(gpu())
            self.meta['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
            save(self.path/'run.json',self.meta)
        clean_gpu()

def generate(url,ids,n=8):
    start=time.monotonic()
    r=post(url+'/generate',json={'input_ids':ids,'sampling_params':{'temperature':0,'max_new_tokens':n,'ignore_eos':True}},timeout=180)
    r.raise_for_status(); body=r.json()
    assert body['meta_info']['completion_tokens']==n,body
    return {'elapsed_s':time.monotonic()-start,'input_len':len(ids),'response':body}

def flush(cluster):
    for name,_,_,port in commands(cluster.label):
        if name!='router':
            r=post(f'http://127.0.0.1:{port}/flush_cache',timeout=30);r.raise_for_status()
    print('Cache flushed',flush=True)

def validate(cluster):
    # Deterministic random ordinary IDs; no trace prefixes are warmed.
    import random
    rng=random.Random(730031)
    ids=[128000]+[rng.randrange(128000) for _ in range(31662)]
    urls=([BASE,BASE] if cluster.label=='sglang_pd' else [BASE]) if cluster.label in ('muxwise_pp','sglang_pd') else [f'http://127.0.0.1:{31000+i}' for i in range(4)]
    outputs=[]
    for i,url in enumerate(urls):
        if cluster.label=='sglang_pd' and i:ids[1]=(ids[1]+1)%128000
        out=generate(url,ids);out['url']=url;outputs.append(out)
        save(cluster.path/'long-warmup.json',outputs)
        print(f'long warmup {url}: {out["elapsed_s"]:.2f}s / {len(ids)} tokens',flush=True)
    flush(cluster)
    if cluster.label in ('chunked','sglang_pd'):
        # Use exactly the same 20 short deterministic token prompts in plain and PD mode.
        rng=random.Random(20020)
        cases=[[128000]+[rng.randrange(100,120000) for _ in range(63+i*3)] for i in range(20)]
        results=[generate('http://127.0.0.1:31000' if cluster.label=='chunked' else BASE,ids,16) for ids in cases]
        save(cluster.path/'correctness20.json',results)
        ref=ROOT/'logs/p3/normal-correctness20.json'
        if cluster.label=='chunked':save(ref,results)
        else:
            expected=json.loads(ref.read_text())
            matches=[r['response']['text']==e['response']['text'] for r,e in zip(results,expected)]
            save(cluster.path/'correctness-comparison.json',{'matches':matches,'passed':sum(matches),'total':20})
            assert all(matches),f'PD output equality: {sum(matches)}/20'
            print('PD correctness: 20/20 exact text matches',flush=True)
        flush(cluster)

def bench(cluster,smoke=False):
    out=cluster.path/'smoke' if smoke else ROOT/'results'
    label=cluster.label+'_smoke' if smoke else cluster.label
    cmd=[PYTHON,str(ROOT/'bench/bench_client.py'),'--trace',str(TRACE),'--base-url',BASE,'--label',label,'--out-dir',str(out),'--warmup','2','--force-close']
    if smoke:cmd+=['--limit','20','--time-scale','0.2','--timeout','55']
    else:
        # An exclusive marker prevents replays even if the client crashes before writing its summary.
        marker=ROOT/'results'/f'{label}.formal-started.json'
        if (ROOT/'archive/superseded/results'/marker.name).exists():
            raise RuntimeError('Archived formal launch already recorded; replay forbidden')
        with marker.open('x') as f:json.dump({'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':cmd,'log_dir':str(cluster.path)},f,indent=2)
    save(cluster.path/('smoke-command.json' if smoke else 'formal-command.json'),cmd)
    print('RUN '+shlex.join(cmd),flush=True)
    with (cluster.path/('smoke-client.log' if smoke else 'formal-client.log')).open('w') as log:
        p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        try: rc=p.wait(timeout=60 if smoke else 4300)
        except subprocess.TimeoutExpired:
            p.send_signal(signal.SIGINT)
            try:rc=p.wait(timeout=20)
            except subprocess.TimeoutExpired:p.kill();p.wait();raise
    summary=json.loads((out/f'{label}_summary.json').read_text())
    if smoke:
        assert rc==0 and summary['completed_requests']==20 and summary['duration_sec']<=60,summary
    else:
        assert summary['planned_requests']==1551,summary
        cluster.meta['formal_returncode']=rc;cluster.meta['formal_summary']=summary
    print(f'{label}: complete={summary["completed_requests"]}, fail={summary["failed_requests"]}, duration={summary["duration_sec"]:.2f}s',flush=True)
    return summary

def main():
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['launch','preflight','experiment']);a.add_argument('label',choices=list(DEFAULT_MEM));a.add_argument('--tag');args=a.parse_args()
    parent=ROOT/'logs/p3'/args.label;parent.mkdir(parents=True,exist_ok=True)
    tag=args.tag or (args.mode+'-'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
    if args.mode=='experiment' and any((ROOT/p/f'{args.label}.formal-started.json').exists() for p in ('results','archive/superseded/results')):raise RuntimeError('Formal launch already recorded; replay forbidden')
    if (parent/tag).exists():raise FileExistsError(parent/tag)
    c=Cluster(args.label,parent/tag)
    def term(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,term)
    try:
        c.start()
        if args.mode=='launch':
            while True:time.sleep(1)
        else:
            validate(c)
            bench(c,smoke=True)
            if args.label=='muxwise':
                flush(c)
                validate(c)
                (c.path/'long-warmup-after-smoke.json').write_text((c.path/'long-warmup.json').read_text())
            if args.mode=='experiment':
                flush(c)
                c.reset_router()
                bench(c)
            c.meta['status']='PASS'
    except BaseException as exc:
        c.meta.update(status='FAIL',error=repr(exc));raise
    finally:c.close()
if __name__=='__main__':main()
