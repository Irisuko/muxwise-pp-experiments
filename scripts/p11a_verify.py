#!/usr/bin/env python3
"""P11a same-topology references, cancellation and development-only stress."""
import argparse
import concurrent.futures
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import time

import pynvml
import p11a_runner as r

b=r.b
ROOT=r.ROOT
OUT=r.OUT


def ordinary(label, kind='plain'):
    large=label.startswith('big_')
    topo=label.split('_',1)[1]
    tp,pp={'pp4':(1,4),'pp2tp2':(2,2),'tp1':(1,1),'tp2':(2,1),'tp4':(4,1)}[topo]
    cmd=[b.PYTHON,'-m','sglang.launch_server','--model-path',str(r.big.RAM) if large else b.MODEL,
         '--tp',str(tp),'--pp-size',str(pp),'--dtype','bfloat16','--port','30000',
         '--disable-overlap-schedule','--context-length','32768' if large else '33792',
         '--schedule-policy','fcfs','--max-running-requests','128','--watchdog-timeout','180',
         '--log-level','debug','--enable-cache-report','--attention-backend','flashinfer',
         '--mem-fraction-static',('0.78' if tp==2 else '0.76') if large else ('0.76' if tp==2 else '0.71'),
         '--cuda-graph-max-bs','32','--cuda-graph-bs','1','4','8','16','32',
         '--chunked-prefill-size','2048' if kind=='chunked' else '-1','--skip-tokenizer-init']
    if tp>1:cmd+=['--disable-custom-all-reduce']
    env=dict(CUDA_VISIBLE_DEVICES=','.join(map(str,range(tp*pp))),PYTHONPATH=str(r.MAIN/'python'))
    if kind=='pdmux':
        cmd+=['--enable-pdmux','--pdmux-config-path',str(ROOT/'configs/pdmux_4090.yml')]
        env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    return [('server',cmd,env,30000)]


def prompts(label):
    if label.startswith('big_'):
        return json.loads((ROOT/'logs/p10a/reference-tp4-serial20/serial20-prompts.json').read_text())
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(b.MODEL,local_files_only=True)
    cases=[]
    for i,n in enumerate([100,128,256,512,768,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000]):
        text=('The garden contains apple and pear trees. Water the plants daily. ' if i%4==0 else f'Report family {i%4}: the robot records temperature and rainfall. ')*1200
        ids=tok.encode(text,add_special_tokens=False)
        suffix=tok.encode(f'\nSummarize report number {i} in two sentences.\nAnswer:',add_special_tokens=False)
        cases.append([128000]+ids[:n-len(suffix)-1]+suffix)
    return cases


def correctness(c,label):
    cases=prompts(label)
    b.save(c.path/'prompts.json',cases)
    for kind in ['serial20','cold20']:
        b.flush(c)
        outputs=[]
        for i,ids in enumerate(cases):
            if kind=='cold20':b.flush(c)
            body=b.generate(b.BASE,ids,64)['response']
            assert len(body['output_ids'])==64
            if kind=='cold20':assert body['meta_info']['cached_tokens']==0
            outputs.append(body)
            b.save(c.path/(kind+'.json'),outputs)
            print(kind,i,flush=True)
        reference=OUT/('reference-'+label)/(kind+'.json')
        if reference.exists() and reference.parent!=c.path:
            expected=json.loads(reference.read_text())
            rows=[]
            for i,(actual,ref) in enumerate(zip(outputs,expected)):
                a,z=actual['output_ids'],ref['output_ids']
                same=[x==y for x,y in zip(a,z)]
                rows.append(dict(request=i,matched=sum(same),exact=all(same),
                                 first_mismatch=next((j for j,v in enumerate(same) if not v),None)))
            result=dict(exact_requests=sum(x['exact'] for x in rows),
                exact_tokens=sum(x['matched'] for x in rows),total_requests=20,total_tokens=1280,rows=rows,reference=str(reference))
            b.save(c.path/(kind+'-comparison.json'),result)
            c.meta.setdefault('correctness',{})[kind]={k:v for k,v in result.items() if k!='rows'}
            c.meta['correctness_status']='EXACT' if all(v['exact_requests']==20 for v in c.meta['correctness'].values()) else 'NUMERICAL_DIFFERENCES'


def cancel(c,label):
    cases=prompts(label)
    bos=cases[0][0]
    # Distinct long prompts exercise queued and multiple not-yet-computed chunks.
    def request(i):
        ids=cases[i%20] if i<4 else [bos]+(cases[17][1:]*4)[:27000+i]
        response=b.post(b.BASE+'/generate',json=dict(rid=f'p11a-abort-{i}',input_ids=ids,
             sampling_params=dict(temperature=0,max_new_tokens=1000,ignore_eos=True)),timeout=180)
        return dict(status=response.status_code,body=response.json())
    b.flush(c)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures=[]
        for i in range(4):
            futures.append(pool.submit(request,i));time.sleep(.15)
        futures += [pool.submit(request,i) for i in range(4,16)]
        time.sleep(.3)
        b.post(b.BASE+'/abort_request',json=dict(rid='p11a-abort-',abort_all=True),timeout=20).raise_for_status()
        outputs=[f.result() for f in futures]
    b.save(c.path/'cancellation.json',outputs)
    assert len(outputs) == 16
    assert all(('abort' in json.dumps(x['body']).lower()) if x['status'] == 400
               else x['status'] == 200 and x['body']['meta_info']['finish_reason']['type'] == 'abort'
               for x in outputs), outputs
    recovered = b.generate(b.BASE,cases[0],16)
    assert len(recovered['response']['output_ids']) == 16
    b.save(c.path/'after-cancel.json',recovered)
    b.flush(c)


def long_correctness(c, label, reference=False):
    # The original cold20 stops at 8000, below an 8192 CPP budget. Exercise
    # the long path separately, including real chunks when CPP is enabled.
    base = prompts(label)[19]
    ids = [base[0]] + (base[1:-64] * 2)[:12288 - 65] + base[-64:]
    assert len(ids) == 12288
    b.flush(c)
    body = b.generate(b.BASE, ids, 64)['response']
    assert len(body['output_ids']) == 64 and body['meta_info']['cached_tokens'] == 0
    budget = next(iter(c.meta['server_info'].values())).get('pdmux_pp_prefill_chunk_size', 0)
    b.save(c.path/'cold12288.json', dict(input_ids=ids, response=body,
                                       cpp_budget=budget, crosses_cpp_budget=bool(budget and len(ids)>budget)))
    if not reference:
        path = OUT/('final-legacy-plain-' + label)/'cold12288.json'
        assert path.exists(), 'Run the final ordinary-path reference before CPP long correctness'
        if path.exists():
            expected = json.loads(path.read_text())
            assert ids == expected['input_ids']
            actual, target = body['output_ids'], expected['response']['output_ids']
            same = [a == z for a, z in zip(actual, target)]
            comparison = dict(exact=all(same), matched=sum(same), total=64,
                              first_mismatch=next((i for i, ok in enumerate(same) if not ok), None),
                              reference=str(path), cpp_budget=budget,
                              crosses_cpp_budget=bool(budget and len(ids)>budget))
            b.save(c.path/'cold12288-comparison.json', comparison)
            c.meta['cold12288'] = comparison
    b.flush(c)


def stress(c,label):
    trace=r.TRACES[label.split('_')[0]]
    cmd=[b.PYTHON,str(ROOT/'bench/bench_client.py'),'--trace',str(trace),'--base-url',b.BASE,
         '--label','stress','--out-dir',str(c.path/'stress'),'--limit','200','--time-scale','0.02',
         '--warmup','2','--force-close','--drain-timeout','300','--progress-interval','30']
    b.save(c.path/'stress-command.json',cmd)
    with (c.path/'stress-client.log').open('x') as log:
        p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        try:rc=p.wait(timeout=600)
        except BaseException:
            p.send_signal(signal.SIGINT)
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            raise
    s=json.loads((c.path/'stress/stress_summary.json').read_text())
    c.meta['stress']=s
    assert rc==0 and (s['completed'],s['failed'],s['unfinished'])==(200,0,0)
    b.flush(c)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['reference','correctness','verify','smoke','candidate-smoke','long'])
    p.add_argument('label',choices=r.LABELS[:4]+['llama_tp1','llama_tp2','big_tp4','llama_pp2dp2'])
    p.add_argument('--kind',choices=['plain','pdmux','chunked'],default='plain')
    p.add_argument('--tag')
    p.add_argument('--source',type=Path,default=r.MAIN)
    p.add_argument('--extra',nargs=argparse.REMAINDER,default=[])
    x=p.parse_args()
    OUT.mkdir(exist_ok=True);r.monitoring.OUT=OUT;pynvml.nvmlInit()
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    tag=x.tag or x.mode+'-'+x.label
    path=OUT/tag
    with (ROOT/'logs/p7a/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not list(OUT.glob('STOP-*.json'))
        r.monitoring.idle(tag+'-before')
        if x.mode=='reference' or x.mode=='smoke' and x.label!='llama_pp2dp2':
            jobs=ordinary(x.label,x.kind)
        else:
            jobs=r.commands(x.label,x.source,path,x.extra)
            for name,cmd,env,port in jobs:
                if name!='router':cmd+=['--skip-tokenizer-init']
        for name,cmd,env,port in jobs:
            if name!='router':env['PYTHONPATH']=str(x.source/'python')
        b.commands=lambda _:jobs
        c=r.p7.Cluster(x.label,path)
        c.meta.update(mode=x.mode,kind=x.kind,commit=r.monitoring.git(x.source,'rev-parse','HEAD'),source=str(x.source),
                      verifier_sha256=r.monitoring.sha(Path(__file__)))
        guard=r.Guard(tag);guard.start()
        try:
            c.start();(path/'source.patch').write_text(r.monitoring.git(x.source,'diff','HEAD','--binary'))
            if x.mode in ['smoke','candidate-smoke']:
                cases=prompts('big_pp4' if x.label.startswith('big_') else 'llama_pp4')
                with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                    outputs=list(pool.map(lambda ids:b.generate(b.BASE,ids,32),cases[::3]))
                b.save(path/'smoke.json',outputs);b.flush(c)
                if x.mode == 'smoke' and x.kind == 'plain' and x.label in r.LABELS[:4]:
                    long_correctness(c, x.label, reference=True)
                if x.label == 'llama_pp2dp2':
                    # Explicitly cover both independent PP workers as well as
                    # the router, with an input that crosses the CPP budget.
                    base = cases[-1]
                    ids = [base[0]] + (base[1:-64] * 2)[:12288 - 65] + base[-64:]
                    workers = {name:b.generate(f'http://127.0.0.1:{port}', ids, 32)
                               for name, _, _, port in jobs if name != 'router'}
                    assert len(workers) == 2
                    assert all(len(v['response']['output_ids']) == 32 for v in workers.values())
                    b.save(path/'both-workers.json', workers)
                    b.flush(c)
            elif x.mode == 'long':
                long_correctness(c, x.label)
            else:
                correctness(c,x.label)
                if x.mode=='verify':
                    long_correctness(c,x.label)
                    cancel(c,x.label);stress(c,x.label)
            time.sleep(4)
            assert all(p.poll() is None for p in c.procs)
            c.meta['status']='PASS'
        except BaseException as exc:
            c.meta.update(status='FAIL',error=repr(exc));raise
        finally:
            try:c.close()
            finally:guard.close()
        assert not guard.error
        time.sleep(3);r.monitoring.idle(tag+'-after')


if __name__=='__main__':main()
