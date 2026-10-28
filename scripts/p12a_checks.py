#!/usr/bin/env python3
"""P12a same-topology correctness, cancellation, stress and legacy checks."""
import argparse
import concurrent.futures
import fcntl
import json
from pathlib import Path
import signal
import time
import pynvml
import p12a_runner as r
import p11a_verify as verify
import p12a_audit as audit

b=r.r.b
ROOT=r.ROOT
OUT=r.OUT

def correctness(c,label,reference,reference_dir):
    previous_out=verify.OUT
    try:
        verify.OUT=c.path/'.generate-reference-only'
        verify.correctness(c,label)
    finally:verify.OUT=previous_out
    if reference:return
    for kind in ['serial20','cold20']:
        actual=json.loads((c.path/(kind+'.json')).read_text())
        expected=json.loads((reference_dir/(kind+'.json')).read_text())
        rows=[]
        for i,(a,z) in enumerate(zip(actual,expected)):
            same=[x==y for x,y in zip(a['output_ids'],z['output_ids'])]
            rows.append(dict(request=i,matched=sum(same),exact=all(same),first_mismatch=next((j for j,v in enumerate(same) if not v),None)))
        result=dict(exact_requests=sum(x['exact'] for x in rows),exact_tokens=sum(x['matched'] for x in rows),
                    total_requests=20,total_tokens=1280,rows=rows,reference=str(reference_dir/(kind+'.json')))
        b.save(c.path/(kind+'-comparison.json'),result)
        c.meta.setdefault('correctness',{})[kind]={k:v for k,v in result.items() if k!='rows'}
    c.meta['correctness_status']='EXACT' if all(v['exact_requests']==20 for v in c.meta['correctness'].values()) else 'NUMERICAL_DIFFERENCES'

def long_case(c,label,reference,reference_dir):
    base=verify.prompts(label)[19]
    ids=[base[0]]+(base[1:-64]*2)[:12288-65]+base[-64:]
    assert len(ids)==12288
    b.flush(c)
    response=b.generate(b.BASE,ids,64)['response']
    assert response['meta_info']['cached_tokens']==0 and len(response['output_ids'])==64
    b.save(c.path/'cold12288.json',dict(input_ids=ids,response=response))
    if not reference:
        ref=json.loads((reference_dir/'cold12288.json').read_text())
        assert ref['input_ids']==ids
        same=[a==z for a,z in zip(response['output_ids'],ref['response']['output_ids'])]
        comparison=dict(exact=all(same),matched=sum(same),first_mismatch=next((i for i,v in enumerate(same) if not v),None))
        b.save(c.path/'cold12288-comparison.json',comparison);c.meta['cold12288']=comparison
        assert comparison['exact'], comparison
    b.flush(c)

def experiment(item,config):
    config=Path(item.get('config',str(config)))
    label=item['label'];mode=item['mode'];tag=item['tag'];source=Path(item.get('source',str(r.r.MAIN)))
    kind=item.get('kind','plain');path=OUT/tag
    reference_dir=OUT/item.get('reference_tag','reference-'+label)
    environment=(json.loads(config.read_text()).get('environment',{}).get(label,{}) if mode=='verify' else {}) | item.get('environment',{})
    assert not list(OUT.glob('STOP-*.json'))
    assert not r.git(source,'status','--porcelain')
    r.monitor.idle(tag+'-before',seconds=10)
    if mode in ['reference','legacy']:
        jobs=verify.ordinary(label,kind)
    else:
        flags=json.loads(config.read_text())['flags'][label]+item.get('extra',[])
        jobs=r.r.commands(label,source,path,flags)
        for name,cmd,env,port in jobs:
            if name!='router':cmd+=['--skip-tokenizer-init']
    mapping=r.monitor.mapping()
    for name,cmd,env,port in jobs:
        if name=='router':continue
        env['PYTHONPATH']=str(source/'python')
        env.update(environment)
        env['CUDA_VISIBLE_DEVICES']=','.join(mapping[int(i)]['uuid'] for i in env['CUDA_VISIBLE_DEVICES'].split(','))
        env.update(NCCL_DEBUG='INFO',NCCL_DEBUG_FILE=str(path/'nccl-%h-%p.log'))
    b.commands=lambda _:jobs;b.clean_gpu=r.clean
    c=r.r.p7.Cluster(label,path)
    c.meta.update(mode=mode,kind=kind,source=str(source),commit=r.git(source,'rev-parse','HEAD'),gpu_mapping=mapping,
                  verifier_sha256=r.monitor.sha(Path(__file__)),
                  config=str(config) if mode=='verify' else None,
                  config_sha256=r.monitor.sha(config) if mode=='verify' else None,
                  environment=environment,reference=str(reference_dir) if mode=='verify' else None)
    guard=r.monitor.Guard(tag);guard.thread.start()
    print('START',tag,r.monitor.now(),flush=True)
    try:
        c.start()
        (path/'verifier.py').write_text(Path(__file__).read_text())
        if mode=='legacy':
            cases=verify.prompts('big_pp4' if label.startswith('big_') else 'llama_pp4')
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                outputs=list(pool.map(lambda ids:b.generate(b.BASE,ids,32),cases[::3]))
            b.save(path/'smoke.json',outputs);b.flush(c)
        else:
            correctness(c,label,mode=='reference',reference_dir)
            long_case(c,label,mode=='reference',reference_dir)
            if mode=='verify':
                gate={}
                for kind_name in ['serial20','cold20']:
                    actual=json.loads((path/(kind_name+'.json')).read_text())
                    expected=json.loads((reference_dir/(kind_name+'.json')).read_text())
                    old=ROOT/'logs/p11a'/('final-verify-'+label)/(kind_name+'.json')
                    previous=json.loads(old.read_text()) if label=='big_pp4' else expected
                    unexpected=[i for i,(a,z,o) in enumerate(zip(actual,expected,previous)) if a['output_ids'] not in [z['output_ids'],o['output_ids']]]
                    gate[kind_name]=dict(exact_plain=sum(a['output_ids']==z['output_ids'] for a,z in zip(actual,expected)),unexpected_cases=unexpected)
                b.save(path/'correctness-gate.json',gate)
                assert not any(v['unexpected_cases'] for v in gate.values()),gate
                verify.cancel(c,label)
                verify.stress(c,label)
        time.sleep(4)
        assert all(p.poll() is None for p in c.procs)
        c.meta['status']='PASS'
    except BaseException as exc:
        c.meta.update(status='FAIL',error=repr(exc));raise
    finally:
        try:c.close()
        finally:guard.close()
    assert not guard.error
    resource=audit.run_audit(path)
    assert not resource['errors'] and resource['all_conserved'] is not False, resource
    time.sleep(3);r.monitor.idle(tag+'-after',seconds=5)
    print('END',tag,r.monitor.now(),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue',type=Path,required=True)
    p.add_argument('--config',type=Path,default=r.SELECTED)
    x=p.parse_args()
    OUT.mkdir(exist_ok=True);r.monitor.OUT=OUT;verify.OUT=OUT;pynvml.nvmlInit()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    with (ROOT/'logs/p7a/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for item in json.loads(x.queue.read_text()):experiment(item,x.config)

if __name__=='__main__':main()
