#!/usr/bin/env python3
"""Development-only runner. Never reads or replays the formal trace."""
import argparse, datetime, json, os, pathlib, subprocess, time, shutil, hashlib
import p3_runner as base
ROOT=base.ROOT
TRACE=ROOT/'bench/dev_toolagent_1200_240_x2.5.jsonl'
p=argparse.ArgumentParser()
p.add_argument('label', choices=['muxwise_pp'])
p.add_argument('--tag', required=True)
p.add_argument('--baseline', action='store_true')
p.add_argument('--profile-v1', action='store_true')
p.add_argument('--limit', type=int, default=300)
p.add_argument('--time-scale', type=float, default=1)
p.add_argument('--extra', nargs=argparse.REMAINDER, default=[])
a=p.parse_args()
assert a.limit == 300 and a.time_scale == 1, 'Performance rounds fixed to dev 300/time-scale=1'
assert not a.baseline and not a.profile_v1, 'Do not rerun v1'
assert (ROOT/'logs/p5a.gpu_done').exists() or (ROOT/'logs/p5a.rc').exists(), 'P5a has not finished'
base.clean_gpu()
original=base.commands
def commands(label):
    jobs=original(label)
    for name,cmd,env,port in jobs:
        if name!='router':
            cmd.extend(a.extra)
            env["SGLANG_PDMUX_PP_ROUND_TRACE"]=str(ROOT/"logs/p5b"/a.tag/"decode-rounds.jsonl")
            if os.getenv('P5B_SOURCE'): env['PYTHONPATH']=os.environ['P5B_SOURCE']+'/python'
            if a.profile_v1:
                env['SGLANG_PDMUX_PP_TIMING_TRACE']=str(ROOT/'logs/p5b'/a.tag/'prefill-timing')
            if a.baseline or a.profile_v1:
                env['PYTHONPATH']='/root/workspace/sglang-p4b-'+('v1' if a.baseline else 'profile')+'/python'
    return jobs
base.commands=commands
path=ROOT/'logs/p5b'/a.tag
cluster=base.Cluster(a.label,path)
source=os.getenv('P5B_SOURCE', '/root/workspace/sglang')
cluster.meta['source']=source
cluster.meta['commit']=subprocess.check_output(['git','-C',source,'rev-parse','HEAD'],text=True).strip()
cluster.meta['dirty']=subprocess.check_output(['git','-C',source,'status','--short'],text=True)
cluster.meta['development_only']=True
try:
    cluster.start()
    (path/'source.patch').write_bytes(subprocess.check_output(['git','-C',source,'diff','--binary']))
    new_profile=pathlib.Path(source)/'python/sglang/srt/multiplex/pp_profile.py'
    if new_profile.exists(): shutil.copy2(new_profile,path/'pp_profile.py.snapshot')
    for name,cmd,env,port in commands(a.label):
        for i,arg in enumerate(cmd):
            if arg=='--pdmux-config-path':
                config=pathlib.Path(cmd[i+1]);shutil.copy2(config,path/(name+'-config.yml'))
    runtime_paths=['multiplex/pp_admission.py','multiplex/pp_decode.py','multiplex/pp_policy.py','multiplex/pp_timing.py','multiplex/multiplexing_pp.py','multiplex/pp_profile.py','managers/tp_worker.py','models/llama.py','model_executor/cuda_graph_runner.py']
    cluster.meta['source_sha256']={v:hashlib.sha256((pathlib.Path(source)/'python/sglang/srt'/v).read_bytes()).hexdigest() for v in runtime_paths if (pathlib.Path(source)/'python/sglang/srt'/v).exists()}
    cluster.meta["gpu_ready"]=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,memory.free","--format=csv"],text=True)
    base.save(path/"run.json",cluster.meta)
    # Exact P3 long warmup; no trace requests used for startup selection.
    import random
    rng=random.Random(730031)
    ids=[128000]+[rng.randrange(128000) for _ in range(31662)]
    urls=[base.BASE] if a.label=='muxwise_pp' else [f'http://127.0.0.1:{31000+i}' for i in range(4)]
    base.save(path/'long-warmup.json',[base.generate(url,ids) for url in urls])
    base.flush(cluster)
    if a.label=='chunked': cluster.reset_router()
    cmd=[base.PYTHON,str(ROOT/'bench/bench_client.py'),'--trace',str(TRACE),'--base-url',base.BASE,'--label',a.tag,'--out-dir',str(path),'--warmup','2','--force-close','--time-scale',str(a.time_scale),'--progress-interval','15']
    if a.limit: cmd+=['--limit',str(a.limit)]
    base.save(path/'dev-command.json',cmd)
    with (path/'client.log').open('w') as f:
        subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1800)
    summary=json.loads((path/f'{a.tag}_summary.json').read_text())
    cluster.meta['summary']=summary
    assert summary['failed_requests']==0 and summary['unsent_requests']==0, summary
    time.sleep(3)
    cluster.meta['status']='PASS'
    print(json.dumps({k:summary[k] for k in ['completed_requests','duration_sec','latencies_sec','throughput','slo']}),flush=True)
except BaseException as e:
    cluster.meta.update(status='FAIL',error=repr(e));raise
finally:
    cluster.close()
