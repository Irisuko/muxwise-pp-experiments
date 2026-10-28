#!/usr/bin/env python3
"""P10c orchestration only; frozen inference, client, traces and selection."""
import argparse, datetime, fcntl, hashlib, json, os, shlex, signal, subprocess, threading, time
from pathlib import Path
import psutil, pynvml
import p3_runner as b
import p7a_runner as p7
import p9a_runner as p9
import p10a_runner as a
import p10b_serve as selected
from p9b_runner import mapping

ROOT=b.ROOT
OUT=ROOT/'logs/p10c'
SOURCE=Path('/root/workspace/sglang')
OLD=Path('/root/workspace/sglang-p9a-v4')
HEAD='4c10135d2e2891d6fb262d4894f0769c86dd5751'
OLDHEAD='f6833f24ef0587278ae799696cfe17c3ade99048'
LABELS=['big_muxwise_tp4','big_chunked_tp4','big_chunked_pp4','big_chunked_pp2tp2','big_muxpp_pp4','big_muxpp_pp2tp2']
FORMAL=ROOT/'bench/big_toolagent_600_80_x7.5.jsonl'
LOCK=ROOT/'configs/p10a_formal.lock.json'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(p,*args):return subprocess.check_output(['git','-C',str(p),*args],text=True).strip()
def source_gate():
    for p,h in [(SOURCE,HEAD),(OLD,OLDHEAD)]:
        assert git(p,'rev-parse','HEAD')==h and not git(p,'status','--porcelain'),str(p)
    assert not [k for k in os.environ if k.startswith(('P3_','SGLANG_PDMUX_PP_','P9A_SOURCE','P10A_STACKS'))]

def snapshot():
    rows=[]
    for i in range(pynvml.nvmlDeviceGetCount()):
        h=pynvml.nvmlDeviceGetHandleByIndex(i)
        apps=pynvml.nvmlDeviceGetComputeRunningProcesses(h)
        memory=pynvml.nvmlDeviceGetMemoryInfo(h,version=pynvml.nvmlMemory_v2)
        rows.append(dict(index=i,uuid=pynvml.nvmlDeviceGetUUID(h),memory_mib=memory.used/2**20,reserved_mib=memory.reserved/2**20,
            utilization=pynvml.nvmlDeviceGetUtilizationRates(h).gpu,
            processes=[dict(pid=x.pid,memory_mib=x.usedGpuMemory/2**20 if x.usedGpuMemory and x.usedGpuMemory<2**60 else 0,
                            visible=psutil.pid_exists(x.pid)) for x in apps]))
    return dict(utc=now(),gpu=rows)

def idle(tag):
    s=snapshot()
    residual=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        cmd=p.info['cmdline'] or []
        if any(v in ['sglang.launch_server','sglang_router.launch_router'] or Path(v).name=='bench_client.py' for v in cmd):
            residual.append(p.info)
        elif (p.info['name'] or '').startswith(('sglang::','SGLang')):residual.append(p.info)
    s['residual_processes']=residual
    b.save(OUT/(tag+'-idle.json'),s)
    if residual or len(s['gpu'])!=4 or any(x['processes'] or x['memory_mib']>4 or x['utilization']!=0 for x in s['gpu']):
        b.save(OUT/'STOP-gpu-occupied.json',s)
        raise RuntimeError('GPU not idle; stop immediately. See STOP-gpu-occupied.json')
    return s

class Guard:
    def __init__(self,tag):self.tag=tag;self.done=threading.Event();self.error=None
    def start(self):self.t=threading.Thread(target=self.run,daemon=True);self.t.start()
    def close(self):self.done.set();self.t.join(timeout=4)
    def run(self):
        hidden_count=0
        while not self.done.wait(2):
            s=snapshot();root=psutil.Process(os.getpid());owned={root.pid}|{p.pid for p in root.children(recursive=True)}
            foreign=[p for g in s['gpu'] for p in g['processes'] if p['pid'] not in owned]
            hidden=any(g['memory_mib']-sum(p['memory_mib'] for p in g['processes'])>512 for g in s['gpu'])
            hidden_count=hidden_count+1 if hidden else 0
            if foreign or hidden_count>=2:
                self.error=dict(tag=self.tag,snapshot=s,foreign=foreign,hidden=hidden,action='Interrupt owned runner and stop sequence; do not kill foreign jobs')
                b.save(OUT/'STOP-external-gpu.json',self.error)
                os.kill(os.getpid(),signal.SIGINT);return

def freeze():
    source_gate();idle('initial')
    paths=[ROOT/'env.sh',*sorted((ROOT/'configs').glob('*.json')),*sorted((ROOT/'configs').glob('*.yml')),
           *sorted((ROOT/'scripts').glob('*.py')),*sorted((ROOT/'scripts').glob('*.sh')),
           *sorted((ROOT/'bench').glob('*.py')),*sorted((ROOT/'bench').glob('*.jsonl'))]
    jobs={label:[dict(name=n,argv=c,env=e,port=p) for n,c,e,p in selected.commands(label)] for label in LABELS}
    # Check against the six exact expanded commands in P10B, not a new selection.
    notes=(ROOT/'P10B_NOTES.md').read_text()
    for label,rows in jobs.items():
        cmd=shlex.split(notes.split('### '+label+'\n',1)[1].split('```bash\n',1)[1].split('\n```',1)[0])
        row=rows[0]
        assert cmd==['env']+[k+'='+v for k,v in row['env'].items()]+row['argv'],label
    value=dict(created=now(),commit=HEAD,old_commit=OLDHEAD,gpu_mapping=mapping(),labels=LABELS,commands=jobs,
               protected_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
               authority='User P10c request explicitly authorizes six one-shot formal runs; P10a preparation lock temporarily archived and restored.')
    with (OUT/'frozen.json').open('x') as f:json.dump(value,f,indent=2)
    (OUT/'RESULTS-before-P10c.md').write_bytes((ROOT/'RESULTS.md').read_bytes())
    (OUT/'topology.txt').write_text(subprocess.check_output(['nvidia-smi','topo','-m'],text=True))
    b.save(OUT/'gpu-mapping.json',value['gpu_mapping'])
    b.save(OUT/'historical-result-sha256.json',{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'results').glob('*') if p.is_file()})
    return value

def gate(frozen):
    source_gate();assert mapping()==frozen['gpu_mapping']
    for rel,digest in frozen['protected_sha256'].items():
        p=ROOT/rel
        if p==LOCK and not p.exists():p=OUT/'p10a_formal.lock.json'
        assert sha(p)==digest,rel

def llama(which,frozen):
    gate(frozen);idle('llama-'+which+'-before')
    source=OLD if which=='old' else SOURCE
    tag='p10c-llama-'+which
    path=ROOT/'logs/p9a'/tag
    assert not path.exists()
    flags=json.loads((ROOT/'configs/p9a_selected.json').read_text())['flags']
    cmd=[b.PYTHON,str(ROOT/'scripts/p9a_runner.py'),'--topology','pp2tp2','--trace','toolagent','--tag',tag,
         '--limit','300','--time-scale','1','--extra','--pdmux-config-path',str(ROOT/'configs/pdmux_pp_4090_v4.yml'),*flags]
    b.save(OUT/('llama-'+which+'-launch.json'),dict(command=cmd,source=str(source),started=now(),gpu_mapping=mapping()))
    guard=Guard(tag);guard.start()
    print('START',tag,now(),flush=True)
    with (OUT/(tag+'-runner.log')).open('x') as f:
        proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,P9A_SOURCE=str(source)))
        try:rc=proc.wait()
        except BaseException:
            proc.send_signal(signal.SIGINT)
            try:proc.wait(timeout=45)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            raise
        finally:guard.close()
    (OUT/('llama-'+which)).symlink_to(path,target_is_directory=True)
    assert not guard.error
    assert rc==0,(tag,rc)
    s=json.loads((path/(tag+'_summary.json')).read_text())
    print('END',tag,s['completed'],s['slo']['ttft_le_5s_and_tpot_le_100ms'],flush=True)
    time.sleep(3);idle('llama-'+which+'-after')

def replay(c,formal):
    label=c.label if formal else c.label+'_smoke'
    out=ROOT/'results' if formal else c.path/'smoke'
    cmd=[b.PYTHON,str(ROOT/'bench/bench_client.py'),'--trace',str(FORMAL if formal else a.DEV),
         '--base-url',b.BASE,'--label',label,'--out-dir',str(out),'--warmup','2','--force-close','--drain-timeout','300','--progress-interval','30']
    if not formal:cmd+=['--limit','20','--time-scale','0.2']
    kind='formal' if formal else 'smoke'
    if formal:
        with (out/(label+'.formal-started.json')).open('x') as f:json.dump(dict(command=cmd,log_dir=str(c.path),started=now(),commit=HEAD),f,indent=2)
    b.save(c.path/(kind+'-command.json'),cmd)
    with (c.path/(kind+'-client.log')).open('x') as f:
        proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
        try:rc=proc.wait(timeout=1100 if formal else 400)
        except BaseException:
            proc.send_signal(signal.SIGINT)
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            raise
    s=json.loads((out/(label+'_summary.json')).read_text())
    c.meta[kind+'_summary']=s;c.meta[kind+'_returncode']=rc;b.save(c.path/'run.json',c.meta)
    if not formal:assert rc==0 and (s['completed'],s['unfinished'],s['failed'])==(20,0,0)
    else:assert s['planned_requests']==555
    print(kind,label,s['completed'],s['unfinished'],s['failed'],s['slo']['ttft_le_5s_and_tpot_le_100ms'],flush=True)

def experiment(label,frozen):
    gate(frozen);idle(label+'-before')
    assert not list((ROOT/'results').glob(label+'*'))
    with (OUT/(label+'.launch-started.json')).open('x') as f:json.dump(dict(label=label,started=now(),commit=HEAD),f)
    jobs=selected.commands(label);b.commands=lambda _:jobs
    c=p7.Cluster(label,OUT/label)
    c.meta.update(mode='experiment',commit=HEAD,source=str(SOURCE),gpu_mapping=mapping(),formal_launch_count=1,restart_count=0,
                  frozen_file=str(OUT/'frozen.json'),trace_sha256=sha(FORMAL))
    guard=Guard(label);guard.start()
    print('START',label,now(),flush=True)
    try:
        c.start();(c.path/'source.patch').write_text(git(SOURCE,'diff','HEAD','--binary'))
        a.warmup(c);replay(c,False)
        if 'mux' in label:
            b.flush(c)
            (c.path/'long-warmup.json').rename(c.path/'long-warmup-before-smoke.json')
            a.warmup(c)
            (c.path/'long-warmup.json').rename(c.path/'long-warmup-after-smoke.json')
        b.flush(c)
        (c.path/'gpu-rank-processes.csv').write_text(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True))
        replay(c,True)
        c.meta['status']='PASS' if c.meta['formal_returncode']==0 else 'FORMAL_ERRORS_WITH_DATA'
    except BaseException as e:c.meta.update(status='FAIL',error=repr(e));raise
    finally:
        try:c.close()
        finally:guard.close()
    assert not guard.error
    time.sleep(3);idle(label+'-after')
    print('END',label,now(),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','ab','formal']);x=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    pynvml.nvmlInit()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    with (ROOT/'logs/p7a/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if x.mode=='freeze':freeze();return
        frozen=json.loads((OUT/'frozen.json').read_text())
        if x.mode=='ab':
            llama('old',frozen);llama('new',frozen)
        else:
            gate(frozen);idle('formal-batch-before');a.ensure()
            assert LOCK.exists() and not (OUT/'p10a_formal.lock.json').exists()
            LOCK.rename(OUT/'p10a_formal.lock.json')
            b.save(OUT/'formal-authorization.json',dict(utc=now(),authorization=frozen['authority'],lock_sha256=sha(OUT/'p10a_formal.lock.json')))
            try:
                for label in LABELS:experiment(label,frozen)
            finally:(OUT/'p10a_formal.lock.json').rename(LOCK)
            b.save(OUT/'batch-finished.json',dict(utc=now(),labels=LABELS,restarts=0))
        idle(x.mode+'-final');gate(frozen)

if __name__=='__main__':main()
