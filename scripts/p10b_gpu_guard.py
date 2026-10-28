#!/usr/bin/env python3
"""Stop an owned P10B runner on confirmed unaccounted GPU occupancy."""
import argparse,json,os,signal,time
from pathlib import Path
import psutil,pynvml

def watch(root_pid,evidence):
    pynvml.nvmlInit();hits=0;last=[]
    while psutil.pid_exists(root_pid):
        try:root=psutil.Process(root_pid);owned=[root,*root.children(recursive=True)]
        except psutil.NoSuchProcess:break
        pids={p.pid for p in owned};rows=[];unexpected=False
        for i in range(pynvml.nvmlDeviceGetCount()):
            h=pynvml.nvmlDeviceGetHandleByIndex(i)
            used=pynvml.nvmlDeviceGetMemoryInfo(h).used
            apps=pynvml.nvmlDeviceGetComputeRunningProcesses(h)
            accounted=sum(a.usedGpuMemory for a in apps if a.usedGpuMemory is not None and a.usedGpuMemory < 2**60)
            foreign=[a.pid for a in apps if a.pid not in pids]
            util=pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            hidden=used-accounted>512*2**20
            unexpected|=bool(foreign or hidden)
            rows.append(dict(gpu=i,used_mib=used/2**20,accounted_mib=accounted/2**20,util=util,compute_pids=[a.pid for a in apps],foreign=foreign,hidden=hidden))
        hits=hits+1 if unexpected else 0
        last.append(dict(at=time.time(),gpu=rows));last=last[-4:]
        if hits>=3:
            Path(evidence).write_text(json.dumps(dict(root_pid=root_pid,samples=last,action='SIGINT owned runner; no foreign process or GPU reset'),indent=2)+'\n')
            stopped=False
            for p in owned:
                try:
                    cmd=p.cmdline()
                    if any(Path(v).name in ['p10b_runner.py','p9a_runner.py','p9a_verify.py'] for v in cmd[1:]):
                        p.send_signal(signal.SIGINT);stopped=True
                except psutil.NoSuchProcess:pass
            if not stopped:root.send_signal(signal.SIGTERM)
            return
        time.sleep(2)
    Path(str(evidence)+'.clean').write_text(json.dumps({'ended':time.time(),'root_pid':root_pid,'last_samples':last},indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=int,required=True);p.add_argument('--evidence',required=True);a=p.parse_args();watch(a.root,a.evidence)
