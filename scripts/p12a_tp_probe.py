#!/usr/bin/env python3
"""Four-GPU TP2/PP2 transport probe on full and green streams; no model/trace."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import pynvml
import p12a_runner as r

def worker(out):
    import torch
    import torch.distributed as dist
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,initialize_model_parallel,set_custom_all_reduce,
        set_pdmux_status,get_tp_group)
    from sgl_kernel import spatial
    rank=int(os.environ['LOCAL_RANK']);torch.set_num_threads(1);torch.cuda.set_device(rank)
    set_custom_all_reduce(False)
    init_distributed_environment(4,rank,local_rank=rank)
    initialize_model_parallel(tensor_model_parallel_size=2,pipeline_model_parallel_size=2,duplicate_tp_group=True)
    set_pdmux_status(True);group=get_tp_group()
    rows=[]
    torch.manual_seed(730+rank)
    matrix=torch.randn(8192,4096,device='cuda',dtype=torch.bfloat16)
    torch.cuda.synchronize()
    full=group.all_reduce(matrix.clone())
    torch.cuda.synchronize()
    equivalence=[]
    for count in [1,32,511,2048,4096,8191]:
        cropped=group.all_reduce(matrix[:count].clone())
        torch.cuda.synchronize()
        exact=torch.equal(full[:count],cropped)
        equivalence.append(dict(rows=count,exact=exact))
        assert exact,(rank,count)
    (out/f'rank{rank}-crop-equivalence.json').write_text(json.dumps(equivalence,indent=2))
    del matrix,full,cropped
    for sm in [128,112,64]:
        if sm==128:stream=torch.cuda.Stream()
        else:stream,_=spatial.create_greenctx_stream_by_value(sm,128-sm,rank)
        for n in [1,32,512,2048,4096,8192,16384]:
            x=torch.full((n,4096),float(rank%2+1),device='cuda',dtype=torch.bfloat16)
            torch.cuda.synchronize();dist.barrier()
            with torch.cuda.stream(stream):
                y=group.all_reduce(x.clone())
            stream.synchronize()
            assert torch.equal(y,torch.full_like(y,3)),(rank,sm,n)
            with torch.cuda.stream(stream):
                for _ in range(3):group.all_reduce(x)
            stream.synchronize();times=[]
            for _ in range(5):
                dist.barrier(group=group.cpu_group)
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(stream):
                    start.record()
                    for _ in range(5):group.all_reduce(x)
                    end.record()
                end.synchronize();times.append(start.elapsed_time(end)/5)
            rows.append(dict(sm=sm,rows=n,bytes=x.numel()*x.element_size(),milliseconds=times))
            (out/f'rank{rank}.json').write_text(json.dumps(rows,indent=2))
        dist.barrier()
    cpu=[]
    flag=torch.ones(1,dtype=torch.int32)
    for _ in range(200):
        t=time.perf_counter();dist.all_reduce(flag,op=dist.ReduceOp.MIN,group=group.cpu_group)
        cpu.append(time.perf_counter()-t)
    (out/f'rank{rank}-cpu.json').write_text(json.dumps(cpu))
    dist.barrier();dist.destroy_process_group()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',type=Path)
    p.add_argument('--tag')
    p.add_argument('--memcpy',type=int,choices=[0,1],default=0)
    x=p.parse_args()
    if x.worker:worker(x.worker);return
    assert x.tag
    r.monitor.OUT=r.OUT;pynvml.nvmlInit()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    with (r.ROOT/'logs/p7a/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not list(r.OUT.glob('STOP-*.json'))
        r.monitor.idle(x.tag+'-before',seconds=10)
        out=r.OUT/x.tag;out.mkdir(exist_ok=False)
        env=dict(os.environ,NCCL_DEBUG='INFO',NCCL_DEBUG_FILE=str(out/'nccl-%h-%p.log'),
                 NCCL_SHM_USE_CUDA_MEMCPY=str(x.memcpy),PYTHONPATH=str(r.r.MAIN/'python'))
        cmd=[r.r.b.PYTHON,'-m','torch.distributed.run','--standalone','--nproc-per-node=4',str(Path(__file__).resolve()),'--worker',str(out)]
        r.r.b.save(out/'command.json',dict(argv=cmd,env={k:env[k] for k in ['NCCL_DEBUG','NCCL_DEBUG_FILE','NCCL_SHM_USE_CUDA_MEMCPY','PYTHONPATH']}))
        guard=r.monitor.Guard(x.tag);guard.thread.start();proc=None
        try:
            with (out/'probe.log').open('x') as log:
                proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                assert proc.wait(timeout=180)==0
        finally:
            if proc and proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            guard.close()
        assert not guard.error
        time.sleep(3);r.monitor.idle(x.tag+'-after',seconds=5)

if __name__=='__main__':main()
