"""Four-GPU transport/graph-buffer integrity smoke; no trace or model load."""
import json, os, pathlib, subprocess, tempfile, time
from types import SimpleNamespace
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from sglang.srt.multiplex.pp_decode import AsyncDecodeTransport


def worker(rank, path):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device=torch.device('cuda', rank)
    dist.init_process_group('gloo',init_method='file://'+path+'/init',rank=rank,world_size=4)
    nccl=dist.new_group(backend='nccl')
    group=SimpleNamespace(rank_in_group=rank,world_size=4,ranks=list(range(4)),device=device,cpu_group=dist.group.WORLD,device_group=nccl)
    from sgl_kernel import spatial
    _, compute=spatial.create_greenctx_stream_by_value(96,32,rank)
    graph_input=torch.empty((2,32,4096),device=device,dtype=torch.bfloat16)
    static_output=torch.empty_like(graph_input)
    # One graph output buffer is deliberately reused for every microbatch.
    with torch.cuda.stream(compute):
        graph_input.fill_(0)
        for _ in range(3):static_output.copy_(graph_input+rank*10)
    compute.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=compute):
        static_output.copy_(graph_input+rank*10)
    dist.barrier()
    transport=AsyncDecodeTransport(group,32,4096,torch.bfloat16,device)
    dist.barrier()
    flight=[None]*4;returns=0
    for step in range(100):
        mb=step%4;bs=0 if step>=96 or step%7==0 else [1,4,8,16,32][step%5]
        if rank==0:
            retired=transport.receive_tokens(flight[mb],mb) if flight[mb] is not None else None
            packet=SimpleNamespace(step=step,mb_id=mb,decode_bs=bs,retired=retired)
        else:
            packet,received=transport.receive_control()
            assert (packet.step,packet.mb_id,packet.decode_bs)==(step,mb,bs)
        if packet.retired:
            old=flight[mb]
            assert packet.retired['step']==old
            assert packet.retired['tokens']==[old+60]*[1,4,8,16,32][old%5],(rank,step,packet.retired)
            returns+=1
        else:assert flight[mb] is None
        if rank<3:transport.send_control(packet)
        if bs:
            with torch.cuda.stream(compute):
                if rank==0:graph_input.fill_(step)
                else:
                    received.wait()
                    graph_input[0,:bs].copy_(received.payload['hidden_states'])
                    graph_input[1,:bs].copy_(received.payload['residual'])
                graph.replay()
                if rank<3:transport.send_proxy(mb,dict(hidden_states=static_output[0,:bs],residual=static_output[1,:bs]))
                else:transport.send_tokens(step,mb,static_output[0,:bs,0].to(torch.int64),True)
                if rank:transport.release_proxy(mb)
        flight[mb]=step if bs else None
        if step%11==rank:time.sleep(0.001)
    assert all(v is None for v in flight)
    assert returns==82
    compute.synchronize()
    dist.barrier()
    pathlib.Path(path,f'{rank}.json').write_text(json.dumps(dict(rank=rank,returns=returns,status='PASS')))
    os._exit(0)


if __name__=='__main__':
    root=pathlib.Path('/root/workspace/muxpp_work')
    assert (root/'logs/p5a.gpu_done').exists() or (root/'logs/p5a.rc').exists()
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    path=root/'logs/p5b/transport-gpu';path.mkdir(exist_ok=False)
    procs=[mp.get_context('spawn').Process(target=worker,args=(rank,str(path))) for rank in range(4)]
    try:
        for p in procs:p.start()
        deadline=time.monotonic()+60
        for p in procs:
            p.join(max(0,deadline-time.monotonic()))
            assert p.exitcode==0,(p.pid,p.exitcode)
        print('PASS: 4 ranks, 100 steps, 82 results, green streams and shared CUDA graph outputs',flush=True)
    finally:
        for p in procs:
            if p.is_alive():p.kill()
            p.join(5)
        (path/'gpu-cleanup.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
