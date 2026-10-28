"""Actual TP2 Llama MLP: chunk collective count, BF16 error, peak, and latency."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
from sglang.srt.distributed.parallel_state import (
    init_distributed_environment, initialize_model_parallel, set_custom_all_reduce,
    set_pdmux_status, get_tp_group,
)
from sglang.srt.models.llama import LlamaMLP
from sglang.srt.model_executor.forward_batch_info import ForwardMode

rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
set_custom_all_reduce(False)
init_distributed_environment(2, rank, local_rank=rank)
initialize_model_parallel(2, duplicate_tp_group=True)
set_pdmux_status(True)
torch.set_default_dtype(torch.bfloat16)
with torch.device('cuda'):
    mlp = LlamaMLP(4096, 14336, 'silu')
with torch.no_grad():
    torch.manual_seed(7300+rank)
    for parameter in mlp.parameters():
        parameter.normal_(std=.01)
    torch.manual_seed(7301)
    x = torch.randn(31663, 4096, device='cuda')
    batch = SimpleNamespace(forward_mode=ForwardMode.SPLIT_PREFILL)
    comm = get_tp_group()
    original = comm.all_reduce
    counter = [0]
    def counted(tensor, *args, **kwargs):
        counter[0] += 1
        return original(tensor, *args, **kwargs)
    comm.all_reduce = counted
    ref = mlp(x, batch)
    rows = []
    for chunk in (0, 8192, 4096, 2048):
        mlp.prefill_chunk_tokens = chunk
        for _ in range(2):
            output = mlp(x, batch)
        torch.cuda.synchronize()
        del output
        torch.cuda.empty_cache()
        counter[0] = 0
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        dist.barrier(group=comm.cpu_group)
        start.record()
        for _ in range(3):
            output = None
            output = mlp(x, batch)
        end.record()
        end.synchronize()
        peak = torch.cuda.max_memory_allocated()-baseline
        error = (ref.float()-output.float()).abs()
        rows.append(dict(chunk=chunk, allreduces_per_forward=counter[0]/3,
                         elapsed_ms=start.elapsed_time(end)/3, incremental_peak_mib=peak/2**20,
                         max_abs=error.max().item(), mean_abs=error.mean().item()))
        assert counter[0] == 3 * (1 if not chunk else (len(x)+chunk-1)//chunk)
        # Record BF16 shape sensitivity rather than treating an arbitrary
        # random-weight tolerance as an end-to-end correctness criterion.
        close = torch.isclose(output, ref, rtol=.02, atol=.002)
        rows[-1].update(within_p5b_tolerance=bool(close.all().item()),
                        mismatched_elements=int((~close).sum().item()), elements=output.numel(),
                        bitwise_equal=bool((output == ref).all().item()))
        Path(f'/root/workspace/muxpp_work/logs/p8a/mlp-tp{rank}.json').write_text(json.dumps(rows, indent=2))
        del close
        del output, error
    Path(f'/root/workspace/muxpp_work/logs/p8a/mlp-tp{rank}.json').write_text(json.dumps(rows, indent=2))
    dist.barrier(group=comm.cpu_group)
dist.destroy_process_group()
