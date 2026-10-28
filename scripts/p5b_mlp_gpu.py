"""BF16 SwiGLU chunk math and allocated-intermediate peak check; no trace."""
import json,pathlib,subprocess
from types import SimpleNamespace
import torch
from torch import nn
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.models.llama import LlamaMLP
from sglang.srt.model_executor.forward_batch_info import ForwardMode

class LinearPair(nn.Linear):
    def forward(self,x,**kwargs):return super().forward(x),None

root=pathlib.Path('/root/workspace/muxpp_work')
assert (root/'logs/p5a.gpu_done').exists() or (root/'logs/p5a.rc').exists()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
torch.set_num_threads(8);torch.manual_seed(51)
with torch.inference_mode():
    mlp=LlamaMLP.__new__(LlamaMLP);nn.Module.__init__(mlp)
    mlp.gate_up_proj=LinearPair(4096,28672,bias=False,device='cuda',dtype=torch.bfloat16)
    mlp.down_proj=LinearPair(14336,4096,bias=False,device='cuda',dtype=torch.bfloat16)
    mlp.act_fn=SiluAndMul();mlp.prefill_chunk_tokens=0
    x=torch.randn((8193,4096),dtype=torch.bfloat16,device='cuda')
    batch=SimpleNamespace(forward_mode=ForwardMode.SPLIT_PREFILL)
    torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    expected=mlp(x,batch);torch.cuda.synchronize();full_peak=torch.cuda.max_memory_allocated()-base
    base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    mlp.prefill_chunk_tokens=4096
    actual=mlp(x,batch);torch.cuda.synchronize();chunk_peak=torch.cuda.max_memory_allocated()-base
    diff=(actual.float()-expected.float()).abs()
    result=dict(shape=list(x.shape),chunk_tokens=4096,dtype=str(x.dtype),full_incremental_peak_MiB=full_peak/2**20,chunk_incremental_peak_MiB=chunk_peak/2**20,max_abs_error=diff.max().item(),mean_abs_error=diff.mean().item(),bitwise_equal=torch.equal(expected,actual),elements=actual.numel())
    torch.testing.assert_close(actual,expected,rtol=0.02,atol=0.002)
    result['status']='PASS'
    (root/'logs/p5b/mlp-gpu.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
