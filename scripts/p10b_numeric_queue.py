import subprocess,time
from pathlib import Path
root=Path('/root/workspace/muxpp_work')
while not (root/'logs/p10b/r0-memory-pp2tp2/gpu-cleanup.txt').exists():time.sleep(2)
for label,tag,chunk in [('big_plain_pp4','numeric-plain-pp4',None),('big_muxpp_pp4','numeric-mux-pp4-4096',4096),('big_muxpp_pp4','numeric-mux-pp4-0',0),('big_muxpp_pp4','numeric-mux-pp4-1024',1024)]:
 cmd=['/root/venvs/sgl/bin/python',str(root/'scripts/p10b_runner.py'),'numeric',label,'--mem','.76','--tag',tag,'--env',f'SGLANG_PDMUX_PP_NUMERIC_TRACE={root}/logs/p10b/{tag}/numeric']
 if chunk is not None:cmd+=['--extra','--pdmux-prefill-mlp-chunk-tokens',str(chunk)]
 with (root/f'logs/p10b/{tag}-driver.log').open('x') as f:rc=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=root)
 print(tag,rc,flush=True)
