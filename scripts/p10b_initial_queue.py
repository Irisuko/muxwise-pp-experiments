import subprocess,time,json
from pathlib import Path
root=Path('/root/workspace/muxpp_work')
while not (root/'logs/p10b/r0-plain-pp4/gpu-cleanup.txt').exists():time.sleep(2)
jobs=[('correctness','big_plain_pp2tp2','r0-plain-pp2tp2','.78',[]),('correctness','big_muxpp_pp4','r0-mux-pp4','.76',[]),('correctness','big_muxpp_pp2tp2','r0-mux-pp2tp2','.78',[]),('preflight','big_muxpp_pp4','r0-memory-pp4','.76',['--env',f'SGLANG_PDMUX_PP_MEMORY_TRACE={root}/logs/p10b/r0-memory-pp4/memory']),('preflight','big_muxpp_pp2tp2','r0-memory-pp2tp2','.78',['--env',f'SGLANG_PDMUX_PP_MEMORY_TRACE={root}/logs/p10b/r0-memory-pp2tp2/memory'])]
for mode,label,tag,mem,extra in jobs:
 cmd=['/root/venvs/sgl/bin/python',str(root/'scripts/p10b_runner.py'),mode,label,'--mem',mem,'--tag',tag]+extra
 with (root/f'logs/p10b/{tag}-driver.log').open('x') as f:rc=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=root)
 print(tag,rc,flush=True)
