#!/usr/bin/env python3
"""Requested Llama v4 regression, using its original P9a development tokenizer/trace."""
import json,subprocess,sys,os
from pathlib import Path
from p9a_serve import commands
from p10b_compare import compare
root=Path('/root/workspace/muxpp_work');log=root/'logs/p10b';py='/root/venvs/sgl/bin/python'
raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
assert all(int(row.split(',')[0])<100 and int(row.split(',')[1])<5 for row in raw.splitlines()), 'GPU occupied: stop without waiting'
with (log/'llama-gpu-guard.log').open('x') as f:
 guard=subprocess.Popen([py,str(root/'scripts/p10b_gpu_guard.py'),'--root',str(os.getpid()),'--evidence',str(log/'llama-external-gpu.json')],stdout=f,stderr=subprocess.STDOUT)
_,argv,env,_=commands('pp2tp2')[0]
cmd=[py,str(root/'scripts/p9a_verify.py'),'muxpp','--suite','correctness','--tp','2','--pp','2','--tag','p10b-llama-serial20','--extra']+argv[3:]
with (log/'llama-serial20-driver.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
a=json.loads((root/'logs/p9a/regressions/tp2-reference-serial-result.json').read_text())['outputs']
b=json.loads((root/'logs/p9a/regressions/p10b-llama-serial20-result.json').read_text())['outputs']
(log/'llama-reference-serial20.json').write_text(json.dumps(a));(log/'llama-serial20.json').write_text(json.dumps(b))
result=compare(log/'llama-reference-serial20.json',log/'llama-serial20.json')
(log/'llama-serial20-comparison.json').write_text(json.dumps(result,indent=2))
print('llama serial20',result['equal_requests'],result['equal_tokens'],flush=True)
assert result['equal_requests']==20
flags=json.loads((root/'configs/p9a_selected.json').read_text())['flags']
cmd=[py,str(root/'scripts/p9a_runner.py'),'--topology','pp2tp2','--trace','toolagent','--tag','p10b-llama-dev300','--extra','--pdmux-config-path',str(root/'configs/pdmux_pp_4090_v4.yml')]+flags
with (log/'llama-dev300-driver.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
(log/'llama-dev300').symlink_to('../p9a/p10b-llama-dev300',target_is_directory=True)
subprocess.run([py,str(root/'scripts/p9a_analyze.py'),str(log/'llama-dev300')],check=True)
