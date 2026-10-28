import json,subprocess,time
from pathlib import Path
root=Path('/root/workspace/muxpp_work');log=root/'logs/p10b';python='/root/venvs/sgl/bin/python'
while not (log/'numeric-mux-pp4-1024/gpu-cleanup.txt').exists():time.sleep(2)
def run(mode,label,tag,mem,env,flags):
 cmd=[python,str(root/'scripts/p10b_runner.py'),mode,label,'--mem',str(mem),'--tag',tag,'--smoke-seconds','120']
 for k,v in env.items():cmd+=['--env',k+'='+v]
 cmd+=['--extra']+flags
 with (log/f'{tag}-driver.log').open('x') as f:rc=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=root)
 print(tag,rc,flush=True);return rc
assert run('numeric','big_muxpp_pp4','numeric-mux-pp4-pad',.76,{'SGLANG_PDMUX_PP_NUMERIC_TRACE':str(log/'numeric-mux-pp4-pad/numeric')},['--pdmux-prefill-mlp-pad-to-chunk'])==0
ref=json.loads((log/'numeric-plain-pp4/numeric-output.json').read_text())['response']['output_ids']
actual=json.loads((log/'numeric-mux-pp4-pad/numeric-output.json').read_text())['response']['output_ids']
assert ref==actual,(ref,actual)
cfg=json.loads((root/'configs/p10b_candidate_r1.json').read_text())
for topo in ['pp4','pp2tp2']:
 label='big_muxpp_'+topo
 assert run('correctness',label,'r1-mux-'+topo,cfg[label]['mem_fraction'],cfg['env'],cfg['flags'])==0
for topo in ['pp4','pp2tp2']:
 label='big_muxpp_'+topo
 assert run('preflight',label,'r1-preflight-'+topo,cfg[label]['mem_fraction'],cfg['env'],cfg['flags'])==0
