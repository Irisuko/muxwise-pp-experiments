import json,subprocess,time
from pathlib import Path
root=Path('/root/workspace/muxpp_work');logs=root/'logs/p10b';py='/root/venvs/sgl/bin/python'
cfg=json.loads((root/'configs/p10b_candidate_r2.json').read_text())
while not (logs/'r2d-dev-pp2tp2/gpu-cleanup.txt').exists():
    if not Path('/proc/43331').exists():raise RuntimeError('R2 queue ended before completing both development runs')
    time.sleep(5)
for topo in ['pp4','pp2tp2']:
    assert json.loads((logs/f'r2d-dev-{topo}/run.json').read_text())['status']=='PASS'
    c=cfg['big_muxpp_'+topo];tag='r2d-profile-'+topo
    cmd=[py,str(root/'scripts/p10b_runner.py'),'profile','big_muxpp_'+topo,'--tag',tag,'--mem',str(c['mem_fraction'])]
    for k,v in {**c['env'],'SGLANG_PDMUX_PP_MEMORY_TRACE':str(logs/tag/'memory')}.items():cmd+=['--env',k+'='+v]
    cmd+=['--extra']+c['flags']
    with (logs/(tag+'-driver.log')).open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    subprocess.run([py,str(root/'scripts/p10b_memory_report.py'),str(logs/tag)],check=True)
subprocess.run([py,str(root/'scripts/p10b_llama_regression.py')],check=True)
