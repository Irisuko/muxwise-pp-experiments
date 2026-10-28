#!/usr/bin/env python3
"""Select validated development candidates and write the six launch wrappers."""
import argparse,datetime,json,shlex,subprocess
from pathlib import Path
import p10a_runner as old
from p10b_compare import compare
from p10b_audit_run import audit
root=old.ROOT;log=root/'logs/p10b'
p=argparse.ArgumentParser(description=__doc__)
for topo in ['pp4','pp2tp2']:
 p.add_argument('--'+topo+'-config',type=Path,required=True)
 p.add_argument('--'+topo+'-prefix',required=True)
 p.add_argument('--'+topo+'-correctness-prefix')
 p.add_argument('--'+topo+'-preflight-prefix')
p.add_argument('--correctness-prefix',default='r2e')
a=p.parse_args();selected={}
for topo in ['pp4','pp2tp2']:
 label='big_muxpp_'+topo;prefix=getattr(a,topo+'_prefix');config=json.loads(getattr(a,topo+'_config').read_text())[label]
 preflight_prefix=getattr(a,topo+'_preflight_prefix') or prefix
 pf=log/f'{preflight_prefix}-preflight-{topo}';dev=log/f'{prefix}-dev-{topo}'
 pre=json.loads((pf/'run.json').read_text());d=json.loads((dev/'run.json').read_text());s=pre['smoke']
 assert pre['status']=='PASS' and s['completed']==20 and not s['failed'] and not s['unfinished']
 assert d['status']=='PASS' and d['sanity']['completed']==150 and not d['sanity']['failed'] and not d['sanity']['unfinished']
 resource=audit(pf);assert resource['all_conserved'] and not resource['errors']
 info=next(iter(pre['server_info'].values()));assert info['mem_fraction_static']==config['mem_fraction']
 flags=config['flags'];i=0
 while i<len(flags):
  key=flags[i];field=key[2:].replace('-','_');i+=1
  if i<len(flags) and not flags[i].startswith('--'):
   assert str(info[field])==flags[i],(field,info[field],flags[i]);i+=1
  else:assert info[field] is True,(field,info[field])
 correctness_prefix=getattr(a,topo+'_correctness_prefix') or a.correctness_prefix
 for kind in ['serial20','cold20']:
  result=compare(log/f'r0-plain-{topo}'/(kind+'.json'),log/f'{correctness_prefix}-mux-{topo}'/(kind+'.json'));assert not result['early']
 selected[label]={**config,'kv_capacity':resource['capacity'],'preflight':str(pf),'development':str(dev),'correctness_prefix':correctness_prefix}
assert json.loads((log/'llama-serial20-comparison.json').read_text())['equal_requests']==20
assert json.loads((log/'llama-dev300/run.json').read_text())['status']=='PASS'
selected['_meta']={'commit':subprocess.check_output(['git','-C',str(old.SOURCE),'rev-parse','HEAD'],text=True).strip(),'selected_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'formal_replayed':False,'llama_regression':str(log/'llama-dev300')}
assessment=log/'llama-regression-assessment.json'
if assessment.exists():
 selected['_meta']['llama_nonregression_status']=json.loads(assessment.read_text())['nonregression_status']
 selected['_meta']['llama_assessment']=str(assessment)
(root/'configs/p10b_selected.json').write_text(json.dumps(selected,indent=2)+'\n')
import p10b_serve as new
labels=[label for label in old.MATRIX if label!='big_plain_tp4']
for label in labels:
 if not label.startswith('big_muxpp_'):assert old.commands(label)==new.commands(label),label
 commands=new.commands(label)
 content='#!/usr/bin/env bash\nset -euo pipefail\nsource /root/workspace/muxpp_work/env.sh\nexec python /root/workspace/muxpp_work/scripts/p10b_serve.py launch '+label+' "$@"\n'
 script=root/'scripts'/('launch_'+label+'.sh');script.write_text(content);script.chmod(0o755)
 subprocess.run(['bash','-n',str(script)],check=True)
manifest={label:[dict(argv=cmd,env=env,command=shlex.join(['env']+[k+'='+v for k,v in env.items()]+cmd)) for _,cmd,env,_ in new.commands(label)] for label in labels}
(log/'final-launch-commands.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Selected six launch commands; four unchanged families verified byte-for-byte as argv/env.')
