import json,re,sys
from pathlib import Path

def audit(path):
 d=json.loads((path/'run.json').read_text());info=next(iter(d.get('server_info',{}).values()),{})
 capacity=info.get('max_total_num_tokens');cfgpath=info.get('pdmux_config_path');cfg=d.get('pdmux_configs',{}).get(cfgpath,Path(cfgpath).read_text() if cfgpath else '');match=re.search(r'pp_prefetch_slots:\s*(\d+)',cfg);expected_slots=int(match.group(1)) if match else 2;text=(path/'server.log').read_text();idle={}
 for match in re.finditer(r'idle self-check PASS stage=(\d+) free_kv=(\d+) evictable=(\d+) free_reqs=(\d+).*?free_receive_slots=(\d+)',text):
  stage,free,evict,reqs,slots=map(int,match.groups());idle[stage]=dict(free=free,evictable=evict,reqs=reqs,slots=slots,conserved=free+evict==capacity and reqs==128 and slots==(0 if stage==0 else expected_slots))
 errors=[l for l in text.splitlines() if any(k in l for k in ['Scheduler hit an exception','torch.OutOfMemoryError','illegal memory access','AssertionError'])]
 result=dict(run=str(path),status=d.get('status'),capacity=capacity,idle=idle,errors=errors,all_conserved=bool(idle) and all(x['conserved'] for x in idle.values()))
 (path/'p10b-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
 return result
if __name__=='__main__':
 for p in sys.argv[1:]:audit(Path(p))
