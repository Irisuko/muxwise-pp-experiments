#!/usr/bin/env python3
import ast,collections,json,re,sys
from pathlib import Path
import numpy as np

def diagnose(path):
 d=json.loads((path/'analysis.json').read_text());s=d['summary'];start=s['run_started_unix'];end=start+s['duration_sec'];text=(path/'server.log').read_text()
 r={k:d[k] for k in ['matched','quantile_decomposition','component_distributions','wait_distributions','cuda_per_request','arrival','full'] if k in d}
 r['metrics']={'ttft':s['latencies_sec']['ttft'],'tpot':s['latencies_sec']['tpot'],'slo':s['slo']['ttft_le_5s_and_tpot_le_100ms']}
 rounds=[json.loads(x) for x in (path/'decode-rounds.jsonl').read_text().splitlines()];rounds=[x for x in rounds if start<=x['at']<=end and x['bs']]
 def q(v):return dict(zip(['p10','p50','p90','p99'],map(float,np.percentile(v,[10,50,90,99])))) if v else {}
 r['decode_rounds']={'count':len(rounds),'bs':q([x['bs'] for x in rounds]),'mean_ctx':q([x['ctx']/x['bs'] for x in rounds]),'seconds':q([x['seconds'] for x in rounds]),'partition_count':dict(collections.Counter(x['partition'] for x in rounds))}
 r['calibration_stage0']={}
 pp_size=next(iter(json.loads((path/'run.json').read_text())['server_info'].values()))['pp_size']
 for l in text.splitlines():
  m=re.search(r'PP0\].*decode calibration partition=(\d+) SMs=(.*?) samples=(.*)',l)
  if m:
   records=ast.literal_eval(m[3]);r['calibration_stage0'][m[1]]={'sm':ast.literal_eval(m[2]),'predicted_pipeline_ms_bs8':{c:pp_size*(max(t for b,x,t in records if b<=8 and x<=c)*1.2+.001)*1000 for c in [4096,8192,16384]}}
 (path/'diagnosis.json').write_text(json.dumps(r,indent=2)+'\n')
 print(path.name,json.dumps({'metrics':r['metrics'],'decode_rounds':r['decode_rounds'],'admission':d.get('full',{}).get('admission',{}),'p50':d['quantile_decomposition']['p50']},indent=2))
if __name__=='__main__':
 for p in sys.argv[1:]:diagnose(Path(p))
