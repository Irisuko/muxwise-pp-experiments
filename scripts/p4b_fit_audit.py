import importlib.util, json, numpy as np
from pathlib import Path
path='/root/workspace/sglang-p4b-v1/python/sglang/srt/multiplex/pp_policy.py'
spec=importlib.util.spec_from_file_location('old_policy',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
root=Path('/root/workspace/muxpp_work/logs/p4b/v1-profile')
class WarmRLS(mod.OnlineRLS):
 def update(self,x,y):
  if self.count>=16:return super().update(x,y)
  x=np.asarray(x,dtype=float)
  pred=self.predict(x);self.errors.append(abs(y-pred)/y)
  px=self.P@x;g=px/(self.forgetting+x@px)
  self.theta+=g*(y-float(self.theta@x))
  self.P=(self.P-np.outer(g,x@self.P))/self.forgetting
  self.count+=1
for i in range(4):
 rows=[json.loads(x) for x in (root/f'prefill-timing.{i}.jsonl').read_text().splitlines()]
 models={'old':mod.OnlineRLS(7,3),'warm_unclipped':WarmRLS(7,3)}
 bad=[]
 for j,r in enumerate(rows):
  x=mod.sm_features(r['base'],[128,96,80,64][r['partition']],128); y=r['seconds']/r['layers']
  pred=models['old'].predict(x)
  if j>len(rows)*.7 and abs(y-pred)/y>.5:bad.append(dict(row=j,base=r['base'],idx=r['partition'],true=round(y,4),pred=round(pred,4)))
  for model in models.values():model.update(x,y)
 print(i,'n',len(rows),{k:dict(mape=v.mape,theta=v.theta.tolist()) for k,v in models.items()},'bad',bad[-4:])
