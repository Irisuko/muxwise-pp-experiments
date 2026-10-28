import json,sys
from pathlib import Path
import torch
root=Path('/root/workspace/muxpp_work/logs/p10b')
ref=root/'numeric-plain-pp4'
for tag in sys.argv[1:]:
 path=root/tag;records=[]
 def compare(a,b,key):
  if isinstance(a,torch.Tensor):
   diff=(a.float()-b.float()).abs(); records.append(dict(file=name,key=key,shape=list(a.shape),max=float(diff.max()),mean=float(diff.mean()),unequal=int((a!=b).sum()),total=a.numel()))
  elif isinstance(a,dict):
   for k,v in a.items():
    if k!='rows' and k!='shape':compare(v,b[k],key+'.'+k)
  elif isinstance(a,list):
   for i,(x,y) in enumerate(zip(a,b)):compare(x,y,key+'.'+str(i))
 for file in sorted(ref.glob('numeric-*.pt')):
  name=file.name;other=path/name
  if not other.exists():continue
  a,b=torch.load(file,weights_only=True),torch.load(other,weights_only=True)
  if 'logits' in name:
   print(tag,name,'ref',torch.topk(a,5).indices.tolist(),torch.topk(a,5).values.tolist(),'actual',torch.topk(b,5).indices.tolist(),torch.topk(b,5).values.tolist())
  compare(a,b,'')
 (path/'numeric-comparison.json').write_text(json.dumps(records,indent=2)+'\n')
 for kind in ['attention','mlp','layer']:
  x=[r for r in records if f'-0-{kind}' in r['file']];print(tag,kind,x)
