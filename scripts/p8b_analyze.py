#!/usr/bin/env python3
"""P8b: CPU-only, read saved logs/results; write analysis_* artifacts only."""
import ast, collections, csv, datetime as dt, hashlib, json, math, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/root/workspace/muxpp_work'); OUT=ROOT/'results'
LABELS=['muxwise_r2','muxwise_tp2dp2','muxwise_tp4','muxpp_pp2dp2','muxwise_pp_v3','chunked_tp2dp2','chunked_pp2dp2','chunked','chunked_tp4']
NAMES=['MuxWise DP4','MuxWise TP2 x DP2','MuxWise TP4','MuxWise-PP PP2 x DP2','MuxWise-PP PP4','Chunked TP2 x DP2','Chunked PP2 x DP2','Chunked DP4','Chunked TP4']
SHORT=dict(zip(LABELS,NAMES)); TZ=dt.timezone(dt.timedelta(hours=8))
MANIFEST={}
def read(p):
 b=p.read_bytes();MANIFEST[str(p.relative_to(ROOT))]=hashlib.sha256(b).hexdigest();return b.decode(errors='replace')
def jread(p):return json.loads(read(p))
def pct(a):
 a=np.asarray(a,dtype=float)
 return {'n':len(a),**({f'p{q}':float(np.percentile(a,q)) for q in [10,50,90,95,99]} if len(a) else {}), 'mean':float(np.mean(a)) if len(a) else None,'max':float(np.max(a)) if len(a) else None}
def stamp(s):return dt.datetime.strptime(s,'%Y-%m-%d %H:%M:%S').replace(tzinfo=TZ).timestamp()
def csvwrite(name,rows):
 if rows:
  keys=list(dict.fromkeys(k for r in rows for k in r));
  with (OUT/name).open('w') as f:
   w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
DATA={}; METRICS={}; ALL_MATCH=[]; ALL_EVENTS=[]
for label in LABELS:
 old=label in ['chunked','muxwise_r2','muxwise_pp_v3']
 s=jread(OUT/f'{label}{"_cut300" if old else ""}_summary.json')
 rs=[json.loads(l) for l in read(OUT/f'{label}_requests.jsonl').splitlines()]
 mark=jread(OUT/f'{label}.formal-started.json'); folder=Path(mark['log_dir']);run=jread(folder/'run.json')
 start=s['run_started_unix'];end=start+s['duration_sec'];dur=s['duration_sec']
 workers={};summaries={}; raw_streams={}; retractions=[]; evidence=[]
 for w,info in run['server_info'].items():
  p=folder/f'{w}.log'
  if not p.exists():continue
  ev=[];summ=[];streams=[]
  for ln,line in enumerate(read(p).splitlines(),1):
   m=re.match(r'\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)([^\]]*)\]',line)
   if not m:continue
   # One logical scheduler: TP0/PP0 only; ignore duplicate rank/stage output.
   if any(int(x)>0 for x in re.findall(r'(?:TP|PP)(\d+)',m[2])):continue
   t=stamp(m[1])-start
   base={'label':label,'worker':w,'line':ln,'t':t}
   g=re.search(r'Adjusting stream groups: (\d+), prefill sm: (\d+), decode sm: (\d+)',line)
   if g:
    streams.append({**base,'kind':'stream','group':int(g[1]),'prefill_sm':int(g[2]),'decode_sm':int(g[3])})
   if 'MuxWise-PP summary stages=' in line:
    a=line.split('summary stages=',1)[1].split(' mbs=',1)[0]
    if 0<=t and t+1<=dur:summ.append((t,ast.literal_eval(a)))
   if not -1<t<=dur:continue
   strict=0<=t and t+1<=dur
   if 'Retract requests' in line:
    n=int(re.search(r'#retracted_reqs: (\d+)',line)[1]);retractions.append({**base,'n':n,'text':line});ev.append({**base,'kind':'retract','n':n})
   pre=re.search(r'Prefill batch\. #new-seq: (\d+), #new-token: (-?\d+), #cached-token: (\d+), token usage: ([\d.]+), #running-req: (\d+), #queue-req: (\d+)',line)
   dec=re.search(r'Decode batch\. #running-req: (\d+), #token: (\d+), token usage: ([\d.]+), cuda graph: (True|False), gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)',line)
   admit=re.search(r'MuxWise-PP admit batch=(\d+) requests=(\d+) tokens=(\d+) cached=(\d+) inflight=(\d+)',line)
   if pre:
    n,new,cached,u,r,q=pre.groups(); obj={**base,'kind':'prefill','n':int(n),'new':int(new),'cached':int(cached),'usage':float(u),'running':int(r),'queue':int(q),'strict':strict}
    # Exclude 16-token client warmups and 1-token cleanup probes in boundary seconds.
    if (label.startswith('muxwise') or label=='muxpp_pp2dp2') and int(new)+int(cached)<892:continue
    if not strict and int(new)+int(cached)<892:continue
    ev.append(obj)
   if dec: 
    n,tokens,u,graph,tps,q=dec.groups();ev.append({**base,'kind':'decode','running':int(n),'tokens':int(tokens),'usage':float(u),'graph':graph=='True','tps':float(tps),'queue':int(q),'strict':strict})
   if admit:
    batch,n,new,cached,inf=map(int,admit.groups())
    if new+cached<892:continue
    ev.append({**base,'kind':'admit','batch':batch,'n':n,'new':new,'cached':cached,'inflight':inf,'strict':strict})
  workers[w]={'capacity':info['max_total_num_tokens'],'events':ev,'info':{k:info.get(k) for k in ['mem_fraction_static','cuda_graph_max_bs','pdmux_prefill_mlp_chunk_tokens','tp_size','pp_size']}}
  summaries[w]=summ;raw_streams[w]=streams
  ALL_EVENTS.extend(ev)
 # GPU timestamps are local UTC+8, router timestamps UTC.
 gpu=[]
 for row in csv.DictReader(read(folder/'gpu.csv').splitlines()):
  v=list(row.values()); t=dt.datetime.strptime(v[0].strip(),'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=TZ).timestamp()-start
  if 0<=t<=dur:gpu.append({'t':t,'gpu':int(v[1]),'memory_mib':int(v[2].strip().split()[0]),'util':int(v[4].strip().split()[0]),'mem_util':int(v[5].strip().split()[0])})
 # Mutual-unique matches, exact (new+cached,cached) signature and overlapping time feasibility.
 batches=[e for w in workers.values() for e in w['events'] if e['kind']=='prefill' and e['n']==1]
 candidates={};reverse=collections.defaultdict(list)
 if label.startswith('muxwise') and label!='muxwise_pp_v3':
  sig=collections.defaultdict(list)
  for r in rs:sig[(r['input_len'],r['cached_tokens'])].append(r)
  for i,b in enumerate(batches):
   cand=[r for r in sig[(b['new']+b['cached'],b['cached'])] if max(b['t'],r['sent_at'])<=min(b['t']+1,r['first_token_at'])]
   candidates[i]=cand
   for r in cand:reverse[r['idx']].append(i)
 matches=[]
 for i,cand in candidates.items():
  if len(cand)!=1 or len(reverse[cand[0]['idx']])!=1:continue
  b=batches[i];r=cand[0];lo=max(b['t'],r['sent_at']);hi=min(b['t']+1,r['first_token_at']);mid=(lo+hi)/2
  matches.append({**b,'idx':r['idx'],'input':r['input_len'],'ttft':r['ttft'],'sent':r['sent_at'],'first':r['first_token_at'],'end':r['finished_at'],'output':r['output_tokens'],'start_lo':lo,'start_hi':hi,'queue_lo':lo-r['sent_at'],'queue_hi':hi-r['sent_at'],'queue_est':mid-r['sent_at'],'service_lo':r['first_token_at']-hi,'service_hi':r['first_token_at']-lo,'service_est':r['first_token_at']-mid})
 # Same-worker emitted decode snapshots strictly inside the guaranteed prefill-to-first-token span.
 for m in matches:
  de=[e for e in workers[m['worker']]['events'] if e['kind']=='decode' and e['t']>=m['start_hi'] and e['t']+1<=m['first']]
  m['decode_snapshots_inside']=len(de)
  peers=[x for x in matches if x['idx']!=m['idx'] and x['worker']==m['worker']]
  emit=0
  for x in peers:
   r=next(r for r in rs if r['idx']==x['idx'])
   emit+=sum(m['start_hi']<c['at']<m['first'] and c['completion_tokens']>1 for c in r['chunks'])
  m['peer_sse_inside']=emit
 ALL_MATCH.extend(matches)
 # Rank-deduplicated stream state dwell, transition and adjustment counts.
 streamstats={};intervals=[]
 for w,st in raw_streams.items():
  state=None;last=0;dwell=collections.Counter();adj=collections.Counter();switch=0
  for e in st:
   if e['t']<0:state=e['group'];continue
   if e['t']>=dur:break
   if state is not None:dwell[state]+=e['t']-last;intervals.append({'worker':w,'lo':last,'hi':e['t'],'group':state})
   if state is not None and state!=e['group']:switch+=1
   adj[e['group']]+=1;state=e['group'];last=e['t']
  if state is not None:dwell[state]+=dur-last;intervals.append({'worker':w,'lo':last,'hi':dur,'group':state})
  streamstats[w]={'dwell_seconds':dict(dwell),'adjustments':dict(adj),'true_switches':switch,'switches_per_min':switch/dur*60}
 metrics={'duration':dur,'workers':{},'streams':streamstats,'retracts':retractions,'gpu_util':pct([g['util'] for g in gpu]),'gpu_mem_util':pct([g['mem_util'] for g in gpu]),'matches':len(matches),'match_service':pct([m['service_est'] for m in matches]),'match_queue':pct([m['queue_est'] for m in matches]),'match_ttft':pct([m['ttft'] for m in matches]),'match_decode_inside':sum(m['decode_snapshots_inside']>0 for m in matches),'match_sse_inside':sum(m['peer_sse_inside']>0 for m in matches)}
 for w,x in workers.items():
  ev=x['events'];d=[e for e in ev if e['kind']=='decode' and e['strict']];p=[e for e in ev if e['kind'] in ['prefill','admit']];q=[e for e in ev if 'usage' in e and e.get('strict')]
  stable=[];last=None;dirty=False
  for e in ev:
   if e['kind'] in ['prefill','retract']:dirty=True
   if e['kind']=='decode':
    if last and not dirty and e['strict'] and last['strict'] and e['running']==last['running'] and e['tokens']-last['tokens']==40*e['running'] and e['tps']>0:
     stable.append({'t':e['t'],'ms_proxy':1000*e['running']/e['tps'],'bs':e['running'],'delta_ms_40step':(e['t']-last['t'])*25})
    last=e;dirty=False
  x['stable']=stable
  metrics['workers'][w]={'capacity':x['capacity'],'decode_usage':pct([e['usage'] for e in d]),'decode_tokens':pct([e['tokens'] for e in d]),'decode_bs':pct([e['running'] for e in d]),'queue':pct([e['queue'] for e in q]),'usage_ge90_fraction':np.mean([e['usage']>=.9 for e in d]).item() if d else None,'queue_nonzero_fraction':np.mean([e['queue']>0 for e in d]).item() if d else None,'queue_and_usage_ge90_fraction':np.mean([e['queue']>0 and e['usage']>=.9 for e in d]).item() if d else None,'queue_and_usage_lt80_fraction':np.mean([e['queue']>0 and e['usage']<.8 for e in d]).item() if d else None,'graph_fraction':np.mean([e['graph'] for e in d]).item() if d else None,'prefill_batches':len(p),'prefill_n':pct([e['n'] for e in p]),'prefill_new':pct([e['new'] for e in p]),'prefill_running0_fraction':np.mean([e.get('running',-1)==0 for e in p]).item() if p else None,'admitted_req':sum(e['n'] for e in p),'admitted_new':sum(e['new'] for e in p),'admitted_cached':sum(e['cached'] for e in p),'stable_step_proxy_ms':pct([e['ms_proxy'] for e in stable])}
 # Difference two in-window cumulative summaries, no startup/warmup mixing.
 pp={}
 for w,ss in summaries.items():
  if len(ss)<2:continue
  ta,a=ss[0];tb,b=ss[-1];pp[w]={'span':tb-ta,'start_t':ta,'end_t':tb,'stages':{}}
  for st in a.keys() & b.keys():
   x=a[st]['timing'];y=b[st]['timing'];z={'elapsed':y['elapsed']-x['elapsed']}
   z['cuda']={k:y['cuda'].get(k,0)-x['cuda'].get(k,0) for k in y['cuda']}
   z['wall']={side:{k:v-x['wall'].get(side,{}).get(k,0) for k,v in vals.items()} for side,vals in y['wall'].items()}
   z['slices']={k:[v[i]-x['slices'].get(k,[0,0])[i] for i in range(2)] for k,v in y['slices'].items()}
   if a[st].get('admission') and b[st].get('admission'):
    z['admission_wall']={k:v-a[st]['admission']['wall'].get(k,0) for k,v in b[st]['admission']['wall'].items()}
   pp[w]['stages'][st]=z
 metrics['pp_cumulative_deltas']=pp
 DATA[label]={'summary':s,'requests':rs,'workers':workers,'gpu':gpu,'matches':matches,'streams':intervals,'folder':str(folder.relative_to(ROOT))};METRICS[label]=metrics
(OUT/'analysis_metrics.json').write_text(json.dumps(METRICS,indent=2,ensure_ascii=False))
csvwrite('analysis_events.csv',ALL_EVENTS);csvwrite('analysis_prefill_matches.csv',ALL_MATCH)
# Match the same trace request across TP1/TP2/TP4, requiring identical cache workload.
matched={l:{m['idx']:m for m in DATA[l]['matches']} for l in LABELS[:3]};pairs=[]
for a,b in [('muxwise_r2','muxwise_tp2dp2'),('muxwise_tp2dp2','muxwise_tp4')]:
 for idx in matched[a].keys() & matched[b].keys():
  x,y=matched[a][idx],matched[b][idx]
  if x['new']!=y['new'] or x['cached']!=y['cached']:continue
  pairs.append({'a':a,'b':b,'idx':idx,'new':x['new'],'a_service':x['service_est'],'b_service':y['service_est'],'a_lo':x['service_lo'],'a_hi':x['service_hi'],'b_lo':y['service_lo'],'b_hi':y['service_hi'],'ratio':x['service_est']/max(y['service_est'],1e-9)})
csvwrite('analysis_prefill_pairs.csv',pairs)
# Publication-ready static figures, English labels avoid missing CJK fonts.
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'figure.dpi':140})
colors=['#4477aa','#228833','#cc6677','#aa3377','#66ccee','#ee7733','#999933']
fig,axs=plt.subplots(4,3,figsize=(15,12),sharex='col')
for col,l in enumerate(LABELS[:3]):
 d=DATA[l]
 for j,(w,x) in enumerate(d['workers'].items()):
  ev=[e for e in x['events'] if 'usage' in e and e['strict']]
  # Unweighted event means in 10s bins: explicitly not continuous-time occupancy.
  for row,key in [(0,'usage'),(1,'queue')]:
   bins=collections.defaultdict(list)
   for e in ev:bins[int(e['t']//10)*10+5].append(e[key])
   axs[row,col].plot(sorted(bins),[np.mean(bins[t]) for t in sorted(bins)],label=w,lw=1)
  ds=[e for e in ev if e['kind']=='decode'];bins=collections.defaultdict(list)
  for e in ds:bins[int(e['t']//10)*10+5].append(e['tokens']/1000)
  axs[2,col].plot(sorted(bins),[np.mean(bins[t]) for t in sorted(bins)],lw=1)
  rt=[r['t'] for r in METRICS[l]['retracts'] if r['worker']==w]
  axs[3,col].step([0]+rt+[d['summary']['duration_sec']],[0]+list(range(1,len(rt)+1))+[len(rt)],where='post',lw=1)
 axs[0,col].set_title(SHORT[l]);axs[0,col].legend(fontsize=8);axs[0,col].set_ylim(0,1.05)
 for row in range(4):axs[row,col].axvline(600,color='gray',ls='--',lw=1)
 axs[3,col].set_xlabel('Seconds from formal replay start');axs[3,col].set_ylim(-.05,2.15);axs[3,col].set_yticks([0,1,2])
for row,y in enumerate(['Token usage (10s event mean)','Queue requests (10s event mean)','Decode resident tokens (thousands)','Cumulative retract events']):axs[row,0].set_ylabel(y)
fig.tight_layout();fig.savefig(OUT/'analysis_capacity_timeline.png');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(15,4.5))
groups=['P128/D0','P96/D32','P80/D48','P64/D64','P0/D128']
for i,l in enumerate(LABELS[:3]):
 ss=METRICS[l]['streams'];den=sum(sum(s['dwell_seconds'].values()) for s in ss.values());share=[sum(s['dwell_seconds'].get(g,0) for s in ss.values())/den*100 for g in range(5)]
 left=0
 for g,v in enumerate(share):axs[0].barh(i,v,left=left,label=groups[g] if i==0 else None,color=plt.get_cmap('tab10')(g));left+=v
 ds=[e['running'] for w in DATA[l]['workers'].values() for e in w['events'] if e['kind']=='decode' and e['strict']]
 axs[1].plot(sorted(ds),np.arange(1,len(ds)+1)/len(ds),label=SHORT[l],color=colors[i])
 gp=DATA[l]['gpu'];bins=collections.defaultdict(list)
 for g in gp:bins[int(g['t']//20)*20+10].append(g['util'])
 axs[2].plot(sorted(bins),[np.mean(bins[t]) for t in sorted(bins)],label=SHORT[l],color=colors[i])
axs[0].set_yticks(range(3),[SHORT[l] for l in LABELS[:3]]);axs[0].set_xlabel('Approximate wall-time share (%)');axs[0].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3)
axs[1].set_xlabel('Decode batch size (log samples)');axs[1].set_ylabel('CDF');axs[1].legend(fontsize=8)
axs[2].set_xlabel('Replay time (s)');axs[2].set_ylabel('GPU utilization (%), 20s / 4-GPU mean');axs[2].legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT/'analysis_mux_overlap_gpu.png',bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(12,9))
for i,l in enumerate(LABELS[:3]):
 ms=DATA[l]['matches'];axs[0,0].scatter([m['new'] for m in ms],[m['service_est'] for m in ms],s=10,alpha=.4,label=SHORT[l],color=colors[i])
 axs[0,1].bar(i,np.mean([m['queue_est'] for m in ms]),color='#bbbbbb')
 axs[0,1].bar(i,np.mean([m['service_est'] for m in ms]),bottom=np.mean([m['queue_est'] for m in ms]),color=colors[i])
for l in [LABELS[1],LABELS[3],LABELS[4],LABELS[5],LABELS[6]]:
 rs=DATA[l]['requests'];a=sorted(r['tpot']*1000 for r in rs if r['tpot'] is not None and r['success'])
 axs[1,0].plot(a,np.arange(1,len(a)+1)/len(a),label=SHORT[l],color=colors[LABELS.index(l)])
 axs[1,1].scatter([r['output_tokens'] for r in rs if r['tpot'] is not None],[r['tpot']*1000 for r in rs if r['tpot'] is not None],s=7,alpha=.25,label=SHORT[l],color=colors[LABELS.index(l)])
axs[0,0].set_xlabel('Uncached tokens in matched single-request batch');axs[0,0].set_ylabel('Estimated batch-to-first-token time (s)');axs[0,0].legend(fontsize=8)
axs[0,1].set_xticks(range(3),['DP4','TP2 x DP2','TP4']);axs[0,1].set_ylabel('Mean TTFT of matched subset (s)');axs[0,1].set_title('Gray: pre-admission wait; color: service estimate')
axs[1,0].set_xscale('log');axs[1,0].set_xlabel('Request TPOT (ms)');axs[1,0].set_ylabel('CDF');axs[1,0].legend(fontsize=8)
axs[1,1].set_xscale('log');axs[1,1].set_yscale('log');axs[1,1].set_xlabel('Output tokens');axs[1,1].set_ylabel('Request TPOT (ms)')
fig.tight_layout();fig.savefig(OUT/'analysis_prefill_latency.png');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,4.5))
l='muxwise_tp2dp2'
for w,x in DATA[l]['workers'].items():
 ps=[e for e in x['events'] if e['kind']=='prefill'];t=[max(0,e['t']) for e in ps]
 axs[0].step(t,np.cumsum([e['n'] for e in ps]),where='post',label=w)
 axs[1].step(t,np.cumsum([e['new']+e['cached'] for e in ps])/1e6,where='post',label=w+' total input',color=colors[0 if w=='worker0' else 1])
 axs[1].step(t,np.cumsum([e['new'] for e in ps])/1e6,where='post',ls='--',label=w+' uncached',color=colors[0 if w=='worker0' else 1])
axs[0].set_ylabel('Cumulative admitted requests');axs[1].set_ylabel('Cumulative tokens (million)')
for ax in axs:ax.set_xlabel('Replay time (s)');ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT/'analysis_router_balance.png');plt.close(fig)
# Verify every read input stayed unchanged; no source checkout or GPU interface is accessed.
changed=[p for p,h in MANIFEST.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
assert not changed,changed
(OUT/'analysis_input_sha256.json').write_text(json.dumps(MANIFEST,indent=2))
print(json.dumps({'labels':len(DATA),'input_files':len(MANIFEST),'inputs_unchanged':not changed,'matched':{l:len(DATA[l]['matches']) for l in LABELS[:3]},'pairs':len(pairs)},indent=2))
# Derived evidence for the report, using only the parsed records above.
derived={}
for l,d in DATA.items():
 rs=d['requests'];ms=d['matches'];de=[e for w in d['workers'].values() for e in w['events'] if e['kind']=='decode' and e['strict']]
 z={'decode_pooled':{k:pct([e[k] for e in de]) for k in ['usage','tokens','running','queue']},'gpu_by_card':{i:{'util':pct([g['util'] for g in d['gpu'] if g['gpu']==i]),'mem_util':pct([g['mem_util'] for g in d['gpu'] if g['gpu']==i])} for i in range(4)},'prefill_pooled':{k:pct([e[k] for w in d['workers'].values() for e in w['events'] if e['kind'] in ['prefill','admit']]) for k in ['n','new']},'decode_proxy_all_ms':pct([1000*e['running']/e['tps'] for e in de if e['tps']>0]),'phase':[],'short_output':{},'itl_gt1s':sum(v>1 for r in rs for v in r['itl']),'ttft_gt5s':sum(r['ttft']>5 for r in rs),'tpot_gt100ms':sum(r['tpot'] is not None and r['tpot']>.1 for r in rs),'finish_overhead_ms':pct([(r['finished_at']-r['chunks'][-1]['at'])*1000 for r in rs]),'match_means':{k:float(np.mean([m[k] for m in ms])) if ms else None for k in ['ttft','queue_est','queue_lo','queue_hi','service_est','service_lo','service_hi']}}
 for lo,hi in [(0,120),(120,300),(300,600),(600,9999)]:
  a=[e for e in de if lo<=e['t']<hi]
  z['phase'].append({'lo':lo,'hi':hi,**{k:pct([e[k] for e in a]) for k in ['usage','queue','running']}})
 for lo,hi in [(2,8),(9,64),(65,99999)]:
  a=[r['tpot']*1000 for r in rs if lo<=r['output_tokens']<=hi];z['short_output'][f'{lo}-{hi}']={'tpot_ms':pct(a),'gt100':sum(x>100 for x in a)}
 # Feasibility diagnostic conditioned on next batch being uniquely matched and already arrived.
 hol_n=hol_bad=0;hol_examples=[]
 for w,x in d['workers'].items():
  a=[e for e in de if e['worker']==w];b=[e for e in x['events'] if e['kind']=='prefill'];mat={m['line']:m for m in ms if m['worker']==w}
  for e in a:
   if not e['queue']:continue
   nxt=next((p for p in b if p['line']>e['line']),None)
   if not nxt or nxt['line'] not in mat:continue
   mt=mat[nxt['line']]
   if mt['sent']>e['t']:continue
   hol_n+=1
   if x['capacity']-e['tokens']<nxt['new']:
    hol_bad+=1
    if len(hol_examples)<4:hol_examples.append({'worker':w,'decode_line':e['line'],'at':e['t'],'tokens':e['tokens'],'queue':e['queue'],'next_prefill_line':nxt['line'],'next_new':nxt['new'],'next_idx':mt['idx'],'next_sent':mt['sent']})
 z['next_matched_batch_headroom']={'insufficient':hol_bad,'eligible_decode_snapshots':hol_n,'examples':hol_examples}
 z['paired_service']={}
 for lo,hi in [(0,99999),(8000,16000),(16000,24000),(24000,40000)]:
  a=[r for r in pairs if r['a']==l and lo<=r['new']<hi]
  if a:z['paired_service'][f'{lo}-{hi}']={'n':len(a),'a_service':pct([r['a_service'] for r in a]),'b_service':pct([r['b_service'] for r in a]),'ratio_of_sums':sum(r['a_service'] for r in a)/sum(r['b_service'] for r in a),'lower':sum(r['a_lo'] for r in a)/sum(r['b_hi'] for r in a),'upper':sum(r['a_hi'] for r in a)/sum(r['b_lo'] for r in a),'a_tok_s':sum(r['new'] for r in a)/sum(r['a_service'] for r in a),'b_tok_s':sum(r['new'] for r in a)/sum(r['b_service'] for r in a)}
 derived[l]=z
(OUT/'analysis_derived.json').write_text(json.dumps(derived,indent=2))
# Full accounting must match the client exactly for the target and TP4 (no retractions).
for l in ['muxwise_tp2dp2','muxwise_tp4','muxpp_pp2dp2','muxwise_pp_v3']:
 a=METRICS[l]['workers'].values();s=DATA[l]['summary']
 assert sum(w['admitted_req'] for w in a)==s['planned_requests'],l
 assert sum(w['admitted_new']+w['admitted_cached'] for w in a)==s['input_tokens'],l
 assert sum(w['admitted_cached'] for w in a)==s['cached_tokens']['total'],l
print('Prefill request/input/cache conservation: PASS (TP2DP2, TP4, PP2DP2, PP4)')
