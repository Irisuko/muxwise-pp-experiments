#!/usr/bin/env python3
"""CPU-only P11b report, recomputed from immutable request events."""
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import p9b_report as r
from p9b_write_results import best_table

ROOT=r.ROOT
OUT=ROOT/'results'
LOG=ROOT/'logs/p11b'
ARCH=ROOT/'archive/superseded/p11b'
MAIN=r.sm.MAIN_SLO
PAIRS={'muxpp_v5_pp4':'muxpp_v4_pp4','muxpp_v5_pp2tp2':'muxpp_v4_pp2tp2',
       'muxpp_v5_pp2dp2':'muxpp_v4_pp2dp2','conv_muxpp_v5_pp4':'conv_muxpp_v4_pp4',
       'conv_muxpp_v5_pp2tp2':'conv_muxpp_v4_pp2tp2',
       'big_muxpp_v5_pp2tp2':'big_muxpp_pp2tp2','big_muxpp_v5_pp4':'big_muxpp_pp4'}


def load(p):return json.loads(p.read_text())
def fmt(v,m=1):return r.sm.format_number(v,m)
def pct(v):return fmt(v,100)+'%'
def quant(s,key):return '/'.join(fmt(s['latencies_sec'][key][q],1000 if key=='tpot' else 1) for q in ['p50','p90','p95','p99'])
def save(p,v):r.save(p,v)


def resolve(p):
    if p.exists():return p
    for archive in [ARCH,ROOT/'archive/superseded/p9b']:
        q=archive/p.relative_to(ROOT)
        if q.exists():return q
    return p


r.resolve=resolve


def gpu_stats(path,s):
    # New samples carry explicit Unix time in the guard. Historical nvidia-smi
    # timestamps are local UTC+8, as in the original P9b/P10c reader.
    guard=LOG/(s['label']+'-guard.jsonl')
    if not guard.exists():return ORIGINAL_GPU_STATS(path,s)
    data=[];external=[]
    with guard.open() as f:
        for line in f:
            a=json.loads(line)
            if s['run_started_unix'] <= a['unix'] <= s['run_started_unix']+s['duration_sec']:
                external.extend(u for g in a['gpu'] for u in g['process_utilization'] if not u['owned'])
    # For the requested GPU utilization statistic use the 5s CSV, with its actual timezone.
    g=ORIGINAL_GPU_STATS(path,s)
    g['external_sm_peak_pct']=max((u['sm'] for u in external),default=None)
    g['exclusive_gpu_assignment']=True
    return g


ORIGINAL_GPU_STATS=r.gpu_stats
r.gpu_stats=gpu_stats


def maps():
    tool={s['label']:(s['matrix']['system'],s['matrix']['configuration']) for s in load(OUT/'matrix_summary.json')}
    conv={s['label']:(s['matrix']['system'],s['matrix']['configuration']) for s in load(OUT/'conv_summary.json')}
    big={s['label']:(s['matrix']['system'],s['matrix']['configuration']) for s in load(OUT/'big_summary.json')}
    for new,old in PAIRS.items():
        target=big if new.startswith('big_') else conv if new.startswith('conv_') else tool
        if new not in target:
            target[new]=('MuxWise-PP',target[old][1].removesuffix(' v4')+' v5')
        if old.startswith('big_') and old in target:target[old]=(target[old][0],target[old][1].removesuffix(' v4')+' v4')
    return tool,conv,big


def compare():
    mapping={}
    for new,old in PAIRS.items():
        topology=new.split('v5_')[1].replace('pp2tp2','PP2×TP2').replace('pp2dp2','PP2DP2').replace('pp4','PP4')
        mapping[new]=('MuxWise-PP',topology+' v5');mapping[old]=('MuxWise-PP',topology+' v4')
    pairs={}
    for new,old in PAIRS.items():
        n=555 if new.startswith('big_') else 735 if new.startswith('conv_') else 1551
        runs=r.prepare({old:mapping[old],new:mapping[new]},n)
        a,b=[s for s,_ in runs]
        pairs[new]=dict(old_label=old,before=a,after=b,
            main_slo_delta_pp=100*(b['slo'][MAIN]['rate']-a['slo'][MAIN]['rate']),
            output_change_pct=100*(b['throughput']['output_tokens_per_sec']/a['throughput']['output_tokens_per_sec']-1),
            ttft_change_pct={q:100*(b['latencies_sec']['ttft'][q]/a['latencies_sec']['ttft'][q]-1)
                             if isinstance(b['latencies_sec']['ttft'][q],(int,float)) else None for q in ['p50','p90','p95','p99']},
            tpot_change_pct={q:100*(b['latencies_sec']['tpot'][q]/a['latencies_sec']['tpot'][q]-1)
                             for q in ['p50','p90','p95','p99']})
    save(LOG/'v4-v5-comparison.json',pairs)
    for new,x in pairs.items():
        print(new,json.dumps({k:v for k,v in x.items() if k not in ['before','after']},ensure_ascii=False))
    return pairs


def archive(decisions):
    manifest=ARCH/'manifest.json'
    if manifest.exists():
        for row in load(manifest)['files']:assert sha(ROOT/row['destination'])==row['sha256']
        return
    ARCH.mkdir(parents=True,exist_ok=True);files=[]
    def transfer(p,q,move):
        assert p.exists() and not q.exists(), (p,q)
        digest=sha(p);q.parent.mkdir(parents=True,exist_ok=True)
        if move:shutil.move(str(p),str(q))
        else:shutil.copy2(p,q)
        assert sha(q)==digest
        files.append(dict(source=str(p.relative_to(ROOT)),destination=str(q.relative_to(ROOT)),sha256=digest,operation='move' if move else 'copy'))
    for p in OUT.iterdir():
        if p.is_file() and p.suffix in ['.png','.svg','.csv','.md']:
            transfer(p,ARCH/'previous_report/results'/p.name,False)
    transfer(LOG/'RESULTS-before-P11b.md',ARCH/'previous_report/RESULTS.md',False)
    for new,d in decisions.items():
        if d['replace']:
            old=PAIRS[new]
            for suffix in ['_requests.jsonl','_summary.json','.formal-started.json']:
                p=OUT/(old+suffix);transfer(p,ARCH/'results'/p.name,True)
    save(manifest,dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),decisions=decisions,files=files,
                      note='Superseded raw requests, summary and formal lock moved with identical bytes; original server logs remain at their historical paths.'))


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def heatmap(runs,prefix,title):
    systems=list(dict.fromkeys(s['matrix']['system'] for s,_ in runs))
    def topology(cfg):return cfg.replace(' v3','').replace(' v4','').replace(' v5','').replace('TP2PP2','PP2×TP2')
    columns=list(dict.fromkeys(topology(s['matrix']['configuration']) for s,_ in runs))
    values=r.np.full((len(systems),len(columns)),r.np.nan);texts={}
    for s,_ in runs:
        m=s['matrix'];cfg=m['configuration'];i=systems.index(m['system']);j=columns.index(topology(cfg))
        v=s['slo'][MAIN]['rate']*100
        values[i,j]=max(v,values[i,j]) if r.np.isfinite(values[i,j]) else v
        version='v5 ' if 'v5' in s['label'] else 'v4 ' if 'v4' in cfg else 'v3 ' if 'v3' in cfg else ''
        texts.setdefault((i,j),[]).append(version+f'{v:.2f}%'+('*' if s['failed'] else ''))
    fig,ax=r.sm.plt.subplots(figsize=(max(9,1.45*len(columns)),max(4.5,1.1*len(systems))))
    cmap=r.sm.plt.get_cmap('YlGnBu').copy();cmap.set_bad('#eeeeee')
    im=ax.imshow(r.np.ma.masked_invalid(values),vmin=0,vmax=100,aspect='auto',cmap=cmap)
    for i in range(len(systems)):
        for j in range(len(columns)):
            ax.text(j,i,'\n'.join(texts.get((i,j),['N/A'])),ha='center',va='center',fontsize=9,color='white' if values[i,j]>55 else '#222')
    ax.set_xticks(range(len(columns)),columns,rotation=25,ha='right');ax.set_yticks(range(len(systems)),systems)
    ax.set_title(title);ax.grid(False)
    ax.set_xlabel('Single measurements; different dates/GPU assignments. All P11b GPUs on NUMA0.\nCoexisting versions both shown; color uses the higher rate. * Failures retained.')
    fig.colorbar(im,ax=ax,label='Main SLO attainment (%)');r.sm.save_figure(fig,OUT,prefix+'slo_heatmap.png')


def saturation(labels):
    rows=[]
    for label in labels:
        s,records=r.sm.load_runs([str(OUT/(label+'_summary.json'))])[0]
        raw=load(OUT/(label+'_summary.json'))
        for k in ['completed','unfinished','failed','duration_sec','latencies_sec','slo','throughput']:assert s[k]==raw[k],(label,k)
        assert s['time_scale']==.05 and s['drain_timeout_sec']==300
        assert s['planned_requests']==(555 if 'big_' in label else 1551)
        s['all_requests_completed_sec']=max(x['finished_at'] for x in records) if s['completed']==s['planned_requests'] else None
        s['matrix']=dict(system='MuxWise-PP' if 'muxpp' in label else 'MuxWise' if 'muxwise' in label else 'Chunked',
                         configuration=label.split('_')[-1].replace('pp2tp2','PP2×TP2').replace('pp2dp2','PP2DP2').replace('tp2dp2','TP2DP2').replace('pp4','PP4').replace('tp4','TP4'))
        meta=load(LOG/label/'run.json')
        s['execution']=dict(commit=meta['commit'],gpu_mapping=meta['gpu_mapping'],
                            log_dir=str(LOG/label),formal_launch_count=meta['formal_launch_count'],
                            restart_count=meta['restart_count'])
        s['gpu_measurement']=gpu_stats(LOG/label/'gpu.csv',s)
        rows.append((s,records))
    r.sm.write_tables(rows,OUT,'sat_');save(OUT/'sat_summary.json',[s for s,_ in rows])
    for fam in ['tool','big']:
        group=[s for s,_ in rows if ('big_' in s['label'])==(fam=='big')]
        fig,axes=r.sm.plt.subplots(1,3,figsize=(14,4.8))
        names=[s['matrix']['system']+'\n'+s['matrix']['configuration'] for s in group]
        for ax,key,title in zip(axes,['output_tokens_per_sec','input_tokens_per_sec','requests_per_sec'],['Output tok/s','Input tok/s (includes cache)','Completed requests/s']):
            values=[s['throughput'][key] for s in group]
            bars=ax.bar(names,values,color=['#d69a36','#426b98','#4b9a86'])
            for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f'{v:.2f}',ha='center',va='bottom')
            ax.set_title(title);ax.set_ylim(0,max(values)*1.18);ax.grid(axis='x',visible=False)
        fig.suptitle(('Tool & Agent 8B' if fam=='tool' else 'Mistral-Small-24B')+': time-scale 0.05, drain 300 s; one run each\nPartial outputs included; four exclusive GPUs on NUMA0')
        r.sm.save_figure(fig,OUT,'sat_'+fam+'_throughput.png')
    return rows


def sat_table(rows):
    lines=['| label | 完成/未完/失败 | 全部完成耗时(s) | 测量时长(s) | 输入/输出 tok/s | 完成req/s | TTFT p50/p90/p95/p99(s) | TPOT p50/p90/p95/p99(ms); n |',
           '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for s,_ in rows:
        t=s['throughput']
        lines.append(f'| {s["label"]} | {s["completed"]}/{s["unfinished"]}/{s["failed"]} | {fmt(s["all_requests_completed_sec"])} | {fmt(s["duration_sec"])} | {fmt(t["input_tokens_per_sec"])}/{fmt(t["output_tokens_per_sec"])} | {fmt(t["requests_per_sec"])} | {quant(s,"ttft")} | {quant(s,"tpot")}; n={s["latencies_sec"]["tpot"]["count"]} |')
    return '\n'.join(lines)


def main():
    pairs=compare()
    if '--compare-only' in sys.argv:return
    decisions=load(LOG/'replacement-decisions.json')
    assert set(decisions)==set(PAIRS)
    mapping=maps();archive(decisions)
    allruns=[]
    for m,count,prefix,title in zip(mapping,[1551,735,555],['matrix_','conv_','big_'],['Tool & Agent 8B','Conversation 8B','Mistral-Small-24B']):
        for new,old in PAIRS.items():
            if new in m and decisions[new]['replace']:m.pop(old,None)
        runs=r.prepare(m,count);best=r.best(runs);allruns.append((runs,best))
        bp='best_' if prefix=='matrix_' else prefix+'best_'
        r.tables(runs,prefix,title+'完整矩阵');r.tables(best,bp,title+'各系统最佳配置')
        r.sm.plot_runs(runs,OUT,prefix);r.sm.plot_runs(best,OUT,bp)
        r.barplot(runs,prefix+'main_slo.png',title+': one run per configuration')
        r.barplot(best,bp+'main_slo.png',title+': best by main SLO')
        heatmap(runs,prefix,title+': TTFT <=5s and TPOT <=100ms')
    # Existing generic aliases must follow the updated Tool & Agent best set.
    for prefix in ['','final_']:
        r.tables(allruns[0][1],prefix,'Tool&Agent 各系统最佳配置')
        r.sm.plot_runs(allruns[0][1],OUT,prefix)
    batch=load(LOG/'batch-finished.json');sat=saturation([l for l in batch['labels'] if l.startswith('sat_')])
    frozen=load(LOG/'frozen.json')
    lines=['# P11b：MuxWise-PP v5 正式实验与三系统饱和吞吐对比','',
           f'本次冻结源码 `{frozen["head"]}`，A 的七个标准速率 label 与 B 的六个饱和 label 各取得一次正式测量；没有根据正式数据调参。启动层重试单独说明。'
           '历史正式对照保留原测量日期与源码条件，v5 的版本替换按本节逐项决定。', '',
           '## 实验设置、外部占用与统计口径','',
           '四张 RTX4090 24GB，无GPU P2P；Llama3.1-8B-Instruct与Mistral-Small-24B-Instruct-2501使用原BF16/上下文/FCFS/KV预留及已选参数。'
           '完整命令与文件SHA256见[冻结记录](logs/p11b/frozen.json)，五套v5命令与P11A逐参数相同（仅CUDA索引绑定到本次UUID），'
           'MuxWise/Chunked沿用[P7A](P7A_NOTES.md)/[P10B](P10B_NOTES.md)。'
           '模型经原 `scripts/p10a_model.py ensure` 从私有盘恢复并完整校验，见[恢复日志](logs/p11b/model-ensure.log)。','',
           '本次按用户2026-10-28换卡后的规则，四卡必须无外部占用。每label前35秒基线、运行中1秒NVML进程级监测、5秒GPU采样；发现外部PID、外部进程利用率或异常未归属显存即停止。'
           '运行中按PID及创建时间保留自有进程归属，跨父进程退出仍追踪；不将本实验高GPU利用率当作外部负载。'
           '显存比例未变，实际空闲显存由原backend查询后决定KV容量；所有容量与峰值按本次实测列出。'
           '此前GPU3约454MiB保活占用已不存在，本次不再使用允许该占用的例外。','',
           '此前因用户换卡手动停止时，`muxpp_v5_pp4`正式客户端刚启动、未产出完整结果。旧正式锁、全部运行日志与脚本已保留在'
           '[archive/aborted_p11b](archive/aborted_p11b/README.md)，日志移动逐文件哈希核对见[清单](archive/aborted_p11b/log-manifest.json)。'
           '本次根据用户明确授权从头执行；这次外部中断不作为一次完整测量，也未用于选参或正式对比。','',
           '本次另有一次启动层修复：`big_muxpp_v5_pp2tp2`首次正式客户端在`load_trace`中被遗留P10a准备锁拒绝，发生在预热和发送任何正式请求之前，未产出请求或summary数据。'
           '按用户“完全无法产出数据时允许修复启动层后再启动一次”的规则，保留[失败与校验清单](logs/p11b/launch-recovery/manifest.json)，'
           '使用[续跑入口](scripts/p11b_resume.py)临时归档准备锁，完整重启该配置一次；前五组已完成测量保持原样。'
           '全部运行结束后准备锁按原字节恢复，见[恢复记录](logs/p11b/preparation-lock-restored.json)。推理源码、客户端与服务参数均未修改。','',
           '| CUDA索引 | GPU UUID | PCI bus | NUMA |','| --- | --- | --- | --- |']
    for g in frozen['gpu_mapping']:lines.append(f'| {g["index"]} | {g["uuid"]} | {g["pci_bus_id"]} | {g["numa"]} |')
    lines+=['','PP2×TP2的TP组为0+1、2+3；DP两副本使用同样双卡组；PP4按0→1→2→3。'
            '**本次四卡全部位于NUMA0，互为NODE；历史正式运行及P11a为两个NUMA节点上的2+2分配。**'
            '同拓扑v4对照中，8B来自2026-10-21（UTC）、提交`f6833f24e`；24B来自2026-10-22（UTC）、提交`4c10135d2`。'
            '跨日期、物理GPU和NUMA拓扑均变化，因此v5与历史v4差异不能完全归因于调度优化，也不能当作同硬件单变量消融。'
            '本次三系统饱和对比使用相同四卡分配。所有测量单次，无置信区间或显著性结论。','',
            '每组按长随机31663-token/8输出预热→flush→对应开发trace前20条冒烟；MuxWise与MuxWise-PP再次长预热，正式前flush并重启DP router。'
            '客户端统一 `--warmup 2 --force-close --drain-timeout 300`；冒烟采用time-scale0.2且要求20/20，沿用P10c完整排空协议。'
            'A为time-scale1；B为0.05。正式数据、启动记录和正式锁均排他写入，共用P7 GPU flock，无自动重试。','',
            '| 负载 | 正式trace | 请求数 | 输入/输出token | 最后计划到达(s) A / B |',
            '| --- | --- | ---: | --- | --- |',
            '| Tool&Agent 8B | toolagent_600_240_x2.5 | 1551 | 11245637 / 261776 | 599.9975 / 29.999875 |',
            '| Conversation 8B | conversation_600_240_x2.5 | 735 | 6718097 / 248112 | 599.9975 / — |',
            '| Tool&Agent 24B | big_toolagent_600_80_x7.5 | 555 | 3997588 / 97918 | 585 / 29.25 |','',
            '主SLO=TTFT≤5s且TPOT≤100ms，分母为全部计划请求。未完成/失败/未发及TPOT未定义均算未达标。'
            'TTFT分位包含全部计划请求，未见首token取+∞并显示`>cutoff`；TPOT仅取截止前完成且输出>1的样本。'
            '截止=最后实际发送+300秒；若全部先结束则提前停止。吞吐分母含排空：输出计截止前所有已观测SSE token（含部分输出），'
            '输入计已交付首token的请求输入（含缓存），请求吞吐只计成功完成。完成全部耗时从回放原点计算，未全部完成时留空。'
            '指标从逐请求记录重算并与summary核对，详见[bench/README](bench/README.md)。','',
            '## A：七个v5正式结果与同拓扑v4替换决定','',
            '| v5 label | 完成/未完/失败 | 主SLO v4→v5 | TTFT p50/p90/p95/p99(s) | TPOT p50/p90/p95/p99(ms) | 输出tok/s v4→v5 | 决定 |',
            '| --- | --- | --- | --- | --- | --- | --- |']
    for new,x in pairs.items():
        a,s=x['before'],x['after'];d=decisions[new]
        lines.append(f'| {new} | {s["completed"]}/{s["unfinished"]}/{s["failed"]} | {pct(a["slo"][MAIN]["rate"])}→{pct(s["slo"][MAIN]["rate"])} | {quant(s,"ttft")} | {quant(s,"tpot")} | {fmt(a["throughput"]["output_tokens_per_sec"])}→{fmt(s["throughput"]["output_tokens_per_sec"])} | '+('替换v4' if d['replace'] else '两版并列')+' |')
    lines.append('')
    for new,d in decisions.items():lines.append(f'- **{new}**：{d["reason"]}')
    lines+=['','被替换v4的逐请求原始数据、summary与正式锁已按原相对路径移动到[archive/superseded/p11b](archive/superseded/p11b/manifest.json)，'
            '逐文件SHA256保持一致，历史服务日志仍在原路径；更新前报告/图表留有快照。未替换的v4继续并列。'
            '[完整v4/v5对照](logs/p11b/v4-v5-comparison.json) · [决定](logs/p11b/replacement-decisions.json)。']
    for (runs,best),prefix,title in zip(allruns,['matrix_','conv_','big_'],['Tool&Agent 8B','Conversation 8B','大模型24B']):
        bp='best_' if prefix=='matrix_' else prefix+'best_'
        lines+=['',f'## {title}完整矩阵（{len(runs)}行）','',r.sm.matrix_table(runs),'',
                f'[全部分位/配置/GPU](results/{prefix}summary.md) · [CSV](results/{prefix}summary.csv)','',
                f'![{title}主SLO矩阵](results/{prefix}slo_heatmap.png)','',
                f'### {title}各系统最佳配置','',best_table([s for s,_ in best]),'',
                f'![{title}最佳主SLO](results/{bp}main_slo.png)','',
                f'![{title}最佳TTFT](results/{bp}ttft_cdf.png)','',
                f'![{title}最佳TPOT](results/{bp}tpot_cdf.png)','']
        rank=sorted((s for s,_ in best),key=lambda s:-s['slo'][MAIN]['rate'])
        lines.append('按已测集合的主SLO排序：'+'；'.join(s['matrix']['system']+' '+s['matrix']['configuration']+' '+pct(s['slo'][MAIN]['rate']) for s in rank)+'。最佳仅针对5s/100ms门限；TTFT、TPOT、吞吐需分别判断，不能推广为所有指标共同排名。')
        if prefix=='big_':
            lines+=['','历史MuxWise TP4主SLO通过109/555条，本次v5 PP2×TP2为108/555条，仅差1条；硬件分配、日期与提交不同，不能据此建立稳定优劣。'
                    'v5 PP4虽较同拓扑v4明显改善，绝对主SLO仍只有54/555条。']
    lines+=['','## B：饱和吞吐对比','',
            '同正式trace压缩到约30秒到达，保持请求长度和客户端参数不变。标准速率下吞吐相近不能证明容量相同；本节测量固定到达压缩和300秒排空截止下的吞吐。'
            'MuxWise与Chunked拓扑按用户指定；MuxWise-PP按A主SLO最高、同分TTFT p50最低选择，见[选择记录](logs/p11b/saturation-selection.json)。'
            '这是指定候选配置的饱和对比，并未遍历所有配置证明全局最大吞吐。','']
    for fam,title in [('tool','Tool&Agent 8B'),('big','Tool&Agent 24B')]:
        group=[x for x in sat if ('big_' in x[0]['label'])==(fam=='big')]
        lines+=['### '+title,'',sat_table(group),'',f'![{title}饱和吞吐](results/sat_{fam}_throughput.png)','']
        rank=sorted((s for s,_ in group),key=lambda s:-s['throughput']['output_tokens_per_sec'])
        leader=rank[0]
        lines.append('输出吞吐排序：'+'；'.join(s['matrix']['system']+' '+fmt(s['throughput']['output_tokens_per_sec'])+' tok/s' for s in rank)+'。'
                     '本次最高的'+leader['matrix']['system']+'比第二名高'+fmt(100*(leader['throughput']['output_tokens_per_sec']/rank[1]['throughput']['output_tokens_per_sec']-1))+'%。'
                     '截止前完成数与TTFT尾部同时见表；部分输出吞吐不等于完成请求吞吐。')
        if fam=='tool':
            lines+=['','v5 PP4本次完成1211条，输出吞吐和完成请求吞吐均为三者最高；但TTFT中位196.363s高于Chunked的172.261s，'
                    'TPOT中位52.050ms高于MuxWise的24.598ms。其TPOT p99为90.960ms，低于另两者；三者TTFT p90/p95/p99均为`>cutoff`。'
                    '因此吞吐领先并不代表所有延迟分位同时领先。']
            lines+=['','`sat_chunked_pp2dp2`保留2条真实客户端失败：idx4176与4370的服务响应`cached_tokens`分别为12288与1024，'
                    '超过对应输入长度7311与901，被原客户端校验拒绝（HTTP200，非正常完成）。'
                    '同次服务日志有KV满后的request retraction，未发现CUDA OOM；未对具体因果作额外归因，也未修改校验、剔除失败或重跑。'
                    '见[精简失败记录](logs/p11b/sat_chunked_pp2dp2/failure-records.json)与[worker0日志](logs/p11b/sat_chunked_pp2dp2/worker0.log)。']
        else:
            lines+=['','v5 PP2×TP2本次输出吞吐、完成数和TTFT中位数均为三者最佳；其TPOT中位66.130ms高于MuxWise的31.269ms，'
                    '但p99为121.852ms，低于MuxWise的172.638ms和Chunked的15242.762ms。'
                    'TPOT按首末token间隔平均，包含生成期间的调度/回退停顿，不等同于GPU纯计算时间。']
    lines+=['','六组均未在截止前完成全部请求，故“全部完成耗时”均留空；不会据部分完成速度外推一个完成全部请求的时间。'
            'TPOT分位仅描述各系统截止前完成且输出超过1 token的请求；未完成较多时，样本集合不同，不能解释为全体请求的decode延迟排名。'
            '输入吞吐包含缓存命中token，不能解释为每秒实际重新执行prefill的token数。','',
            '[饱和完整JSON](results/sat_summary.json) · [全部统计CSV](results/sat_summary.csv)。PNG另有同名SVG。','',
            '## MuxWise-PP v5 调度优化','',
            '以下机制和开发数字来自[P11A_NOTES.md](P11A_NOTES.md)，不充当P11b正式结果。stage0对整批FCFS输入生成token块计划，短请求与长请求尾部可以打包；'
            '各stage FIFO执行，子块共享已分配的完整输入KV并保留radix锁，按块READY/接收槽和CUDA event管理生命周期，整父批完成后在MERGE处返回首token。'
            '这减少长prefill造成的流水线等待；块内仍沿用v4按层切片、RLS/ALP与原分区控制，没有新增动态chunk-size控制器。','',
            '可选token在途额度取代批数门控；按末级完成的实际token归还额度，16384-token回补门槛减少零碎准入。'
            '完整KV和未来decode输出预留保持；超额度FCFS单请求仅在无未完成工作时进入，避免饿死。'
            '中间块跳过norm/head/采样，终结span使用basic slice视图，避免Python列表GPU索引同步。'
            '可选CPU长度、前缀和与FlashInfer规划元数据减少GPU→CPU读回，KV索引仍在GPU构造。','',
            '| 所选拓扑（开发轮次） | 开发标准主SLO v4→v5 | 开发标准输出tok/s v4→v5 | 开发饱和输出tok/s v4→v5 |',
            '| --- | --- | --- | --- |',
            '| 8B PP4 (R5) | 85.333%→93.000% | 342.74→345.06 | 474.46→514.26 |',
            '| 8B PP2×TP2 (R2) | 73.333%→84.333% | 354.80→352.04 | 488.12→478.78 |',
            '| 24B PP2×TP2 (R6) | 30.000%→43.333% | 111.75→112.38 | 160.60→167.54 |',
            '| 24B PP4 (R5) | 4.000%→23.333% | 95.66→105.84 | 109.07→133.45 |','',
            '已知取舍：8B PP2×TP2开发主SLO提高，但三档输出吞吐略降；24B PP2×TP2启用CPU元数据后标准宽松SLO为68.0%，低于v4的69.333%及R2的80.0%。'
            'PP2×DP2附测无成对v4基线，标准主SLO89.667%，饱和仅351.37 tok/s，伴随worker负载不均；因此不能仅凭低中位TTFT推断饱和容量最佳。'
            '最终24B PP4 cold20仍为19/20 exact，与v4已知分叉相同，不宣称全模型逐token位等价。开发样本仅8B前300/24B前150条，不能与正式1551/555条混排。','',
            '## 审计、历史证据与清理','',
            '原始数据位于 `results/<label>_requests.jsonl` 与summary；每label正式锁指向 `logs/p11b/<label>`。'
            '运行目录保存命令、环境、预热/冒烟、server_info、GPU/rank映射及5秒采样，旁侧保存逐秒外部监测与启动前基线。'
            '[运行进度](logs/p11b/progress.jsonl) · [最终审计](logs/p11b/final-audit.json)。','',
            '历史Chunked PP4/TP2PP2失败数据和PD截止未完成均保留在矩阵。旧版实现与异常、P10c Llama A/B及24B显存优化说明见'
            '[更新前完整报告](logs/p11b/RESULTS-before-P11b.md)、[P10B](P10B_NOTES.md)、[P11A](P11A_NOTES.md)。'
            '这些测量跨源码提交、日期和GPU/NUMA分配；单次排序不证明稳定优势。','',
            'CPU重建：`source env.sh && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/p11b_report.py`。'
            '该入口只读已保存请求，不启动模型或重跑label。最终GPU/进程/端口状态见最终审计。','']
    lines+=['### 本次正式客户端启动时间（UTC）','',
            '各行使用上表相同UUID与NUMA0分配；历史数据日期保留在各summary的`run_started_unix`及原正式锁中。','',
            '| label | 正式启动UTC | GPU启动前基线 |','| --- | --- | --- |']
    for label in batch['labels']:
        marker=load(OUT/(label+'.formal-started.json'))
        baseline=load(LOG/(label+'-before-baseline.json'))
        memory=[g['memory_mib'] for v in baseline for g in v['gpu']]
        utilization=[g['utilization'] for v in baseline for g in v['gpu']]
        lines.append(f'| {label} | {marker["started"]} | 四卡显存{min(memory):.4f}–{max(memory):.4f}MiB，利用率{min(utilization)}–{max(utilization)}% |')
    lines+=['','全部GPU实验结束后执行35秒空闲检查；最终审计再次核对四卡显存/利用率、容器内进程和服务端口。'
            '运行中未检测到外部GPU占用。NVML设备显存与进程显存计数非原子采样，少量差额按驱动/采样差异记录；不代表允许外部进程。','']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines))
    save(LOG/'report-index.json',dict(standard=[[s['label'] for s,_ in runs] for runs,_ in allruns],
         best=[[s['label'] for s,_ in best] for _,best in allruns],saturated=[s['label'] for s,_ in sat]))


if __name__=='__main__':main()
