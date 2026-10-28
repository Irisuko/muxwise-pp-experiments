#!/usr/bin/env python3
"""Render P10B notes from completed development evidence and selected commands."""
import json,subprocess
from pathlib import Path
root=Path('/root/workspace/muxpp_work');log=root/'logs/p10b'
read=lambda p:json.loads(Path(p).read_text())
sel=read(root/'configs/p10b_selected.json');head=subprocess.check_output(['git','-C','/root/workspace/sglang','rev-parse','HEAD'],text=True).strip()
body_file=log/'notes-evidence-section.md'
if not body_file.exists():
 old=(root/'P10B_NOTES.md').read_text()
 body_file.write_text(old[old.index('## 实验边界与证据'):old.index('## 预检与开发轮次')])
body=body_file.read_text()
body=body.replace('源码两次提交的GPU验证范围分别标明', '每次提交的GPU验证范围分别标明')
body=body.replace('`correctness-r0.json`、`correctness-r1b.json` 与各run的comparison.json', '`correctness-r0.json`、`correctness-r1b.json`、`correctness-r2e.json` 及 `r4-mux-pp4/*-comparison.json`')
body=body.replace('R1b全20条验证已消除pos0–2阻断', 'R1b、R2e及最终R4全20条验证已消除pos0–2阻断')
body=body.replace('| 新R2候选 | 未跑 | 未跑 | 资源阻断 |','| R2e PP4 | 20/20，1280/1280 | 19/20，1252/1280 | 仅req12 pos35；无早期分叉 |\n| R2e PP2×TP2 | 20/20，1280/1280 | 20/20，1280/1280 | 无 |\n| R4 PP4 | 20/20，1280/1280 | 19/20，1252/1280 | 仅req12 pos35；无早期分叉 |')
text=f'''# P10B：大模型正确性、显存与调度开发复核

已完成显存与早期数值分叉修复、四轮开发验证、一次Llama回归及六配置启动命令整理。**验收限制：PP4仍有显著延迟差距；Llama回归TTFT/主SLO较历史基准变差，未达到“确认未退化”的要求。** 这些限制保留为未解决项，不能将本次选择当作正式性能结论。未跑正式实验，未push。

起点`03e2fe794`，分支muxwise-pp。提交：`89ccd11ce`（MLP尾块与主要显存修复）、`a713881f9`（unused wrapper/MLP生命周期）、`4c10135d2`（可选附加上下文校准）。最终HEAD `{head}`。

'''+body
text+='''## 最终配置显存剖析

最终两组MuxWise-PP保留双prefill接收槽、8个分区、每分区graph bs1/4/8/16/32；FlashInfer prefill/decode各独立128MiB float workspace。以下是独立31k剖析，不混入性能测量。单位GiB。

| 拓扑 | stage/TP | weights | KV+pool | backend | graph增量 | receive/TX/RX | peak allocated | peak reserved | peak inactive_split |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
'''
memory_sources=[]
for topo in ['pp4','pp2tp2']:
 cfg=sel['big_muxpp_'+topo];prefix=Path(cfg['development']).name.split('-dev-')[0];path=log/f'{prefix}-profile-{topo}';m=read(path/'memory-report.json')
 for v in m['stages']:
  vals=[(v['init'][k]['allocated']-(v['init'][p]['allocated'] if p else 0))/2**30 for k,p in [('weights',None),('kv','weights'),('backend','kv'),('graphs','backend'),('slots','calibration')]]
  vals += [v['peak_allocated']/2**30,v['peak_reserved']/2**30,v['peak_inactive_split']/2**30]
  text+=f"| {topo} | {v['stage']}/{v['tp']} | "+' | '.join(f'{x:.3f}' for x in vals)+' |\n'
 memory_sources.append(f'`{path.relative_to(root)}/memory-report.json`')
text+='\n来源：'+'、'.join(memory_sources)+'。\n\n'
text+='''GPU验证提交范围：R2e两拓扑为a713881f9，R4 PP4和Llama回归为最终4c10135d2。后者只新增可选校准点；PP2×TP2的extra列表仍为空，其校准网格与R2e逐点相同，因此沿用R2e数据，未将它标成最终HEAD重新测得。

最终backend阶段从原1.605降到约.344GiB/卡。R2逐pointer清单显示：各decode backend的wrapper整数工作区48MiB（基础8MiB加5个graph wrapper各8MiB），8个backend合计384MiB；prefill backend另有ragged/paged/decode三个8MiB整数工作区。40个graph的wrapper部分包含在graph初始化增量内，不与backend/graph列重复相加。decode float workspace128MiB由8分区共享，prefill另有独立128MiB。每个decode backend的KV indices约4MiB；原始清单见ready.backend_tensors，分组表见`r2e-profile-pp4/graph-workspace-breakdown.json`。private graph pool在R2快照中保留192MiB，全部graph共享，不是每分区192MiB。

`a713881f9`仅在opt-in路径于SwiGLU后释放已无用gate/up，避免它与down投影输出重叠；再省去decode-only ragged与无speculative用途的verify wrapper约136MiB常驻。31k attention仍完整计算，未引入token chunk的调度语义。最终长请求峰值与结束资源守恒均有实际GPU证据。

## 预检与四轮开发结果

| 预检 | 31k耗时s | 20条冒烟完成/失败 | 冒烟耗时s | KV |
| --- | ---: | --- | ---: | ---: |
'''
for prefix,topos in [('r1b',['pp4','pp2tp2']),('r2e',['pp4','pp2tp2']),('r3',['pp4']),('r4',['pp4'])]:
 for topo in topos:
  path=log/f'{prefix}-preflight-{topo}';d=read(path/'run.json');w=read(path/'long-warmup.json');s=d['smoke'];info=next(iter(d['server_info'].values()))
  text+=f"| {prefix}/{topo} | {w['elapsed_s']:.3f} | {s['completed']}/{s['failed']} | {s['duration_sec']:.3f} | {info['max_total_num_tokens']} |\n"
text+='''
P10B冒烟是开发前20条、time-scale=.2、原输出长度、客户端上限120s，要求20/20；P10A为60s限时窗口，不混作同一门槛。所有最终stage的free_kv+evictable=capacity、free_reqs=128、接收槽全部归还，见各run的p10b-audit.json。

R1：单接收槽、384MiB float workspace、原.75系数。R2：双槽、128MiB workspace、PP4系数.90，PP2保持.75。R3只将PP4在途上限6→4；R4恢复6，系数.99，新增12k/24k校准点，混合分区改为104/24、96/32、88/40、80/48、72/56、64/64，总数仍8。R3/R4只改PP4，PP2沿用R2，未重复不变配置。总共四轮，所有性能行均150完成/0失败/0未完成。

主SLO为TTFT≤5s且TPOT≤100ms，以下都是指定开发前150条、time-scale=1的单次记录。

| 轮次/拓扑 | TTFT p50/p90/p95/p99 s | TPOT p50/p90/p99 ms | 主SLO | cache加权比 | 总时长s |
| --- | --- | --- | ---: | ---: | ---: |
'''
results=read(log/'results.json')
for d in results:
 text+=f"| {d['round']}/{d['topology']} | "+' / '.join(f"{d['ttft'][k]:.3f}" for k in ['p50','p90','p95','p99'])+' | '+' / '.join(f"{d['tpot'][k]*1000:.3f}" for k in ['p50','p90','p99'])+f" | {d['slo']['passed']}/150 | {d['cache']['token_weighted_ratio']*100:.3f}% | {d['duration']:.3f} |\n"
text+='''
最终PP4 KV相对P10A增加52.91%，达到同拓扑Chunked的88.64%；PP2×TP2增加24.72%，达到对应Chunked的85.76%。容量目标达到，长预热没有OOM。

## PP4归因与选择

每轮P9a分析匹配150/150，既保留p50/p90对应请求分解，也保留每分量独立分布。下表p50按对应请求插值后分组，非各分量的独立中位数。

| PP4轮次 | p50 TTFT | 准入 | stage0本地FIFO | 四级prefill | 三级链路/队列 | 其余 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
'''
for prefix in ['r1','r2e','r3','r4b']:
 d=read(log/f'{prefix}-dev-pp4/analysis.json');q=d['quantile_decomposition']['p50'];c=q['components'];ad=c['admission_queue'];local=c['stage0_local_queue'];compute=sum(v for k,v in c.items() if k.endswith('_prefill_span'));links=sum(v for k,v in c.items() if k.startswith('link'));rest=q['ttft']-ad-local-compute-links
 text+=f"| {prefix} | {q['ttft']:.3f} | {ad:.3f} | {local:.3f} | {compute:.3f} | {links:.3f} | {rest:.3f} |\n"
text+='''
| PP4轮次 | inflight阻塞s | KV阻塞s | decode reserve阻塞s | 平均准入+stage0队列s | stage3等上游% |
| --- | ---: | ---: | ---: | ---: | ---: |
'''
for prefix in ['r1','r2e','r3','r4b']:
 d=read(log/f'{prefix}-dev-pp4/analysis.json');w=d['full']['admission']['wall'];means=d['component_distributions'];u=d['full']['stages']['3']['wall_percent']['prefill'].get('upstream',0)
 text+=f"| {prefix} | {w.get('inflight',0):.3f} | {w.get('kv',0):.3f} | {w.get('decode_reserve',0):.3f} | {means['admission_queue']['mean']+means['stage0_local_queue']['mean']:.3f} | {u:.3f} |\n"
text+='''
阻塞时间是可匹配summary窗口的状态驻留，不是可相加的逐请求延迟。R1的单槽、保守分区和队列共同限制流水线；R2恢复双槽后，stage0 READY驻留从约32%降至8%，末级等上游从约41%降至28%。但R2 KV阻塞时protected中位数174976/181066，准入仍受长上下文活跃请求占用影响。R3减小在途上限，KV/预留阻塞减少，却增加inflight等待，未改善主SLO，因此不选。

R2实际decode微批以1–3条为主；约19.22%的decode观测mean context在8–12k，7.97%在16–24k，旧上取整网格把它们按16k/32k估计。R4补测12k/24k并增加40/56 SM中间档，以减少分区选择的粗粒度损失。仍使用实测上界、单调上包络、至少1.2在线安全修正、1ms/级开销，不插值或外推；PP4预测预算99ms，完整未来输出KV预留、FCFS和PP/TP协议未放松。新增pp_decode_extra_contexts默认空，Llama v4原始校准点不变。

'''
for topo in ['pp4','pp2tp2']:
 cfg=sel['big_muxpp_'+topo];d=read(Path(cfg['development'])/'analysis.json');s=d['summary'];text+=f"最终{topo}选择`{Path(cfg['development']).name}`：TTFT p50 {s['latencies_sec']['ttft']['p50']:.3f}s、主SLO {s['slo']['ttft_le_5s_and_tpot_le_100ms']['passed']}/150、KV {cfg['kv_capacity']}。\n\n"
text+='''R4末级混合分区驻留主要为88/40约19.77%、80/48约39.04%，旧R1的64/64约74.03%降为0；减少的decode保留SM留给prefill，但TPOT尾部仍高，说明预测预算并非端到端SLO保证。

R1/P10A与恢复后的R2–R4使用了不同GPU UUID分配，虽然均为4×4090且NUMA拓扑一致，不能将跨分配的全部差值归于单项代码变化。R2–R4在同一恢复分配上运行，但仍为单次开发结果。PP4的主SLO、尾延迟及相对PP2差距均按上表披露；四轮上限到达后不增加第五轮，也不宣称PP4所有性能问题已消失。

## Llama8B v4回归与测试

'''
comp=read(log/'llama-serial20-comparison.json');now=read(next((log/'llama-dev300').glob('*_summary.json')));base=read(next((root/'logs/p9a/r1-pp2tp2-toolagent').glob('*_summary.json')))
text+=f"最终HEAD PP2×TP2 v4 serial20：{comp['equal_requests']}/20请求、{comp['equal_tokens']}/1280 token完全一致，参照为P9a普通TP2归档。新增三项显存/数值开关均为off，原v4 YAML SHA256仍为`3bd2a392dcaae12cac690aa48de8f671de5b8cbff76769a5a388236d3aa784ae`。Tool&Agent仅回归原Llama开发trace前300条一次；未使用Mistral tokenizer的trace冒充Llama对照。\n\n"
text+='| Llama PP2×TP2 | 完成/失败/未完成 | TTFT p50/p90/p95/p99 s | TPOT p50/p90 ms | 主SLO | in/out tok/s |\n| --- | --- | --- | --- | ---: | --- |\n'
for name,d in [('P9a v4开发基准',base),('P10B最终HEAD回归',now)]:
 text+=f"| {name} | {d['completed']}/{d['failed']}/{d['unfinished']} | "+' / '.join(f"{d['latencies_sec']['ttft'][k]:.3f}" for k in ['p50','p90','p95','p99'])+' | '+' / '.join(f"{d['latencies_sec']['tpot'][k]*1000:.3f}" for k in ['p50','p90'])+f" | {d['slo']['ttft_le_5s_and_tpot_le_100ms']['passed']}/300 | {d['throughput']['input_tokens_per_sec']:.2f}/{d['throughput']['output_tokens_per_sec']:.2f} |\n"
text+='''
**Llama性能非退化未获确认。** TTFT p50增加27.95%，主SLO下降19条（6.33个百分点）；TPOT p50改善3.69%，吞吐仅下降约0.04%。共同server参数仅random_seed、trace路径和等内容的v3/v4 YAML路径不同；KV仍390622，三个新开关均关闭，新增context网格为空。历史性能数据来自P9a选定R1提交a17263c91，并非在本次GPU分配下测得的P10B改动前基线，因此不能据单次差异断言P10B代码是原因，也不能宣称未退化。

300/300 TTFT事件匹配：平均准入基本持平（.891890→.893128s），本地FIFO .669859→.833156s、级间队列 .207217→.319282s，两级prefill均值合计1.413155→1.454491s。增加的平均TTFT主要集中在本地/级间排队；分区驻留和GPU计算占比接近历史。按要求只运行一次300条，没有为追平历史单次数据改动Llama参数或追加回放。判定及分解保存于`logs/p10b/llama-regression-assessment.json`，selected元数据明确标为NOT_ESTABLISHED。

回归证据：`logs/p10b/llama-serial20-comparison.json`、`logs/p10b/llama-dev300/analysis.json`；两次历史环境分配差异同样保留，不给统计显著性承诺。最终相关CPU测试71 passed（unit-r4.log），客户端10 passed（bench-unit-final.log）；Black及git diff --check通过。附加网格默认与v4完全相同、上界覆盖和非法配置拒绝均有测试，另覆盖PP控制、准入、KV预留、TP重放、单槽生命周期、MLP数学/尾块/张量生命周期及head引用环释放。

## 最终六配置启动命令

`configs/p10b_selected.json`记录选择和验收路径，六个`launch_big_*.sh`已接入`p10b_serve.py`。仅两组MuxWise-PP改变参数；MuxWise TP4与三组Chunked的argv和env已逐项与原P10A比较，完全相同。这四组容量继承P10A证据，本任务没有重跑其开发实验。

| label | mem | KV容量 | 来源 |
| --- | ---: | ---: | --- |
'''
legacy={'big_muxwise_tp4':(.76,167692),'big_chunked_tp4':(.90,252288),'big_chunked_pp4':(.90,204263),'big_chunked_pp2tp2':(.90,236846)}
commands=read(log/'final-launch-commands.json')
for label in commands:
 if label in legacy:mem,kv=legacy[label];source='P10A，命令不变'
 else:mem,kv=sel[label]['mem_fraction'],sel[label]['kv_capacity'];source=Path(sel[label]['preflight']).name
 text+=f'| {label} | {mem:.2f} | {kv} | {source} |\n'
text+='\n先执行 `source /root/workspace/muxpp_work/env.sh`。对应六个绝对路径脚本均可直接启动；下面是其展开命令（不包含开发trace插桩）：\n\n'
for label,jobs in commands.items():text+=f'### {label}\n\n```bash\n'+'\n'.join(v['command'] for v in jobs)+'\n```\n\n'
text+='''## 失败记录、环境恢复与清理

- R0剖析PP4的31k预热通过，但旧60s冒烟18/20，run记FAIL，非OOM。P10B改用明确标注的120s完整冒烟要求。
- `r1-mux-pp4`暴露单槽idle检查仍硬编码2、unused head bound-method引用环未GC；修复后以r1b重跑，旧证据保留。
- `r2-preflight-pp4`在31k MLP down投影申请40MiB时真实OOM；a713881f9回收额外工作区和无用gate/up后，R2e起长请求及并发冒烟均通过。
- 前一会话r2b/r2c启动时GPU2/3已有容器内不可见占用，权重加载OOM，未计有效开发轮次。该会话已清理本任务进程并报告阻断。用户恢复独占后重新确认四卡1MiB/0%；UUID变更记录于resume-gpu-map.csv、resume-topology.txt。
- /dev/shm模型丢失后从私有盘恢复，16文件、47,162,178,592 bytes完整SHA256通过、363张量与索引一致，见resume-model-validation.json。没有更换模型快照。
- R4首次开发启动前，四卡显存均1MiB，但GPU1单次利用率52%，guard立即退出，未启动服务或发送trace。随后六次采样20秒全部1MiB/0%，运行guard未见foreign PID，更符合清理后的利用率窗口滞后；原始信号与复核均保留。退出后增加2秒采样稳定时间，以r4b新标签完成同一R4，未额外增加开发迭代。

本次恢复后独占监测没有确认新的持续外部占用；若出现外部PID或连续无法归属的GPU显存，guard会中断本任务runner，不杀外部进程或reset GPU。最终GPU/进程/端口清理与formal文件审计见`logs/p10b/final-cleanup.json`、`logs/p10b/final-audit.json`。正式锁、P10A冻结/选择文件及正式启动标记保持不变，没有新增正式回放，没有push。
'''
(root/'P10B_NOTES.md').write_text(text)
print('Wrote P10B_NOTES.md from completed evidence')
