#!/usr/bin/env python3
"""CPU-only P10c reporting from saved one-shot measurements."""
import json, sys, shlex, subprocess
from pathlib import Path
import p9b_report as r
ROOT=r.ROOT
OUT=ROOT/'results'
LOG=ROOT/'logs/p10c'
MAP={'big_muxwise_tp4':('MuxWise','TP4'),'big_chunked_tp4':('Chunked','TP4'),
     'big_chunked_pp4':('Chunked','PP4'),'big_chunked_pp2tp2':('Chunked','PP2×TP2'),
     'big_muxpp_pp4':('MuxWise-PP','PP4'),'big_muxpp_pp2tp2':('MuxWise-PP','PP2×TP2')}
MAIN=r.sm.MAIN_SLO
def load(p):return json.loads(p.read_text())
def fmt(v):return f'{v:.3f}' if isinstance(v,(int,float)) else str(v)
def pct(v):return f'{v*100:.3f}%'
def q(s,k,m=1):return '/'.join(fmt(s['latencies_sec'][k][p]*m) if isinstance(s['latencies_sec'][k][p],(int,float)) else s['latencies_sec'][k][p] for p in ['p50','p90','p95','p99'])

def ab():
    rows=[]
    for which,head in [('old','f6833f24e'),('new','4c10135d2')]:
        path=LOG/('llama-'+which)
        summary=path/f'p10c-llama-{which}_summary.json'
        s,records=r.sm.load_runs([str(summary)])[0]
        raw=load(summary)
        for k in ['completed','unfinished','failed','latencies_sec','slo','throughput','cached_tokens']:assert raw[k]==s[k],k
        meta=load(path/'run.json')
        rows.append(dict(which=which,head=head,summary=s,meta=meta))
    a,b=[v['summary'] for v in rows]
    delta=100*(b['slo'][MAIN]['rate']-a['slo'][MAIN]['rate'])
    worse=delta < -5 and b['latencies_sec']['ttft']['p50']>a['latencies_sec']['ttft']['p50']
    value=dict(rows=rows,main_slo_delta_pp=delta,ttft_p50_delta_pct=100*(b['latencies_sec']['ttft']['p50']/a['latencies_sec']['ttft']['p50']-1),
               meets_regression_threshold=worse,criterion='new main SLO lower by >5 percentage points AND TTFT p50 worse')
    r.save(LOG/'llama-ab.json',value)
    lines=['| 版本 | 完成/未完成/失败 | TTFT p50/p90/p95/p99(s) | TPOT p50/p90/p95/p99(ms) | 主SLO | 输入/输出tok/s | cache |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        s=row['summary'];sl=s['slo'][MAIN];t=s['throughput']
        lines.append(f'| {row["head"]} | {s["completed"]}/{s["unfinished"]}/{s["failed"]} | {q(s,"ttft")} | {q(s,"tpot",1000)} | {sl["passed"]}/300 ({pct(sl["rate"])}) | {fmt(t["input_tokens_per_sec"])}/{fmt(t["output_tokens_per_sec"])} | {pct(s["cached_tokens"]["token_weighted_ratio"])} |')
    conclusion=(f'本次同分配顺序A/B：新版主SLO变化{delta:+.3f}个百分点，TTFT p50变化{value["ttft_p50_delta_pct"]:+.2f}%。'
        +('达到预先规定的“明显更差”门槛，应标注代码回归嫌疑；这对单次实验仍不能定位到某一提交。' if worse else
          '未达到预先规定的“主SLO下降超过5个百分点且TTFT p50更差”门槛，本次未复现所定义的明显退化。单次结果不能证明严格性能等价，也不能把历史差异全部归因于GPU分配。'))
    text='\n'.join(lines)+'\n\n'+conclusion+'\n'
    olda,newa=[load(LOG/('llama-'+w)/'analysis.json') for w in ['old','new']]
    text+=('\n两版逐请求TTFT均匹配300/300。新版TTFT p90为'+fmt(b['latencies_sec']['ttft']['p90'])+
           's（旧版'+fmt(a['latencies_sec']['ttft']['p90'])+'s），因此不声称所有分位改善。平均分解如下，各分量均值可加，但不等于中位数分解。\n\n'
           '| 平均TTFT分量(s) | 旧版 | 新版 |\n| --- | ---: | ---: |\n')
    for key,title in [('admission_queue','准入'),('stage0_local_queue','stage0 FIFO'),('stage0_prefill_span','stage0 prefill'),
                      ('link0_and_stage1_queue','链路/stage1队列'),('stage1_prefill_span','stage1 prefill')]:
        text+=f'| {title} | {fmt(olda["component_distributions"][key]["mean"])} | {fmt(newa["component_distributions"][key]["mean"])} |\n'
    (LOG/'llama-ab.md').write_text(text)
    return text,value

def main():
    abtext,abdata=ab()
    if '--ab-only' in sys.argv:print(abtext);return
    runs=r.prepare(MAP,555);best=r.best(runs)
    from p10c_stages import report as stage_report
    stage_text=stage_report()
    r.tables(runs,'big_','大模型完整矩阵：Mistral-Small-24B，六配置各一次')
    r.tables(best,'big_best_','大模型三系统最佳配置（按5s/100ms主SLO选择）')
    for prefix in ['big_','big_best_']:
        p=OUT/(prefix+'summary.md')
        p.write_text(p.read_text().replace('cache命中率†','cache命中率').replace('† PD cache仅为decode响应。','').replace('PD decode响应的缓存字段不代表prefill实际命中。',''))
    r.sm.plot_runs(runs,OUT,'big_');r.sm.plot_runs(best,OUT,'big_best_')
    r.barplot(runs,'big_main_slo.png','Mistral-Small-24B: six configurations, one measurement each')
    r.barplot(best,'big_best_main_slo.png','Mistral-Small-24B: best measured configuration per system')
    frozen=load(LOG/'frozen.json')
    commands=['source /root/workspace/muxpp_work/env.sh']
    configlines=['| label | 拓扑 | mem fraction | 实测实例KV(token) | 完整命令 |','| --- | --- | ---: | ---: | --- |']
    bylabel={s['label']:s for s,_ in runs}
    for label,jobs in frozen['commands'].items():
        for job in jobs:commands+=['', '# '+label,shlex.join(['env']+[k+'='+v for k,v in job['env'].items()]+job['argv'])]
        s=bylabel[label];m=s['matrix']
        configlines.append(f'| {label} | {m["configuration"]} | {next(iter(m["mem_fraction"].values())):.2f} | {next(iter(m["kv_capacity_tokens_per_instance"].values()))} | [argv/env](logs/p10c/{label}/commands.json) |')
    (OUT/'big_commands.sh').write_text('\n'.join(commands)+'\n')
    bestlines=['| 系统 | 本次最佳配置 | 主SLO（达标/全部） | TTFT p50/p99(s) | TPOT p50/p99(ms) | 输出tok/s |',
               '| --- | --- | --- | --- | --- | --- |']
    for s,_ in best:
        sl=s['slo'][MAIN];l=s['latencies_sec'];m=s['matrix']
        bestlines.append(f'| {m["system"]} | {m["configuration"]} | {pct(sl["rate"])} ({sl["passed"]}/555) | {fmt(l["ttft"]["p50"])}/{fmt(l["ttft"]["p99"])} | {fmt(1000*l["tpot"]["p50"])}/{fmt(1000*l["tpot"]["p99"])} | {fmt(s["throughput"]["output_tokens_per_sec"])} |')
    breakdown=['| label | 达标 | 仅TTFT>5s | 仅TPOT>100ms | 两项超标 | TPOT未定义 | 未完成/未发 |',
               '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for s,_ in runs:
        x=s['main_slo_breakdown']
        breakdown.append(f'| {s["label"]} | {x["passed"]} | {x["ttft_only"]} | {x["tpot_only"]} | {x["both"]} | {x["undefined_tpot"]} | {x["not_completed"]+x["unsent"]} |')
    gm=['| CUDA索引 | UUID | PCI bus | NUMA |','| --- | --- | --- | --- |']
    for g in frozen['gpu_mapping']:gm.append(f'| {g["index"]} | {g["uuid"]} | {g["pci_bus_id"]} | {g["numa"]} |')
    rank=sorted((s for s,_ in best),key=lambda s:-s['slo'][MAIN]['rate'])
    ranking='；'.join(f'{s["matrix"]["system"]} {s["matrix"]["configuration"]} {pct(s["slo"][MAIN]["rate"])}' for s in rank)
    quantiles=['| label | TTFT p50/p90/p95/p99(s) | TPOT p50/p90/p95/p99(ms) | E2E p50/p90/p95/p99(s) |',
               '| --- | --- | --- | --- |']
    for s,_ in runs:quantiles.append(f'| {s["label"]} | {q(s,"ttft")} | {q(s,"tpot",1000)} | {q(s,"e2e")} |')
    pp4,pp2=bylabel['big_muxpp_pp4'],bylabel['big_muxpp_pp2tp2']
    ppdelta=100*(pp2['slo'][MAIN]['rate']-pp4['slo'][MAIN]['rate'])
    winner,second=rank[:2]
    comparison=(f'本次最佳与次名的主SLO差为{100*(winner["slo"][MAIN]["rate"]-second["slo"][MAIN]["rate"]):.3f}个百分点，'
                f'即{winner["slo"][MAIN]["passed"]-second["slo"][MAIN]["passed"]}条请求。')
    comparison+='三系统最佳配置的TPOT p99依次为'+ '；'.join(
        f'{s["matrix"]["system"]} {fmt(1000*s["latencies_sec"]["tpot"]["p99"])}ms' for s,_ in best)+'。'
    cp4=bylabel['big_chunked_pp4']
    comparison+=(f'同拓扑PP4中，MuxWise-PP的TPOT p99为{fmt(1000*pp4["latencies_sec"]["tpot"]["p99"])}ms，'
                 f'Chunked为{fmt(1000*cp4["latencies_sec"]["tpot"]["p99"])}ms；'
                 f'但其TTFT p50为{fmt(pp4["latencies_sec"]["ttft"]["p50"])}s，高于Chunked的{fmt(cp4["latencies_sec"]["ttft"]["p50"])}s。'
                 '应分别看待decode尾部与首token排队，不把主SLO排名等同于所有指标支配。')
    text='''
## 大模型场景（Mistral-Small-24B，必须跨4卡）

P10c在同一时段、同一四卡物理分配下先做Llama开发A/B，再按预定顺序完成六个大模型配置。每个正式配置只运行一次；推理源码固定为`4c10135d2`，沿用P10B已选参数，没有根据正式结果调参。A/B不计入正式结果。

### 场景、模型与负载

Llama-8B在本机用TP2已足够，可用两组双卡实例增加服务并行度。24B的本任务BF16/32k场景必须跨四卡：权重47,144,806,400 bytes加单条32k序列KV 5,368,709,120 bytes，已超过两卡物理容量51,514,441,728 bytes，尚未计入激活、graph、工作区和通信。这不是“权重文件本身装不进两卡”的结论。四张RTX4090没有GPU P2P，TP4每层归约需跨PCIe/主机通信；这为PP或PP2×TP2提供了减少TP通信的动机，但PP的串行stage、队列和KV约束也会产生代价，胜负由本次测量判断。

模型为`Mistral-Small-24B-Instruct-2501`：40层、hidden5120、MLP32768、32 attention heads、8 KV heads、head_dim128、BF16、context32768。运行位置`/dev/shm/models/Mistral-Small-24B-Instruct-2501`；持久备份`/root/autodl-pvt-A40-hdd-1/models/Mistral-Small-24B-Instruct-2501`。shm会随容器重启丢失，启动前调用原`p10a_model.py ensure`校验/恢复。快照为16文件、47,162,178,592 bytes、363张量，来源、逐文件SHA256与存储复核见[P10A_NOTES](P10A_NOTES.md)及[manifest](configs/p10a_model_manifest.json)。

正式负载取Mooncake Tool&Agent的[600,680)秒窗口，到达时间拉长7.5倍，按Mistral tokenizer生成确定性token和hash前缀，过滤输入+输出>32768的请求。普通token为1000–131071、BOS=1，不复用Llama tokenizer的输入。开发冒烟另用[1200,1280)秒窗口。首次hash出现口径理论复用率不等于实际cache命中。

| 负载统计 | 正式值 |
| --- | ---: |
| 窗口请求/过滤/保留 | 570/15/555 |
| 规划窗口/最后到达(s) | 600/585 |
| 总输入/输出token | 3,997,588/97,918 |
| 输入mean/p50/p90/p95/p99/max | 7202.861/6178/15163.2/20614.1/29040.12/31612 |
| 输出mean/p50/p90/p95/p99/max | 176.429/28/482/571.3/795.38/1987 |
| 规划输入/输出token/s（600s分母） | 6662.647/163.197 |
| 理论前缀复用率 | 45.9797% |

[完整负载统计](bench/big_toolagent_600_80_x7.5.stats.json)。实际吞吐分母为客户端运行时长（含排空），不与规划600s速率混用。

### 配置、执行与物理分配

四卡始终按0,1,2,3排列；PP2×TP2的同NUMA TP组为0+1与2+3，PP跨NUMA。PP4按0→1→2→3。[拓扑](logs/p10c/topology.txt)、[冻结文件](logs/p10c/frozen.json)保存完整环境与哈希。

'''+ '\n'.join(gm)+'''

所有服务统一BF16、FlashInfer、FCFS、disable-overlap、max-running128、context32768、graph bs1/4/8/16/32，TP>1禁用custom allreduce。Chunked使用chunk2048，只有TP4开启mixed chunk。MuxWise使用原pdmux配置；MuxWise-PP保留整请求prefill、4096行MLP分块、budget16384、双接收槽、PP4/PP2在途6/3，最终显存开关及PP4校准/分区见P10B。

'''+ '\n'.join(configlines)+'''

容量是四卡单实例的逻辑token位置数，不乘stage/rank数。[六条完整展开命令](results/big_commands.sh)与[选参依据](P10B_NOTES.md)逐参数一致。

执行入口为`source env.sh && python scripts/p10c_runner.py formal`。沿用P7生命周期：固定31663-token/8输出长预热、flush、开发前20条time-scale0.2冒烟；MuxWise/MuxWise-PP冒烟后再次长预热；正式前flush。客户端统一`--warmup 2 --force-close --drain-timeout 300`，正式time-scale1、无限制条数。冒烟同样使用300s排空并要求20/20；未沿用早期P7/P10A会截断PP4长输出的55s单请求/60s外层上限。日志记录实际冒烟时长，不把此变化当作性能收益。

每次启动前确认四卡约1MiB、0%利用率且无残留进程；运行中监测外部占用，5s采样GPU。P7 GPU flock、每label启动记录与排他的formal-started标记防止重跑。P10a准备阶段锁按本次明确授权临时归档，六组结束后原样恢复；客户端/metrics未修改。[编排](scripts/p10c_runner.py)保存commands、server_info、日志、rank映射及清理证据。

### 六配置完整结果

统计口径沿用本文前节和bench/README：SLO分母为全部555条，未完成/失败/未发/未定义均不达标；TTFT/E2E保留全样本及无穷尾部，TPOT仅统计已完成且输出>1者。cache为成功响应的输入token加权命中率。

'''+r.sm.matrix_table(runs)+'\n\n'+'\n'.join(quantiles)+'''

[全部TTFT/TPOT/E2E p50/p90/p95/p99、吞吐、辅助SLO及GPU统计](results/big_summary.md) · [完整精度CSV](results/big_summary.csv) · [JSON](results/big_summary.json)

![大模型主SLO](results/big_main_slo.png)
![大模型TTFT CDF](results/big_ttft_cdf.png)
![大模型TPOT CDF](results/big_tpot_cdf.png)
![大模型辅助SLO](results/big_slo_attainment.png)

### 三系统最佳配置与限制

按主SLO最高、同分取TTFT p50最低，从本次已测配置中选择：

'''+ '\n'.join(bestlines)+f'''

本次主SLO排序：{ranking}。最佳只针对已测集合和5s/100ms门限，不能推广为所有负载、硬件或延迟指标的统一排名。六组采用同一物理分配，但固定顺序、每个一次，没有重复、随机化、置信区间或显著性检验；吞吐受到达速率与排空共同限制，接近不等于饱和服务能力相同。

{comparison}

与前文Llama结果相比，模型、tokenizer、窗口及到达缩放同时改变，不能将跨场景差值归因于模型规模这一个因素。无P2P上的TP4归约代价是场景动机，本次未单独测量allreduce占时，不能据此预先判定PP必胜。

MuxWise-PP PP4相对PP2×TP2的主SLO差距为{ppdelta:.3f}个百分点（PP2减PP4）；TTFT p50分别{fmt(pp4['latencies_sec']['ttft']['p50'])}s和{fmt(pp2['latencies_sec']['ttft']['p50'])}s。显存优化提高KV容量并不保证端到端延迟最优。P10B独立开发插桩表明PP4受准入、stage0 FIFO、四级prefill及三级链路/队列共同限制，R4开发p50的17.061s中分别为7.000、1.769、3.803、4.347s；末级等上游约32.316%。这些是开发负载的机制证据，不能当作本次正式窗口的精确分解。正式数据不单独识别计算、通信、调度及cache的因果贡献。

'''+ stage_text+'\n\n'+'\n'.join(breakdown)+'''

上述类别互斥且加和555；这是未达标条件分解，不能直接解释机制因果。[三系统最佳完整表](results/big_best_summary.md)、[最佳CSV](results/big_best_summary.csv)、[最佳主SLO图](results/big_best_main_slo.png)、[最佳TTFT CDF](results/big_best_ttft_cdf.png)、[最佳TPOT CDF](results/big_best_tpot_cdf.png)、[全配置E2E CDF](results/big_e2e_cdf.png)。PNG另有同名SVG。

### P10a/P10b代码改动与正确性边界

- **Mistral放行**：`99a756bfa`扩展MuxWise-PP白名单。Mistral继承Llama实现，共享层切分、权重加载和split-prefill；没有重写模型计算。
- **Chunked上游PP KV预算bug基线修复**：`3285a4325`修复共享KV pool在其他微批推进后使续块预算非正的问题。旧续块路径缺少普通请求已有检查，可将−69写入extend_input_len并触发FlashInfer负长度；新路径暂停续块，让decode释放容量，req_pool_idx释放后置None，避免连续暂停重复释放。正预算计算不变。负/零预算、恢复及槽位守恒有CPU复现与回归，修复后的开发GPU运行无该错误。历史Llama矩阵中未修复的失败数据仍保留；不能把大模型新旧差异当成单纯模型效应。
- **PP decode传输尺寸修复**：`03e2fe794`按实际microbatch裁剪hidden/residual再复制到TX槽。graph bs=4可能用于实际bs=2，旧发送4行与接收2行不匹配可停滞；修复保留graph与v4策略，涵盖PP4及PP2×TP2。
- **显存优化**：默认关闭的stage-local-lm-head释放非末级未用head并GC引用环；compact-attention去除decode-only后端未用工作区；分离prefill/decode工作区各128MiB、保留双槽及8分区/每分区5档graph。最终backend常驻从约1.605降至0.344GiB/卡，PP4逻辑KV从118415增至181066（+52.91%），PP2×TP2从162858增至203111（+24.72%）。这些是组合优化和所选mem fraction的结果，不归因于单个开关；显存总量与PyTorch allocated/reserved口径不混算。
- **MLP尾块补齐**：默认关闭，本次大模型MuxWise-PP显式开启。BF16 GEMM对形状敏感；cold req17的7000-token输入，首次差异定位在4096后的MLP尾块，attention/MLP输入相同。尾块补零到4096并裁回有效行，dummy行只经过逐token MLP，不进入attention/KV。相同拓扑普通PP作参照，greedy/ignore_eos/64输出，20条100–8000输入，serial保留cache及cold逐条flush并断言cached_tokens=0。补齐后414个逐层采样张量与完整logits比较相同，采样覆盖块边界和末32行全部hidden维；并非全部token逐位证明。所选PP2×TP2在a713881f9的R2e验证中serial/cold均20/20、1280/1280；最终HEAD仅新增附加校准点且PP2列表仍为空，不能标成最终HEAD重新测得。PP4在4c10135d2的R4验证中serial20/20，cold19/20、1252/1280，仅req12输出位置35后期分叉，早期pos0–2分叉已消除。**不声称全模型bitwise等价**。

详细方法、各提交GPU验证范围与原始证据见[P10A_NOTES.md](P10A_NOTES.md)、[P10B_NOTES.md](P10B_NOTES.md)。以上正确性结果引用开发复核，本次不额外跑正确性或调参试验。

### Llama-8B同GPU回归A/B（开发结果，非正式排名）

同一GPU顺序与物理UUID，依次独立启动`f6833f24e`（独立worktree）与`4c10135d2`，使用P9a PP2×TP2 v4原命令、原Llama tokenizer开发Tool&Agent前300条、time-scale1、warmup2、force-close、drain300，先长预热/flush，未调参。新增三项显存/MLP开关均关闭，额外校准context为空，原v4 YAML SHA256不变。每版一次，固定旧→新顺序。

'''+abtext+'''

历史P9a开发为TTFT p50 3.031s、262/300；P10B为3.878s、243/300，跨运行日期/时段和GPU物理分配。P9a历史开发实际使用R1提交a17263c91，本次旧版按要求使用最终v4提交f6833f24e；不能把历史行当成本次配对旧版。A/B压缩了环境差异，仍保留顺序、温度、校准和系统调度波动。

默认路径差异审查：decode send_proxy每轮新增实际batch_size查询、shape检查和hidden/residual切片；MLP即使pad=false，仍新增属性分支及输出view切片，计算尺寸保持原值；idle self-check由长度判断变为sorted/free完整性检查。memory/numeric诊断模块有初始化导入及调用，但无环境变量时立即返回、不安装hook、不做GPU同步。Mistral白名单和配置验证/校准网格包装在初始化执行，Llama默认网格不变；stage-local head、compact attention、尾块补齐及提前释放gate_up仅显式开关开启时生效。Chunked续块预算修复不在本次chunked-prefill=-1的MuxWise-PP整请求prefill路径。因此“所有新增开关关闭”不等于执行路径逐指令相同；这些差异是候选来源，端到端一次A/B不能量化各自贡献。

[A/B数据与阈值判断](logs/p10c/llama-ab.json)、[旧版日志](logs/p10c/llama-old)、[新版日志](logs/p10c/llama-new)、[源码diff](logs/p10c/llama-default-path.diff)。

### 审计与复现

正式原始数据为`results/<label>_requests.jsonl`与对应summary（label完整名称以上表为准），每条正式锁指向`logs/p10c/<label>`。报告从请求事件重算并逐项核对summary；冻结输入、历史结果、源码HEAD/干净状态和清理检查见[最终审计](logs/p10c/final-audit.json)。首次启动前NVML v1将484MiB驱动保留空间算入used，监测退出；NVML v2与nvidia-smi均证实实际约1MiB、无GPU进程。修正仅涉及P10c监测脚本，发生在任何服务启动之前，保留[原始误报](logs/p10c/preflight-nvml-v1-reserved-false-alarm.json)。

CPU重建：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/p10c_report.py`；只读已保存请求并生成图表/报告，不启动服务。结束状态见[GPU快照](logs/p10c/final-gpu.txt)及[进程/端口](logs/p10c/final-processes.json)。

首组正式回放全部结束后，清理期间四个已知scheduler脱离退出中的父进程，初版guard瞬时子树判断将其误报为外部进程并中断编排。四个PID与正式前rank记录逐一相同，告警发生在测量结束后，GPU随后恢复1MiB/0%；未发现真正的外部占用。保留[原始告警](logs/p10c/cleanup-guard-false-alarm.json)与[核验](logs/p10c/cleanup-guard-false-alarm-review.json)。仅以新编排[续跑脚本](scripts/p10c_resume.py)按PID+创建时间保留子进程归属，继续未启动的五组；首组未重跑，原始请求与summary未改，原冻结runner/推理代码/选参未改。首组run.json的清理状态补记保留了修改前副本。
'''
    audit=load(LOG/'final-audit.json')
    assert audit['formal_runs']==6 and audit['restarts']==0 and not any(audit['cleanup'].values())
    text+='\n最终核验（'+audit['completed_utc']+'）：六个正式label各1次、重启0次，全部555完成/0未完成/0失败；14项现有CPU统计/客户端测试通过。'
    text+='推理源码HEAD及干净状态、冻结参数/客户端/trace哈希、'+str(audit['historical_files_unchanged'])+'个历史结果文件哈希均通过。'
    text+='四卡各约1MiB、利用率0%，无GPU计算进程、服务/客户端残留或实验端口占用，P10a准备锁已原样恢复。\n'
    text=text.replace('cache命中率†','cache命中率')
    text=text.replace('每版一次，固定旧→新顺序。','每版一次，固定旧→新顺序；两版KV均390622，实际命令除源码和运行日志路径外一致，见[命令核对](logs/p10c/llama-command-audit.json)。')
    text=text.replace('平均分解如下，各分量均值可加，但不等于中位数分解。','下表列出主要分量的均值（未列客户端传输及结果回传等小项），不等于中位数分解。')
    text=text.replace('P10c在同一时段、同一四卡物理分配下','P10c于2026-10-22 13:24–14:51 UTC，在同一四卡物理分配下')
    baseline=(LOG/'RESULTS-before-P10c.md').read_text()
    (ROOT/'RESULTS.md').write_text(baseline.rstrip()+'\n'+text)
    r.save(LOG/'report-index.json',dict(labels=list(MAP),best=[s['label'] for s,_ in best],ab=abdata['meets_regression_threshold']))
    (LOG/'final-tables.md').write_text(abtext+'\n'+r.sm.matrix_table(runs)+'\n\n'+'\n'.join(bestlines)+'\n')
    print('\n'.join(bestlines))

if __name__=='__main__':main()
