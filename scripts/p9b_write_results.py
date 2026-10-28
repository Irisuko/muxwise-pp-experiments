#!/usr/bin/env python3
"""Render P9b report from validated summaries; no inference workloads."""
import json
from pathlib import Path
import p9b_report as r


def pct(x): return f'{100*x:.3f}%'
def fmt(x,scale=1): return r.sm.format_number(x,scale)
def main_slo(s): return s['slo'][r.sm.MAIN_SLO]['rate']
def name(s): return s['matrix']['system']+' '+s['matrix']['configuration']


def best_table(rows):
    lines=['| 系统 | 最佳配置 | 主SLO（达标/全部） | TTFT p50/p99(s) | TPOT p50/p99(ms); n | 完成/未完/失败 | 输出 tok/s |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for s in rows:
        q=s['latencies_sec'];v=s['slo'][r.sm.MAIN_SLO]
        lines.append(f'| {s["matrix"]["system"]} | {s["matrix"]["configuration"]} | {pct(v["rate"])} ({v["passed"]}/{v["total"]}) | '
            +'/'.join(fmt(q['ttft'][k]) for k in ['p50','p99'])+' | '
            +'/'.join(fmt(q['tpot'][k],1000) for k in ['p50','p99'])+f'; n={q["tpot"]["count"]} | '
            +'/'.join(str(s[k]) for k in ['completed','unfinished','failed'])
            +f' | {fmt(s["throughput"]["output_tokens_per_sec"])} |')
    return '\n'.join(lines)


def main():
    load=r.load
    tool=load(r.OUT/'matrix_summary.json'); tb=load(r.OUT/'best_summary.json')
    conv=load(r.OUT/'conv_summary.json'); cb=load(r.OUT/'conv_best_summary.json')
    dec=load(r.AUDIT/'replacement-decisions.json')
    comparison=load(r.AUDIT/'v3-v4-comparison.json')
    frozen=load(r.ROOT/'configs/p9b_frozen.json')
    index=load(r.AUDIT/'report-index.json')
    ts=load(r.ROOT/'bench/toolagent_600_240_x2.5.stats.json')
    cs=load(r.ROOT/'bench/conversation_600_240_x2.5.stats.json')
    text=['# P9b：Conversation三系统对比与Tool&Agent MuxWise-PP v4', '',
          f'本次按指定顺序执行9个正式label，每个一次、重启0次：Conversation六配置、Tool&Agent v4三配置。'
          f'Tool&Agent当前矩阵为{len(tool)}行；历史对照仅用于比较和版本取舍，未重新运行。'
          '所有新增实验均使用同一冻结HEAD `f6833f24ef0587278ae799696cfe17c3ade99048`。', '',
          '## 实验设置与冻结信息', '',
          '- 硬件：4×RTX4090 24GB。模型：本地Llama3.1-8B-Instruct，BF16；'
          'disable-overlap-schedule、context33792、FCFS、max-running128、enable-cache-report。',
          '- MuxWise沿用[P7A_NOTES.md](P7A_NOTES.md)所选TP2DP2 mem0.76、TP4 mem0.80；'
          'Chunked PP2DP2/TP2DP2均mem0.88、chunk2048，PP关闭mixed chunk、TP开启。'
          'TP禁用custom allreduce，FlashInfer、graph max-bs32。',
          '- MuxWise-PP严格沿用[P9A_NOTES.md](P9A_NOTES.md)的v4命令：mem0.71、MLP chunk4096、'
          '预算16384、PP2 inflight3/PP4 inflight6、graph bs1/4/8/16/32、ALP及decode reserve等保持。'
          '仅latency-partition开启，其余P9a实验开关关闭。',
          '- P7入口原冻结提交为f91e8a18d，P8为d8dfbfaba。按本任务授权新增'
          '[P9b冻结记录](configs/p9b_frozen.json)，将本次九个label统一校验为f6833f24e；'
          '原P7/P8选择文件与历史证据保留不改。新入口只增加P9b调度分支，'
          '推理/调度源码、benchmark客户端/metrics及已选服务参数均未修改。',
          '- 旧P7 UUID部分已不在当前机器；本次保留并行布局，将双卡实例映射到当前同NUMA的0+1、2+3，'
          '四卡顺序0,1,2,3，并用UUID固定。P9a v4中的CUDA索引仅替换为等价UUID。'
          '因此本次与P7历史结果存在物理GPU/NUMA差异；v3→v4正式对照不是同日单因素消融。',
          '- 正式客户端统一 `--warmup 2 --force-close --drain-timeout 300`，time-scale=1、无限制条数。'
          '每次先对所有副本做固定随机31,663-token/8输出预热和flush，使用对应开发trace '
          '`--limit 20 --time-scale 0.2`冒烟；客户端timeout55秒、外层60秒硬上限，drain仍300秒。'
          'MuxWise及MuxWise-PP在冒烟后再次flush、长预热、flush；正式前清缓存，DP重启router。',
          '- 共用P7 GPU flock；每label同时有排他启动记录和正式标记。每次保存完整argv/env、'
          'server info、客户端/服务日志、GPU/rank映射及5秒GPU采样。正式期间不调参、不补跑；'
          '未用正式trace做预检。', '',
          '[顺序与UTC时间](logs/p9b/progress.md) · [入口](scripts/run_experiment.sh) · '
          '[编排脚本](scripts/p9b_runner.py) · [全部命令](configs/p9b_frozen.json)', '',
          '### 本次GPU UUID/NUMA映射', '',
          '| CUDA索引 | GPU UUID | PCI bus ID | NUMA |', '| --- | --- | --- | --- |']
    for g in frozen['gpu_mapping']:
        text.append(f'| {g["index"]} | {g["uuid"]} | {g["pci_bus_id"]} | {g["numa"]} |')
    text += ['', 'PP2×TP2：stage0的TP组0+1、stage1的TP组2+3；PP跨NUMA。'
             'TP2DP2/PP2DP2：副本0用0+1，副本1用2+3。PP4按0→1→2→3。'
             '[拓扑](logs/p9b/topology.txt) · [映射JSON](logs/p9b/gpu-mapping.json)。', '',
             '## 统计口径', '',
             '沿用[bench/README.md](bench/README.md)与未修改的[metrics.py](bench/metrics.py)。'
             '最后实际发送后最多排空300秒，提前完成立即结束；吞吐分母包含排空时间。'
             'SLO分母为全部计划请求（Tool&Agent 1551、Conversation 735）；failed、unfinished、'
             '未发送及指标未定义均算不达标。主SLO为TTFT≤5s且TPOT≤100ms。', '',
             'TTFT/E2E分位含全部计划请求，未观测首token/未完成尾部取+∞并显示`>cutoff`；'
             'TPOT仅取截止前完成且输出>1的请求，样本数n单列。单token请求不达成联合SLO。'
             '输出吞吐包含截止前观测到的部分输出；输入仅计已交付首token的请求、包含缓存。'
             'cache是成功响应的输入token加权比例；PD decode值不代表prefill命中。'
             'TP/PP容量是同一逻辑KV空间的分片，不能乘rank数；DP副本容量独立列出。', '',
             'TTFT/E2E CDF分母为全计划请求，TPOT CDF分母为有效已完成样本。'
             'GPU均值仅取正式测量窗口内每5秒采样（含排空）；GPU忙也可能包含通信等待。'
             '原始请求重算与保存summary逐项核对。所有结果均为单次测量，无置信区间。', '',
             f'## Tool&Agent完整矩阵：{len(tool)}行', '',
             r.sm.matrix_table([(s,[]) for s in tool]), '',
             '[完整表与全部分位/SLO](results/matrix_summary.md) · [完整精度CSV](results/matrix_summary.csv)', '',
             '![Tool&Agent主SLO矩阵](results/matrix_slo_heatmap.png)', '',
             'MuxWise原版没有PP路径；MuxWise-PP已支持PP2×TP2，PP1的TP2DP2/TP4归原MuxWise。'
             'PD按P/D配比列出。未测/不适用单元不伪造结果。', '',
             '### Tool&Agent各系统最佳配置', '', best_table(tb), '']
    winner=max(tb,key=main_slo)
    text += [f'按主SLO最高、同分取TTFT p50最低选择，本矩阵最佳为{name(winner)}，'
             f'主SLO={pct(main_slo(winner))}。这不是所有指标的共同排名；'
             'TTFT、TPOT、吞吐及完成率须分别比较。历史Chunked/PD/MuxWise行保留原运行日期和源码条件。', '',
             '[最佳表](results/best_summary.md) · [CSV](results/best_summary.csv)', '',
             '![Tool&Agent最佳主SLO](results/best_main_slo.png)', '',
             '![Tool&Agent最佳TTFT CDF](results/best_ttft_cdf.png)', '',
             '![Tool&Agent最佳TPOT CDF](results/best_tpot_cdf.png)', '',
             '### Tool&Agent v4三配置及v3替换决定', '',
             '| v3 → v4 | 主SLO v3→v4 | TTFT p50/p90/p99(s) v3→v4 | TPOT p50/p99(ms) v3→v4 | 输出tok/s v3→v4 | 决定 |',
             '| --- | --- | --- | --- | --- | --- |']
    for new in ['muxpp_v4_pp2tp2','muxpp_v4_pp4','muxpp_v4_pp2dp2']:
        c=comparison[new];a,b=c['before'],c['after']
        tt=lambda s:'/'.join(fmt(s['latencies_sec']['ttft'][q]) for q in ['p50','p90','p99'])
        tp=lambda s:'/'.join(fmt(s['latencies_sec']['tpot'][q],1000) for q in ['p50','p99'])
        text.append(f'| {c["old_label"]} → {new} | {pct(main_slo(a))}→{pct(main_slo(b))} | '
                    f'{tt(a)}→{tt(b)} | {tp(a)}→{tp(b)} | '
                    f'{fmt(a["throughput"]["output_tokens_per_sec"])}→{fmt(b["throughput"]["output_tokens_per_sec"])} | '
                    +('v4替换v3' if dec[new]['replace'] else '两版并列')+' |')
    text.append('')
    for new,d in dec.items():
        text.append(f'- **{new}**：{d["reason"]}')
    text += ['', '替换按用户指定的“全面或基本全面更优”规则判断，不要求每项次要指标严格下降。'
             '吞吐小差异与TPOT中位数的资源权衡均保留，不将单次结果描述为显著改善。'
             '被替换v3的原始请求、summary、正式锁及正式日志移入'
             '[archive/superseded/p9b/](archive/superseded/p9b/)，逐文件SHA256核对，未删除。'
             '[版本对照完整表](results/p9b_v3_v4_summary.md) · '
             '[决定JSON](logs/p9b/replacement-decisions.json) · '
             '[归档清单](archive/superseded/p9b/manifest.json)。', '',
             '## Conversation负载对比', '',
             '### 负载统计与选择理由', '',
             '采用Mooncake Conversation，取窗方式与Tool&Agent相同：从600秒开始取4分钟，'
             '到达拉长2.5倍至600秒，过滤输入+输出总长>32768的请求。'
             '理论未复用prefill负载几乎相同，约8.9–8.95k tok/s；'
             '按生成stats精确计算分别为8,894.978与8,894.615 tok/s。'
             'Conversation平均输出338 token（Tool&Agent 169）、平均输入9.1k，'
             '预期KV驻留压力更重，适合检验单实例大KV容量（PP）与decode隔离（MuxWise）的表现。'
             '其理论前缀复用仅约20%，对单一radix cache的共享优势不利；'
             '理论复用不等于实测缓存，实际命中仍以表中响应统计为准。', '',
             '| 统计 | Conversation | Tool&Agent |', '| --- | ---: | ---: |']
    stats=[('窗口请求/过滤/保留',lambda x:f'{x["window_requests"]}/{x["filtered_requests"]}/{x["requests"]}'),
           ('总输入/输出token',lambda x:f'{x["total_input_tokens"]:,}/{x["total_output_tokens"]:,}'),
           ('平均输入/输出token',lambda x:f'{x["input_length"]["mean"]:.3f}/{x["output_length"]["mean"]:.3f}'),
           ('输入/输出p50',lambda x:f'{x["input_length"]["p50"]:.0f}/{x["output_length"]["p50"]:.0f}'),
           ('输入/输出token/s（600s窗）',lambda x:f'{x["total_input_tokens"]/600:.3f}/{x["total_output_tokens"]/600:.3f}'),
           ('理论前缀复用率',lambda x:pct(x['theoretical_prefix_reuse_ratio'])),
           ('最后计划到达(s)',lambda x:f'{x["last_arrival_sec"]:.4f}')]
    for label,fn in stats:text.append(f'| {label} | {fn(cs)} | {fn(ts)} |')
    text += ['', '未复用速率按首次hash块口径估算，不包含连续前缀、并发完成和驱逐约束；'
             '不能把两负载的理论速率接近等同于GPU实际计算量完全相同。'
             '[Conversation stats](bench/conversation_600_240_x2.5.stats.json) · '
             '[Tool&Agent stats](bench/toolagent_600_240_x2.5.stats.json)。', '',
             '### 六个配置完整结果', '', r.sm.matrix_table([(s,[]) for s in conv]), '',
             '[完整分位、全部SLO、GPU及配置表](results/conv_summary.md) · '
             '[CSV](results/conv_summary.csv) · [JSON](results/conv_summary.json)', '',
             '![Conversation六配置主SLO](results/conv_main_slo.png)', '',
             '![Conversation TTFT CDF](results/conv_ttft_cdf.png)', '',
             '![Conversation TPOT CDF](results/conv_tpot_cdf.png)', '',
             '![Conversation SLO](results/conv_slo_attainment.png)', '',
             '### Conversation三系统最佳配置', '', best_table(cb), '',
             'Chunked PP2DP2以625/735达标，比MuxWise TP2DP2多20条（+2.721个百分点），'
             '比MuxWise-PP v4 PP4多118条（+16.054个百分点）。它的TTFT p50/p99更低，'
             '但TPOT p99为552.443ms，明显高于MuxWise的30.189ms及PP4的59.917ms；'
             'MuxWise的E2E p50为8.053s，也低于Chunked的18.957s。'
             '因此主SLO胜出不意味着decode平稳性或整体响应时间全面胜出，20条达标差距也不能证明稳定排名。', '',
             '已有辅助SLO也呈现不同取舍：TTFT≤2s且TPOT≤50ms时，三系统上述最佳配置分别为'
             'MuxWise 51.020%、Chunked 30.204%、MuxWise-PP 9.796%；'
             'TTFT≤10s且TPOT≤100ms时为97.279%、89.660%、97.687%。'
             '这些是预先存在的辅助门限，主排序仍固定使用5s/100ms。', '',
             'MuxWise-PP内部，PP4比PP2×TP2多23条达标（+3.129个百分点），'
             'TTFT p50/p99由3.915/11.565s降至3.836/10.489s；'
             'TPOT p50则从40.530ms升至44.082ms，输出吞吐从400.860降至397.451 tok/s。'
             '其单实例359,498/390,622-token逻辑KV容量没有转化为本负载上的主SLO最优。'
             '六配置实际cache仅5.580%–6.029%，低于理论20.558%；单一radix缓存的复用空间有限，'
             '但本次数据不足以把PP与其它系统的差距单独归因于缓存。', '',
             '![Conversation最佳主SLO](results/conv_best_main_slo.png)', '',
             '[最佳表/CSV](results/conv_best_summary.md) · '
             '[最佳TTFT CDF](results/conv_best_ttft_cdf.png) · '
             '[最佳TPOT CDF](results/conv_best_tpot_cdf.png) · '
             '[E2E CDF](results/conv_e2e_cdf.png) · [TTFT时间线](results/conv_ttft_over_time.png)。'
             '所有PNG另提供同名SVG。', '',
             '### 主SLO未达标构成', '',
             '| label | 达标 | 仅TTFT>5s | 仅TPOT>100ms | 两项均超标 | TPOT未定义 | 未完成/未发 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for s in conv:
        b=s['main_slo_breakdown']
        text.append(f'| {s["label"]} | {b["passed"]} | {b["ttft_only"]} | {b["tpot_only"]} | '
                    f'{b["both"]} | {b["undefined_tpot"]} | {b["not_completed"]+b["unsent"]} |')
    text += ['', '类别互斥且加和为735；已完成单token请求归TPOT未定义，'
             '不从分母中删除。此表描述指标未达标条件，不是机制因果分解。', '',
             '### 与Tool&Agent结论的异同', '']
    tb3=sorted([s for s in tb if s['matrix']['system']!='PD'],key=main_slo,reverse=True)
    cb3=sorted(cb,key=main_slo,reverse=True)
    for workload,rows in [('Conversation',cb3),('Tool&Agent',tb3)]:
        text.append(workload+'按主SLO的三系统最佳排名：'+'；'.join(
            f'{name(s)} {pct(main_slo(s))}' for s in rows)+'。')
        text.append('')
    text.append('两种负载的系统排名'+('一致' if [s['matrix']['system'] for s in tb3]==[s['matrix']['system'] for s in cb3] else '发生变化')+
                '。具体配置和延迟分布如下；最佳配置仅从各负载已测集合选择，两个集合并不相同。')
    text += ['', '| 系统 | Tool&Agent最佳/主SLO | Conversation最佳/主SLO | TTFT p50变化(s) | TPOT p50变化(ms) |',
             '| --- | --- | --- | --- | --- |']
    for c in cb:
        a=next(t for t in tb if t['matrix']['system']==c['matrix']['system'])
        text.append(f'| {c["matrix"]["system"]} | {a["matrix"]["configuration"]}/{pct(main_slo(a))} | '
                    f'{c["matrix"]["configuration"]}/{pct(main_slo(c))} | '
                    f'{fmt(a["latencies_sec"]["ttft"]["p50"])}→{fmt(c["latencies_sec"]["ttft"]["p50"])} | '
                    f'{fmt(a["latencies_sec"]["tpot"]["p50"],1000)}→{fmt(c["latencies_sec"]["tpot"]["p50"],1000)} |')
    text += ['', '两种负载中MuxWise均以TP2DP2优于TP4，Chunked均以PP2DP2优于TP2DP2。'
             'MuxWise TP2DP2主SLO由Tool&Agent的82.656%变为Conversation的82.313%，接近；'
             'Chunked PP2DP2则从58.672%升至85.034%。其TTFT p50实际上从0.783s升到0.965s，'
             '而TPOT p99从2637.747ms降至552.443ms，说明联合SLO变化不能只看TTFT。'
             'Conversation输出中位数354，Tool&Agent为29；短输出请求的TPOT对个别长间隔更敏感，'
             '但本次没有对输出长度、调度或GPU映射做单独消融，不能将改善全部归因于长度分布。', '',
             'Conversation的输出长度、输入长度与缓存复用同时变化，'
             '其结果可以检验配置在更重KV驻留和更低复用下的表现，但不能把性能差异全部归因于KV容量或decode隔离。'
             '本次没有增加机制插桩；容量、TTFT、TPOT与cache是端到端证据，'
             '不能单独量化通信、准入等待、计算及缓存的因果贡献。'
             'Tool&Agent的历史MuxWise/Chunked运行还存在跨日期GPU映射差异，跨负载对比也受此限制。', '',
             '## MuxWise-PP v4调度优化', '',
             '[P9A_NOTES.md](P9A_NOTES.md)的逐请求TTFT分解显示：3–5秒不是硬编码下限。'
             'Tool&Agent PP2×TP2 v3开发基线平均准入等待1.272s、本地FIFO0.657s、'
             '两级prefill1.607s、级间等待0.338s；结果回传、MERGE及发送合计约0.063s。'
             'TTFT p90对应分解中准入占4.763s；Conversation p90的准入占7.247s。'
             '主要问题是prefill服务时间及其引发的准入/流水线队列，而非末级首token回传。', '',
             'v4采用R1：在原校准及在线修正形成的decode安全候选集合内，选择prefill SM最多的分区。'
             'FCFS、KV检查、未来decode预留、stage0全局决策、TP重放和双流排空约束不变；'
             '预算16384、PP2/PP4在途上限3/6不变，v3/v4 YAML逐字节相同。'
             '开发结果表明收益主要来自prefill服务加速及准入/级间排队缩短，'
             '本地FIFO均值并未普遍下降；TPOT中位数可能因资源倾斜而升高。', '',
             '用户建议的**上游decode感知已实现，但收益不稳定，默认关闭**。'
             '控制通道携带上游decode预计耗时/到达时间，TP0一致决定是否缩短prefill slice；'
             '这是发射前预测，不是实测GPU起点。典型每层prefill约54–57ms，decode轮转约36–43ms，'
             '许多空隙放不下一整层；开发测试中Conversation有收益而Tool&Agent退化。'
             '按可计算队首选分区、工作量自适应inflight、CPU token MERGE及延迟预算计算也未形成跨负载稳定收益，'
             '均默认关闭。P9a为开发trace证据，不能与本次正式数据混合排名。', '',
             '## 运行异常、证据与限制', '',
             '历史Tool&Agent Chunked PP4与TP2PP2分别完成390/587条，失败1161/964条，'
             '原PrefillSplitQOKVIndptr负长度崩溃保留在矩阵；PD未完成请求仍按全分母统计。'
             '这些历史异常没有因本次报告更新而被删除或补跑。'
             '[P8c原报告快照](logs/p9b/RESULTS-before-P9b.md)保留历史异常、PD预分配分析和PP2×TP2预测检验。', '',
             '本次新增运行的逐label完成/失败状态见[progress.md](logs/p9b/progress.md)，'
             '错误扫描及最终资源检查见[final-audit.json](logs/p9b/final-audit.json)。'
             '单次测量、固定顺序、无随机化，无误差条或置信区间；小差异不等于稳定优势。', '',
             '### 原始证据与CPU复现', '',
             '| label | 正式日志 | 请求明细 | summary |', '| --- | --- | --- | --- |']
    for s in tool+conv:
        label=s['label'];folder=Path(s['matrix']['log_dir']).relative_to(r.ROOT)
        summary=r.summary_path(label).relative_to(r.ROOT)
        text.append(f'| {label} | [日志]({folder}) | [requests](results/{label}_requests.jsonl) | [summary]({summary}) |')
    text += ['', '```bash', 'source /root/workspace/muxpp_work/env.sh', 'cd /root/workspace/muxpp_work',
             'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/p9b_report.py',
             'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/p9b_write_results.py',
             'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/p9b_audit.py', '```', '',
             '以上命令只读已保存请求并生成报告/图，不启动服务或重跑trace。'
             'P7/P8旧汇总入口保留作历史实现文档；当前矩阵使用P9b入口重建。', '',
             '## 完成核验', '',
             '9个新label各正式运行1次、重启0次。推理源码HEAD及干净状态通过核验，'
             '所选参数、benchmark客户端/metrics、正式/开发trace哈希未变；'
             '历史原始数据及本次归档逐文件SHA256核对。'
             '14项现有CPU统计/配置/选择测试通过，原始明细与summary重算一致。'
             '结束时GPU计算进程、服务/router/客户端及实验端口均无残留。'
             '[最终审计](logs/p9b/final-audit.json) · [GPU快照](logs/p9b/final-gpu.txt) · '
             '[进程与端口](logs/p9b/final-processes.json) · [测试日志](logs/p9b/bench-unit-tests.log)。', '']
    (r.ROOT/'RESULTS.md').write_text('\n'.join(text))
    print('Wrote RESULTS.md')


if __name__=='__main__':main()
