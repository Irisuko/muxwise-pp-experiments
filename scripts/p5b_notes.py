"""Assemble the final P5b evidence report from saved development artifacts."""
import json
from pathlib import Path

root = Path("/root/workspace/muxpp_work")
log = root / "logs/p5b"
audit = json.loads((log / "regression-audit.json").read_text())
assert audit["v3-cold-correctness"]["inputs_exceed_chunk"] == 8
assert audit["pdmux-chunk-cold-correctness"]["inputs_exceed_chunk"] == 8
paired = json.loads((log / "matched-serial-decode.json").read_text())
mlp = json.loads((log / "mlp-gpu-warmed.json").read_text())
tags = ["r1-v2-instrumented", "r2-async", "r3-capacity", "r4-inflight8"]
runs = {tag: json.loads((log / tag / "analysis.json").read_text()) for tag in tags}
tables = (log / "tables.md").read_text().split("\n\n")
command = (root / "scripts/p5b_v3_serve.sh").read_text()
report = """# P5b：MuxWise-PP v3 —— 开发验证结果

推荐第3轮 `r3-capacity`：开发联合SLO **47.00%**、TTFT p50 **5.094s**、TPOT p50 **32.296ms**，明显优于既有v1的15.67%、8.61s、64.23ms。相对v1，TTFT p50下降40.8%，联合SLO增加31.33个百分点，输出吞吐提高15.3%。**仍未胜过chunked的TTFT p50=1.07s、联合SLO=53.00%。**

真实allocator数据确认v2的KV/完整未来输出预留是主要准入阻塞，inflight上限是其次；异步decode显著缩短驻留后，容量阻塞几乎消失。MLP分块允许提高KV容量。第4轮实际验证了每stage **360,997 token**，但该轮inflight=8的延迟退化，故最终选择inflight=6、容量**359,498 token**的第3轮。推荐配置距离360k目标差502 token；没有把未测的“0.712 + inflight6”组合当作已验证方案。

仅完成四轮开发性能迭代和指定回归；未运行正式trace，未push。代码在`/root/workspace/sglang`的`muxwise-pp`分支，起点`ebdf39aa0`，最终HEAD `f91e8a18da502106a07bb67c67dbf351a933bfd9`。

## 1. 范围、门禁与复现口径

- 已阅读P4B_NOTES、P2_NOTES、P1_NOTES、RESULTS及P1/P2一致性实现。没有改动P5a的`/root/workspace/sglang-p4a`。
- CPU阶段每120秒检查P5a完成标记和GPU计算PID；首次获准使用GPU为 **2026-10-20 05:30:44 UTC**，当时`p5a.gpu_done`、`p5a.rc`均存在，compute PID为空。证据：`logs/p5b/gpu-gate.jsonl`。每次后续启动仍检查标记和空卡。
- 性能只重放`bench/dev_toolagent_1200_240_x2.5.jsonl`前300条，`time-scale=1 --warmup 2 --force-close`。四轮均300成功/0失败，输入2,113,735 token、输出43,109 token，到达窗口104.9975秒。另有独立31,663-token随机输入预热，随后flush cache；预热不计入性能结果。
- 串行正确性沿用P1的20个固定输入（100–8000 token，每条64输出token）；取消、smoke、GPU通信/数学探针均不使用正式trace。压力回归只用同一开发trace前200条、time-scale=0.02、warmup2、force-close。
- `scripts/p5b_runner.py`仅允许本次开发PP配置，保存完整argv、配置文本、源码hash/diff、客户端请求和GPU采样。旧v1和chunked结果直接引用P4B，不重跑。
- R1冻结在`/root/workspace/sglang-p5b-baseline`的`da74a65a6`，只新增归因与decode轮时延取样，差异保存在`r1-v2-instrumented/source.patch`。R2冻结在`sglang-p5b-async`的`714fd92a5`。R3/R4及最终v3回归使用干净的`f91e8a18d`。

延迟是成功请求分位；TPOT=(E2E−TTFT)/(输出token−1)；联合SLO要求TTFT≤5s且TPOT≤100ms，分母为全部300条。吞吐包含到达前空闲及末尾排空。GPU均值是四卡在完整客户端窗口内的NVML均值（5秒采样）；它包含NCCL等待kernel，**不等于SM有效算力占用率**。

准入与lane表采用完全落在到达窗口内的summary边界，约100秒；精确边界见各`analysis.json`中的`admission_arrival.strict_window`和`timing_arrival_strict_window_unix`。allocator以0.5秒间隔取样。CUDA事件与CPU墙钟、prefill与decode可以重叠，不能把百分比直接相加。

## 2. 先验证准入阻塞归因

新增`multiplex/pp_admission.py`和stage0准入插桩。每次waiting非空但本次没有准入时记录互斥原因：

| 字段 | 定义 |
|---|---|
| inflight | 已达在途prefill批数上限 |
| kv | 锁定候选共享前缀后，候选本身的输入+输出+PP安全余量已不满足free+evictable |
| decode_reserve | 候选本身可容纳，但再加所有已准入请求的完整未来输出预留后不足 |
| budget | 当前准入token预算不足 |
| max_running_requests | 请求槽/运行请求上限不足 |
| other | 其余原因；本轮未出现 |

保留旧策略严格的`needed >= available`拒绝判定，没有为获得更好数字放松输出预留。按执行检查顺序记录首个阻塞，inflight检查优先；不能由互斥计数推断其它限制同时不存在。部分准入后停止另记`stops`；它不属于“本次未准入”。`blocked`是次数，`wall`是两次决策之间沿用上一状态的时间加权估计，不是精确硬件因果计时。

记录真实allocator：`used=capacity−free`、`protected=used−evictable`，同时记录可驱逐、空闲、waiting、inflight、新输入token和free_reqs，随10秒summary输出，导出各轮`allocator.csv`。**Σctx不作为KV使用量**；radix共享前缀只在allocator中算一次。

R1在waiting非空时，protected p50/p90/max=**286,334.5/290,483.8/292,166**，容量299,528；free+evictable p50=13,193.5、p10=9,044.2。KV和decode_reserve合计**44.95秒**，占所有阻塞时间约**68.95%**；inflight为20.24秒，占31.05%。次数上inflight占56.95%，说明高频决策次数不能代替等待时长。部分准入后的budget停止31次，但单独因budget而零准入为0。

这证实了“真实容量/保守未来输出预留不足”而不是仅由Σctx推断。R1 stage0 prefill lane空队列44.78%，四stage decode CUDA仅38.50–46.60%墙钟，stage0 send/token为44.56%/39.41%；与P4B v2-r4剖析一致。R1 TPOT p50=77.72ms、联合SLO18.00%，也与旧v2-r4的78.05ms、17.67%接近；不把两次TTFT波动解释成插桩收益。

"""
report += tables[1] + "\n\n"
report += """所有轮次budget/max_running_requests/other的“完全未准入”计数都是0。R2只有0.99秒未来输出预留等待，R3/R4没有KV或预留阻塞。R3的20.79秒inflight等待是真实的剩余限制，但盲目增大在途数并未改善端到端延迟（见第5节）。

allocator以下是到达窗口全部取样的独立中位数，不能把不同字段的中位数相加；更直接的waiting条件下protected和可用量在上表。

| 轮次 | 容量 | used p50 | evictable p50 | free p50 | waiting时inflight新输入 p50/max |
|---|---:|---:|---:|---:|---|
"""
for tag, a in runs.items():
    d = a["admission_arrival"]
    s = d["allocator"]
    w = d["waiting_allocator"]["inflight_new_tokens"]
    report += (
        f"| {tag} | {s['capacity']['p50']:.0f} | {s['used']['p50']:.1f} | "
        f"{s['evictable']['p50']:.1f} | {s['free']['p50']:.1f} | {w['p50']:.1f}/{w['max']:.0f} |\n"
    )
report += """
![实际allocator占用和等待队列](logs/p5b/allocator-comparison.png)

图中protected/evictable/free使用统一纵轴，虚线为最后一次计划到达；原始CSV和可导出SVG一并保留。采样随summary批量输出，排空尾端不足一个summary周期的点可能未输出。即使used接近容量，只要大量缓存可驱逐，也不能称作活跃KV占满。

## 3. 异步decode环与协议

参考只读下载的SGLang wheel：0.5.5的scheduler仍为阻塞式，进一步参考0.5.8的`async_send`/`P2PWork(work,payload)`；FastPP本机`cb3ce5b39c085a671b2c79477ac6d673b55f8eac`的`isend_tensor`用于静态shape发送参考。没有安装wheel或更换Torch/CUDA依赖。路径与SHA256见`logs/p5b/references.json`。

新`multiplex/pp_decode.py`实现`AsyncDecodeTransport`：

- 每microbatch有固定RX/TX GPU槽。下游控制接收线程根据控制中的batch size提前post `irecv`；计算线程仅消费前等待。发送线程`isend`，保留Work与payload，直到槽复用才wait。
- CUDA graph输出在下次replay覆盖前复制到本transport拥有的TX槽。残差可能别名RX槽，因此复制完成后才记录RX消费事件、允许再次接收。线程显式设置各rank的GPU设备。
- 逐步控制包合并proxy shape和该microbatch的retired token，后台预取；不再单独进行逐microbatch proxy元数据握手。每步控制仍存在，保持P1/P2对齐协议。
- 末stage一次pinned异步D2H，后台线程等事件后通过Gloo返回stage0；stage0在下一次该microbatch控制中向所有stage重放CPU token。去掉原token GPU逐stage转发及各stage重复D2H；CPU seq_lens由同序Req长度构造，避免每步`seq_lens.cpu()`。
- retirement从旧循环的前一槽尾部移到该槽下一次构建起点；stage0仍唯一决定ADMIT/MERGE/abort/retract/migration。step/mb、batch size、结果step、eviction journal继续校验，只有主调度线程能改请求与cache。
- 空控制队列也限制深度，防止idle时无限积压。异常清理不阻塞在满队列；发送Work启动等待检查线程异常。修复MERGE先于发送轮询时接收槽未释放的竞态，在删除在途状态前等待launch并加CUDA事件依赖。

默认`pp_async_decode=false`；只有`configs/pdmux_pp_4090_v3.yml`显式启用。原PP及pp=1维持原通信路径。Torch为2.8.0+cu128，未设置TORCH_NCCL_BLOCKING_WAIT。NCCL `Work.wait()`在当前CUDA stream建立完成依赖，保留Work和缓冲区直到安全复用；参考[PyTorch 2.8 Work.wait文档](https://docs.pytorch.org/docs/2.8/distributed.html#torch.distributed.Work.wait)。

R1→R2仅引入异步环和对应生命周期/CPU同步修正，保持mem=0.63、inflight6、无MLP分块：TPOT p50 **77.719→34.505ms，下降55.6%**。同microbatch成员不变时stage0相邻launch周期中位数**63.871→26.314ms**。按(bs, Σctx/8192桶, stage0分区)匹配、每轮每桶至少20点，27桶合计权重1423，加权周期下降**44.7%**；这是近似shape匹配，不能称为严格同一batch。原始分桶在`matched-development-bins.json`。

"""
report += (
    f"额外使用指定串行20条正确性语料做严格shape核对：固定bs=1、精确ctx、stage0纯decode分区7，"
    f"剔除warmup和分区切换后的第一个区间，共**{paired['pairs']}对**唯一匹配。"
    f"v2/v3 launch周期中位数为**{paired['baseline_median_ms']:.3f}/{paired['v3_median_ms']:.3f}ms**，"
    f"下降**{paired['median_reduction_pct']:.1f}%**；均值下降{paired['mean_reduction_pct']:.1f}%。"
    "同一输入、输出token和decode shape均已验证一致。它检验空载串行batch=1的环开销，"
    "不代表所有负载batch的固定形状微基准；v3内存配置也与v2不同。"
    "该测试是串行正确性回归附带计时，没有新增第五轮开发300条调参。"
    "数据见`matched-serial-decode.json`、两份`*-serial-rounds.jsonl`。\n\n"
)
report += """## 4. MLP分块、容量与数值边界

新增`--pdmux-prefill-mlp-chunk-tokens`，默认0关闭，当前支持TP1的Llama pdmux，pp=1/pp>1共用。每层只在SPLIT_PREFILL下将MLP输入按token行切片，保持SwiGLU公式、权重、attention与token顺序不变。decode和普通EXTEND不进入此路径；不支持的模型显式拒绝启用。

chunk=4096限制gate/up和激活中间张量，attention仍整批，输入/输出/残差仍有O(tokens×hidden)大小；没有宣称所有激活或所有模型都与批长无关。没有改变attention或logits数学。实际31,663-token预热、开发负载和回归都未OOM，因此这次不再额外拆attention之外其它张量。

"""
report += (
    "独立GPU数学/内存探针使用8193×4096 BF16输入、14336宽SwiGLU，预热两种GEMM形状后："
    f"完整调用增量allocated峰值**{mlp['full_incremental_peak_MiB']:.3f}MiB**，"
    f"chunk4096为**{mlp['chunk_incremental_peak_MiB']:.3f}MiB**，"
    f"下降{100*(1-mlp['chunk_incremental_peak_MiB']/mlp['full_incremental_peak_MiB']):.1f}%。"
    "证据`mlp-gpu-warmed.json`；原首次调用含cuBLAS workspace初始化的探针也保留在`mlp-gpu.json`，不把它与预热结果混用。\n\n"
    "数学上token行独立，CPU float64测试误差≤1e-12；BF16因GEMM形状改变不是逐位相等："
    f"独立探针max abs={mlp['max_abs_error']:.8f}，mean abs={mlp['mean_abs_error']:.8g}，"
    "33,558,528元素，通过rtol=0.02/atol=0.002检查。完整模型v3及pp=1 chunk均补充了逐请求flush cache的20条、1280输出token对照，"
    "每组8条新输入超过4096，实际触发分块，均与本次TP1及P1保存结果逐token完全相同。"
    "原带cache的serial20虽有长输入，但新输入均≤4096，本身不能覆盖分块路径；本次审计发现后用冷cache回归补齐。"
    "不能由有限样本推断任意输入都位级一致。\n\n"
)
report += """R3将mem fraction从0.63提高到0.71，KV从299,528到359,498；R4在0.712达到360,997。R3 summary中的PyTorch进程峰值allocated：stage0/1/2/3为20,643.20/20,456.48/20,457.42/20,447.81 MiB；max reserved约23,244–23,306 MiB。它包含启动/预热至该summary的峰值，与5秒NVML峰值的定义不同。

检查了在途KV计费：ADMIT时新输入已经从allocator分配，`inflight_new_tokens`记录的就是这些实际块，不能再次从free扣除；完整未来输出预留覆盖decode、尚未MERGE的prefill及pending状态。R3 waiting时在途输入中位数83,249.5 token、最大109,956，已包含在protected中。没有放松未来输出预留来冒险扩大准入。

## 5. 四轮配置、完整指标与剩余瓶颈

共用TP1×PP4、8个SM分区（128/0，120/8，112/16，104/24，96/32，80/48，64/64，0/128）、slice budget65536、16k max_prefill/max_budget、CUDA graph bs=1/4/8/16/32、ALP、tpot_coeff=0.75、完整decode预留、balance_merge/rebalance、max_running128。16k是准入批预算；继承的策略允许单个超预算长请求成为该批首条以避免饿死，并非每批激活绝不超过16k。配置及命令完整快照在各`run.json`。

| 轮次 | 变更 | mem fraction | inflight | MLP chunk | 每stage KV |
|---|---|---:|---:|---:|---:|
| R1 | v2-r4起点+归因插桩 | 0.63 | 6 | 关闭 | 299528 |
| R2 | 异步decode | 0.63 | 6 | 关闭 | 299528 |
| R3（推荐） | 分块与更高容量 | 0.71 | 6 | 4096 | 359498 |
| R4 | 扩inflight并验证≥360k | 0.712 | 8 | 4096 | 360997 |

"""
report += tables[0] + "\n\n"
report += """每轮准入直方图和KV占用在第2节；下表为到达窗口decode环分解。send/token/control_recv/control为主调度线程墙钟分类，CUDA为事件区间。异步后的downstream `control_recv`包含retired token到达前的等待，因此不能把类别迁移单独算作加速。stage0 send从44.56%降至1.07%，其余等待与CUDA执行更多重叠。CPU与GPU类别不能加总成100%。

"""
report += tables[2] + "\n\n"
report += """R2→R3联合SLO33%→47%、TTFT p50 5.55→5.09s；这是分块、容量和动态ALP反馈的联合结果。R2已经几乎没有KV准入等待，不能把全部改善都归因于容量。未做更多因子拆分实验。

R4的inflight阻塞时间20.79→12.65秒，但TTFT p50反而5.09→6.66s、联合SLO47%→23.33%。stage0 prefill READY等待1.89%→7.81%，prefill CUDA53.46%→56.43%，64/64分区驻留40.80%→53.94%。数据与“更多请求移入流水线、背压增加”相符；R4同时将mem fraction增加0.002，故这不是严格单变量因果证明。保留inflight6，不继续第五轮调参。

budget在部分准入后停止次数R1/R2/R3/R4为31/50/44/46；完全因budget零准入始终0，max_running_requests也始终0。因此没有证据支持继续提高16k预算或128请求上限。当前主要剩余问题是prefill服务时间、流水线排队/背压及控制等待，而不是持续KV枯竭。

| 轮次 | 准入请求 | 实际新输入token | 复用前缀token | prefill批次 | token加权cache命中 |
|---|---:|---:|---:|---:|---:|
"""
for tag in tags:
    a = json.loads((log / tag / "admit-audit.json").read_text())
    report += (
        f"| {tag} | {a['requests']} | {a['tokens']} | {a['cached']} | "
        f"{a['batches']} | {a['cache_rate']*100:.2f}% |\n"
    )
report += """
这些数按客户端精确窗口审计，四轮新输入+cache都等于2,113,735，未混入31k预热或遗漏请求。

## 6. 与既有开发结果比较

| 配置 | TTFT p50/p90/p95/p99 (s) | TPOT p50/p90/p95/p99 (ms) | 联合SLO | 输出tok/s | GPU均值 |
|---|---|---|---:|---:|---:|
| v1（P4B既有） | 8.61/14.94/18.44/19.62 | 64.23/93.06/106.54/133.91 | 15.67% | 294.1 | 60.61% |
| v2-r4（P4B既有） | 8.98/20.27/23.19/27.36 | 78.05/103.21/142.55/182.17 | 17.67% | 278.3 | 57.94% |
| v2+插桩（本次R1） | 8.03/20.19/22.35/25.47 | 77.72/119.41/134.05/260.35 | 18.00% | 281.5 | 60.38% |
| v3（本次R3） | 5.09/8.39/9.63/11.11 | 32.30/48.04/56.36/100.28 | 47.00% | 339.1 | 89.35% |
| chunked（P4B既有） | 1.07/8.71/12.45/17.38 | 33.75/210.23/254.02/283.91 | 53.00% | 342.0 | 85.84% |

v3已满足本次“明显改善v1”的开发推荐门槛，但不是正式通过或全面胜过chunked的结论。四轮各一次、未做置信区间或重复性统计；历史结果来自同开发trace，不是本次重新控制的A/B。

## 7. 正确性、压力与原路径回归

CPU单测共39项通过（准入、协议/策略/计时、四进程Gloo异步环、MLP数学及边界）。Gloo测试52控制步、动态batch/空batch、41结果重放；GPU原生transport测试四卡100步、82结果，覆盖green stream和跨microbatch共享CUDA graph输出缓冲区，均通过。详见`mlp-unit.log`、`transport-gpu-driver.log`和`transport-gpu/*.json`。

| 检查 | 结果 |
|---|---|
| 本次TP1 serial20 | 20×64输出token全部完成 |
| 推荐v3 serial20 | 与本次TP1、P1保存结果20/20、1280/1280 token完全相同 |
| v3取消 | 16次取消；随后flush和可继续服务检查通过 |
| v3压力 | 开发200条，time-scale=0.02，warmup2/force-close；200成功、0失败 |
| 原pdmux pp=1 | chunk默认关闭，7条smoke通过 |
| 原PP pp=2 | pdmux关闭，7条smoke通过 |
| pdmux pp=1 + chunk4096 | 带cache的serial20与TP1全部token相同 |
| v2固定shape对照 | 冻结baseline serial20与TP1/v3全部token相同 |
| 冷cache TP1/v3/pp1 chunk | 每条请求前flush；每组20条cached_tokens=0，其中8条输入4500–8000，实际触发分块；全部1280输出token与TP1/P1一致 |

"""
stress = audit["v3-stress"]
report += (
    f"压力回归耗时{stress['duration_sec']:.3f}秒。v3正确性、冷cache正确性、取消、压力每次均有四stage最终idle自检："
    "free_kv+evictable=359,498，free_reqs=128，下游free_receive_slots=2（stage0为0）。"
    "没有CUDA OOM、CUDA error、assert或scheduler exception，KV/请求槽/接收槽均无泄漏。"
    "审计详情：`logs/p5b/regression-audit.json`；每个测试均保存服务日志、命令、源码hash和result.json。\n\n"
)
report += """## 8. 推荐的后续正式服务启动命令（仅准备，未执行正式实验）

这是**已验证R3**的服务参数，脚本为`scripts/p5b_v3_serve.sh`；本次没有调用该启动脚本，没有准备或调用正式trace重放。先source现有环境，使用本地commit及v3配置。KV目标≥360k的R4已经测过但不是最终推荐。

```bash
"""
report += command + "```\n\n"
report += """`pp_async_decode: true`位于`configs/pdmux_pp_4090_v3.yml`，仓库与工作目录副本一致。该脚本不启用逐步JSONL round trace；summary与原策略参数保留，逐步round trace仅为开发诊断。

## 9. 提交与产物

| 本地commit | 内容 |
|---|---|
| da74a65a6 | allocator真实占用及准入原因/时间序列 |
| e50f75191 | 异步decode环、控制预取、Work/槽生命周期及回归单测 |
| 714fd92a5 | 空控制背压和非阻塞异常清理 |
| b0a107c21 | 默认关闭的Llama split-prefill MLP token分块 |
| f91e8a18d | PP summary记录CUDA allocated/reserved峰值 |

主工作树干净，所有代码已commit，未push。参考源码仅在/tmp和现有FastPP目录只读使用。保留baseline/async冻结worktree用于复核，没有修改P5a工作树。

主要产物：

- `logs/p5b/r{1,2,3,4}-*/`：服务/客户端/GPU日志、run.json、请求JSONL、analysis.json、allocator.csv、decode-rounds.jsonl、admit-audit.json。
- `logs/p5b/allocator-comparison.png/.svg`、`tables.md`：可导出图表。
- `logs/p5b/matched-development-bins.json`、`matched-serial-decode.json`：decode轮对照。
- `logs/p5b/regression-audit.json`、`mlp-gpu-warmed.json`、`transport-gpu/`：回归和数值证据。
- `scripts/p5b_runner.py`、`p5b_verify.py`、`p5b_analyze.py`、`p5b_tables.py`、`p5b_paired_decode.py`、`p5b_check_regressions.py`：开发复现与审计。

最终清理证据见`logs/p5b/cleanup-final.json`：本任务启动的服务器、router、监控和GPU计算进程已退出，服务端口已释放；没有留下运行中的正式或开发实验。
"""
(root / "P5B_NOTES.md").write_text(report)
print(root / "P5B_NOTES.md")
