# P11a：MuxWise-PP v5 开发优化结果

最终代码 `1dcbb285e958936113e373d4b0ff1b30f63aaed2`，分支 `muxwise-pp`；起点 `4c10135d2e2891d6fb262d4894f0769c86dd5751`。本次仅跑指定开发 trace，完成六轮优化矩阵，没有运行正式实验，没有 push。最终推荐按拓扑选择已测配置，见 [p11a_selected.json](configs/p11a_selected.json)。

## 方法、范围与可比性

已阅读 RESULTS.md 的 Tool&Agent、Conversation、大模型三部分以及 P9A_NOTES.md/P10B_NOTES.md。历史正式数字只用于定位瓶颈；下面均为本次重新测得的开发结果。

- 8B：`bench/dev_toolagent_1200_240_x2.5.jsonl` 前300条；24B：`bench/dev_big_toolagent_1200_80_x7.5.jsonl` 前150条。标准/2倍/饱和分别为 time-scale 1/0.5/0.05。客户端统一 `--warmup 2 --force-close --drain-timeout 300`。
- 主 SLO：TTFT≤5s 且 TPOT≤100ms；宽松：TTFT≤10s 且 TPOT≤100ms。goodput=主SLO达标请求数/测量总时长；吞吐和goodput分母均包含排空。输入tok/s包含cache命中，不能当作新计算prefill速率。
- 每配置启动一次，顺序为标准→2倍→饱和。每档额外31663-token固定输入/8输出预热，前后flush；DP的router按档重置。RLS在同一进程内继续学习。每项只有一次测量，没有置信区间，亚百分比差异不作统计显著性结论。
- 四张RTX4090独占；0/1在NUMA0、2/3在NUMA1。每次使用前检查GPU和进程，运行期间按PID及创建时间追踪自有进程；未发现容器外占用。KV、完整未来输出预留、mem fraction和原分区配置没有放宽。
- 原始argv/env、源码commit及patch、trace/config哈希、逐请求结果、GPU采样、逐stage lane/请求事件和decode批大小均在 `logs/p11a/`。完整所有速率的TTFT/TPOT p50/p90/p95/p99、输入/输出tok/s、SLO与GPU见 [comparison.md](logs/p11a/comparison.md) / [JSON](logs/p11a/comparison.json)；每stage计时、分区驻留、准入阻塞直方图和decode批大小见 [stage-details.md](logs/p11a/stage-details.md) 及每档 `analysis.json`。DP2的独立worker分别计算计时差分，避免跨worker相减。stage计时取测量窗口内首末summary差分，窗口两端不足10s不计；CUDA与双stream驻留会重叠，不能相加。

基线特例：v4 24B PP4标准档150/150完成后，下一档的31k预热分配256MiB时仅余253.81MiB而OOM，发生在double客户端启动之前。修正编排为预热前也flush，然后用同一冻结4c版本/配置另起进程续跑double、saturated，保留 `baseline-big_pp4` 原失败记录及 `baseline-resume-big_pp4` 续跑记录。因此该基线后两档RLS重置；不能声称三档同一进程。测量档本身均无失败/未完成。

## 有效实现与边界

stage0对完整准入批生成FCFS token块计划，短请求和长请求尾部可打包。每stage一次性分配完整输入KV、保留原radix锁；子块只持有张量视图，后块prefix包含前块KV，每stage FIFO执行。块内继续使用原按层切片和RLS/ALP分区控制。每块有独立wire ID与READY/接收槽生命周期，child metadata由CUDA event覆盖；末级完成后按序归还精确token额度。整父批全部完成后才在原MERGE点插入radix并返回首token。取消标志共享到子块，完成在途计算/通信后释放，避免KV或接收槽提前复用。

中间块跳过norm/head/采样，终结span按FCFS必为行前缀，用basic slice视图采样，避免Python列表GPU索引的同步。接收buffer按块预算分配。24B保留原MLP尾块padding；可选QKV/O投影padding只影响投影dummy行，不进入attention/KV。追加块改用完整上下文paged causal attention，并禁用自动KV split，减少BF16归约路径变化。

可选token在途上限替代批数上限；完整KV/output预留独立保留。单条超额度的FCFS请求仅在未完成工作为0时允许进入，避免饿死。可选refill门槛等待已有工作归还足够额度后再准入，门槛受当前ALP预算限制，不等待未来请求凑批。可选CPU元数据让整批prefill首次split无需GPU→CPU读回，同时用CPU前缀和、query/KV indptr及last-page长度供FlashInfer规划；KV索引仍在GPU构造，整数元数据等价。新增参数默认0/关闭，原pdmux、普通PP、普通Chunked路径保持原行为。

本次采用实测选择的固定4096/8192块预算，没有实现新的动态chunk-size控制器；块内沿用v4层切片，RLS/ALP继续用于原有分区与准入决策。KV容量和decode微批上限保持v4。剖析表明主要问题是长prefill气泡以及小批带来的CPU/READY固定开销。

## 本次开发基线（18档）

| 配置 | 标准主/宽松SLO % | TTFT p50 s / TPOT p50 ms | 三档主SLO % | 三档输出 tok/s | 饱和时长 s | 三档goodput req/s | 标准GPU均值 % |
| --- | --- | --- | --- | --- | ---: | --- | ---: |
| 8B PP4 | 85.333 / 99.333 | 3.044 / 51.519 | 85.333 / 12.667 / 5.000 | 342.74 / 449.04 / 474.46 | 90.859 | 2.0354 / 0.3958 / 0.1651 | 84.91 |
| 8B PP2×TP2 | 73.333 / 98.333 | 3.137 / 46.459 | 73.333 / 9.667 / 5.333 | 354.80 / 472.68 / 488.12 | 88.316 | 1.8107 / 0.3180 / 0.1812 | 84.56 |
| 24B PP2×TP2 | 30.000 / 69.333 | 6.248 / 59.211 | 30.000 / 0.667 / 0.000 | 111.75 / 146.50 / 160.60 | 141.224 | 0.2217 / 0.0065 / 0.0000 | 80.64 |
| 24B PP4 | 4.000 / 14.667 | 18.650 / 83.158 | 4.000 / 0.000 / 0.000 | 95.66 / 103.55 / 109.07 | 207.952 | 0.0253 / 0.0000 / 0.0000 | 83.34 |
| 8B MuxWise TP2×DP2 | 84.333 / 93.000 | 1.831 / 19.834 | 84.333 / 15.000 / 6.000 | 360.70 / 486.86 / 489.23 | 88.116 | 2.1169 / 0.5082 / 0.2043 | 91.03 |
| 24B MuxWise TP4 | 38.000 / 74.000 | 5.083 / 29.077 | 38.000 / 10.667 / 8.667 | 116.54 / 166.60 / 179.79 | 126.150 | 0.2929 / 0.1175 / 0.1031 | 79.98 |

## R1：4096固定CPP，保留v4批数上限

源码 `c690b5d0d`。长输入气泡减少，但采样列表索引造成CPU同步，8B吞吐退化；24B冷cache出现新增数值分叉，本轮不推荐。

| 配置 | 标准主/宽松SLO % | TTFT p50 s / TPOT p50 ms | 三档主SLO % | 三档输出 tok/s | 饱和时长 s | 三档goodput req/s | 标准GPU均值 % |
| --- | --- | --- | --- | --- | ---: | --- | ---: |
| 8B PP4 | 87.000 / 94.000 | 2.448 / 55.770 | 87.000 / 17.333 / 4.667 | 343.90 / 456.63 / 467.22 | 92.266 | 2.0821 / 0.5508 / 0.1517 | 87.40 |
| 8B PP2×TP2 | 75.667 / 93.333 | 3.529 / 48.921 | 75.667 / 9.667 / 5.333 | 348.79 / 421.92 / 440.03 | 97.968 | 1.8366 / 0.2838 / 0.1633 | 90.26 |
| 24B PP2×TP2 | 50.667 / 78.000 | 4.542 / 61.016 | 50.667 / 3.333 / 2.667 | 112.68 / 151.90 / 168.81 | 134.359 | 0.3776 / 0.0335 / 0.0298 | 79.35 |
| 24B PP4 | 23.333 / 41.333 | 8.757 / 84.065 | 23.333 / 1.333 / 0.000 | 104.50 / 122.27 / 130.65 | 173.600 | 0.1613 / 0.0108 / 0.0000 | 83.47 |

## R2：修复采样视图、MLP尾块、paged追加attention、KV split和接收buffer；8192 CPP/投影，保留v4批数准入

源码 `563b32fe9`。8B PP4及两种24B拓扑三档主SLO/输出吞吐均不低于v4；8B PP2×TP2主SLO改善，但容量仍略退。性能前数值预检通过无新增差异门槛。

| 配置 | 标准主/宽松SLO % | TTFT p50 s / TPOT p50 ms | 三档主SLO % | 三档输出 tok/s | 饱和时长 s | 三档goodput req/s | 标准GPU均值 % |
| --- | --- | --- | --- | --- | ---: | --- | ---: |
| 8B PP4 | 91.000 / 99.667 | 2.437 / 51.755 | 91.000 / 16.667 / 5.333 | 345.65 / 492.53 / 515.98 | 83.548 | 2.1890 / 0.5713 / 0.1915 | 91.34 |
| 8B PP2×TP2 | 84.333 / 100.000 | 3.411 / 47.222 | 84.333 / 12.333 / 5.667 | 352.04 / 462.12 / 478.78 | 90.039 | 2.0661 / 0.3966 / 0.1888 | 89.43 |
| 24B PP2×TP2 | 40.000 / 80.000 | 6.004 / 61.124 | 40.000 / 2.000 / 1.333 | 112.43 / 153.25 / 165.22 | 137.281 | 0.2974 / 0.0203 / 0.0146 | 81.47 |
| 24B PP4 | 14.000 / 37.333 | 10.460 / 82.101 | 14.000 / 0.000 / 0.000 | 103.02 / 121.36 / 126.65 | 179.089 | 0.0954 / 0.0000 / 0.0000 | 85.10 |

## R3：8192 CPP，65536 token额度，准入budget8192；8B去投影padding

源码 `563b32fe9`。小批过多：两个PP2×TP2高负载吞吐低于v4；8B PP4标准SLO略升但容量低于R2。

| 配置 | 标准主/宽松SLO % | TTFT p50 s / TPOT p50 ms | 三档主SLO % | 三档输出 tok/s | 饱和时长 s | 三档goodput req/s | 标准GPU均值 % |
| --- | --- | --- | --- | --- | ---: | --- | ---: |
| 8B PP4 | 92.000 / 99.333 | 2.067 / 53.080 | 92.000 / 17.000 / 5.333 | 344.79 / 488.93 / 507.87 | 84.882 | 2.2075 / 0.5784 / 0.1885 | 87.49 |
| 8B PP2×TP2 | 82.333 / 99.333 | 2.634 / 47.683 | 82.333 / 7.333 / 4.667 | 347.52 / 436.24 / 454.50 | 94.850 | 1.9912 / 0.2226 / 0.1476 | 90.78 |
| 24B PP2×TP2 | 44.000 / 69.333 | 5.270 / 60.725 | 44.000 / 2.000 / 2.667 | 112.20 / 144.77 / 153.22 | 148.024 | 0.3265 / 0.0191 / 0.0270 | 81.36 |
| 24B PP4 | 15.333 / 43.333 | 8.792 / 83.747 | 15.333 / 0.667 / 0.000 | 103.28 / 117.06 / 124.50 | 182.171 | 0.1047 / 0.0052 / 0.0000 | 84.89 |

## R4：恢复准入budget16384、token额度65536；8B PP2关闭CPP，24B PP4使用4096 CPP/投影

源码 `563b32fe9`。PP4两组改善；PP2×TP2高负载容量仍退。8B PP2关闭CPP也未解决小批固定成本。

| 配置 | 标准主/宽松SLO % | TTFT p50 s / TPOT p50 ms | 三档主SLO % | 三档输出 tok/s | 饱和时长 s | 三档goodput req/s | 标准GPU均值 % |
| --- | --- | --- | --- | --- | ---: | --- | ---: |
| 8B PP4 | 92.000 / 100.000 | 2.294 / 51.470 | 92.000 / 20.667 / 5.667 | 343.73 / 484.82 / 520.55 | 82.814 | 2.2007 / 0.6973 / 0.2053 | 86.72 |
| 8B PP2×TP2 | 87.333 / 99.333 | 3.038 / 48.859 | 87.333 / 6.000 / 5.667 | 353.24 / 439.37 / 458.03 | 94.119 | 2.1469 / 0.1835 / 0.1806 | 91.50 |
| 24B PP2×TP2 | 39.333 / 78.000 | 6.516 / 61.337 | 39.333 / 2.667 / 2.667 | 112.39 / 148.28 / 158.21 | 143.359 | 0.2924 / 0.0262 / 0.0279 | 81.28 |
| 24B PP4 | 23.333 / 46.667 | 9.158 / 83.594 | 23.333 / 0.667 / 0.667 | 105.40 / 122.38 / 130.07 | 174.374 | 0.1626 / 0.0054 / 0.0057 | 87.21 |

## R5：R4上增加16384 token回补门槛，cap/KV预留不变

源码 `7e1adaecf`。检查额度逐块归还时能否减少零碎准入；与R4的SLO/容量取舍保留在表中。

| 配置 | 标准主/宽松SLO % | TTFT p50 s / TPOT p50 ms | 三档主SLO % | 三档输出 tok/s | 饱和时长 s | 三档goodput req/s | 标准GPU均值 % |
| --- | --- | --- | --- | --- | ---: | --- | ---: |
| 8B PP4 | 93.000 / 100.000 | 2.293 / 52.730 | 93.000 / 21.667 / 5.667 | 345.06 / 503.12 / 514.26 | 83.828 | 2.2332 / 0.7586 / 0.2028 | 89.96 |
| 8B PP2×TP2 | 80.000 / 99.667 | 3.217 / 48.371 | 80.000 / 8.333 / 5.667 | 353.66 / 468.42 / 478.16 | 90.156 | 1.9689 / 0.2716 / 0.1886 | 84.82 |
| 24B PP2×TP2 | 43.333 / 79.333 | 5.363 / 60.971 | 43.333 / 2.667 / 2.000 | 112.42 / 144.49 / 163.38 | 138.824 | 0.3222 / 0.0255 / 0.0216 | 81.49 |
| 24B PP4 | 23.333 / 50.667 | 7.517 / 83.161 | 23.333 / 1.333 / 0.667 | 105.84 / 129.81 / 133.45 | 169.955 | 0.1633 / 0.0114 / 0.0059 | 86.41 |

## R6：PP整批首split复用CPU长度元数据；按拓扑采用此前有效配置

源码 `1dcbb285e`。最后一轮。具体参数及选择理由见最终配置；不再追加调参轮次。

| 配置 | 标准主/宽松SLO % | TTFT p50 s / TPOT p50 ms | 三档主SLO % | 三档输出 tok/s | 饱和时长 s | 三档goodput req/s | 标准GPU均值 % |
| --- | --- | --- | --- | --- | ---: | --- | ---: |
| 8B PP4 | 91.000 / 99.667 | 2.657 / 52.074 | 91.000 / 20.667 / 5.667 | 345.64 / 491.87 / 517.31 | 83.332 | 2.1889 / 0.7074 / 0.2040 | 86.40 |
| 8B PP2×TP2 | 84.000 / 100.000 | 2.898 / 47.815 | 84.000 / 12.333 / 5.667 | 351.56 / 462.42 / 478.76 | 90.043 | 2.0551 / 0.3969 / 0.1888 | 90.92 |
| 24B PP2×TP2 | 43.333 / 68.000 | 5.382 / 61.876 | 43.333 / 2.000 / 1.333 | 112.38 / 156.32 / 167.54 | 135.375 | 0.3221 / 0.0207 / 0.0148 | 81.42 |
| 24B PP4 | 17.333 / 48.667 | 7.325 / 84.804 | 17.333 / 0.667 / 0.667 | 105.68 / 126.49 / 132.40 | 171.313 | 0.1211 / 0.0056 / 0.0058 | 84.88 |

## 剖析与未采用的方案

R1将标准窗口末级等上游比例从8B PP4的39.2%降至19.5%、24B PP4的32.3%降至15.7%、24B PP2的17.5%降至9.8%，但末级prefill_launch CPU驻留分别由1.85/6.29/10.20s升至14.08/21.10/17.27s。无拷贝采样修复后R2的8B PP4与24B PP4该驻留降至1.87/3.46s。长请求流水线重叠有效，但固定CPU成本必须一起处理。

R3的8B PP2两倍速率准入批数从R2的73增至135，末级CPU launch 6.72→9.43s，stage0 READY 7.53→13.56s，末级等上游11.38→18.62s；decode平均bs反而9.70→9.28。R4整请求prefill仍有89批，CUDA prefill计算量与v4近似，但CPU/READY开销更大。R5的refill门槛针对这种零碎credit准入；R6进一步去掉整批首次长度、前缀求和以及FlashInfer planner索引指针的读回同步。

24B数值诊断使用独立插桩服务，未混入性能测量：R1 cold20在PP2/PP4均17/20一致。初始R2的MLP尾块修复不足（16/20与17/20）；逐层定位到CPP边界attention的ragged/paged BF16合并。改paged causal后为18/20与17/20。短尾块QKV shape变化由投影padding修复；自动KV split变化由禁用split修复，两条4500/5500输入的完整64步logits及采样层组件恢复完全一致。4096块的PP2在8000输入仍有O投影GEMM归约差异（第0层起约6.1e-5），导致输出26近似并列logits分叉；关闭全局BF16 reduced-precision反而增加差异，未采用。PP2因此采用8192块，≤8192保留整批fastpath。

8192候选的serial20/cold20：8B两组和24B PP2均20/20；24B PP4 serial20/20、cold19/20，所有输出与v4归档完全相同。已知PP4差异是cold请求12输出35起，与本次普通PP4相比1252/1280 token一致；不能宣称所有拓扑逐token完全等价。最终另测12288-token冷请求，覆盖真实跨块执行，避免把最长8000的cold20误当作CPP数学等价证明。数值证据见 `r2e-correctness-gate.json`、`numeric-r2*-*`；失败方案均留档，没有覆盖结果。

## 最终选择、v4对比与MuxWise差距

最终配置按拓扑选取各轮冻结源码的实测点，表中注明轮次。后续新增功能有独立开关，未选择的功能保持关闭。所选配置在最终HEAD统一执行正确性与资源回归；六轮之后不追加性能矩阵。

| 拓扑 | 所选轮次 | 速率 | v4→v5 主SLO % | v4→v5 输出tok/s | v5 goodput req/s | 相对MuxWise主SLO百分点 / 输出% |
| --- | --- | --- | --- | --- | ---: | --- |
| llama_pp4 | r5 | standard | 85.333→93.000 | 342.74→345.06 | 2.2332 | +8.667 / -4.34 |
| llama_pp4 | r5 | double | 12.667→21.667 | 449.04→503.12 | 0.7586 | +6.667 / +3.34 |
| llama_pp4 | r5 | saturated | 5.000→5.667 | 474.46→514.26 | 0.2028 | -0.333 / +5.12 |
| llama_pp2tp2 | r2 | standard | 73.333→84.333 | 354.80→352.04 | 2.0661 | +0.000 / -2.40 |
| llama_pp2tp2 | r2 | double | 9.667→12.333 | 472.68→462.12 | 0.3966 | -2.667 / -5.08 |
| llama_pp2tp2 | r2 | saturated | 5.333→5.667 | 488.12→478.78 | 0.1888 | -0.333 / -2.14 |
| big_pp2tp2 | r6 | standard | 30.000→43.333 | 111.75→112.38 | 0.3221 | +5.333 / -3.57 |
| big_pp2tp2 | r6 | double | 0.667→2.000 | 146.50→156.32 | 0.0207 | -8.667 / -6.17 |
| big_pp2tp2 | r6 | saturated | 0.000→1.333 | 160.60→167.54 | 0.0148 | -7.333 / -6.81 |
| big_pp4 | r5 | standard | 4.000→23.333 | 95.66→105.84 | 0.1633 | -14.667 / -9.19 |
| big_pp4 | r5 | double | 0.000→1.333 | 103.55→129.81 | 0.0114 | -9.333 / -22.08 |
| big_pp4 | r5 | saturated | 0.000→0.667 | 109.07→133.45 | 0.0059 | -8.000 / -25.77 |


8B PP4默认候选R5：标准主SLO93%，两倍吞吐503.12 tok/s；R4饱和吞吐520.55高于R5的514.26，但标准主SLO少1个百分点、两倍吞吐较低。两者相对v4三档主SLO和输出均不降；默认优先R5的标准/两倍goodput。


8B PP2×TP2候选R2存在取舍：标准主SLO73.333→84.333%，三档goodput均升；输出吞吐标准/两倍/饱和约退0.8%/2.2%/1.9%。不宣称同时提升吞吐；吞吐优先可保持v4整请求/批数准入参数（禁用CPP/token额度/投影padding/refill/CPU元数据）。


24B PP2×TP2选择R6：相对v4三档主SLO/输出吞吐均提高；标准主SLO30→43.333%，饱和输出160.60→167.54 tok/s。宽松SLO标准68.0%低于v4的69.333%及R2的80.0%，追求宽松达标率可用R2（同8192 CPP/投影、批数准入，关闭CPU元数据）；R2标准主SLO40%，饱和165.22 tok/s。宽松达标率的下降主要体现在TTFT≤10s比例（R2 82.0%→R6 70.667%），TPOT≤100ms为98.0%→97.333%。


24B PP4选择R5：标准主SLO4.0→23.333%，三档输出95.66/103.55/109.07→105.84/129.81/133.45 tok/s；R6标准主SLO17.333%、两倍126.49、饱和132.39，故该拓扑关闭CPU元数据。


8B PP2×DP2采用8192 CPP、65536 token额度、16384 refill/准入预算，不启用投影padding和CPU元数据；最终另做三档开发与两个worker的跨块冒烟，无成对v4 PP2×DP2基线。


### 所选配置的标准窗口剖析

| 配置 | 末级等上游占比 v4→v5 | 末级CPU prefill_launch占比 v4→v5 | decode平均bs v4→v5 | v5主要准入阻塞（次数/驻留s） |
| --- | --- | --- | --- | --- |
| 8B PP4 | 39.15%→21.35% | 1.54%→1.51% | 3.95→3.43 | inflight_refill: 3280/13.02s；inflight_tokens: 736/8.23s |
| 8B PP2×TP2 | 22.17%→15.97% | 6.73%→7.04% | 6.29→6.14 | inflight: 3995/38.37s |
| 24B PP2×TP2 | 17.49%→7.57% | 5.36%→5.10% | 3.23→3.14 | inflight: 9957/41.28s |
| 24B PP4 | 32.34%→10.26% | 2.72%→1.66% | 2.32→2.19 | inflight_refill: 9236/24.65s；inflight_tokens: 545/11.17s；kv: 330/6.75s |

上述占比按各自stage观测窗口归一化；完整各stage、三种速率及准入直方图仍以stage-details.md/analysis.json为准。

![开发SLO与饱和吞吐](logs/p11a/slo-throughput.png)

可导出图：[PDF](logs/p11a/slo-throughput.pdf)。图仅表示本次单次开发测量。


### 8B PP2×DP2补充验证

对最终配置仅做一次三档开发验证，无本次v4 PP2×DP2成对基线，不宣称相对v4提升。

| 配置 | 标准主/宽松SLO % | TTFT p50 s / TPOT p50 ms | 三档主SLO % | 三档输出 tok/s | 饱和时长 s | 三档goodput req/s | 标准GPU均值 % |
| --- | --- | --- | --- | --- | ---: | --- | ---: |
| 8B PP2×DP2 | 89.667 / 99.667 | 1.551 / 49.447 | 89.667 / 20.000 / 6.333 | 344.19 / 418.35 / 351.37 | 122.689 | 2.1477 / 0.5823 / 0.1549 | 86.40 |

这组附测不作为8B默认推荐。饱和档GPU0/1利用率均值为87.0%/99.4%，GPU2/3仅56.9%/61.3%；worker0的KV/完整输出预留阻塞驻留为61.05/34.58s，worker1的empty状态为54.55s，伴随明显worker负载不均。两个worker分别计时，未跨worker相减；验证文件为analyzer-validation.json。该配置在标准档中位TTFT较低，但高负载吞吐落后PP4，未在六轮之外继续调参。

## 最终回归与资源守恒

| 配置 | serial20 exact | cold20 exact | 新增分叉 | 冷12288 exact | 取消数 | stress200 完成/失败/未完成 |
| --- | --- | --- | --- | --- | ---: | --- |
| 8B PP4 | 20/20 | 20/20 | [] | True | 16 | 200/0/0 |
| 8B PP2×TP2 | 20/20 | 20/20 | [] | True | 16 | 200/0/0 |
| 24B PP2×TP2 | 20/20 | 20/20 | [] | True | 16 | 200/0/0 |
| 24B PP4 | 20/20 | 19/20 | [] | True | 16 | 200/0/0 |

serial/cold每请求比较64个原始token，参照为同拓扑普通实例。取消16个请求后检查abort结果并再生成16-token请求；压力为指定开发trace前200条、time-scale0.02，客户端仍warmup2/force-close/drain300。所有PP/TP rank在排空后检查free KV+可驱逐KV=容量、128个请求槽全部归还、接收槽回收；保留 `final-verify-*/run.json` 和 [正确性门禁](logs/p11a/final-correctness-gate.json)。

原路径共12项冒烟：原pdmux的8B TP1/TP2、24B TP4；普通PP四拓扑；普通Chunked四拓扑；最终8B PP2×DP2。DP2包含router并发与两个worker分别执行12288-token请求。CPU最终相关套件141项全部通过，格式检查通过，见 `logs/p11a-unit-r6-full.log` / `logs/p11a-format-r6.log`。所有配置默认关闭新功能时仍走原路径。

结束审计见 [final-audit.json](logs/p11a/final-audit.json)：复核220个既有trace/结果/笔记哈希，重算每次测量指标，检查每rank资源守恒。服务器日志中的唯一执行异常是上述v4 PP4测量前预热OOM；数值预检中的失败方案已在上文单独记录。未发现外部GPU占用。结束时四卡空闲、无残留服务/监测进程，工作树干净。代码已commit，未push。

## 最终v5配置与完整启动命令

配置文件：[p11a_selected.json](configs/p11a_selected.json)。下面命令从最终配置生成；执行前确认四卡没有容器外占用。同一时刻只启动一组拓扑。带GPU独占检查和退出清理的入口为 `source /root/workspace/muxpp_work/env.sh && python /root/workspace/muxpp_work/scripts/p11a_serve.py launch <label>`；按Ctrl-C停止。以下是可核查的完整展开命令，DP2需先在两个终端启动workers、待ready后启动router。

| 配置 | CPP token | 投影padding | 未完成token额度 | refill token | CPU元数据 | 实测来源 |
| --- | ---: | ---: | --- | ---: | --- | --- |
| 8B PP4 | 8192 | 0 | 65536 | 16384 | 关闭 | r5 |
| 8B PP2×TP2 | 8192 | 8192 | 沿用v4批数门控 | 0 | 关闭 | r2 |
| 24B PP2×TP2 | 8192 | 8192 | 沿用v4批数门控 | 0 | 开启 | r6 |
| 24B PP4 | 4096 | 4096 | 65536 | 16384 | 关闭 | r5 |
| 8B PP2×DP2 | 8192 | 0 | 65536 | 16384 | 关闭 | 最终固定配置附测 |

### 8B PP4

```bash
source /root/workspace/muxpp_work/env.sh
# worker0
env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/root/workspace/sglang/python /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --tp 1 --pp-size 4 --dtype bfloat16 --port 30000 --disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --enable-pdmux --chunked-prefill-size -1 --pdmux-config-path /root/workspace/muxpp_work/configs/pdmux_pp_4090_v4.yml --mem-fraction-static 0.71 --cuda-graph-max-bs 32 --cuda-graph-bs 1 4 8 16 32 --max-prefill-tokens 16384 --pdmux-pp-max-budget 16384 --pdmux-pp-max-inflight-prefill 6 --pdmux-pp-radix-eviction lru --pdmux-pp-policy alp --pdmux-pp-slo-ttft 5.0 --pdmux-pp-slo-tpot 0.1 --pdmux-pp-tpot-coeff 0.75 --pdmux-pp-reserve-decode --pdmux-pp-balance-merge --pdmux-pp-rebalance --pdmux-pp-alp-admission --pdmux-pp-rebalance-unit 8 --pdmux-pp-summary-interval 10 --pdmux-prefill-mlp-chunk-tokens 4096 --attention-backend flashinfer --pdmux-pp-latency-partition --no-pdmux-pp-runnable-partition --no-pdmux-pp-decode-aware-slices --no-pdmux-pp-adaptive-inflight --no-pdmux-pp-cpu-merge-tokens --no-pdmux-pp-lazy-admission-budget --pdmux-pp-prefill-chunk-size 8192 --pdmux-pp-max-inflight-tokens 65536 --pdmux-pp-max-budget 16384 --pdmux-pp-token-refill-size 16384
```

### 8B PP2×TP2

```bash
source /root/workspace/muxpp_work/env.sh
# worker0
env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/root/workspace/sglang/python /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --tp 2 --pp-size 2 --dtype bfloat16 --port 30000 --disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --enable-pdmux --chunked-prefill-size -1 --pdmux-config-path /root/workspace/muxpp_work/configs/pdmux_pp_4090_v4.yml --mem-fraction-static 0.71 --cuda-graph-max-bs 32 --cuda-graph-bs 1 4 8 16 32 --max-prefill-tokens 16384 --pdmux-pp-max-budget 16384 --pdmux-pp-max-inflight-prefill 3 --pdmux-pp-radix-eviction lru --pdmux-pp-policy alp --pdmux-pp-slo-ttft 5.0 --pdmux-pp-slo-tpot 0.1 --pdmux-pp-tpot-coeff 0.75 --pdmux-pp-reserve-decode --pdmux-pp-balance-merge --pdmux-pp-rebalance --pdmux-pp-alp-admission --pdmux-pp-rebalance-unit 8 --pdmux-pp-summary-interval 10 --pdmux-prefill-mlp-chunk-tokens 4096 --disable-custom-all-reduce --attention-backend flashinfer --pdmux-pp-latency-partition --no-pdmux-pp-runnable-partition --no-pdmux-pp-decode-aware-slices --no-pdmux-pp-adaptive-inflight --no-pdmux-pp-cpu-merge-tokens --no-pdmux-pp-lazy-admission-budget --pdmux-pp-prefill-chunk-size 8192 --pdmux-pp-prefill-projection-pad-to 8192
```

### 8B PP2×DP2

```bash
source /root/workspace/muxpp_work/env.sh
# worker0
env CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=/root/workspace/sglang/python /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --tp 1 --pp-size 2 --dtype bfloat16 --port 31000 --disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --enable-pdmux --chunked-prefill-size -1 --pdmux-config-path /root/workspace/muxpp_work/configs/pdmux_pp_4090_v4.yml --mem-fraction-static 0.71 --cuda-graph-max-bs 32 --cuda-graph-bs 1 4 8 16 32 --max-prefill-tokens 16384 --pdmux-pp-max-budget 16384 --pdmux-pp-max-inflight-prefill 3 --pdmux-pp-radix-eviction lru --pdmux-pp-policy alp --pdmux-pp-slo-ttft 5.0 --pdmux-pp-slo-tpot 0.1 --pdmux-pp-tpot-coeff 0.75 --pdmux-pp-reserve-decode --pdmux-pp-balance-merge --pdmux-pp-rebalance --pdmux-pp-alp-admission --pdmux-pp-rebalance-unit 8 --pdmux-pp-summary-interval 10 --pdmux-prefill-mlp-chunk-tokens 4096 --attention-backend flashinfer --pdmux-pp-latency-partition --no-pdmux-pp-runnable-partition --no-pdmux-pp-decode-aware-slices --no-pdmux-pp-adaptive-inflight --no-pdmux-pp-cpu-merge-tokens --no-pdmux-pp-lazy-admission-budget --pdmux-pp-prefill-chunk-size 8192 --pdmux-pp-max-inflight-tokens 65536 --pdmux-pp-max-budget 16384 --pdmux-pp-token-refill-size 16384
# worker1
env CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=/root/workspace/sglang/python /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --tp 1 --pp-size 2 --dtype bfloat16 --port 31001 --disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --enable-pdmux --chunked-prefill-size -1 --pdmux-config-path /root/workspace/muxpp_work/configs/pdmux_pp_4090_v4.yml --mem-fraction-static 0.71 --cuda-graph-max-bs 32 --cuda-graph-bs 1 4 8 16 32 --max-prefill-tokens 16384 --pdmux-pp-max-budget 16384 --pdmux-pp-max-inflight-prefill 3 --pdmux-pp-radix-eviction lru --pdmux-pp-policy alp --pdmux-pp-slo-ttft 5.0 --pdmux-pp-slo-tpot 0.1 --pdmux-pp-tpot-coeff 0.75 --pdmux-pp-reserve-decode --pdmux-pp-balance-merge --pdmux-pp-rebalance --pdmux-pp-alp-admission --pdmux-pp-rebalance-unit 8 --pdmux-pp-summary-interval 10 --pdmux-prefill-mlp-chunk-tokens 4096 --attention-backend flashinfer --pdmux-pp-latency-partition --no-pdmux-pp-runnable-partition --no-pdmux-pp-decode-aware-slices --no-pdmux-pp-adaptive-inflight --no-pdmux-pp-cpu-merge-tokens --no-pdmux-pp-lazy-admission-budget --pdmux-pp-prefill-chunk-size 8192 --pdmux-pp-max-inflight-tokens 65536 --pdmux-pp-max-budget 16384 --pdmux-pp-token-refill-size 16384
# router
env /root/venvs/sgl/bin/python -m sglang_router.launch_router --host 127.0.0.1 --port 30000 --policy cache_aware --request-timeout-secs 3600 --max-concurrent-requests 4096 --log-level info --worker-urls http://127.0.0.1:31000 http://127.0.0.1:31001
```

### 24B PP4

```bash
source /root/workspace/muxpp_work/env.sh
# server
env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/root/workspace/sglang/python PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FLASHINFER_WORKSPACE_SIZE=134217728 /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /dev/shm/models/Mistral-Small-24B-Instruct-2501 --tp 1 --pp-size 4 --dtype bfloat16 --port 30000 --disable-overlap-schedule --context-length 32768 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --enable-pdmux --chunked-prefill-size -1 --pdmux-config-path /root/workspace/muxpp_work/configs/pdmux_pp_4090_big_r4.yml --mem-fraction-static 0.81 --cuda-graph-max-bs 32 --cuda-graph-bs 1 4 8 16 32 --max-prefill-tokens 16384 --pdmux-pp-max-budget 16384 --pdmux-pp-max-inflight-prefill 6 --pdmux-pp-radix-eviction lru --pdmux-pp-policy alp --pdmux-pp-slo-ttft 5.0 --pdmux-pp-slo-tpot 0.1 --pdmux-pp-tpot-coeff 0.99 --pdmux-pp-reserve-decode --pdmux-pp-balance-merge --pdmux-pp-rebalance --pdmux-pp-alp-admission --pdmux-pp-rebalance-unit 8 --pdmux-pp-summary-interval 10 --pdmux-prefill-mlp-chunk-tokens 4096 --attention-backend flashinfer --pdmux-pp-latency-partition --no-pdmux-pp-runnable-partition --no-pdmux-pp-decode-aware-slices --no-pdmux-pp-adaptive-inflight --no-pdmux-pp-cpu-merge-tokens --no-pdmux-pp-lazy-admission-budget --pdmux-pp-stage-local-lm-head --pdmux-pp-compact-attention --pdmux-prefill-mlp-pad-to-chunk --pdmux-pp-prefill-chunk-size 4096 --pdmux-pp-prefill-projection-pad-to 4096 --pdmux-pp-max-inflight-tokens 65536 --pdmux-pp-max-budget 16384 --pdmux-pp-token-refill-size 16384
```

### 24B PP2×TP2

```bash
source /root/workspace/muxpp_work/env.sh
# server
env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/root/workspace/sglang/python PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FLASHINFER_WORKSPACE_SIZE=134217728 /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /dev/shm/models/Mistral-Small-24B-Instruct-2501 --tp 2 --pp-size 2 --dtype bfloat16 --port 30000 --disable-overlap-schedule --context-length 32768 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --enable-pdmux --chunked-prefill-size -1 --pdmux-config-path /root/workspace/muxpp_work/configs/pdmux_pp_4090_v4.yml --mem-fraction-static 0.82 --cuda-graph-max-bs 32 --cuda-graph-bs 1 4 8 16 32 --max-prefill-tokens 16384 --pdmux-pp-max-budget 16384 --pdmux-pp-max-inflight-prefill 3 --pdmux-pp-radix-eviction lru --pdmux-pp-policy alp --pdmux-pp-slo-ttft 5.0 --pdmux-pp-slo-tpot 0.1 --pdmux-pp-tpot-coeff 0.75 --pdmux-pp-reserve-decode --pdmux-pp-balance-merge --pdmux-pp-rebalance --pdmux-pp-alp-admission --pdmux-pp-rebalance-unit 8 --pdmux-pp-summary-interval 10 --pdmux-prefill-mlp-chunk-tokens 4096 --disable-custom-all-reduce --attention-backend flashinfer --pdmux-pp-latency-partition --no-pdmux-pp-runnable-partition --no-pdmux-pp-decode-aware-slices --no-pdmux-pp-adaptive-inflight --no-pdmux-pp-cpu-merge-tokens --no-pdmux-pp-lazy-admission-budget --pdmux-pp-stage-local-lm-head --pdmux-pp-compact-attention --pdmux-prefill-mlp-pad-to-chunk --pdmux-pp-prefill-chunk-size 8192 --pdmux-pp-prefill-projection-pad-to 8192 --pdmux-pp-cpu-prefill-metadata
```

## 代码与复现入口

```text
0dbb8554b Add opt-in token-chunk pipeline prefill with ordered KV ownership
b4a0bc93e Validate ordered terminal-stage token credits for prefill chunks
fea8d1343 Test chunk terminal sampling and token admission credits
74654b9ae Fence chunk metadata copies before asynchronous prefill
55e290b4a Cover prefill chunk metadata stream ordering in admission test
fbbf7ee43 Retain whole-batch fast path below the pipeline chunk budget
c690b5d0d Retain terminal request view for chunk sampler weak references
b43c85b44 Keep token credits independent of the adaptive batch-count gate
8d9bab795 Avoid terminal sampling index copies that synchronize pipeline prefill
9cd605ea7 Preserve configured MLP tail padding across pipeline chunks
eb1702183 Format pipeline chunk implementation and regression tests
09e3a6a56 Bound receive storage by the chunk plan and credit whole-batch completion
48c02faf2 Attend to appended pipeline chunks in one causal KV kernel
286e958cc Pad short CPP linear projections without adding attention or KV rows
60ae81f76 Preserve unconfigured attention subclasses in projection padding path
563b32fe9 Keep CPP tail attention reduction independent of automatic KV splitting
7e1adaecf Coalesce token credit refills before admitting pipeline prefill batches
1dcbb285e Avoid CUDA readback for pipeline prefill sequence and prefix metadata
```

开发复现：`scripts/p11a_runner.py --round <新标签> --source <冻结worktree> --config configs/p11a_selected.json`。正确性/原路径入口分别为 `p11a_final_checks.py`、`p11a_legacy.py`。分析入口为 `p11a_analyze.py`、`p11a_compare.py`、`p11a_selection_report.py`、`p11a_plot.py`、`p11a_audit.py --final`。不调用任何正式trace。
