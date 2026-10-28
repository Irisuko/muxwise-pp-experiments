# P10B：大模型正确性、显存与调度开发复核

已完成显存与早期数值分叉修复、四轮开发验证、一次Llama回归及六配置启动命令整理。**验收限制：PP4仍有显著延迟差距；Llama回归TTFT/主SLO较历史基准变差，未达到“确认未退化”的要求。** 这些限制保留为未解决项，不能将本次选择当作正式性能结论。未跑正式实验，未push。

起点`03e2fe794`，分支muxwise-pp。提交：`89ccd11ce`（MLP尾块与主要显存修复）、`a713881f9`（unused wrapper/MLP生命周期）、`4c10135d2`（可选附加上下文校准）。最终HEAD `4c10135d2e2891d6fb262d4894f0769c86dd5751`。

## 实验边界与证据

大模型回放仅用 `bench/dev_big_toolagent_1200_80_x7.5.jsonl`：开发前150条、time-scale=1、warmup2、force-close、drain300；每次先31663-token固定输入、8输出的长预热并flush。所有结果是单次开发验证，没有正式结论或统计显著性保证。

正确性读取P10A原始20条输入 `logs/p10a/reference-tp4-serial20/serial20-prompts.json`，长度100–8000，temperature=0、ignore_eos、64输出、skip-tokenizer-init。普通PP4/TP1和普通PP2/TP2分别作为参照，非pdmux、FlashInfer、相同graph bs 1/4/8/16/32、禁用token chunk。serial保留缓存；cold逐条flush并断言cached_tokens=0，8条4500–8000输入触发4096 MLP分块。以下请求编号和输出位置都从0开始。

日志保存在 `logs/p10b/`，每run有commands、run.json、GPU采样、server/client日志、source.patch及清理记录。每次提交的GPU验证范围分别标明，不把CPU通过当成GPU通过。31k剖析插桩同步当前CUDA stream，单独运行，不用于性能测量。

## 正确性复核

| 实现/拓扑 | serial20 请求/token相同 | cold20 请求/token相同 | cold首分叉 |
| --- | --- | --- | --- |
| 原始R0 PP4 | 20/20，1280/1280 | 16/20，1151/1280 | req12 pos58；req14 pos41；req16 pos19；req17 pos1 |
| 原始R0 PP2×TP2 | 20/20，1280/1280 | 19/20，1265/1280 | req12 pos45 |
| R1b PP4 | 20/20，1280/1280 | 19/20，1252/1280 | 仅req12 pos35；无pos0–2分叉 |
| R1b PP2×TP2 | 20/20，1280/1280 | 20/20，1280/1280 | 无 |
| R2e PP4 | 20/20，1280/1280 | 19/20，1252/1280 | 仅req12 pos35；无早期分叉 |
| R2e PP2×TP2 | 20/20，1280/1280 | 20/20，1280/1280 | 无 |
| R4 PP4 | 20/20，1280/1280 | 19/20，1252/1280 | 仅req12 pos35；无早期分叉 |

同拓扑普通PP4也与P10A归档MuxWise-PP PP4 serial20完全一致，原TP4跨拓扑比较不能作为PP4错误的依据。完整对比见 `correctness-r0.json`、`correctness-r1b.json`、`correctness-r2e.json` 及 `r4-mux-pp4/*-comparison.json`。

真正的早期问题来自cold req17（7000 token）：第0层attention输出、MLP输入均相同；MLP前4096行采样相同，尾块从4096起出现BF16差异，第0层MLP最大采样差0.001953125。输出位置1参照token2782/5135 logits并列22.75；未补齐尾块时为22.625/22.75，greedy改变。PP4为TP1，排除了TP归约差异；首次分叉前的层输入及attention相同，也不支持层区间/PP proxy截断错误。Mistral使用Llama共享实现，修复位置在共同MLP。

关闭MLP分块的诊断：414个逐层采样张量/完整logits比较均相同。改1024分块虽前4输出相同，hidden仍不同，未选为修复。末尾不足4096行时补零到4096，独立MLP后裁回有效行：414个采样张量/完整logits比较均相同，前4输出也相同。采样包括块边界和最后32行，每行全部hidden维度；这不是全部token行逐位证明。证据为 `numeric-plain-pp4`、`numeric-mux-pp4-{4096,0,1024,pad}` 的torch记录、numeric-comparison.json及report.txt。

补齐尾块选项默认关闭；dummy行只经过MLP，不进入attention/KV。R1b、R2e及最终R4全20条验证已消除pos0–2阻断，但PP4 cold仍有一条后期差异，**不声称全模型bitwise等价**。

## 显存构成与修复

容量按整个实例可容纳的token位置数报告，不乘stage/TP数量。两种拓扑各rank的裸KV均为capacity×40960 bytes；下表pool还包含请求映射与allocator元数据。weights列是权重初始化阶段allocated，略含buffer；实际每PP4 stage的10层参数10.3518GiB，embedding/head各1.25GiB；PP2×TP2每rank20层参数10.3519GiB、embedding/head分片各.625GiB。

旧PP4 .80 OOM栈在FlashInfer ragged+prefix `merge_state` 的248MiB分配；不能只归因于MLP。未使用head、后台工作区、双接收槽和长prefill临时张量共同挤压空间。原.76 KV118415；PP2×TP2 .78 KV162858。

以下单位GiB，每行按stage/TP rank；graph是该初始化阶段allocated增量，不能与graph reserved或workspace重复相加。

| 配置 | stage / TP rank | weights | KV+pool | attention backend | graph | prefill/decode槽 | 31k peak allocated | peak reserved | peak inactive_split |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R0 PP4 | 0 / 0 | 12.867 | 4.548 | 1.605 | 0.446 | 0.002 | 21.476 | 22.586 | 0.239 |
| R0 PP4 | 1 / 0 | 11.617 | 4.548 | 1.605 | 0.446 | 1.255 | 21.175 | 22.732 | 0.437 |
| R0 PP4 | 2 / 0 | 11.617 | 4.548 | 1.605 | 0.446 | 1.255 | 21.175 | 22.732 | 0.437 |
| R0 PP4 | 3 / 0 | 11.617 | 4.548 | 1.605 | 0.436 | 1.252 | 21.163 | 22.730 | 0.431 |
| R0 PP2×TP2 | 0 / 0 | 11.618 | 6.268 | 1.605 | 0.446 | 0.002 | 21.572 | 22.441 | 0.192 |
| R0 PP2×TP2 | 0 / 1 | 11.618 | 6.268 | 1.605 | 0.446 | 0.002 | 21.572 | 22.441 | 0.192 |
| R0 PP2×TP2 | 1 / 0 | 10.993 | 6.268 | 1.605 | 0.436 | 1.252 | 21.884 | 22.461 | 0.170 |
| R0 PP2×TP2 | 1 / 1 | 10.993 | 6.268 | 1.605 | 0.436 | 1.252 | 21.884 | 22.461 | 0.170 |
| R1 PP4 | 0 / 0 | 11.617 | 6.924 | 0.977 | 0.438 | 0.002 | 22.003 | 22.363 | 0.000 |
| R1 PP4 | 1 / 0 | 10.367 | 6.924 | 0.977 | 0.438 | 0.630 | 21.079 | 21.424 | 0.000 |
| R1 PP4 | 2 / 0 | 10.367 | 6.924 | 0.977 | 0.438 | 0.630 | 21.079 | 21.424 | 0.000 |
| R1 PP4 | 3 / 0 | 11.617 | 6.924 | 0.977 | 0.428 | 0.627 | 22.317 | 22.672 | 0.000 |
| R1 PP2×TP2 | 0 / 0 | 10.993 | 7.765 | 0.977 | 0.438 | 0.002 | 21.844 | 22.203 | 0.000 |
| R1 PP2×TP2 | 0 / 1 | 10.993 | 7.765 | 0.977 | 0.438 | 0.002 | 21.844 | 22.203 | 0.000 |
| R1 PP2×TP2 | 1 / 0 | 10.993 | 7.765 | 0.977 | 0.428 | 0.627 | 22.158 | 22.496 | 0.000 |
| R1 PP2×TP2 | 1 / 1 | 10.993 | 7.765 | 0.977 | 0.428 | 0.627 | 22.158 | 22.496 | 0.000 |

原始每分区graph捕获added_reserved约86MiB，8分区保留、每分区bs1/4/8/16/32未删减。PP4全部graph初始化free显存差约.74GiB，PP2约.85GiB；private graph pool约184–192MiB，其余包括graph后端索引/元数据，详见各memory JSONL的ready.graph_pools和backend_tensors。上述数字观察口径不同，不能相加。NVML使用量还含CUDA context/通信库/驱动开销，不等于PyTorch allocated。

原始FlashInfer prefill与decode分别持有独立384MiB float workspace，保证两条并发stream不共享冲突；额外9个backend的wrapper缓冲使backend阶段共1.605GiB，其中8个decode backend各有未用的64MiB verify索引缓冲，以及ragged/paged/verify整数工作区。R1 compact decode backend使用skip_prefill，backend阶段降到.977GiB。

非首stage默认两个32768×5120×BF16 hidden/residual接收槽，总1280MiB；单槽640MiB。decode TX/RX仅stage0 2.5MiB、内级5MiB、末级2.5MiB。R1单槽达到容量目标，但可能增加上游等待，因此R2恢复双槽，不把节省显存直接当成调度收益。

31k实测attention临时峰值：PP4 R0/R1=1375.9/1372.1MiB；PP2×TP2每rank=687.8/686.0MiB。MLP临时峰值：PP4约1118/1157.2MiB，PP2每rank734/773.2MiB；R1尾块补齐带来最多约40MiB额外hidden。MLP按4096行计算宽中间张量，但整个请求attention和hidden/residual保持完整，chunked-prefill-size=-1，没有改变整请求prefill语义。Mistral中间宽32768，若整个31663行的gate/up和SwiGLU同时展开，仅这两项约5.8GiB；4096分块将宽张量限制为512+256MiB/PP4 rank，另有完整hidden输出、输入、residual及pad等。

R1使用expandable_segments后，PyTorch报告inactive_split峰值为0；这不等于所有reserved缓存/graph保留消失。R0 PP4 reserved比allocated峰值多约1.1–1.6GiB，不能全标为碎片。完整逐层临时峰值和pointer去重后的workspace清单保存在各memory-report.json及原始memory-*.jsonl。

实现改动：

- 默认关闭的stage-local-lm-head只释放非末stage不用的head；容量探测前gc.collect打破weight_loader bound-method引用环，确保真正回收。末级head、tied embedding保护保留。
- 默认关闭的compact-attention减少decode-only工作区。默认不改变其他路径。
- 默认关闭的MLP-tail-pad固定末块形状；默认Llama8B仍沿用原分块行为。
- YAML pp_prefetch_slots支持1/2，默认2；idle self-check按实际槽数验证归还。
- 增加仅环境变量启用的逐层数值/显存诊断，不在性能运行开启。

## 最终配置显存剖析

最终两组MuxWise-PP保留双prefill接收槽、8个分区、每分区graph bs1/4/8/16/32；FlashInfer prefill/decode各独立128MiB float workspace。以下是独立31k剖析，不混入性能测量。单位GiB。

| 拓扑 | stage/TP | weights | KV+pool | backend | graph增量 | receive/TX/RX | peak allocated | peak reserved | peak inactive_split |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pp4 | 0/0 | 11.617 | 6.924 | 0.344 | 0.438 | 0.002 | 21.331 | 21.678 | 0.000 |
| pp4 | 1/0 | 10.367 | 6.924 | 0.344 | 0.438 | 1.255 | 21.031 | 21.365 | 0.000 |
| pp4 | 2/0 | 10.367 | 6.924 | 0.344 | 0.438 | 1.255 | 21.031 | 21.365 | 0.000 |
| pp4 | 3/0 | 11.617 | 6.924 | 0.344 | 0.428 | 1.252 | 22.270 | 22.613 | 0.000 |
| pp2tp2 | 0/0 | 10.993 | 7.765 | 0.344 | 0.438 | 0.002 | 21.172 | 21.520 | 0.000 |
| pp2tp2 | 0/1 | 10.993 | 7.765 | 0.344 | 0.438 | 0.002 | 21.172 | 21.520 | 0.000 |
| pp2tp2 | 1/0 | 10.993 | 7.765 | 0.344 | 0.428 | 1.252 | 22.111 | 22.438 | 0.000 |
| pp2tp2 | 1/1 | 10.993 | 7.765 | 0.344 | 0.428 | 1.252 | 22.111 | 22.438 | 0.000 |

来源：`logs/p10b/r4b-profile-pp4/memory-report.json`、`logs/p10b/r2e-profile-pp2tp2/memory-report.json`。

GPU验证提交范围：R2e两拓扑为a713881f9，R4 PP4和Llama回归为最终4c10135d2。后者只新增可选校准点；PP2×TP2的extra列表仍为空，其校准网格与R2e逐点相同，因此沿用R2e数据，未将它标成最终HEAD重新测得。

最终backend阶段从原1.605降到约.344GiB/卡。R2逐pointer清单显示：各decode backend的wrapper整数工作区48MiB（基础8MiB加5个graph wrapper各8MiB），8个backend合计384MiB；prefill backend另有ragged/paged/decode三个8MiB整数工作区。40个graph的wrapper部分包含在graph初始化增量内，不与backend/graph列重复相加。decode float workspace128MiB由8分区共享，prefill另有独立128MiB。每个decode backend的KV indices约4MiB；原始清单见ready.backend_tensors，分组表见`r2e-profile-pp4/graph-workspace-breakdown.json`。private graph pool在R2快照中保留192MiB，全部graph共享，不是每分区192MiB。

`a713881f9`仅在opt-in路径于SwiGLU后释放已无用gate/up，避免它与down投影输出重叠；再省去decode-only ragged与无speculative用途的verify wrapper约136MiB常驻。31k attention仍完整计算，未引入token chunk的调度语义。最终长请求峰值与结束资源守恒均有实际GPU证据。

## 预检与四轮开发结果

| 预检 | 31k耗时s | 20条冒烟完成/失败 | 冒烟耗时s | KV |
| --- | ---: | --- | ---: | ---: |
| r1b/pp4 | 12.479 | 20/0 | 83.596 | 181066 |
| r1b/pp2tp2 | 8.091 | 20/0 | 52.661 | 203111 |
| r2e/pp4 | 12.491 | 20/0 | 80.481 | 181066 |
| r2e/pp2tp2 | 8.055 | 20/0 | 52.610 | 203111 |
| r3/pp4 | 12.537 | 20/0 | 78.000 | 181066 |
| r4/pp4 | 12.483 | 20/0 | 78.495 | 181066 |

P10B冒烟是开发前20条、time-scale=.2、原输出长度、客户端上限120s，要求20/20；P10A为60s限时窗口，不混作同一门槛。所有最终stage的free_kv+evictable=capacity、free_reqs=128、接收槽全部归还，见各run的p10b-audit.json。

R1：单接收槽、384MiB float workspace、原.75系数。R2：双槽、128MiB workspace、PP4系数.90，PP2保持.75。R3只将PP4在途上限6→4；R4恢复6，系数.99，新增12k/24k校准点，混合分区改为104/24、96/32、88/40、80/48、72/56、64/64，总数仍8。R3/R4只改PP4，PP2沿用R2，未重复不变配置。总共四轮，所有性能行均150完成/0失败/0未完成。

主SLO为TTFT≤5s且TPOT≤100ms，以下都是指定开发前150条、time-scale=1的单次记录。

| 轮次/拓扑 | TTFT p50/p90/p95/p99 s | TPOT p50/p90/p99 ms | 主SLO | cache加权比 | 总时长s |
| --- | --- | --- | ---: | ---: | ---: |
| P10A/pp4 | 60.640 / 77.009 / 81.228 / 95.438 | 70.553 / 100.346 / 437.935 | 1/150 | 37.568% | 296.053 |
| P10A/pp2tp2 | 7.599 / 13.137 / 16.054 / 17.623 | 57.785 / 76.870 / 90.758 | 50/150 | 44.233% | 202.049 |
| r1/pp2tp2 | 7.816 / 17.953 / 19.064 / 26.352 | 58.521 / 75.941 / 147.900 | 43/150 | 44.233% | 202.930 |
| r1/pp4 | 49.138 / 75.472 / 75.475 / 93.390 | 70.831 / 107.268 / 345.796 | 0/150 | 43.101% | 294.596 |
| r2e/pp2tp2 | 8.336 / 15.653 / 15.955 / 18.677 | 58.667 / 82.554 / 119.716 | 39/150 | 44.233% | 202.629 |
| r2e/pp4 | 27.535 / 45.518 / 50.174 / 59.000 | 79.313 / 117.717 / 385.415 | 6/150 | 40.831% | 247.115 |
| r3/pp4 | 28.264 / 41.911 / 44.883 / 48.208 | 78.972 / 113.853 / 354.722 | 3/150 | 43.524% | 245.690 |
| r4b/pp4 | 17.061 / 36.784 / 41.379 / 50.420 | 86.155 / 114.357 / 332.323 | 8/150 | 40.123% | 234.780 |

最终PP4 KV相对P10A增加52.91%，达到同拓扑Chunked的88.64%；PP2×TP2增加24.72%，达到对应Chunked的85.76%。容量目标达到，长预热没有OOM。

## PP4归因与选择

每轮P9a分析匹配150/150，既保留p50/p90对应请求分解，也保留每分量独立分布。下表p50按对应请求插值后分组，非各分量的独立中位数。

| PP4轮次 | p50 TTFT | 准入 | stage0本地FIFO | 四级prefill | 三级链路/队列 | 其余 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r1 | 49.138 | 22.770 | 7.472 | 15.014 | 3.701 | 0.181 |
| r2e | 27.535 | 0.104 | 8.700 | 5.260 | 12.411 | 1.060 |
| r3 | 28.264 | 15.682 | 1.798 | 4.221 | 6.457 | 0.105 |
| r4b | 17.061 | 7.000 | 1.769 | 3.803 | 4.347 | 0.141 |

| PP4轮次 | inflight阻塞s | KV阻塞s | decode reserve阻塞s | 平均准入+stage0队列s | stage3等上游% |
| --- | ---: | ---: | ---: | ---: | ---: |
| r1 | 144.998 | 2.175 | 13.648 | 33.007 | 40.586 |
| r2e | 84.033 | 18.584 | 13.884 | 16.240 | 27.831 |
| r3 | 109.006 | 14.226 | 8.855 | 15.475 | 31.653 |
| r4b | 66.234 | 18.113 | 7.953 | 11.676 | 32.316 |

阻塞时间是可匹配summary窗口的状态驻留，不是可相加的逐请求延迟。R1的单槽、保守分区和队列共同限制流水线；R2恢复双槽后，stage0 READY驻留从约32%降至8%，末级等上游从约41%降至28%。但R2 KV阻塞时protected中位数174976/181066，准入仍受长上下文活跃请求占用影响。R3减小在途上限，KV/预留阻塞减少，却增加inflight等待，未改善主SLO，因此不选。

R2实际decode微批以1–3条为主；约19.22%的decode观测mean context在8–12k，7.97%在16–24k，旧上取整网格把它们按16k/32k估计。R4补测12k/24k并增加40/56 SM中间档，以减少分区选择的粗粒度损失。仍使用实测上界、单调上包络、至少1.2在线安全修正、1ms/级开销，不插值或外推；PP4预测预算99ms，完整未来输出KV预留、FCFS和PP/TP协议未放松。新增pp_decode_extra_contexts默认空，Llama v4原始校准点不变。

最终pp4选择`r4b-dev-pp4`：TTFT p50 17.061s、主SLO 8/150、KV 181066。

最终pp2tp2选择`r2e-dev-pp2tp2`：TTFT p50 8.336s、主SLO 39/150、KV 203111。

R4末级混合分区驻留主要为88/40约19.77%、80/48约39.04%，旧R1的64/64约74.03%降为0；减少的decode保留SM留给prefill，但TPOT尾部仍高，说明预测预算并非端到端SLO保证。

R1/P10A与恢复后的R2–R4使用了不同GPU UUID分配，虽然均为4×4090且NUMA拓扑一致，不能将跨分配的全部差值归于单项代码变化。R2–R4在同一恢复分配上运行，但仍为单次开发结果。PP4的主SLO、尾延迟及相对PP2差距均按上表披露；四轮上限到达后不增加第五轮，也不宣称PP4所有性能问题已消失。

## Llama8B v4回归与测试

最终HEAD PP2×TP2 v4 serial20：20/20请求、1280/1280 token完全一致，参照为P9a普通TP2归档。新增三项显存/数值开关均为off，原v4 YAML SHA256仍为`3bd2a392dcaae12cac690aa48de8f671de5b8cbff76769a5a388236d3aa784ae`。Tool&Agent仅回归原Llama开发trace前300条一次；未使用Mistral tokenizer的trace冒充Llama对照。

| Llama PP2×TP2 | 完成/失败/未完成 | TTFT p50/p90/p95/p99 s | TPOT p50/p90 ms | 主SLO | in/out tok/s |
| --- | --- | --- | --- | ---: | --- |
| P9a v4开发基准 | 300/0/0 | 3.031 / 5.884 / 6.970 / 8.572 | 48.685 / 70.581 | 262/300 | 17344.17/353.73 |
| P10B最终HEAD回归 | 300/0/0 | 3.878 / 6.345 / 7.058 / 8.628 | 46.891 / 72.960 | 243/300 | 17336.56/353.57 |

**Llama性能非退化未获确认。** TTFT p50增加27.95%，主SLO下降19条（6.33个百分点）；TPOT p50改善3.69%，吞吐仅下降约0.04%。共同server参数仅random_seed、trace路径和等内容的v3/v4 YAML路径不同；KV仍390622，三个新开关均关闭，新增context网格为空。历史性能数据来自P9a选定R1提交a17263c91，并非在本次GPU分配下测得的P10B改动前基线，因此不能据单次差异断言P10B代码是原因，也不能宣称未退化。

300/300 TTFT事件匹配：平均准入基本持平（.891890→.893128s），本地FIFO .669859→.833156s、级间队列 .207217→.319282s，两级prefill均值合计1.413155→1.454491s。增加的平均TTFT主要集中在本地/级间排队；分区驻留和GPU计算占比接近历史。按要求只运行一次300条，没有为追平历史单次数据改动Llama参数或追加回放。判定及分解保存于`logs/p10b/llama-regression-assessment.json`，selected元数据明确标为NOT_ESTABLISHED。

回归证据：`logs/p10b/llama-serial20-comparison.json`、`logs/p10b/llama-dev300/analysis.json`；两次历史环境分配差异同样保留，不给统计显著性承诺。最终相关CPU测试71 passed（unit-r4.log），客户端10 passed（bench-unit-final.log）；Black及git diff --check通过。附加网格默认与v4完全相同、上界覆盖和非法配置拒绝均有测试，另覆盖PP控制、准入、KV预留、TP重放、单槽生命周期、MLP数学/尾块/张量生命周期及head引用环释放。

## 最终六配置启动命令

`configs/p10b_selected.json`记录选择和验收路径，六个`launch_big_*.sh`已接入`p10b_serve.py`。仅两组MuxWise-PP改变参数；MuxWise TP4与三组Chunked的argv和env已逐项与原P10A比较，完全相同。这四组容量继承P10A证据，本任务没有重跑其开发实验。

| label | mem | KV容量 | 来源 |
| --- | ---: | ---: | --- |
| big_muxwise_tp4 | 0.76 | 167692 | P10A，命令不变 |
| big_chunked_tp4 | 0.90 | 252288 | P10A，命令不变 |
| big_chunked_pp4 | 0.90 | 204263 | P10A，命令不变 |
| big_chunked_pp2tp2 | 0.90 | 236846 | P10A，命令不变 |
| big_muxpp_pp4 | 0.81 | 181066 | r4-preflight-pp4 |
| big_muxpp_pp2tp2 | 0.82 | 203111 | r2e-preflight-pp2tp2 |

先执行 `source /root/workspace/muxpp_work/env.sh`。对应六个绝对路径脚本均可直接启动；下面是其展开命令（不包含开发trace插桩）：

### big_muxwise_tp4

```bash
env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/root/workspace/sglang/python PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /dev/shm/models/Mistral-Small-24B-Instruct-2501 --tp 4 --pp-size 1 --dtype bfloat16 --port 30000 --disable-overlap-schedule --context-length 32768 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --attention-backend flashinfer --mem-fraction-static 0.76 --cuda-graph-max-bs 32 --cuda-graph-bs 1 4 8 16 32 --disable-custom-all-reduce --enable-pdmux --chunked-prefill-size -1 --pdmux-config-path /root/workspace/muxpp_work/configs/pdmux_4090.yml
```

### big_chunked_tp4

```bash
env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/root/workspace/sglang/python /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /dev/shm/models/Mistral-Small-24B-Instruct-2501 --tp 4 --pp-size 1 --dtype bfloat16 --port 30000 --disable-overlap-schedule --context-length 32768 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --attention-backend flashinfer --mem-fraction-static 0.9 --cuda-graph-max-bs 32 --cuda-graph-bs 1 4 8 16 32 --disable-custom-all-reduce --chunked-prefill-size 2048 --enable-mixed-chunk
```

### big_chunked_pp4

```bash
env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/root/workspace/sglang/python /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /dev/shm/models/Mistral-Small-24B-Instruct-2501 --tp 1 --pp-size 4 --dtype bfloat16 --port 30000 --disable-overlap-schedule --context-length 32768 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --attention-backend flashinfer --mem-fraction-static 0.9 --cuda-graph-max-bs 32 --cuda-graph-bs 1 4 8 16 32 --chunked-prefill-size 2048
```

### big_chunked_pp2tp2

```bash
env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/root/workspace/sglang/python /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /dev/shm/models/Mistral-Small-24B-Instruct-2501 --tp 2 --pp-size 2 --dtype bfloat16 --port 30000 --disable-overlap-schedule --context-length 32768 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --attention-backend flashinfer --mem-fraction-static 0.9 --cuda-graph-max-bs 32 --cuda-graph-bs 1 4 8 16 32 --disable-custom-all-reduce --chunked-prefill-size 2048
```

### big_muxpp_pp4

```bash
env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/root/workspace/sglang/python PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FLASHINFER_WORKSPACE_SIZE=134217728 /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /dev/shm/models/Mistral-Small-24B-Instruct-2501 --tp 1 --pp-size 4 --dtype bfloat16 --port 30000 --disable-overlap-schedule --context-length 32768 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --enable-pdmux --chunked-prefill-size -1 --pdmux-config-path /root/workspace/muxpp_work/configs/pdmux_pp_4090_big_r4.yml --mem-fraction-static 0.81 --cuda-graph-max-bs 32 --cuda-graph-bs 1 4 8 16 32 --max-prefill-tokens 16384 --pdmux-pp-max-budget 16384 --pdmux-pp-max-inflight-prefill 6 --pdmux-pp-radix-eviction lru --pdmux-pp-policy alp --pdmux-pp-slo-ttft 5.0 --pdmux-pp-slo-tpot 0.1 --pdmux-pp-tpot-coeff 0.99 --pdmux-pp-reserve-decode --pdmux-pp-balance-merge --pdmux-pp-rebalance --pdmux-pp-alp-admission --pdmux-pp-rebalance-unit 8 --pdmux-pp-summary-interval 10 --pdmux-prefill-mlp-chunk-tokens 4096 --attention-backend flashinfer --pdmux-pp-latency-partition --no-pdmux-pp-runnable-partition --no-pdmux-pp-decode-aware-slices --no-pdmux-pp-adaptive-inflight --no-pdmux-pp-cpu-merge-tokens --no-pdmux-pp-lazy-admission-budget --pdmux-pp-stage-local-lm-head --pdmux-pp-compact-attention --pdmux-prefill-mlp-pad-to-chunk
```

### big_muxpp_pp2tp2

```bash
env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/root/workspace/sglang/python PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FLASHINFER_WORKSPACE_SIZE=134217728 /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /dev/shm/models/Mistral-Small-24B-Instruct-2501 --tp 2 --pp-size 2 --dtype bfloat16 --port 30000 --disable-overlap-schedule --context-length 32768 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --enable-pdmux --chunked-prefill-size -1 --pdmux-config-path /root/workspace/muxpp_work/configs/pdmux_pp_4090_v4.yml --mem-fraction-static 0.82 --cuda-graph-max-bs 32 --cuda-graph-bs 1 4 8 16 32 --max-prefill-tokens 16384 --pdmux-pp-max-budget 16384 --pdmux-pp-max-inflight-prefill 3 --pdmux-pp-radix-eviction lru --pdmux-pp-policy alp --pdmux-pp-slo-ttft 5.0 --pdmux-pp-slo-tpot 0.1 --pdmux-pp-tpot-coeff 0.75 --pdmux-pp-reserve-decode --pdmux-pp-balance-merge --pdmux-pp-rebalance --pdmux-pp-alp-admission --pdmux-pp-rebalance-unit 8 --pdmux-pp-summary-interval 10 --pdmux-prefill-mlp-chunk-tokens 4096 --disable-custom-all-reduce --attention-backend flashinfer --pdmux-pp-latency-partition --no-pdmux-pp-runnable-partition --no-pdmux-pp-decode-aware-slices --no-pdmux-pp-adaptive-inflight --no-pdmux-pp-cpu-merge-tokens --no-pdmux-pp-lazy-admission-budget --pdmux-pp-stage-local-lm-head --pdmux-pp-compact-attention --pdmux-prefill-mlp-pad-to-chunk
```

## 失败记录、环境恢复与清理

- R0剖析PP4的31k预热通过，但旧60s冒烟18/20，run记FAIL，非OOM。P10B改用明确标注的120s完整冒烟要求。
- `r1-mux-pp4`暴露单槽idle检查仍硬编码2、unused head bound-method引用环未GC；修复后以r1b重跑，旧证据保留。
- `r2-preflight-pp4`在31k MLP down投影申请40MiB时真实OOM；a713881f9回收额外工作区和无用gate/up后，R2e起长请求及并发冒烟均通过。
- 前一会话r2b/r2c启动时GPU2/3已有容器内不可见占用，权重加载OOM，未计有效开发轮次。该会话已清理本任务进程并报告阻断。用户恢复独占后重新确认四卡1MiB/0%；UUID变更记录于resume-gpu-map.csv、resume-topology.txt。
- /dev/shm模型丢失后从私有盘恢复，16文件、47,162,178,592 bytes完整SHA256通过、363张量与索引一致，见resume-model-validation.json。没有更换模型快照。
- R4首次开发启动前，四卡显存均1MiB，但GPU1单次利用率52%，guard立即退出，未启动服务或发送trace。随后六次采样20秒全部1MiB/0%，运行guard未见foreign PID，更符合清理后的利用率窗口滞后；原始信号与复核均保留。退出后增加2秒采样稳定时间，以r4b新标签完成同一R4，未额外增加开发迭代。

本次恢复后独占监测没有确认新的持续外部占用；若出现外部PID或连续无法归属的GPU显存，guard会中断本任务runner，不杀外部进程或reset GPU。最终GPU/进程/端口清理与formal文件审计见`logs/p10b/final-cleanup.json`、`logs/p10b/final-audit.json`。正式锁、P10A冻结/选择文件及正式启动标记保持不变，没有新增正式回放，没有push。
