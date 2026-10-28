# P9a：MuxWise-PP 调度优化与 v4 开发配置

## 范围和结论

本任务只回放两个 `dev_*_1200_240_x2.5.jsonl` 的前300条；没有运行任何正式 trace。Conversation 正式文件只生成和统计。未修改16行正式矩阵 `RESULTS.md`，不将开发结果填入正式结果。代码起点 `d8dfbfaba`，分支 `muxwise-pp`，本地提交、不 push。

v4采用R1的策略：在**原有校准/在线修正后的decode安全候选集合内**，选prefill SM最多的分区。FCFS顺序、KV容量检查、未来输出token预留、stage0全局决策、TP重放与双流排空协议保持原约束。批预算16384；PP2在途上限3、PP4上限6。分区布局和校准数据未改变，v4 YAML与v3逐字节相同。

3–5秒并非硬编码TTFT下限，主要由prefill服务时间和由此产生的准入/流水线FIFO排队造成。原策略只给足以覆盖估计到达率的prefill SM，延长了负载下的队列。增加安全范围内的prefill资源有效；盲目增加在途和频繁切换分区则会加剧排队或切换成本。末级结果回传和首token输出路径不是秒级延迟的主因。

上游decode通知、按可计算队首选分区、工作量约束的自适应在途、CPU token直接MERGE处理和延迟计算准入预算均已实现开关并做开发评估，但没有在两种负载上超过R1组合，最终默认关闭。R5两个控制开销优化作为组合测试，没有做单独消融，不能据此断言其中每个机制独立有害。

## Trace和评估口径

用 `bench/build_trace.py --source /root/ext/mooncake/FAST25-release/traces/conversation_trace.jsonl --start-sec {1200,600} --duration-sec 240 --stretch 2.5 --max-total-len 32768` 生成；分别输出下列开发/正式文件。速率分母是拉伸后的600秒。

| 文件 | 请求数 | 输入token | 输出token | 输入/输出 token/s | 理论前缀复用率 |
| --- | ---: | ---: | ---: | --- | ---: |
| dev_conversation_1200_240_x2.5.jsonl | 780 | 7,259,444 | 257,453 | 12,099.073 / 429.088 | 22.1515% |
| conversation_600_240_x2.5.jsonl（仅生成统计） | 735 | 6,718,097 | 248,112 | 11,196.828 / 413.520 | 20.5581% |

理论复用按已出现hash块的token比例计算，不等于实际radix命中，不包含并发完成、驱逐及连续前缀约束。原始统计及生成日志在 `bench/*conversation*.stats.json` 和 `bench/*conversation_build.log`。性能表另记录实际cached token比例；开发Conversation前300条实际约5.4%。

R0–R5共六组配置，每组两种开发负载、PP2×TP2各300条；R0和选定v4另外验证PP4两负载。每组独立启动、31663-token固定随机预热、flush，然后 `--limit 300 --time-scale 1 --warmup 2 --force-close --drain-timeout 300`。硬件为四张RTX4090；GPU0/1、GPU2/3各在同NUMA节点，跨两组为SYS，详见 `logs/p9a/topology.txt` 和 `gpu-map.csv`。所有性能评估都启用相同请求/round插桩，正常启动默认关闭请求trace。

吞吐按完整测量时间（含排空）计算；GPU利用率取客户端测量窗口内5秒采样均值。stage lane、准入和分区指标取到达窗口内首末10秒summary的差分，具体窗口见明细，不把预热和排空混入准入阻塞比较。CUDA事件包含stream等待，prefill/decode可重叠，其累计时间不能相加当作GPU利用率；NCCL等待也可能表现为GPU忙。

## 第0步：逐请求TTFT分解

新增 `--pdmux-pp-request-trace PATH_PREFIX`，每stage仅TP0写JSONL。记录stage0收到请求、ADMIT、FIFO队首、hidden元数据、接收槽等待/READY、prefill开始和每slice发射/完成观测、CUDA事件、hidden发送、末级首token结果、stage0接收结果、MERGE、首token交给detokenizer。客户端保存响应 `meta_info.id` 和首token时刻，以请求ID关联。

插桩不加CUDA同步；完成时间为CPU首次观测CUDA event完成的时刻，含轮询间隔。客户端→stage0还含HTTP、tokenizer和scheduler收件轮询，当前没有把这几个子项进一步拆开。READY的两端时间及接收槽等待另存诊断分布，与主链分量可能重叠，不能重复加总。逐请求结果见各目录 `ttft-requests.jsonl`；每组300/300关联。

Tool&Agent PP2×TP2基线TTFT p50=3.515454s、p90=6.727835s。下表先按TTFT排序，再对分位对应两个请求的分量使用相同线性插值，因此严格加和为该TTFT分位；不是各分量独立分位数的和。

| 分量(s) | p50对应分解 | p90对应分解 |
| --- | ---: | ---: |
| 客户端发出→stage0接收 | 0.150018 | 0.202202 |
| stage0接收→准入 | 0.075128 | 4.762609 |
| 准入→stage0 prefill开始 | 1.368258 | 0.459795 |
| stage0 prefill服务区间 | 0.839737 | 0.274746 |
| stage0完成→stage1开始（通信/背压/队列） | 0.159274 | 0.707441 |
| stage1 prefill服务区间 | 0.830224 | 0.280491 |
| 末级完成→结果发出 | 0.000038 | 0.000024 |
| 结果返回stage0 | 0.021715 | 0.013215 |
| 返回→MERGE | 0.055565 | 0.022680 |
| MERGE→首token交给detokenizer | 0.013487 | 0.003981 |
| detokenizer/tokenizer/HTTP→客户端 | 0.002009 | 0.000650 |

全体请求平均：准入前1.272s、stage0本地FIFO0.657s、两级prefill1.607s、级间0.338s；结果回传+MERGE+发送约0.063s。每请求所属批次的两级CUDA事件均值0.767/0.754s，接近prefill服务区间0.811/0.796s。到达窗口inflight阻塞41.365s/5131次，未见KV或decode reserve阻塞。stage0/1在64/64分区分别驻留31.91%/42.35%。

Conversation基线p50=5.801634s，其中准入3.260325、本地FIFO0.262795、两级prefill1.330558、级间0.859664s；p90=9.797055s，准入就占7.246553s。inflight阻塞131.373s/6188次，同样没有KV/reserve阻塞。PP4 Tool&Agent p50=5.451710s，四级prefill合计2.748714s、本地与级间等待2.628366s；末级prefill lane等待上游占40.743%。

优先级据此确定：先缩短prefill服务、再检查分区空转与上游到达预测、再检验在途策略、最后处理CPU/输出小开销。不提前在MERGE之外输出首token，也不改变radix插入时点。

## 每轮改动与取舍

| 轮次 | 提交 | 配置/改变 | 结论 |
| --- | --- | --- | --- |
| R0 | ab4779fea（插桩） | v3策略，PP2 cap3/PP4 cap6，预算16k | 建立逐请求基线 |
| R1 | a17263c91 | latency-partition=on；同一decode安全集合内最多prefill SM | 两负载均改善；选为v4 |
| R2 | 4e059d1ba | R1+runnable-partition；仅本地FIFO头compute/receiving算prefill需求 | Tool主SLO略增，Conversation下降；默认off |
| R3 | 6a87446b3 | R2+decode-aware-slices；上游控制包附decode预测，TP一致缩短slice | Conversation有收益，Tool退化；默认off |
| R4 | 14cf4d5ca | R1+adaptive-inflight；预算8k，PP2 hard4/soft3；R2/R3 off | 短请求比例改善，整体队列和主SLO恶化；默认off |
| R5 | 1af92a575 | R1+cpu-merge-tokens+lazy-admission-budget，恢复16k/cap3 | 局部开销降低，两负载主SLO均低于R1；默认off |
| 冻结v4 | f6833f24ef0587278ae799696cfe17c3ade99048 | 仅latency-partition默认on；R1策略，其他实验开关off | PP4关键验证和最终回归使用该HEAD |

R1：Tool两级prefill服务均值由0.811/0.796s降到0.706/0.707s，准入排队均值1.272→0.892s、级间等待0.338→0.207s；Conversation两级服务均值0.890/0.891→0.797/0.789s，准入2.875→2.445s、级间0.488→0.381s。本地FIFO均值没有下降，收益主要来自服务加速及准入/级间排队缩短，不能描述为每类等待都变短。

R2：Conversation stage1分区切换wall由R1的0.330%升至1.964%，全decode驻留由10.626%升至27.420%，但没有转化为更快prefill服务。双流排空成本使“远端等待时立即释放prefill SM”不能单凭空转比例判断收益。

R3：已有控制通道携带step、mb、bs、Σctx、主机发出时间、预测decode耗时和下游预计到达时间，每个上游stage转发时更新。它是发射前预测，不能当作实测GPU起点。只接受同主机、年龄≤TPOT目标、模型已就绪的通知；TP0算层数后广播。按每层预测留20%余量及1ms交接余量；仅在至少一整层能放进间隙时缩短，不超过原token-layer上界。Tool stage1共356次slice计划中26次缩短，累计208层延后到后续slice计算，**不是减少208层计算**。测得典型每层约54–57ms，decode轮转约36–43ms，许多间隙装不下一层；因此未继续加入预测驱动的SM切换或反向背压协议。

R4：已有工作token数≤soft_limit×预算且下一FCFS请求≤预算才允许额外槽；真实radix锁定探测后再检验请求长度，防止到达时cache估计过期放进长请求。KV和未来decode预留检查不变。两负载inflight阻塞分别升到42.715/150.200s，故v4仍保留3/6硬上限。

R5：CPU token已经随控制包返回，MERGE在原控制点保留GPU副本供后续forward/split_batch，同时用CPU副本处理输出；不改变radix位置。Tool平均MERGE→emit由11.695ms降到5.013ms。lazy预算在Tool到达窗口仅计算77次、跳过11320次、计算累计15.789ms，但整体主SLO低于R1。因迭代预算没有继续拆分验证，两个开关均保留off。

每次修改后先跑针对性的CPU测试，再跑两负载GPU开发评估。CPU通过数随覆盖增长：R0 27、R1 45、R2 46、R3 48；最终完整PP/MLP测试52项通过，客户端10项通过。早期测试fixture缺字段的失败及修复记录保留在logs，没有覆盖原日志。

## 全部开发指标

主SLO为每请求TTFT≤5s且TPOT≤100ms。以下全为开发前300条，单次回放，不给统计显著性或正式结果保证。`r1v4-pp4-*`是冻结v4对选定R1策略的关键拓扑验证，不是第七轮调参。PP2×TP2的v4数值来自R1冻结源码；最终HEAD只把其选为默认、保留其他关闭分支及增加诊断计数，最终正确性/压测在最终HEAD执行。

| run | done/fail | TTFT p50/p90/p95/p99 s | TPOT p50/p90/p95/p99 ms | main SLO % | TTFT≤2s % | in/out tok/s | GPU % | cache % | inflight wall s |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| r0-pp2tp2-conversation | 300/0 | 5.802 / 9.797 / 11.120 / 13.963 | 38.532 / 48.567 / 52.520 / 60.662 | 38.333 | 8.000 | 11559.07 / 395.32 | 91.96 | 5.43 | 131.373 |
| r0-pp2tp2-toolagent | 300/0 | 3.515 / 6.728 / 8.575 / 9.407 | 36.396 / 68.859 / 85.272 / 93.607 | 67.333 | 8.667 | 17258.66 / 351.99 | 88.02 | 48.62 | 41.365 |
| r0-pp4-conversation | 300/0 | 6.628 / 10.767 / 12.113 / 14.702 | 35.645 / 54.602 / 56.077 / 61.178 | 25.667 | 2.333 | 11255.58 / 384.94 | 94.27 | 5.41 | 82.411 |
| r0-pp4-toolagent | 300/0 | 5.452 / 9.878 / 10.213 / 11.766 | 33.721 / 61.658 / 75.670 / 101.495 | 32.000 | 3.333 | 16655.03 / 339.67 | 89.38 | 48.04 | 25.306 |
| r1-pp2tp2-conversation | 300/0 | 4.928 / 9.384 / 10.851 / 12.919 | 42.619 / 49.338 / 53.652 / 62.629 | 51.333 | 13.667 | 11492.86 / 393.06 | 92.64 | 5.43 | 116.412 |
| r1-pp2tp2-toolagent | 300/0 | 3.031 / 5.884 / 6.970 / 8.572 | 48.685 / 70.581 / 77.230 / 89.274 | 87.333 | 17.667 | 17344.17 / 353.73 | 85.75 | 48.91 | 32.136 |
| r1v4-pp4-conversation | 300/0 | 4.454 / 7.603 / 9.047 / 11.346 | 45.847 / 50.313 / 53.435 / 62.445 | 57.000 | 11.333 | 11194.47 / 382.85 | 90.21 | 5.41 | 47.242 |
| r1v4-pp4-toolagent | 300/0 | 3.201 / 5.107 / 7.027 / 8.575 | 49.961 / 66.446 / 70.007 / 80.778 | 87.000 | 18.333 | 16809.60 / 342.83 | 85.91 | 48.04 | 9.901 |
| r2-pp2tp2-conversation | 300/0 | 5.371 / 8.927 / 10.533 / 13.040 | 44.180 / 49.973 / 55.994 / 61.296 | 48.333 | 12.000 | 11493.04 / 393.06 | 93.14 | 5.43 | 117.194 |
| r2-pp2tp2-toolagent | 300/0 | 3.104 / 5.926 / 6.922 / 8.616 | 48.723 / 71.232 / 73.931 / 83.592 | 88.000 | 17.333 | 17343.77 / 353.72 | 86.98 | 48.62 | 32.793 |
| r3-pp2tp2-conversation | 300/0 | 4.748 / 9.219 / 10.860 / 13.168 | 43.365 / 49.823 / 55.799 / 67.697 | 53.333 | 12.000 | 11480.55 / 392.63 | 93.30 | 5.43 | 113.535 |
| r3-pp2tp2-toolagent | 300/0 | 3.810 / 5.865 / 7.002 / 8.652 | 46.178 / 66.770 / 73.309 / 87.611 | 77.667 | 19.667 | 17349.68 / 353.84 | 87.61 | 48.62 | 31.780 |
| r4-pp2tp2-conversation | 300/0 | 6.660 / 13.333 / 14.719 / 18.124 | 45.347 / 50.724 / 54.443 / 60.337 | 39.000 | 11.667 | 11369.38 / 388.83 | 92.37 | 5.47 | 150.200 |
| r4-pp2tp2-toolagent | 300/0 | 3.481 / 7.271 / 7.756 / 9.693 | 47.083 / 66.474 / 73.756 / 99.491 | 72.333 | 27.667 | 17106.02 / 348.87 | 84.60 | 48.93 | 42.715 |
| r5-pp2tp2-conversation | 300/0 | 4.911 / 9.055 / 10.517 / 12.929 | 43.442 / 50.197 / 53.834 / 63.110 | 50.667 | 13.333 | 11488.13 / 392.89 | 90.51 | 5.43 | 117.045 |
| r5-pp2tp2-toolagent | 300/0 | 3.508 / 6.485 / 7.032 / 8.579 | 48.178 / 72.515 / 76.946 / 86.403 | 79.333 | 25.333 | 17374.57 / 354.35 | 85.86 | 48.62 | 34.254 |

## v3→v4开发对比

| 拓扑/负载 | 主SLO v3→v4 | TTFT p50(s) v3→v4 | TPOT p50(ms) v3→v4 | TTFT≤2s v3→v4 |
| --- | --- | --- | --- | --- |
| pp2tp2/toolagent | 67.333%→87.333% | 3.515→3.031 | 36.396→48.685 | 8.667%→17.667% |
| pp2tp2/conversation | 38.333%→51.333% | 5.802→4.928 | 38.532→42.619 | 8.000%→13.667% |
| pp4/toolagent | 32.000%→87.000% | 5.452→3.201 | 33.721→49.961 | 3.333%→18.333% |
| pp4/conversation | 25.667%→57.000% | 6.628→4.454 | 35.645→45.847 | 2.333%→11.333% |

PP2×TP2 TTFT中位数分别下降13.78%/15.06%，主SLO增加20/13个百分点。TPOT中位数分别上升约33.77%/10.61%，这体现prefill与decode的资源权衡；所选目标是用户指定的主SLO，不能说所有指标都改善。它也没有消除所有秒级TTFT排队。这里不与正式trace上的55.5%或TP2×DP2的82.7%直接作性能比较。

逐轮各stage的CUDA累计秒数、prefill/decode lane状态驻留秒数、完整8分区驻留百分比、准入blocked/decisions/wall直方图见 [stage-details.md](logs/p9a/stage-details.md)。每组 `analysis.json` 另有p50/p90 TTFT对应分解、分量独立分布、READY/receive slot诊断、GPU均值/峰值显存和完整排空窗口；[comparison.json](logs/p9a/comparison.json)为机器可读总表。

## 正确性与回归

| 项目 | 结果 |
| --- | --- |
| pp2tp2-serial | 20/20请求、1280/1280 token相同 |
| pp2tp2-cold | 20/20请求、1280/1280 token相同 |
| pp4-serial | 20/20请求、1280/1280 token相同 |
| pp4-cold | 20/20请求、1280/1280 token相同 |
| pp2tp2-cancel | PASS；资源守恒=True；取消16/16 |
| pp2tp2-stress | PASS；资源守恒=True；完成200/200，失败0，未完成0 |
| pp4-cancel | PASS；资源守恒=True；取消16/16 |
| pp4-stress | PASS；资源守恒=True；完成200/200，失败0，未完成0 |
| pdmux-tp1 | PASS |
| pdmux-tp2 | PASS |
| plain-pp2 | PASS |
| PP2×DP2启动/路由 | PASS |

最终测试 `logs/p9a/v4-unit.log`：52 passed；`bench-unit.log`：10 passed。正确性比较使用同一BF16/FlashInfer、greedy、20条100–8000-token输入、每条64输出；serial保留缓存，cold每请求前flush并断言cached_tokens=0。PP2×TP2参照普通TP2，PP4参照普通TP1。取消覆盖16条不同长度请求，验证取消后生成与flush仍可用。压测仅用Tool&Agent开发前200条、time-scale=0.02、warmup2、force-close、drain300。

资源守恒审计要求每rank最后idle时free_kv+evictable=capacity、free_reqs=128、stage0接收槽0/其他stage2，并检查scheduler/CUDA错误和TP控制序列相同。原pdmux TP1/TP2及普通PP2各做7条基础/并发请求；原pdmux TP2沿用P8a回归的mem=0.76。额外PP2×DP2用v4启动两副本及cache_aware router，检查每副本直连和并发路由，不用任何trace。

## v4开关与启动命令

最终提交：`f6833f24ef0587278ae799696cfe17c3ade99048`。配置：[p9a_selected.json](configs/p9a_selected.json)、[pdmux_pp_4090_v4.yml](configs/pdmux_pp_4090_v4.yml)。模型Llama3.1-8B-Instruct，BF16，mem=0.71，max_running=128，context=33792，FlashInfer，MLP chunk4096，graph bs1/4/8/16/32，ALP、TPOT系数0.75、decode reserve、balance_merge、rebalance、FCFS。PP2×DP2只验证启动/功能，本任务没有对其做性能调参。

| 参数 | v4默认 | 用途 |
| --- | --- | --- |
| --[no-]pdmux-pp-latency-partition | on | 在decode安全集合内优先prefill |
| --[no-]pdmux-pp-runnable-partition | off | 本级队首状态决定prefill SM需求 |
| --[no-]pdmux-pp-decode-aware-slices | off | 上游decode到达预测缩短slice |
| --[no-]pdmux-pp-adaptive-inflight | off | 工作量约束的软/硬在途上限 |
| --[no-]pdmux-pp-cpu-merge-tokens | off | 同一MERGE点使用CPU token处理输出 |
| --[no-]pdmux-pp-lazy-admission-budget | off | 只在可能准入时求批预算 |
| --pdmux-pp-request-trace PREFIX | 未设置 | 每stage TP0请求事件JSONL |

生产式启动入口（不自动跑任何trace；Ctrl-C/TERM会清理子进程）：

```bash
source /root/workspace/muxpp_work/env.sh
cd /root/workspace/muxpp_work
python scripts/p9a_serve.py launch pp2tp2
# 或：python scripts/p9a_serve.py launch pp4
# 或：python scripts/p9a_serve.py launch pp2dp2
```

完整展开命令如下。PP2×DP2两worker和router要并行运行；推荐上面的lifecycle入口自动管理并清理。

### pp2tp2

```bash
source /root/workspace/muxpp_work/env.sh
env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/root/workspace/sglang/python /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --tp 2 --pp-size 2 --dtype bfloat16 --port 30000 --disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --enable-pdmux --chunked-prefill-size -1 --pdmux-config-path /root/workspace/muxpp_work/configs/pdmux_pp_4090_v4.yml --mem-fraction-static 0.71 --cuda-graph-max-bs 32 --cuda-graph-bs 1 4 8 16 32 --max-prefill-tokens 16384 --pdmux-pp-max-budget 16384 --pdmux-pp-max-inflight-prefill 3 --pdmux-pp-radix-eviction lru --pdmux-pp-policy alp --pdmux-pp-slo-ttft 5.0 --pdmux-pp-slo-tpot 0.1 --pdmux-pp-tpot-coeff 0.75 --pdmux-pp-reserve-decode --pdmux-pp-balance-merge --pdmux-pp-rebalance --pdmux-pp-alp-admission --pdmux-pp-rebalance-unit 8 --pdmux-pp-summary-interval 10 --pdmux-prefill-mlp-chunk-tokens 4096 --disable-custom-all-reduce --attention-backend flashinfer --pdmux-pp-latency-partition --no-pdmux-pp-runnable-partition --no-pdmux-pp-decode-aware-slices --no-pdmux-pp-adaptive-inflight --no-pdmux-pp-cpu-merge-tokens --no-pdmux-pp-lazy-admission-budget
```

### pp4

```bash
source /root/workspace/muxpp_work/env.sh
env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/root/workspace/sglang/python /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --tp 1 --pp-size 4 --dtype bfloat16 --port 30000 --disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --enable-pdmux --chunked-prefill-size -1 --pdmux-config-path /root/workspace/muxpp_work/configs/pdmux_pp_4090_v4.yml --mem-fraction-static 0.71 --cuda-graph-max-bs 32 --cuda-graph-bs 1 4 8 16 32 --max-prefill-tokens 16384 --pdmux-pp-max-budget 16384 --pdmux-pp-max-inflight-prefill 6 --pdmux-pp-radix-eviction lru --pdmux-pp-policy alp --pdmux-pp-slo-ttft 5.0 --pdmux-pp-slo-tpot 0.1 --pdmux-pp-tpot-coeff 0.75 --pdmux-pp-reserve-decode --pdmux-pp-balance-merge --pdmux-pp-rebalance --pdmux-pp-alp-admission --pdmux-pp-rebalance-unit 8 --pdmux-pp-summary-interval 10 --pdmux-prefill-mlp-chunk-tokens 4096 --attention-backend flashinfer --pdmux-pp-latency-partition --no-pdmux-pp-runnable-partition --no-pdmux-pp-decode-aware-slices --no-pdmux-pp-adaptive-inflight --no-pdmux-pp-cpu-merge-tokens --no-pdmux-pp-lazy-admission-budget
```

### pp2dp2

```bash
source /root/workspace/muxpp_work/env.sh
env CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=/root/workspace/sglang/python /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --tp 1 --pp-size 2 --dtype bfloat16 --port 31000 --disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --enable-pdmux --chunked-prefill-size -1 --pdmux-config-path /root/workspace/muxpp_work/configs/pdmux_pp_4090_v4.yml --mem-fraction-static 0.71 --cuda-graph-max-bs 32 --cuda-graph-bs 1 4 8 16 32 --max-prefill-tokens 16384 --pdmux-pp-max-budget 16384 --pdmux-pp-max-inflight-prefill 3 --pdmux-pp-radix-eviction lru --pdmux-pp-policy alp --pdmux-pp-slo-ttft 5.0 --pdmux-pp-slo-tpot 0.1 --pdmux-pp-tpot-coeff 0.75 --pdmux-pp-reserve-decode --pdmux-pp-balance-merge --pdmux-pp-rebalance --pdmux-pp-alp-admission --pdmux-pp-rebalance-unit 8 --pdmux-pp-summary-interval 10 --pdmux-prefill-mlp-chunk-tokens 4096 --attention-backend flashinfer --pdmux-pp-latency-partition --no-pdmux-pp-runnable-partition --no-pdmux-pp-decode-aware-slices --no-pdmux-pp-adaptive-inflight --no-pdmux-pp-cpu-merge-tokens --no-pdmux-pp-lazy-admission-budget
env CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=/root/workspace/sglang/python /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --tp 1 --pp-size 2 --dtype bfloat16 --port 31001 --disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --enable-pdmux --chunked-prefill-size -1 --pdmux-config-path /root/workspace/muxpp_work/configs/pdmux_pp_4090_v4.yml --mem-fraction-static 0.71 --cuda-graph-max-bs 32 --cuda-graph-bs 1 4 8 16 32 --max-prefill-tokens 16384 --pdmux-pp-max-budget 16384 --pdmux-pp-max-inflight-prefill 3 --pdmux-pp-radix-eviction lru --pdmux-pp-policy alp --pdmux-pp-slo-ttft 5.0 --pdmux-pp-slo-tpot 0.1 --pdmux-pp-tpot-coeff 0.75 --pdmux-pp-reserve-decode --pdmux-pp-balance-merge --pdmux-pp-rebalance --pdmux-pp-alp-admission --pdmux-pp-rebalance-unit 8 --pdmux-pp-summary-interval 10 --pdmux-prefill-mlp-chunk-tokens 4096 --attention-backend flashinfer --pdmux-pp-latency-partition --no-pdmux-pp-runnable-partition --no-pdmux-pp-decode-aware-slices --no-pdmux-pp-adaptive-inflight --no-pdmux-pp-cpu-merge-tokens --no-pdmux-pp-lazy-admission-budget
env /root/venvs/sgl/bin/python -m sglang_router.launch_router --host 127.0.0.1 --port 30000 --policy cache_aware --request-timeout-secs 3600 --max-concurrent-requests 4096 --log-level info --worker-urls http://127.0.0.1:31000 http://127.0.0.1:31001
```


开发复现：

```bash
source /root/workspace/muxpp_work/env.sh
cd /root/workspace/muxpp_work
P9A_SOURCE=/root/workspace/sglang-p9a-v4 python scripts/p9a_runner.py   --topology pp2tp2 --trace toolagent --tag unique-v4-toolagent
# --trace conversation；--topology pp4。tag必须唯一，不覆盖证据。
python scripts/p9a_analyze.py logs/p9a/unique-v4-toolagent
```

runner只有两个开发trace白名单，没有formal模式。冻结worktree保留 `sglang-p9a-baseline`、`sglang-p9a-r1`…`sglang-p9a-r5`、`sglang-p9a-v4`。完整命令、源码hash/patch、server/client/GPU日志、请求数据和清理日志保留在每个run目录。最终回归命令位于 `logs/p9a/regressions/queue-commands.json`，启动与测试脚本都在 `scripts/p9a_*`。

## 证据边界与清理

首个R0 Tool run完成300条和全部资源自检后，driver打印旧字段名触发 `KeyError('latencies')`。保留原FAIL状态；summary、逐请求事件和GPU清理完整，审计将workload单独判为通过。之后修复为`latencies_sec`。该首run启动时插桩尚未提交，启动期间只发生格式化/提交，因此meta记录d8dfbfaba且后取的patch为空；保留 `baseline-instrumentation.patch` 重建d8→ab477插桩。其余基线和所有后续轮次使用冻结worktree；后续runner已改为启动前保存hash/patch，避免这个取证竞态。不能把首run描述成启动前已经冻结在ab477。

正式矩阵、P8a选择配置、原正式trace和全部16个正式启动标记的hash审计见 [audit.json](logs/p9a/audit.json)。未新增正式启动标记，未push。

最终清理已完成，GPU计算进程/相关服务器/客户端/监控进程及监听端口均为空；见 [final-cleanup.json](logs/p9a/final-cleanup.json)。
