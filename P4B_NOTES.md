# P4b：MuxWise-PP v2 开发验证

状态：已完成开发验证与本地提交；没有运行正式 trace。所有性能调参仅使用 `bench/dev_toolagent_1200_240_x2.5.jsonl`。

## 隔离与基线

- P4a 完成前只做 CPU 工作，不提交。`logs/p4b/gpu-gate.jsonl` 每 120 秒记录 P4a 完成标记与 GPU 计算 PID；每次启动另检查标记存在、GPU 为空。
- 精确 v1 基线为独立 worktree `/root/workspace/sglang-p4b-v1` 的 `bd849d628`，原 `pdmux_pp_4090.yml` 和 P3b 参数。独立剖析 worktree `/root/workspace/sglang-p4b-profile` 只叠加 `logs/p4b/instrumentation.patch`，不改变策略或执行顺序。
- 共享工作树只修改 PP 相关文件；不 stash/reset 其他任务的内容，不 push。

## P3b 日志复核（只读，不重放）

按原客户端 `run_started_unix` 至排空结束截取，64 条 summary：`required=inf` 56/64；budget=16384 为 60/64，另有 12800 一次、15360 三次；inflight=4 为 40/64，3 为 11/64、2 为 7/64、1 为 2/64、0 为 4/64。因此“预算始终在上限、inflight 始终为4”并非整个时间窗内严格成立。

各 stage 分区1（96/32）出现次数为 51、57、56、53/64；GPU 平均 utilization 为 71.68%、68.09%、67.48%、73.34%。证据：`logs/p4b/p3-audit.json`。

需要区分两类损失：小 decode batch 对预留 SM 的利用不足，和整卡无 kernel 的时间段。`nvidia-smi --help-query-gpu` 将 utilization.gpu 定义为采样周期内至少一个 kernel 执行的时间比例，不是 SM occupancy。因此不能直接用 32/128 SM 解释 70% utilization；必须结合 lane 剖析。帮助文本保存于 `logs/p4b/nvidia-smi-query-semantics.txt`。

旧 `choose_partition` 在 required=inf 且吞吐预测有限时已取 TPOT 安全集合中 prefill SM 最大者；新增分区的关键是消除 decode 小 SM 分区的无样本外推，并显式处理 inf 吞吐预测边界。

## 开发 trace

生成命令：

```bash
source /root/workspace/muxpp_work/env.sh
python /root/workspace/muxpp_work/bench/build_trace.py \
  --start-sec 1200 --duration-sec 240 --stretch 2.5 --max-total-len 32768 \
  --out /root/workspace/muxpp_work/bench/dev_toolagent_1200_240_x2.5.jsonl
```

| 窗口 | 有效请求 | 过滤请求 | 输入 token | 输入 token/s（600s） | 理论前缀复用率 |
|---|---:|---:|---:|---:|---:|
| 正式 600s（仅引用已有统计） | 1551 | 52 | 11,245,637 | 18,742.73 | 52.54% |
| 开发 1200s | 1620 | 49 | 11,769,826 | 19,616.38 | 53.14% |

输入负载高 4.66%，不更换 1800s 起点。理论复用率是 build_trace 的 hash 首次出现后可重用 token 估计，不等于实测 radix 命中。性能对比统一前300条、time-scale=1、warmup=2、force-close；压力回归200条/time-scale=0.02也只用开发 trace。

## 计时口径

每个 stage summary 增加累计时间。prefill 的 wall 状态互斥：compute_slot、upstream、ready、send、switch、empty；`cuda.prefill` 是各 slice CUDA event 区间之和，可用于估计计算槽内的 CPU 发射/轮询间隙。decode wall 区分 control、control_recv、proxy、token、send、retire、launch、prefill_launch、switch、empty；CUDA decode 计算单独累计。双 lane 能重叠，CUDA 时间与 CPU wall 类别不可直接相加。运行期只 query 已完成 event，不为统计添加 synchronize。partition 分布同时有按墙钟加权的驻留时间；原 summary 瞬时分区计数保留供比较。

## 剖析结果与优先级

独立 `v1-profile` 开发运行的到达阶段（按10秒summary边界差分，约101秒）如下。百分比以各 stage 的同一墙钟窗口为分母：

| stage | prefill CUDA | compute_slot−CUDA | 等上游hidden | 等READY | 发送完成 | 队列空 | 切分区同步 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 47.02% | 5.58% | 0 | 27.40% | 2.44% | 17.57% | 0.00073% |
| 1 | 44.42% | 5.89% | 15.15% | 21.94% | 2.75% | 9.86% | 0.00060% |
| 2 | 44.54% | 6.23% | 26.25% | 13.62% | 2.26% | 7.10% | 0.00054% |
| 3 | 40.06% | 6.87% | 45.97% | 0 | 0 | 7.11% | 0.00032% |

首级在等下游READY、末级在等上游hidden，实际发送只占2–3%，分区同步可忽略。优先处理流水线背压与供给气泡，其次是每层slice的发射/轮询间隙；单纯增加prefill SM没有解决这些等待。`compute_slot−CUDA`是槽内总间隙估计，不是独立CPU profiler的精确CPU执行时间。decode CUDA分别占38.16%、40.43%、38.57%、47.97%，会与prefill重叠；其proxy/token/control/send墙钟分项见各运行的analysis.json，不可与CUDA占比直接相加。开发负载下具体占比不能当作原正式运行70%利用率的精确分解，但机制与正式日志的排队/在途饱和一致。

末级norm/logits只占窗口0.116%，sampling占0.0017%，约为prefill CUDA的0.3%；同批末片与中间片CUDA时间比中位数1.033。因此末级head不是60% MAPE的主要解释。原RLS存在冷启动失败：先把首个长prefill时间放入截距，再将所有残差截断为预测值的±40%；线性预测一旦负值并截到1µs，就只能作0.4µs修正，协方差却继续收缩。使用同一批每stage624条CUDA样本离线重放，原算法末128样本MAPE为99.61%、82.76%、98.20%、20.66%；仅取消前16样本残差截断并从零拟合后为1.91%、1.70%、1.77%、2.12%。这是可复现的预测失效机制；正式日志没有逐片样本，无法断言其stage3每个误差均由该机制产生。不同启动可在不同stage失败，不能把v1与v1-profile差异全归因于计时开销。证据：`logs/p4b/fit-audit.log`和`v1-profile/prefill-timing.*.jsonl`。

## 实现

1. 分区为128/0、120/8、112/16、104/24、96/32、80/48、64/64、0/128。混合分区满足sm_89每份≥4 SM、步长2。`required=inf`明确从有限Td且满足TPOT约束的候选中取prefill SM最多者，吞吐预测inf也不能误进“足够吞吐”分支。
2. `pp_profile_decode`启动时直接校准各分区已有graph：bs={1,4,8,16,32}，平均上下文={2048,4096,8192,16384,33792}，均匀及最大偏斜两种形状，每形状预热1次、计时3次。每stage实测约15–17秒，55秒超限检查。临时映射/输入在finally恢复，无请求KV所有权变化。没有用小SM的RLS外推作安全门；采用实测表上格点单调包络、至少1.2倍余量、1ms采样余量，在线观测可提高余量；逐micro-batch取最慢者，越界返回inf；当前分区预测已不安全时绕过驻留/迟滞等待，立即请求切到安全分区（实际切换仍服从在途event守卫）。若所有混合分区都不满足预测门限，则回退到decode SM最多的64/64；这不是端到端TPOT硬保证，通信、CPU以及安全切流等待仍会造成违约。prefill模型使用真实预热/请求在线拟合，在模型未就绪时也按decode安全门选最大prefill分区，故无需用不可信的synthetic prefill外推初始化。
3. RLS前16样本以及负预测恢复时使用普通未截断残差；后续保持抗离群截断。prefill目标改成整批CUDA时间/本地层数，SM交互特征按slice层数加权，增加请求数特征；覆盖首级embed和末级norm/logits/sampling。保留旧slice模型及其整批误差作诊断。R1末级新模型MAPE约6.1%，旧slice约75.8%、旧整批约89.9%，但新旧窗口样本数不同，严格配对证据以上述离线重放为准。
4. `pp_prefetch`启用两份固定hidden/residual接收槽和独立receive stream，提前接收下一批，同时当前批计算。READY仍在缓冲区准备好后发送；R2暴露decode D2H等待与scheduler处理READY之间的循环等待后，R3改由READY通信线程直接提交事先登记的发送，不依赖下一次decode/control轮次；保留launched/event守卫，不同步未匹配NCCL。因为residual可能沿层栈别名输入，槽在末级计算或中间级发送完成后才释放；取消仍沿原有有序协议排空。空闲自检加入两个槽全部归还。预分配避免运行时cudaMalloc与未匹配NCCL互相等待。
5. slice token-layer budget从8192试到65536；graph档位从1/2/4/8/16/24/32减到1/4/8/16/32。R1每分区新增reserved显存106MiB；减少档位后为86MiB，8分区共省160MiB。三个新增分区在旧7档下额外318MiB/stage，在5档下额外258MiB/stage。每个非首级两个接收槽总1056MiB，最多多保留一份528MiB输入；因此后续mem fraction改0.63。显存峰值仍需看实际31k与压力测试，不能只凭静态估计。
6. 原pdmux(pp=1)和普通PP不启用PP计时、校准、预接收；配置新项默认false。没有修改数学计算或KV/请求的生命周期协议。

## 开发运行

所有性能运行使用开发trace前300条，计划到达跨度104.9975s，输入2,113,735/output43,109 token；time-scale=1、force-close。每次独立启动，先验证31,663 token随机单请求，再清cache。表中吞吐包含最终排空；GPU为同一客户端窗口5秒采样均值。单次开发运行不提供统计显著性结论。

详细配置、命令、代码diff/hash、原始请求、summary、GPU采样、分区时间分布保存在`logs/p4b/<tag>/`。v1精确基线无新计时，分区只能用10秒summary瞬时计数；其他PP运行同时有墙钟加权驻留比例。

### 四轮配置与选择

| 轮次 | 分区/graph档数 | slice budget | max-prefill / max-budget | inflight | mem fraction | TPOT系数 | 变更与结果 |
|---|---|---:|---:|---:|---:|---:|---|
| v1 | 5 / 7 | 8192 | 16384 / 16384 | 4 | 0.65 | 1.0 | bd849d628精确基线 |
| R1 | 8 / 7 | 8192 | 16384 / 16384 | 4 | 0.65 | 1.0 | 校准+RLS修正；完成，整体退化 |
| R2 | 8 / 5 | 65536 | 16384 / 16384 | 4 | 0.63 | 0.75 | 双接收槽；74完成后停滞，失败 |
| R3 | 8 / 5 | 65536 | 32768 / 32768 | 6 | 0.63 | 0.75 | READY线程直接发送；300完成，32k未OOM |
| R4（v2候选） | 8 / 5 | 65536 | 16384 / 16384 | 6 | 0.63 | 0.75 | 与R3同代码，只减budget；300完成 |
| static对照 | 与R4一致 | 65536 | 16384 / 16384 | 6 | 0.63 | 0.75 | 仅policy=static；其fallback准入固定16384 |

这是有顺序的开发迭代，不是控制所有变量的因子实验；R1→R3同时改变通信、slice和容量参数，不能把总差值单独归功于某个参数。R3/R4只改变budget上限，可直接观察该方向。超过budget的单个请求仍允许作为独立批次进入，因此16k预算不意味着只能处理16k请求；所有性能运行均额外验证31,663 token随机单请求，开发前300条也含31,573 token输入。

服务/GPU原始日志采用宿主机UTC+8时间，客户端summary另存UTC时间和Unix时间戳；分析按Unix时间对齐。R2的失败有完整保留：本地10:38:48开始不再完成，stage1 faulthandler堆栈停在`get_model_worker_batch -> seq_lens.cpu()`；已提交receive等待对应send，而send仍要等被阻塞的scheduler消费READY。10:40:21主动终止自有stage1以取堆栈，226个未完成请求因终止失败，不能用其延迟作为性能比较。修复是把已经登记的发送由READY通信线程直接提交，保持READY缓冲区安全条件，而不依赖decode轮次。详见`v2-r2/diagnosis.json`、`pp4.log`；直接py-spy attach被环境权限拒绝，未绕过权限。R2清理时首次GPU查询短暂仍见一个退出中的PID，随后确认GPU为空才启动R3。

R3实际最大多请求批次为8请求/29,879新token，最大单请求为31,663输入token（其中31,662 uncached）；没有声称测到恰好32,768新token的多请求批次。R3的32k预算峰值24,076MiB，R4为24,022MiB（nvidia-smi每5秒采样，可能漏瞬时峰值）。R4选为后续回归候选，因联合SLO和尾延迟优于R3且预算更小；不是因为它整体胜过v1。R4 TTFT p50比v1高约4.3%，TPOT p50高约21.5%，输出吞吐低约5.4%；联合SLO仅增加2个百分点，单次运行不足以认为有稳定收益。当前不建议把v2当作已证实的性能升级，也不建议直接据此开启正式比较。

R4到达阶段的首级READY等待约11.63%（v1剖析27.40%），stage1/2/3的槽内非CUDA间隙约1.92/2.96/2.61%（此前5.89/6.23/6.87%），说明通信与slice改动处理了部分空闲。可是首级空队列增至41.27%，末级等上游仍45.73%；整体GPU时间利用率没有提升。小SM decode变慢、请求KV驻留和有序decode/control进度会反过来限制prefill供给，不能将全部TTFT归咎于prefill SM不足。该因果链是结合计时和吞吐的推断，不是额外硬件occupancy实验。

### 结果和分区/计时明细

GPU均值为四卡客户端完整窗口的均值，含排空；延迟为成功请求分位，联合SLO分母为全部300请求。下面的分区驻留及lane表使用完全落在客户端窗口内的summary边界（`timing_strict*`），排除可能混入的31k预热；窗口各自约90–101秒，精确Unix边界在analysis.json。前面的v1剖析表使用最邻近边界，其起点比客户端早0.073秒，约101秒。原`timing`近邻边界结果仍保留以便复核。

分区顺序：v1为128/0、96/32、80/48、64/64、0/128；v2为128/0、120/8、112/16、104/24、96/32、80/48、64/64、0/128。R2的分区和计时只覆盖失败前片段，不能与完整运行比较。decode墙钟send/token等类别与CUDA区间存在重叠，不能相加得到占用率。

| 运行 | 完成/失败 | TTFT p50/p90/p95/p99 (s) | TPOT p50/p90/p95/p99 (ms) | 联合SLO | 输入/输出 tok/s | GPU均值 | 峰值MiB |
|---|---:|---|---|---:|---:|---:|---:|
| v1 | 300/0 | 8.61/14.94/18.44/19.62 | 64.23/93.06/106.54/133.91 | 15.67% | 14418.8/294.1 | 60.61% | 23976 |
| v1-profile | 300/0 | 10.41/17.01/18.14/22.28 | 67.03/93.78/105.34/159.70 | 5.67% | 14138.1/288.3 | 60.90% | 24046 |
| chunked | 300/0 | 1.07/8.71/12.45/17.38 | 33.75/210.23/254.02/283.91 | 53.00% | 16767.7/342.0 | 85.84% | 22401 |
| v2-r1 | 300/0 | 9.53/22.66/24.74/28.08 | 88.12/124.50/140.44/207.90 | 7.67% | 13353.1/272.3 | 60.94% | 24064 |
| v2-r2（失败；延迟不作比较） | 74/226 | — | — | 8.33% | — | — | 24014 |
| v2-r3 | 300/0 | 8.76/20.52/23.85/28.42 | 78.48/105.43/142.28/208.20 | 11.67% | 13550.4/276.4 | 58.62% | 24076 |
| v2-r4 | 300/0 | 8.98/20.27/23.19/27.36 | 78.05/103.21/142.55/182.17 | 17.67% | 13645.3/278.3 | 57.94% | 24022 |
| v2-static | 300/0 | 9.82/25.96/29.54/33.64 | 86.19/130.95/164.95/207.67 | 15.33% | 13053.3/266.2 | 52.41% | 24058 |

| 运行 | stage | 分区驻留百分比（按配置分区顺序） |
|---|---:|---|
| v1（summary采样频率） | 0 | 6.7/46.7/6.7/26.7/13.3 |
| v1（summary采样频率） | 1 | 6.7/53.3/0.0/26.7/13.3 |
| v1（summary采样频率） | 2 | 6.7/33.3/0.0/46.7/13.3 |
| v1（summary采样频率） | 3 | 6.7/46.7/6.7/26.7/13.3 |
| v1-profile | 0 | 0.0/37.9/5.5/33.8/22.8 |
| v1-profile | 1 | 0.0/52.1/1.0/32.7/14.3 |
| v1-profile | 2 | 0.0/44.3/4.7/41.1/9.9 |
| v1-profile | 3 | 0.0/69.5/1.5/19.5/9.5 |
| v2-r1 | 0 | 5.1/3.1/45.3/0.3/1.1/1.4/7.1/36.6 |
| v2-r1 | 1 | 6.3/4.2/52.9/4.2/2.1/0.5/7.8/22.1 |
| v2-r1 | 2 | 5.2/7.7/59.5/0.0/1.5/1.1/9.1/16.0 |
| v2-r1 | 3 | 5.0/1.5/55.4/14.9/1.5/0.7/8.4/12.6 |
| v2-r2 | 0 | 18.0/5.1/16.0/0.0/6.4/3.9/13.8/36.8 |
| v2-r2 | 1 | 22.7/5.1/23.9/0.0/2.5/8.8/23.7/13.2 |
| v2-r2 | 2 | 18.5/4.8/33.2/0.0/3.9/3.6/33.0/3.0 |
| v2-r2 | 3 | 19.4/0.0/27.4/17.2/6.1/8.4/21.3/0.1 |
| v2-r3 | 0 | 5.8/0.4/25.9/0.3/6.2/0.7/10.1/50.5 |
| v2-r3 | 1 | 6.8/0.9/3.8/35.1/7.7/3.2/11.1/31.6 |
| v2-r3 | 2 | 5.7/1.2/45.4/0.2/11.5/0.5/12.5/22.9 |
| v2-r3 | 3 | 6.0/0.0/8.0/43.5/0.6/10.9/14.5/16.4 |
| v2-r4 | 0 | 5.1/1.2/1.6/24.2/8.5/3.1/8.8/47.5 |
| v2-r4 | 1 | 6.1/1.1/28.0/6.2/1.3/13.5/13.2/30.6 |
| v2-r4 | 2 | 5.1/1.9/36.3/0.4/13.9/2.5/17.6/22.3 |
| v2-r4 | 3 | 5.4/0.0/6.2/36.3/1.8/9.3/24.8/16.2 |
| v2-static | 0 | 4.5/6.3/23.9/13.7/0.0/0.0/0.0/51.7 |
| v2-static | 1 | 5.6/7.7/34.9/16.5/0.0/0.0/0.0/35.3 |
| v2-static | 2 | 4.4/9.7/40.9/18.8/0.1/0.0/0.0/26.1 |
| v2-static | 3 | 4.7/11.4/44.4/20.4/0.1/0.0/0.0/19.0 |

| 运行 | stage | prefill CUDA | 槽内间隙 | upstream | READY | send | empty | decode CUDA | decode proxy/token/send/control_recv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v1-profile | 0 | 49.99% | 5.84% | 0.00% | 29.85% | 2.64% | 11.68% | 42.30% | 0.00% / 41.67% / 43.20% / 0.00% |
| v1-profile | 1 | 47.38% | 6.39% | 16.35% | 23.83% | 2.96% | 3.09% | 44.72% | 2.18% / 56.41% / 28.50% / 2.09% |
| v1-profile | 2 | 47.91% | 6.44% | 28.08% | 15.11% | 2.45% | 0.00% | 42.78% | 2.25% / 69.34% / 14.36% / 2.56% |
| v1-profile | 3 | 43.24% | 7.31% | 49.46% | 0.00% | 0.00% | 0.00% | 53.17% | 2.10% / 22.62% / 61.78% / 2.89% |
| v2-r1 | 0 | 35.30% | 7.46% | 0.00% | 24.93% | 2.95% | 29.35% | 41.87% | 0.00% / 44.31% / 41.33% / 0.00% |
| v2-r1 | 1 | 34.65% | 6.42% | 16.86% | 20.79% | 2.54% | 18.75% | 45.19% | 1.46% / 55.05% / 26.36% / 7.25% |
| v2-r1 | 2 | 32.70% | 5.94% | 34.69% | 13.06% | 2.94% | 10.67% | 47.42% | 1.65% / 68.20% / 13.56% / 6.57% |
| v2-r1 | 3 | 33.54% | 6.67% | 52.40% | 0.00% | 0.00% | 7.39% | 52.68% | 1.45% / 23.71% / 56.62% / 8.92% |
| v2-r2 | 0 | 40.29% | 3.15% | 0.00% | 2.35% | 3.63% | 50.57% | 30.00% | 0.00% / 32.85% / 47.15% / 0.00% |
| v2-r2 | 1 | 40.24% | 2.08% | 20.52% | 2.75% | 2.28% | 32.13% | 31.93% | 1.62% / 41.22% / 31.58% / 14.24% |
| v2-r2 | 2 | 38.40% | 2.65% | 35.80% | 3.18% | 2.84% | 17.14% | 33.33% | 1.94% / 53.49% / 19.41% / 13.06% |
| v2-r2 | 3 | 33.31% | 2.68% | 49.76% | 0.00% | 0.00% | 14.25% | 39.86% | 1.50% / 19.73% / 48.34% / 19.56% |
| v2-r3 | 0 | 37.25% | 4.06% | 0.00% | 12.70% | 3.23% | 42.77% | 37.67% | 0.00% / 39.20% / 44.84% / 0.00% |
| v2-r3 | 1 | 39.41% | 2.57% | 19.99% | 12.73% | 2.17% | 23.12% | 37.80% | 1.81% / 50.04% / 29.11% / 7.02% |
| v2-r3 | 2 | 36.20% | 2.81% | 34.70% | 9.02% | 2.31% | 14.97% | 43.41% | 1.70% / 64.60% / 14.87% / 6.59% |
| v2-r3 | 3 | 38.13% | 2.84% | 49.56% | 0.00% | 0.00% | 9.47% | 46.67% | 1.72% / 24.12% / 54.20% / 9.24% |
| v2-r4 | 0 | 39.86% | 4.46% | 0.00% | 11.63% | 2.77% | 41.27% | 36.03% | 0.00% / 36.41% / 46.73% / 0.00% |
| v2-r4 | 1 | 40.61% | 1.92% | 16.66% | 14.78% | 2.71% | 23.31% | 41.10% | 1.80% / 49.77% / 29.19% / 7.31% |
| v2-r4 | 2 | 39.36% | 2.96% | 31.94% | 8.66% | 2.15% | 14.93% | 42.61% | 1.91% / 64.41% / 15.31% / 6.63% |
| v2-r4 | 3 | 42.11% | 2.61% | 45.73% | 0.00% | 0.00% | 9.55% | 45.79% | 1.65% / 24.21% / 53.88% / 8.95% |
| v2-static | 0 | 34.92% | 5.26% | 0.00% | 9.75% | 3.23% | 46.84% | 37.19% | 0.00% / 45.33% / 41.80% / 0.00% |
| v2-static | 1 | 34.15% | 3.40% | 18.21% | 8.91% | 3.02% | 32.31% | 41.57% | 1.54% / 54.99% / 27.41% / 6.37% |
| v2-static | 2 | 33.30% | 3.11% | 32.55% | 5.96% | 3.16% | 21.91% | 42.97% | 1.68% / 67.50% / 13.87% / 5.90% |
| v2-static | 3 | 34.13% | 2.99% | 48.08% | 0.00% | 0.00% | 14.79% | 54.92% | 1.35% / 19.16% / 61.66% / 7.81% |


### ALP 对照结论

同R4配置、同开发请求，仅改policy=static：static TTFT p50=9.82s、TPOT p50=86.19ms、联合SLO=15.33%、输出266.2tok/s；ALP分别为8.98s、78.05ms、17.67%、278.3tok/s。此次ALP相对static改善TTFT中位数约8.6%、TPOT中位数约9.4%、输出吞吐约4.5%、联合SLO约2.33个百分点；因此保留ALP作为v2候选策略。该结论仅限本次开发对照，未用正式trace验证，也不能掩盖v2相对v1的整体退化。

## 最终候选完整启动命令（未执行正式实验）

后续若进行正式验证，v2候选可使用以下在R4及回归中验证的服务参数。当前性能证据不足以推荐替换v1；这是一条可复现的v2候选命令，不是上线/正式实验通过结论。脚本存为`scripts/p4b_v2_serve.sh`，本任务没有调用该脚本，也没有重放正式trace。

```bash
#!/usr/bin/env bash
# Reviewed v2 candidate only; this script was NOT executed for a formal experiment.
set -euo pipefail
source /root/workspace/muxpp_work/env.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m sglang.launch_server \
  --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct \
  --tp 1 --pp-size 4 --dtype bfloat16 --port 30000 \
  --disable-overlap-schedule --context-length 33792 \
  --schedule-policy fcfs --max-running-requests 128 \
  --watchdog-timeout 180 --log-level debug --enable-cache-report \
  --enable-pdmux --chunked-prefill-size -1 \
  --pdmux-config-path /root/workspace/muxpp_work/configs/pdmux_pp_4090_v2.yml \
  --mem-fraction-static 0.63 --cuda-graph-max-bs 32 --cuda-graph-bs 1 4 8 16 32 \
  --max-prefill-tokens 16384 --pdmux-pp-max-budget 16384 \
  --pdmux-pp-max-inflight-prefill 6 --pdmux-pp-radix-eviction lru \
  --pdmux-pp-policy alp --pdmux-pp-slo-ttft 5.0 --pdmux-pp-slo-tpot 0.1 \
  --pdmux-pp-tpot-coeff 0.75 --pdmux-pp-reserve-decode \
  --pdmux-pp-balance-merge --pdmux-pp-rebalance --pdmux-pp-alp-admission \
  --pdmux-pp-rebalance-unit 8 --pdmux-pp-summary-interval 10
```

## 回归与交付

| 项目 | 结果 | 证据 |
|---|---|---|
| CPU策略/协议/计时/缓冲区测试 | 29 passed；black、isort、git diff --check通过 | `logs/p4b/unit-final.log`、`black-final.log`、`isort-final.log` |
| P1串行正确性 | 20请求×64输出token，temperature=0；与本次TP1及P1保存的TP1输出全部相同，0 mismatch | `tp1-correctness-result.json`、`v2-correctness-result.json`、`regression-audit.json` |
| 取消 | 16请求全部abort（含约27k长输入）；取消后生成正常，unsupported=400，flush=200 | `v2-control-result.json`及server.log |
| 快速压力 | 开发trace前200条、time-scale=0.02、force-close；200完成/0失败，100.960s，无OOM | `v2-stress_summary.json`、`v2-stress-result.json` |
| 原pdmux(pp=1) | 原配置，1条短请求及6条并发请求，7/7 PASS | `original-pdmux-result.json` |
| 原普通PP | pp=2、未启用pdmux，1条短请求及6条并发请求，7/7 PASS | `original-pp-result.json` |

正确性、取消和压力末尾四个stage都通过KV/请求/树缓存自检。压力后每stage `free_kv=4909 + evictable=294619 = capacity=299528`，`free_reqs=128`；stage1/2/3两个接收槽均归还，stage0不配置接收槽。取消后的最后一次idle记录也是free+evictable=299528、free_reqs=128、接收槽全归还；缓存保留为可驱逐状态不算泄漏。压力期间1秒GPU采样峰值分别为23732、23830、24026、23688MiB。

回归统一使用R4候选参数；代码运行时7个关键文件hash与R4/static快照完全一致。`scripts/p4b_regressions.sh`为串行驱动，`p4b_check_regressions.py`验证token一致性、池计数、错误日志和所有性能summary的trace路径；结构化结果为`logs/p4b/regression-audit.json`。原模式回归是冒烟覆盖，不声称覆盖整个上游SGLang测试套件。

## 提交、清理与复现入口

- 本地代码commit：`ebdf39aa002f8733fa20a0b6ebf946dcb4c483d7`（`Profile PP lane waits and calibrate prefill-heavy SM partitions`），位于`muxwise-pp`；未push。
- 仅显式git add本任务12个文件；P4a三次提交保持在父历史中。P4a完成标记为0，首次GPU使用前门禁确认无计算PID；没有在P4a结束前提交。
- 配置：`/root/workspace/sglang/configs/pdmux_pp_4090_v2.yml`与`/root/workspace/muxpp_work/configs/pdmux_pp_4090_v2.yml`一致。精确v1和只加计时的worktree保留，便于重现；原始日志未覆盖。
- 2026-10-20 03:00:30 UTC确认GPU计算PID为空、SGLang/router服务进程为空、30000/31000–31003端口无监听。证据：`logs/p4b/cleanup-final.json`、`nvidia-smi-final.txt`。主工作树提交后干净。
- 开发驱动`p4b_runner.py`只允许开发trace路径；每次记录启动命令、配置、代码diff/hash、请求明细、summary、GPU.csv。`p4b_analyze.py`和`p4b_tables.py`可重建分析与表格；正式trace仅做了已有统计/日志的只读对照，没有用它测试或调参。

本任务停止在四轮开发迭代和指定回归。下一步的性能问题仍是decode/控制推进和KV驻留引发的prefill供给空洞；本次没有证实v2整体胜过v1，更没有证实能追平chunked。保留此负结果，避免把局部计时改善或较低MAPE误写成正式性能提升。

