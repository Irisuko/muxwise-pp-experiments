# 任务 P9a：MuxWise-PP 调度深度优化（迭代开发，不跑正式）

## 背景与必读
- 用户要求：检查 MuxWise-PP 可以改进的地方，修改代码让调度尽可能好；每次修改后测试，再找下一处可优化点。用户特别提示：**每个 stage 决定提交多少层 prefill（slice 层数）或分配 SM 时，可以考虑上游 stage 的 decode 信息**。
- 必读：RESULTS.md（16 行矩阵）、ANALYSIS_MUXWISE_TP2DP2.md、P5B_NOTES.md、P8A_NOTES.md、P1/P2_NOTES.md。代码 /root/workspace/sglang（muxwise-pp，HEAD d8dfbfaba），继续 commit，不 push。环境 source /root/workspace/muxpp_work/env.sh。GPU 全部归你使用。
- 现状（Tool&Agent 正式）：MuxWise-PP 最佳 PP2×TP2 主 SLO 55.5%、TTFT p50 4.74s、TPOT p50 34.8ms；MuxWise TP2×DP2 82.7%、1.79s、20.2ms。MuxWise-PP 的主要短板是 TTFT 中位数（存在约 3–5s 的 TTFT 下限）以及 PP2×TP2 准入受在途上限限制。

## 开发负载（禁止使用任何正式 trace 做测试或调参）
- Tool&Agent 开发 trace：bench/dev_toolagent_1200_240_x2.5.jsonl（已存在）。
- 新建 Conversation trace（下一阶段的新负载）：用 bench/build_trace.py 对 /root/ext/mooncake/FAST25-release/traces/conversation_trace.jsonl 生成
  - 开发：--start-sec 1200 --duration-sec 240 --stretch 2.5 --max-total-len 32768 → bench/dev_conversation_1200_240_x2.5.jsonl
  - 正式（只生成和统计，**本任务不得使用**）：--start-sec 600 --duration-sec 240 --stretch 2.5 --max-total-len 32768 → bench/conversation_600_240_x2.5.jsonl
  记录两者统计（请求数、输入/输出 token 速率、前缀复用率）。
- 每轮评估：PP2×TP2（当前最佳）在两个开发 trace 的前 300 条（time-scale=1，--warmup 2 --force-close --drain-timeout 300）；关键轮次同时验证 PP4。最多 6 轮迭代，每轮记录配置/改动/指标（TTFT/TPOT 分位、主 SLO TTFT≤5s&TPOT≤100ms、TTFT≤2s、吞吐、GPU 利用率、各 stage lane 计时、准入阻塞直方图、分区驻留）。

## 第 0 步：TTFT 分解（先做，决定优先级）
给每个请求记录：到达 → stage0 准入（排队）→ 各 stage prefill 开始/结束（含等待上游 hidden、READY、在途槽）→ 末级首 token → 结果回到 stage0 → MERGE/首 token 发往 tokenizer → 客户端收到。用开发 trace 跑一次，给出 TTFT 中位数/p90 的分解，指出 3–5s 下限的主要来源。

## 候选优化方向（按第 0 步结论排序取舍，也可以提出更好的）
1. **跨 stage 的 decode 感知 prefill 调度（用户提示）**：上游 stage 在发射 decode micro-batch 时，把轻量通知（mb、bs、Σctx、发射时刻、预测耗时）随现有控制/通信通道发给下游；下游据此预测下一个 decode micro-batch 的到达时刻与工作量，用于：
   - 决定 prefill slice 的层数：让 slice 在 decode 到达前完成（层数 = 可用间隙 / 预测的每层时间），避免切换分区时等待长 slice、或 decode 与大 slice 争抢显存带宽；
   - 决定 SM 分区：decode 间隙内用偏 prefill（甚至 128/0）分区，decode 到达前切换到满足 TPOT 的混合分区；分区切换需要两条流排空，因此与 slice 层数联动；
   - 反向信息：下游 prefill 队列/背压与 decode 负载回传上游，上游据此调整 SM 与 slice（例如下游堵塞时上游少给 prefill SM、多给 decode）。
2. **流水线平衡的分区选择**：各 stage 的 prefill 速度受各自分区影响，整体吞吐由最慢 stage 决定；协调各 stage 分区（考虑首/末 stage 额外的 embed/logits 开销、stage0 的 CPU 调度开销）使 prefill stage 时间平衡。
3. **准入/在途自适应**：PP2×TP2 主要准入阻塞是在途上限（固定 3）；改为按流水线占用、批大小与预测完成时间动态决定在途数和批预算（队列短时小批低延迟，积压时大批高吞吐），保持 KV 安全预留不放松。
4. **降低首 token 路径延迟**：例如首 token 在结果回到 stage0 后立即发送给客户端，而不是等到并入 decode micro-batch 的 MERGE 时点（需保证一致性与 radix 插入时机不变）；减少 READY 握手往返等。
5. 其它从剖析中发现的开销（控制环、CPU 调度开销、slice 发射间隙等）。

## 约束
- 所有新机制用参数开关（默认开启最佳组合），保持 TP1/TP2、PP2/PP4 均可运行；保持 stage0 唯一决策与 TP 内一致性协议；不放松 KV 预留与正确性。
- 调度策略仍为 FCFS 准入顺序（与对照系统一致），不要引入按长度排序之类会改变公平性比较基础的策略。
- 回归：PP2×TP2 与 PP4 的 serial20 / 冷 cache20 正确性（参照分别为普通 TP2 与 TP1）、取消、200 条 time-scale=0.02 压测无泄漏、原 pdmux 与原 PP 回归。

## 产出
- 代码 commit；/root/workspace/muxpp_work/P9A_NOTES.md：TTFT 分解结论、每轮改动与结果表（两个开发 trace × PP2×TP2，关键轮 PP4）、最终推荐配置（v4）与完整启动命令（PP2×TP2、PP4、PP2×DP2 三种）。
- 不跑正式实验；结束确认 GPU/进程清理。最后一条消息：简明总结（瓶颈、有效改动、v3→v4 开发结果对比）。
