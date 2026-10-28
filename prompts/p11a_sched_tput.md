# 任务 P11a：继续优化 MuxWise-PP 调度 —— 兼顾 SLO 与吞吐（迭代开发，不跑正式）

## 背景
- 用户要求：继续优化 MuxWise-PP 的调度；除了 SLO 以外，吞吐量也要尽可能大。
- 必读：RESULTS.md（Tool&Agent 矩阵、Conversation、大模型三部分）、P9A_NOTES.md、P10B_NOTES.md。代码 /root/workspace/sglang（muxwise-pp，HEAD 4c10135d2），继续 commit，不 push。环境 source env.sh；大模型 /dev/shm/models/Mistral-Small-24B-Instruct-2501（缺失则 scripts/p10a_model.py ensure）。GPU 全归你（每次用前确认无容器外占用；发现占用立即停止并在最后报告）。
- 现状要点：
  - Llama-8B：MuxWise-PP v4 PP4 主 SLO 72.8%、PP2×TP2 70.1%（Tool&Agent 正式），落后 MuxWise TP2×DP2 82.7%；TTFT 中位数约 3.5s，TTFT≤2s 仅约 14%。
  - Mistral-24B：MuxWise-PP PP2×TP2 主 SLO 17.5%、宽松 SLO 51.0%；PP4 TTFT p50 59s（准入受 inflight 与 KV 约束、末级约 30% 时间在等上游），输出吞吐 128.5 tok/s，明显低于 PP2×TP2 / TP4 的约 160 tok/s。
  - 根本限制：整请求 prefill 不切块，一个长请求（最长 31k）必须逐 stage 顺序通过流水线，单请求 TTFT ≈ PP 级数 × 单级时间；不同批大小不一导致流水线气泡；在途上限按批次计而非按 token。

## 优化方向（大脑建议，按剖析结果取舍，也可提出更好的）
1. **分块流水线 prefill（Chunked Pipeline Prefill，参考 Mooncake 的 CPP 与 FastPP 的 chunk sizing）**：stage0 在准入时把一个 prefill 批按 token 预算切成若干块（长请求切成多段，短请求可打包到同一块），各块依次进入流水线：块 i+1 可以在 stage0 计算的同时，块 i 在 stage1 计算，从而同一长请求在多个 stage 上并行推进，单请求 TTFT 从 “级数×单级(全长)” 降到约 “(块数+级数−1)×单级(块长)”，同时块大小均衡可减少气泡、提高吞吐。
   - 一致性：块计划由 stage0 决定并写入 plan；每个 stage 按 FIFO 处理块，保证块 i 的 KV 在块 i+1 的 attention 之前写入本 stage；KV 在准入时整体分配（或按块分配但必须与 stage0 一致）；radix 插入时机不变（整请求完成后 MERGE）；abort/取消覆盖未完成的块。
   - 块大小：自适应（例如由 RLS 预测单级块耗时，使其接近目标，如 decode 一轮或若干层切片时长的整数倍；或者沿用 FastPP 的 catch-up 吞吐需求选择），并与分区选择、slice 层数联动。
   - MuxWise 原有的按层切片（split prefill）在块内继续使用。
2. **按 token 计的在途控制**：在途上限从“批数”改为“流水线中未完成的 prefill token 数/预测工作量”，配合分块后更细的粒度；保持 KV 安全预留不放松。
3. **decode 吞吐**：在 TPOT 有余量时增大 decode micro-batch（例如合入策略、更积极的准入使 decode bs 变大）、减少每步固定开销；评估 micro-batch 数与 pp 的关系。
4. **流水线平衡**：利用各 stage 的计时（含首/末级额外开销、stage0 的 CPU 调度开销）平衡分区与块大小，降低“等上游/等 READY”占比。
5. 其它剖析中发现的瓶颈。

## 评估方法（只用开发 trace，禁止任何正式 trace）
- 每轮至少评估：
  a. **标准速率 SLO**：Llama-8B PP4 与 PP2×TP2 在 bench/dev_toolagent_1200_240_x2.5.jsonl 前 300 条（time-scale=1）；Mistral-24B PP2×TP2 与 PP4 在 bench/dev_big_toolagent_1200_80_x7.5.jsonl 前 150 条（time-scale=1）。
  b. **吞吐/容量**：同样的请求以 2 倍速率（--time-scale 0.5）跑一次，记录输出/输入 tok/s、SLO 达成率；以及饱和吞吐（--time-scale 0.05，所有请求几乎同时到达），记录完成全部请求的总时长与输出/输入 tok/s。并计算 goodput（满足主 SLO 的请求数/秒）。
  - 客户端统一 `--warmup 2 --force-close --drain-timeout 300`。
- 在第 1 轮之前，用同样的 a/b 方法测一次参照：MuxWise-PP 当前版本（4c10135d2）以及 MuxWise TP2×DP2（8B）、MuxWise TP4（24B）各一次，作为基线；之后每轮只测 MuxWise-PP。
- 最多 6 轮迭代；每轮记录改动、TTFT/TPOT 分位、主/宽松 SLO、标准/2 倍/饱和吞吐、goodput、GPU 利用率、各 stage lane 计时、准入阻塞直方图。最终选择以“主 SLO 与吞吐同时提升或不降”为原则，若存在取舍，明确列出。
- 回归：PP4 / PP2×TP2 在 8B 与 24B 上的 serial20 与冷 cache20 正确性（同拓扑普通实例为参照）；取消；200 条 time-scale=0.02 压测无泄漏；原 pdmux、原 PP、Chunked 路径不受影响。

## 产出
- 代码 commit；/root/workspace/muxpp_work/P11A_NOTES.md：剖析、每轮改动与结果表、基线对照、最终推荐 v5 配置与完整启动命令（8B：PP4、PP2×TP2、PP2×DP2；24B：PP4、PP2×TP2）。
- 不跑正式实验；结束确认 GPU/进程清理。最后一条消息：简明总结（有效改动、v4→v5 在 SLO 与吞吐上的开发对比、与 MuxWise 基线的差距）。
