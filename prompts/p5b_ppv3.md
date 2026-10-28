# 任务 P5b：MuxWise-PP v3 —— 解决准入阻塞与 decode 环开销（只做开发验证）

## 必读
- /root/workspace/muxpp_work/P4B_NOTES.md（v2 剖析与开发结果；v2 未胜过 v1）、P2_NOTES.md、P1_NOTES.md、RESULTS.md。
- 代码：/root/workspace/sglang，muxwise-pp 分支（HEAD ebdf39aa0），继续 commit，不 push。环境 source /root/workspace/muxpp_work/env.sh。
- GPU 协调：另一个任务 P5a（PD 传输评估）在使用 GPU。只有当 /root/workspace/muxpp_work/logs/p5a.gpu_done 或 logs/p5a.rc 存在、且 `nvidia-smi --query-compute-apps=pid --format=csv,noheader` 为空时才能用 GPU；之前只做 CPU 侧分析/编码/单测，每 2 分钟检查一次。P5a 只改 /root/workspace/sglang-p4a，不会动你的工作树。

## 大脑的诊断（请先用数据验证）
开发 trace v2-r4 的 summary（logs/p4b/v2-r4/pp4.log）显示：负载期间 decode micro-batch 的 Σctx 合计长期在 216k–288k，而每 stage max_total_num_tokens=299,528；同时 waiting 队列非空，但 stage0 prefill lane 有 41% 时间是“队列空”（无批可算）。推断链：PP decode 慢（TPOT p50 64–78ms，chunked 33ms）→ 请求驻留 KV 时间约为 chunked 的 2 倍 → KV 接近占满 → stage0 准入被阻塞 → prefill 供给不足、GPU 利用率仅约 58–60% → TTFT 排队。decode lane 计时：decode CUDA 只占墙钟约 36–46%，其余大量时间在等 token / send（阻塞式 p2p 环）。注意 Σctx 按请求累加，radix 共享前缀会被重复计数，因此 “KV 占满” 必须用 allocator 实际占用验证。

## 要做
1. **准入阻塞归因**（先做）：在 stage0 记录每次“waiting 非空但未准入”的原因直方图（inflight 上限 / KV 不足（可用+可驱逐 < 需求）/ 预算 / max_running_requests / decode 预留 / 其它），以及 allocator 实际已用、可驱逐、空闲 token 随时间的序列，汇入 summary。用开发 trace 跑一次（可与第 2 步的基线合并），给出结论。
2. **异步 decode 环**：当前 decode lane 沿用上游 0.5.3 的阻塞式 send/recv 环（metadata 走 gloo + NCCL 张量，逐 micro-batch 串行握手），并有每步的 gloo 控制包。参考更新版 SGLang 的 PP 实现（可 `pip download sglang==0.5.5 --no-deps` 等较新版本到 /tmp 只读参考 scheduler_pp_mixin.py 中的异步 send / P2PWork 处理）与 FastPP 的 isend，改为：非阻塞 isend（保留 work 句柄，延迟到必要时 wait）、提前 post 的 irecv、控制包与 proxy 元数据合并或预取、去掉不必要的每步 D2H 同步与 CPU 往返。保持 P1/P2 的一致性协议（stage0 决策、控制步对齐）不变。目标：同等 batch 下 decode 每轮（TPOT）明显下降（目标 ≥30%）。
3. **扩大 KV 容量**（若第 1 步确认 KV 是主要阻塞）：mem fraction 0.63/0.65 是为 prefill 激活峰值（16k–32k token 批次的 MLP 中间张量、logits）与接收槽预留的。实现 split prefill 层内按 token 分块执行 MLP（以及必要时 attention 之外的大张量），使激活峰值与批 token 数解耦，从而提高 mem fraction（目标每 stage KV ≥ 360k token）；同时评估把 in-flight prefill 批占用的 KV 计入准入、in-flight 数/预算的取值。该分块实现应同样可用于 pp=1 的原 MuxWise（为以后改善 MuxWise 基线做准备，默认关闭、参数开启），数学结果必须一致。
4. 根据第 1 步结论处理其它主要阻塞源。

## 开发验证规则
- 只用开发 trace /root/workspace/muxpp_work/bench/dev_toolagent_1200_240_x2.5.jsonl（前 300 条，time-scale=1，--warmup 2 --force-close）；禁止用正式 trace。v1 与 chunked 在开发 trace 上的结果已在 P4B_NOTES 中（v1：TTFT p50 8.61s、联合 SLO 15.67%；chunked：1.07s、53.00%），无需重跑；以 v2-r4 配置为本轮起点基线（若第 1 步需要新插桩，基线可用插桩版本重跑一次）。
- 迭代不超过 4 轮，每轮记录配置、TTFT/TPOT 分位、联合 SLO（TTFT≤5s&TPOT≤100ms）、吞吐、GPU 利用率、准入阻塞直方图、KV 占用、decode 环分解。
- 回归：串行 20 条正确性与 TP1 一致；取消测试；200 条 time-scale=0.02 压测无泄漏；原 pdmux(pp=1) 与原 PP 回归；若实现了层内分块，pp=1 开启分块的正确性也要测。

## 产出
- 代码 commit；/root/workspace/muxpp_work/P5B_NOTES.md：归因结论（含数据）、改动、每轮结果表、与 v1/v2/chunked 的开发对比、最终推荐的 v3 正式启动命令（若 v3 在开发 trace 上优于 v1 明显，例如联合 SLO 或 TTFT p50 有实质改善）。
- 不要运行正式实验。结束确认 GPU/进程清理。最后一条消息：简明总结。
