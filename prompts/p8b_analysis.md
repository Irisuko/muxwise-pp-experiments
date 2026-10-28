# 任务 P8b：分析 MuxWise TP2×DP2 为什么表现最好（只读日志，不跑实验）

## 约束
- 只读已有日志与结果，不使用 GPU，不启动服务，不修改 /root/workspace/sglang 源码（另一个任务 P8a 正在那里开发并使用 GPU）。可以用 base 或 /root/venvs/sgl 的 python 做分析与画图。
- 材料：/root/workspace/muxpp_work/RESULTS.md、results/（matrix_summary.csv、各 label 的 *_summary.json 与 *_requests.jsonl）、logs/p7b/、logs/p7a/<label>/p7b-formal/（各配置正式运行的服务端日志、gpu.csv、router 日志）、P7A_NOTES.md（各配置命令、mem fraction、KV 容量），以及 muxwise_r2（MuxWise DP4）、muxpp（MuxWise-PP）、chunked 各配置的正式日志。

## 要回答（用数据说话，不确定的地方明确标为推断）
1. 容量：每实例 KV 容量（DP4 32k vs TP2×DP2 161k vs TP4 476k）与运行中实际 token usage、#queue-req、retract 次数随时间的变化；KV 是否是 DP4 的主瓶颈。
2. prefill 与 decode 的复用：pdmux 日志中 SM 分区（stream_idx / "Adjusting stream groups"）的分布与切换频率；decode batch size 分布；prefill split 批次大小与耗时；prefill 和 decode 实际重叠程度。
3. 延迟分解：TTFT = 排队 + prefill 计算的估计（结合服务端 prefill 批次日志与客户端时间戳）；TP2 相对 TP1 的单请求 prefill 加速比估计；TPOT 稳定性（为何 TPOT p50 20ms、p99 66ms 而 Chunked PP2×DP2 p99 达 2.6s）。
4. TP 通信代价：TP2 vs TP4 的对比（TP4 主 SLO 仅 2.5%）—— 从 GPU 利用率、decode 步时间、prefill 吞吐推断 allreduce（无 P2P，走 NCCL SHM）的开销；为什么 TP4 崩塌而 TP2 不。
5. 负载均衡：router cache_aware 下两个实例的请求/token 分布、cache 命中率。
6. 与 MuxWise-PP（PP2×DP2、PP4）、Chunked（TP2×DP2、PP2×DP2）的机制对比：一句话总结每个系统在这个负载上的瓶颈。
7. 对 MuxWise-PP 的启示：如果加上 TP（PP2×TP2，P8a 正在实现），预期会在哪些指标上接近/超过 MuxWise TP2×DP2，哪些不会（给出理由，作为可被 P8a 实验检验的预测）。

## 产出
- /root/workspace/muxpp_work/ANALYSIS_MUXWISE_TP2DP2.md（含表格与图，图放 results/analysis_*.png）。
- 在 RESULTS.md 末尾追加一个简短小节“MuxWise TP2×DP2 机制分析”，概括要点并链接到分析文档（只追加，不改动已有内容）。
- 最后一条消息：要点摘要（5–8 条）。
