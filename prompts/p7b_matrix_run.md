# 任务 P7b：正式运行 11 个并行配置（每个只跑一次）并生成完整矩阵报告

## 必读
- /root/workspace/muxpp_work/P7A_NOTES.md（11 个配置的最终启动命令、mem fraction、预检结论）、RESULTS.md、bench/README.md（截断统计口径）、scripts/run_experiment.sh。
- 环境 source /root/workspace/muxpp_work/env.sh；不得修改推理/调度源码与已选定的启动参数。

## 运行
- 按 P7A_NOTES 中的启动命令，依次正式运行：chunked_tp2dp2、chunked_pp2dp2、chunked_tp4、chunked_pp4、chunked_tp2pp2、muxwise_tp2dp2、muxwise_tp4、pd_1p1d_tp2、pd_3p1d、pd_1p3d、muxpp_pp2dp2。
- 正式 trace bench/toolagent_600_240_x2.5.jsonl；客户端 `--warmup 2 --force-close --drain-timeout 300`；31,663-token 长预热、≤1 分钟冒烟、flush cache、router 重启、日志、5s GPU 采样、正式锁、结束清理与之前一致。
- 每个配置只跑一次，不补跑不重跑。只有当某配置根本无法产出数据（启动失败/崩溃）时，可在修复启动层面问题（不改源码与参数选择）后再启动一次，并在报告中注明；否则如实记录。
- 每完成一个配置，在 /root/workspace/muxpp_work/logs/p7b/progress.md 追加一行（label、开始/结束时间、完成/未完成/失败、主 SLO），便于中途检查。

## 报告
- RESULTS.md 改写为最终版（只含最终版本，不含已归档的老版本）：
  1. 实验设置与统计口径（截断规则：最后请求发出后排空 300s；SLO 分母为全部 1551 条；TTFT/E2E 分位含 +∞；TPOT 仅完成请求并标注样本数）。
  2. **完整矩阵表**：行 = 系统 × 并行配置（共 15 个：4 个已有 + 11 个新增；已有四个使用 `_cut300` 重算结果），列 = 完成/未完成/失败、TTFT p50/p90/p99、TPOT p50/p99（样本数）、E2E p50、输入/输出吞吐、主 SLO（TTFT≤5s&TPOT≤100ms）、TTFT≤2s、TTFT≤10s&TPOT≤100ms、cache 命中率、每实例 KV 容量。不支持的组合列出并注明原因（MuxWise 原版无 PP；MuxWise-PP 仅 TP1）。
  3. **各系统最优配置对比**：每个系统按主 SLO 选出最佳并行配置（同分看 TTFT p50），给出四系统最佳配置对比表与图。
  4. 分析：并行方式对各系统的影响（TP 在无 P2P PCIe 上的 allreduce 代价、PP 的流水线延迟与 KV 容量、PD 配比与 decode KV 预分配瓶颈等），以数据为准，不夸大；单次测量、无置信区间的说明。
- summarize.py：生成 results/matrix_summary.md/csv（15 行）、results/best_summary.md/csv（4 行），以及最佳配置的 TTFT/TPOT CDF、SLO 柱状图、矩阵热力图（系统×配置的主 SLO）。
- 结束确认 GPU/进程清理。最后一条消息：完整矩阵的主 SLO/TTFT p50 简表 + 四系统最佳配置对比 + 关键结论。
