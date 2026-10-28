# 任务 P9b：正式实验 —— Conversation 负载三系统对比 + Tool&Agent 上 MuxWise-PP v4（每个只跑一次）

## 必读
- /root/workspace/muxpp_work/P9A_NOTES.md（v4 启动命令，HEAD f6833f24e）、P7A_NOTES.md（MuxWise/Chunked 各配置的已选命令与 mem fraction）、RESULTS.md、bench/README.md、scripts/run_experiment.sh。环境 source /root/workspace/muxpp_work/env.sh。不得修改源码与已选参数。
- 所有系统均使用当前代码 HEAD f6833f24e 运行（MuxWise/Chunked 路径不受 MuxWise-PP 改动影响，P9a 已做回归；若 run_experiment 的冻结提交校验要求旧提交，请更新冻结信息并在报告中说明）。
- 统一：客户端 `--warmup 2 --force-close --drain-timeout 300`；31,663-token 长预热、≤1 分钟冒烟（使用对应负载的开发 trace，--limit 20 --time-scale 0.2）、flush cache、router 重启、日志、5s GPU 采样、正式锁、清理均同 P7b。每个 label 只跑一次；只有完全无法产出数据（启动失败/崩溃）时可在修复启动层面问题后再启动一次并注明。记录本次 GPU UUID/NUMA 映射。

## A. Conversation 负载（正式 trace：bench/conversation_600_240_x2.5.jsonl，735 条）
负载选择理由（写进报告）：Mooncake Conversation，取窗方式与 Tool&Agent 相同（600s 起 4 分钟、到达拉长 2.5 倍、过滤总长 >32k），未缓存 prefill 负载与 Tool&Agent 几乎相同（约 8.95k tok/s），但平均输出 338 token（Tool&Agent 169）、平均输入 9.1k，KV 驻留更重，更能体现单实例大 KV 容量（PP）与 decode 隔离（MuxWise）；前缀复用低（约 20%），对单一 radix cache 优势不利，需如实说明。
依次运行（label）：
1. `conv_muxwise_tp2dp2`、2. `conv_muxwise_tp4`（命令同 P7A 中 muxwise_tp2dp2 / muxwise_tp4）
3. `conv_chunked_pp2dp2`、4. `conv_chunked_tp2dp2`（同 P7A 中 chunked_pp2dp2 / chunked_tp2dp2）
5. `conv_muxpp_v4_pp2tp2`、6. `conv_muxpp_v4_pp4`（P9A_NOTES 中 v4 的 pp2tp2 / pp4 命令）

## B. Tool&Agent 负载（正式 trace：bench/toolagent_600_240_x2.5.jsonl）
7. `muxpp_v4_pp2tp2`、8. `muxpp_v4_pp4`、9. `muxpp_v4_pp2dp2`（P9A_NOTES 中 v4 三种命令）

## 报告（更新 RESULTS.md）
- Tool&Agent 矩阵：加入 v4 三行。按用户规则判断：若 v4 某配置相对同配置 v3（muxpp_pp2tp2 / muxwise_pp_v3 / muxpp_pp2dp2）全面或基本全面更优（主 SLO、TTFT 分位、吞吐等主要指标更好，个别次要指标略差可视为基本全面，例如 TPOT 中位数上升但仍远低于 SLO），则在报告中用 v4 替换该行，v3 原始数据移到 archive/superseded/；否则两版并列并说明取舍。重新生成矩阵与最佳配置表/图。
- 新增 “Conversation 负载对比” 章节：负载统计与选择理由、6 个配置的完整表格（指标同矩阵）、各系统最佳配置对比、CDF/SLO 图（results/conv_*.png、results/conv_summary.md/csv），以及与 Tool&Agent 结论的异同分析（客观，单次测量无置信区间）。
- 新增 “MuxWise-PP v4 调度优化” 小节：概述 P9a 的 TTFT 分解结论与有效改动（链接 P9A_NOTES.md），包括用户建议的“上游 decode 感知”已实现但收益不稳定、默认关闭的事实。
- 结束确认 GPU/进程清理。最后一条消息：Conversation 6 个配置结果表 + 三系统最佳对比、Tool&Agent v4 三行结果及替换决定。
