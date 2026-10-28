# 任务 P10c：Llama 回归 A/B 澄清 + 大模型 6 配置正式实验（每个只跑一次）

## 必读
- /root/workspace/muxpp_work/P10B_NOTES.md（最终 6 配置命令，HEAD 4c10135d2）、P10A_NOTES.md、RESULTS.md、bench/README.md、scripts/run_experiment.sh。环境 source env.sh；模型 /dev/shm/models/Mistral-Small-24B-Instruct-2501（缺失时 scripts/p10a_model.py ensure）。不得修改源码与已选参数。
- 开始前和每个运行前确认 4 卡空闲（显存约 1MiB、利用率 0%，且容器内无残留进程）；若发现容器外进程占用 GPU，立即停止、记录并在最后一条消息中说明，不要带着干扰继续跑正式实验。

## 1. Llama-8B 回归 A/B（开发 trace，不计入正式结果）
P10B 中 Llama v4 PP2×TP2 开发 300 条为 TTFT p50 3.88s / 主 SLO 243/300，而 P9a 历史为 3.03s / 262/300；两次运行于不同 GPU 物理分配，新增开关均默认关闭。为区分代码退化与环境差异：在同一时段、同一 GPU 分配上，依次用 f6833f24e（P9a v4，用独立 worktree）与 4c10135d2 各跑一次 Llama PP2×TP2 v4（命令同 P9a），开发 Tool&Agent trace 前 300 条（time-scale=1，--warmup 2 --force-close --drain-timeout 300）。只记录，不调参。若 4c10135d2 明显更差（主 SLO 差距 > 5 个百分点且 TTFT p50 更差），在报告中标注并分析差异来源（代码 diff 中哪些在默认路径上生效）。

## 2. 大模型正式实验
- 正式 trace：bench/big_toolagent_600_80_x7.5.jsonl；客户端 `--warmup 2 --force-close --drain-timeout 300`；长预热、冒烟（开发 trace，--limit 20 --time-scale 0.2）、flush、日志、5s GPU 采样、正式锁、清理同 P7b。
- 依次运行（label）：big_muxwise_tp4、big_chunked_tp4、big_chunked_pp4、big_chunked_pp2tp2、big_muxpp_pp4、big_muxpp_pp2tp2。每个只跑一次；只有完全无法产出数据（启动失败/崩溃）时可修复启动层面问题后再启动一次并注明。记录 GPU UUID/NUMA 映射。

## 3. 报告（更新 RESULTS.md）
- 新增“大模型场景（Mistral-Small-24B，必须跨 4 卡）”章节：场景动机（Llama-8B 上 TP2 已足够；24B 只能用 4 卡实例，TP4 的 allreduce 在无 P2P 的 PCIe 上代价高）、模型与存放、负载构造（600s 起 80s 窗口、到达拉长 7.5 倍、过滤 >32k，用 Mistral tokenizer；负载统计）、6 个配置的命令/KV 容量、完整结果表（指标同矩阵：完成/未完成/失败、TTFT/TPOT/E2E 分位、吞吐、主 SLO、TTFT≤2s、TTFT≤10s&TPOT≤100ms、cache 命中）、三系统最佳配置对比、CDF/SLO 图（results/big_*.png、results/big_summary.md/csv）。
- 说明 P10a/P10b 的代码改动：Mistral 放行；Chunked 的上游 PP KV 预算 bug 修复（基线修复，说明根因）；MuxWise-PP 的 PP decode 传输尺寸修复、显存开销降低（KV 118k→181k / 163k→203k）、MLP 尾块补齐（BF16 形状敏感，默认关闭，说明正确性复核方法与结论）；PP4 在大模型上仍落后的事实与归因。
- 附 Llama 回归 A/B 的结果与结论。
- 客观分析，不夸大；单次测量、无置信区间；不同日期与 GPU 分配的说明。
- 结束确认 GPU/进程清理。最后一条消息：Llama A/B 结论、6 个大模型配置结果表、三系统最佳对比与关键结论。
