# 任务 P11b：MuxWise-PP v5 正式实验 + 三系统饱和吞吐对比（每个 label 只跑一次）

## 必读
- /root/workspace/muxpp_work/P11A_NOTES.md（v5 五套启动命令，HEAD 1dcbb285e）、RESULTS.md、bench/README.md、scripts/run_experiment.sh、P7A/P10B_NOTES（MuxWise/Chunked 已选命令）。环境 source env.sh；大模型 /dev/shm/models/Mistral-Small-24B-Instruct-2501（缺失时 scripts/p10a_model.py ensure）。不得修改源码与已选参数。
- 每个运行前确认 4 卡空闲且无容器外占用；发现占用立即停止并在最后报告。
- 统一：客户端 `--warmup 2 --force-close --drain-timeout 300`；长预热、冒烟（对应开发 trace）、flush、router 重启、日志、5s GPU 采样、正式锁、清理同 P7b。每个 label 只跑一次；只有完全无法产出数据时可修复启动层面问题后再启动一次并注明。记录 GPU UUID/NUMA。

## A. v5 标准速率正式运行（替换 v4 行的候选）
- Tool&Agent 8B（bench/toolagent_600_240_x2.5.jsonl）：`muxpp_v5_pp4`、`muxpp_v5_pp2tp2`、`muxpp_v5_pp2dp2`
- Conversation 8B（bench/conversation_600_240_x2.5.jsonl）：`conv_muxpp_v5_pp4`、`conv_muxpp_v5_pp2tp2`
- 大模型 24B（bench/big_toolagent_600_80_x7.5.jsonl）：`big_muxpp_v5_pp2tp2`、`big_muxpp_v5_pp4`

## B. 饱和吞吐正式对比（用户要求吞吐尽可能大）
用同一正式 trace、所有请求几乎同时到达（`--time-scale 0.05`），其余客户端参数相同（drain 300s 后截止）。每个负载取三系统的最佳配置：
- Tool&Agent 8B：`sat_muxwise_tp2dp2`、`sat_chunked_pp2dp2`、`sat_muxpp_v5_<A 中主 SLO 最佳的拓扑>`
- 大模型 24B：`sat_big_muxwise_tp4`、`sat_big_chunked_pp2tp2`、`sat_big_muxpp_v5_<A 中最佳拓扑>`
指标：截止前完成数、完成全部请求的耗时（如在截止前完成）、输入/输出 tok/s、请求吞吐、TTFT/TPOT 分位。

## 报告（更新 RESULTS.md）
- 各负载表中加入 v5 行；按用户规则判断：v5 相对同拓扑 v4 全面或基本全面更优（主 SLO、TTFT 分位、吞吐等主要指标更好，次要指标略差可视为基本全面）则替换 v4 行并把 v4 原始数据移到 archive/superseded/；否则并列说明取舍。重新生成各负载的矩阵/最佳配置表与图。
- 新增“饱和吞吐对比”章节（两负载各三系统：表 + 柱状图 results/sat_*.png）。
- 新增“MuxWise-PP v5 调度优化”小节：分块流水线 prefill、按 token 的在途控制与回补门槛、减少 CPU/GPU 元数据同步等改动与开发结果（链接 P11A_NOTES.md），以及已知取舍。
- 客观分析，单次测量无置信区间，注明不同日期/GPU 分配。
- 结束确认 GPU/进程清理。最后一条消息：A 中 7 个 v5 结果与替换决定、B 中饱和吞吐对比表、关键结论。
