# 任务 P12b：新拓扑上补跑 MuxWise / Chunked 最佳配置的标准速率基线（6 个 label，每个只跑一次）

## 背景
- 必读：RESULTS.md、P7A_NOTES.md（8B 各配置已选命令）、P10B_NOTES.md（24B 已选命令）、bench/README.md、scripts/run_experiment.sh。环境 source env.sh；大模型 /dev/shm/models/Mistral-Small-24B-Instruct-2501（缺失时 scripts/p10a_model.py ensure）。
- 原因：容器换卡后 4 张 4090 均在 NUMA0（互为 NODE），而 MuxWise/Chunked 的标准速率基线均在旧拓扑上运行；MuxWise-PP v5 与饱和对比已在新拓扑上。用户同意在新拓扑上补跑各系统最佳配置的标准速率基线，使三系统的 SLO 对比在同一硬件上。这不是重新选配置。
- 不得修改源码与已选参数（MuxWise/Chunked 路径使用当前 HEAD；如 P12a 已提交新代码，确认其未触及 MuxWise pp=1 / Chunked 路径，否则用 P11b 使用的 1dcbb285e 独立 worktree 运行并说明）。
- 开始前确认 P12a 已结束（/root/workspace/muxpp_work/logs/p12a.rc 存在）且 4 卡显存约 1MiB、利用率 0%、容器内无残留进程；每个 label 前重新确认。发现容器外占用立即停止并在最后报告。记录 GPU UUID/NUMA。
- 先做一次卡间通信微基准并存档（logs/p12b/topology-bench/）：nvidia-smi topo -m、各卡对 H2D/D2H/D2D（经主机）带宽、NCCL allreduce（TP2 各对、TP4）在 1MB/16MB/64MB 的延迟与带宽（torch.distributed 脚本即可，与推理一致关闭 custom allreduce），及 NCCL 实际选择的 transport。不超过 10 分钟。

## 运行（客户端 --warmup 2 --force-close --drain-timeout 300，time-scale=1；长预热、冷烟（对应开发 trace）、flush、router 重启、日志、5s GPU 采样、正式锁、清理同 P7b；每个 label 只跑一次，只有完全无法产出数据时可修复启动层面问题后再启动一次并注明）
1. `nt_muxwise_tp2dp2`、`nt_chunked_pp2dp2`：Tool&Agent 8B（bench/toolagent_600_240_x2.5.jsonl），命令同 muxwise_tp2dp2 / chunked_pp2dp2。
2. `nt_conv_muxwise_tp2dp2`、`nt_conv_chunked_pp2dp2`：Conversation 8B（bench/conversation_600_240_x2.5.jsonl），命令同 conv_muxwise_tp2dp2 / conv_chunked_pp2dp2。
3. `nt_big_muxwise_tp4`、`nt_big_chunked_pp2tp2`：24B（bench/big_toolagent_600_80_x7.5.jsonl），命令同 big_muxwise_tp4 / big_chunked_pp2tp2（HEAD 上的 Chunked KV 预算修复保持）。

## 报告（更新 RESULTS.md）
- 新增“新拓扑上的三系统最终对比”章节并置于最前：三个负载（Tool&Agent 8B、Conversation 8B、24B）各一张表，列出 MuxWise、Chunked 的新拓扑标准速率结果与 MuxWise-PP 在新拓扑上的最佳版本（v5 及若 P12a 有正式结果的版本），指标同矩阵（完成/未完成/失败、TTFT/TPOT/E2E 分位、吞吐、主 SLO、TTFT≤2s、宽松 SLO、cache），并附新拓扑饱和吞吐对比（引用 P11b）。生成 results/final_newtopo_*.md/csv/png（CDF、SLO 柱状图）。
- 旧拓扑的同配置结果保留在矩阵章节中，标注“旧拓扑，用于选择并行配置”；给出新旧拓扑同配置的差异表与简要分析（结合通信微基准）。
- 客观分析，单次测量无置信区间。
- 结束确认 GPU/进程清理。最后一条消息：微基准要点、三负载新拓扑三系统对比表、新旧拓扑差异、关键结论。
