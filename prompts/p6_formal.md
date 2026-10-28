# 任务 P6：正式运行 muxwise_pp_v3 与 muxwise_r2（每个版本只跑一次），并整理最终报告

## 必读
- /root/workspace/muxpp_work/RESULTS.md（已有 P3b/P4a/P5a 结果与正式运行机制）、P5B_NOTES.md、scripts/run_experiment.sh、scripts/p3_runner.py、scripts/p5b_v3_serve.sh。
- 环境：source /root/workspace/muxpp_work/env.sh；代码 /root/workspace/sglang，muxwise-pp 分支 HEAD f91e8a18d（不要再修改推理/调度源码；若正式前发现阻断性 bug，停下并在报告中说明，不要自行改代码后直接跑正式）。
- 用户规则：每个版本正式实验只跑一次；代码修复/改进后的新版本可以再正式跑一次（新 label，旧结果保留）。

## 1. muxwise_pp_v3（先做）
- 服务参数严格使用 scripts/p5b_v3_serve.sh 中的命令（P5b 开发 R3 配置），label `muxwise_pp_v3`，正式 trace bench/toolagent_600_240_x2.5.jsonl，客户端 --warmup 2 --force-close，其余流程（31,663-token 长预热、≤1 分钟冒烟、flush cache、日志、5s GPU 采样、清理、正式锁防重复）与 P3b 一致。只跑一次。

## 2. muxwise_r2（MuxWise 基线改进版）
- 背景：P3b 的 muxwise 因单卡整请求 prefill 的 MLP 激活 OOM，被迫使用 Triton attention、关闭 CUDA graph、mem fraction 0.826（每卡 KV 约 32k）。P5b 新增的 `--pdmux-prefill-mlp-chunk-tokens`（pp=1 同样支持，冷 cache 正确性已验证）可以降低激活峰值。
- 配置：4 个 TP1 实例 + sglang-router cache_aware（与 P3b muxwise 相同），`--enable-pdmux --chunked-prefill-size -1 --pdmux-config-path configs/pdmux_4090.yml --pdmux-prefill-mlp-chunk-tokens 4096`，attention 恢复默认 FlashInfer，恢复 CUDA graph（--cuda-graph-max-bs 32，与 chunked 一致；若显存不够可减少 bs 档位，记录），其余统一参数同 P3b。
- 选择 mem fraction：只用启动 + 31,663-token 长预热 + ≤1 分钟冒烟做选择（与 P3b SELECTION 方法一致），取能稳定通过的较高值（参考范围 0.80–0.88），记录过程；若 FlashInfer + graph 在任何合理 mem fraction 下仍 OOM，则保留 Triton/no-graph 但用 MLP 分块提高 mem fraction，写明原因。
- 正式前再用**开发 trace** 前 300 条（time-scale=1，--warmup 2 --force-close）跑一次作为健全性检查并记录（不得据此调参超过 1 轮）。
- 正式运行 label `muxwise_r2`，只跑一次。

## 3. 最终报告
- 更新 RESULTS.md：新增 P6 小节（两次正式运行的精确命令、配置、结果、观察：MuxWise-PP v3 的准入阻塞直方图、allocator 占用、分区驻留、decode 环分解等 summary 数据）。
- 新增“最终对比”章节放在 RESULTS.md 最前面：每个系统采用其最终版本 —— Chunked（chunked）、MuxWise（muxwise_r2，若失败则说明）、SGLang-PD（sglang_pd_r2）、MuxWise-PP（muxwise_pp_v3）；表格含完成/失败、TTFT/TPOT/E2E 分位、吞吐、SLO 达成率（主口径 TTFT≤5s&TPOT≤100ms，并列出 TTFT≤2/5/10s、TPOT≤50/100/200ms）、cache 命中率；另列“版本演进”表（muxwise→r2，sglang_pd→r2，muxwise_pp→v3）以及每个版本的代码/配置差异。客观分析，不夸大；注明每个版本单次测量、无置信区间、各系统的已知混杂因素（如 PD 的 decode KV 预分配限制、MuxWise r1 的 Triton/no-graph 等）。
- summarize.py：生成包含全部 label 的 summary 与一组只含四个最终版本的图（results/final_*.png 与 results/final_summary.md/csv）。
- 结束确认 GPU/进程清理。最后一条消息：最终四系统对比表 + 关键结论。
