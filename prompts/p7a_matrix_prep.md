# 任务 P7a：并行配置矩阵的准备工作（工具、清理、启动脚本、预检；不跑正式）

## 背景
- 读 /root/workspace/muxpp_work/RESULTS.md、results/final_summary.md、scripts/run_experiment.sh、scripts/p3_runner.py、bench/bench_client.py、bench/summarize.py、scripts/p5b_v3_serve.sh。环境 source /root/workspace/muxpp_work/env.sh；代码 /root/workspace/sglang（muxwise-pp，f91e8a18d），不要改推理/调度源码（若发现某配置因 bug 不能运行，记录为“不支持”并说明，不要修改源码）。
- 用户新要求：
  (1) 新版本若全面或基本全面优于老版本，可不保留老版本信息。大脑已核对：muxwise_r2 优于 muxwise（r1），sglang_pd_r2 优于 sglang_pd（r1，1523 失败），muxwise_pp_v3 全面优于 muxwise_pp（v1）。
  (2) 各系统按 PP×TP×DP=4 补全配置矩阵；每次实验的**排空时间为 5 分钟**：最后一个请求发出后最多再等 300 秒，到时停止客户端，不必等所有请求跑完。

## 1. 客户端与统计口径
- bench_client.py 新增 `--drain-timeout SEC`：最后一个计划请求发出后最多等待 SEC 秒；到时取消所有未完成请求（关闭连接），这些请求记为 unfinished（不同于 failed），已收到的首 token 时间和已生成 token 仍记录。summary 新增字段：completed / unfinished / failed / cutoff_time。
- 统一统计口径（summarize 与 summary 都实现，并在 README/RESULTS 写明）：
  - SLO 达成率分母为全部计划请求，unfinished/failed 计为未达标；
  - TTFT 与 E2E 分位数按全部计划请求计算，到截止时刻仍无首 token（或未完成）的请求取 +∞，分位数落在 +∞ 时显示为 “>cutoff”；
  - TPOT 分位数按截止前完成的请求计算，并标注样本数；
  - 吞吐 = 截止前交付的输出（及输入）token / min(实际时长, 截止时刻)。
- 对 4 个已有最终正式运行（chunked、muxwise_r2、sglang_pd_r2、muxwise_pp_v3）用同一规则做**事后截断重算**：截止时刻 = 最后一个请求的发送时刻 + 300s；finished 晚于截止的视为 unfinished，首 token 晚于截止的 TTFT 取 +∞。生成 `<label>_cut300_summary.json`，不修改原始文件。

## 2. 清理老版本
- RESULTS.md、results/summary.*、图表中去掉 muxwise（r1）、sglang_pd（r1）、muxwise_pp（v1）及 MuxWise-PP v2（只在开发 trace 上跑过）的结果与描述。对应的原始结果与日志移到 /root/workspace/muxpp_work/archive/superseded/（移动，不删除；报告中不再引用）。P*_NOTES.md 作为实现文档保留，但其中与报告冲突的“当前结论”无需维护。
- 已有的四个最终运行在矩阵中对应：chunked→Chunked DP4；muxwise_r2→MuxWise DP4；sglang_pd_r2→PD 2P2D(TP1)；muxwise_pp_v3→MuxWise-PP PP4。

## 3. 新配置（共 11 个；label 命名如下）
统一：Llama3.1-8B bf16、--disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128、--enable-cache-report、FlashInfer + CUDA graph（除非该配置必须例外，需记录原因）、多实例用 sglang-router cache_aware（PD 用 mini_lb PD 模式，同 r2）。GPU 分配尽量让同一实例的卡在同一 NUMA（先用 nvidia-smi topo -m 与 CUDA UUID 核对实际 NUMA 映射，P5a 发现 GPU0 在 NUMA0、GPU1/2/3 在 NUMA1）。4090 之间无 P2P：TP 的 allreduce 走 NCCL SHM/host，这是硬件事实，照常测。
- Chunked（--chunked-prefill-size 2048 --enable-mixed-chunk；若 PP 下 mixed chunk 不支持则关闭并记录）：
  `chunked_tp2dp2`、`chunked_pp2dp2`、`chunked_tp4`、`chunked_pp4`、`chunked_tp2pp2`
- MuxWise 原版（--enable-pdmux --chunked-prefill-size -1 --pdmux-config-path configs/pdmux_4090.yml；PP 不适用；MLP 分块仅支持 TP1，TP2/TP4 下不开启，若 OOM 则按需降低 mem fraction 或改 Triton/关 graph 并记录）：
  `muxwise_tp2dp2`、`muxwise_tp4`
- SGLang-PD（Mooncake TCP，沿用 sglang_pd_r2 的全部传输环境变量与参数、隔离 worktree /root/workspace/sglang-p4a 的冻结提交）：
  `pd_1p1d_tp2`（P、D 各 TP2）、`pd_3p1d`（TP1）、`pd_1p3d`（TP1）
- MuxWise-PP v3（参数同 scripts/p5b_v3_serve.sh；MuxWise-PP 只支持 TP1，故 TP 组合记为不支持）：
  `muxpp_pp2dp2`：两个 PP2 实例 + router cache_aware；`--pdmux-pp-max-inflight-prefill` 按 PP4 的比例缩放为 3（6/4×2），其余参数不变。
- 每个配置写 launch 脚本，并接入 run_experiment.sh（含正式锁，防止重复运行）。

## 4. 预检（每个新配置）
- 选 mem fraction：只用启动 + 31,663-token 长预热 + ≤1 分钟冒烟（--limit 20 --time-scale 0.2）选择能稳定通过的较高值（Chunked 参考 0.88，其余参考已有同类配置），方法与 P3b SELECTION 一致，记录过程。
- muxpp_pp2dp2 额外在**开发 trace**前 300 条（time-scale=1，--warmup 2 --force-close --drain-timeout 300）跑一次健全性检查，只记录、不调参。
- 预检不得使用正式 trace。
- 结果写 /root/workspace/muxpp_work/P7A_NOTES.md：每个配置的最终启动命令、mem fraction、每实例 KV 容量、预检结论（通过/不支持及原因）。

## 产出
- 工具改动的单测（mock server 验证 drain-timeout 与统计口径）；P7A_NOTES.md；更新后的 RESULTS.md（只含最终版本，并用截断口径重算的四个已有结果）。
- 不运行任何正式实验。结束确认 GPU/进程清理。最后一条消息：11 个配置的预检状态表、截断重算后四个已有结果的主指标。
