# 任务 P12a：排查并修复 MuxWise-PP v5 PP2×TP2 的正式退步（开发验证，不跑正式）

## 背景
- 必读：RESULTS.md（P11b 部分）、P11A_NOTES.md、P9A_NOTES.md、P8A_NOTES.md、P10B_NOTES.md。代码 /root/workspace/sglang（muxwise-pp，HEAD 1dcbb285e），可 commit 不 push。环境 source env.sh。大模型 /dev/shm/models/Mistral-Small-24B-Instruct-2501（缺失时 scripts/p10a_model.py ensure）。GPU 全归你；每次用前确认无容器外占用，发现占用立即停止并在最后报告。
- 现象：8B PP2×TP2 在 P11a 开发 trace 上 v5 主 SLO 84.3%（v4 73.3%），但 P11b 正式运行（换卡后 4 卡均在 NUMA0，UUID 见 RESULTS）Tool&Agent 34.9%（v4 正式 70.1%）、Conversation 35.4%（v4 65.9%），TTFT p50 6–7s；同期 PP4 与 PP2×DP2 v5 均大幅改善。24B PP2×TP2 v5 仅 19.5%（v4 17.5%），也没有开发上的提升幅度。
- 候选原因（需用数据验证，不要预设）：(1) 新 GPU/拓扑下 TP allreduce 路径（NCCL SHM/P2P 选择、custom allreduce 关闭）变化；(2) v5 分块流水线 prefill 在 TP 下每块额外的 TP 同步（stage 内 MIN allreduce、分区广播、块级状态共识）在正式负载密度下放大；(3) token 在途控制/回补门槛对 PP2（只有 2 级）参数不合适；(4) 开发 300 条与正式全量负载的差异（队列积压后的行为）。

## 要做
1. **同机 A/B 定位**：在当前 GPU 上，用开发 trace（bench/dev_toolagent_1200_240_x2.5.jsonl）分别跑 v4（f6833f24e，独立 worktree）与 v5（1dcbb285e）的 8B PP2×TP2：前 300 条 time-scale=1，以及**全量**（1620 条，time-scale=1，--drain-timeout 300）各一次；同时 v5 PP4 全量一次作对照。客户端统一 --warmup 2 --force-close。开启 P9a 的请求级 TTFT 分解与准入归因、lane 计时、NCCL 日志（确认 TP 通信走 SHM/P2P/哪种 transport）。禁止使用正式 trace。
2. 根据分解结论定位根因（例如 TP 下块级共识等待、某 rank 的等待/空转、在途额度在 PP2 下过紧/过松导致排队、READY 握手或预接收槽争用等），给出证据。
3. 修复：最小改动，保持 FCFS、KV 安全预留、stage0 唯一决策与 TP 一致性协议。修复后重测 8B PP2×TP2（开发前 300 条 + 全量）与 24B PP2×TP2（bench/dev_big_toolagent_1200_80_x7.5.jsonl 前 150 条 + 全量 time-scale=1），并确认 PP4（8B 全量、24B 前 150 条）与 PP2×DP2（8B 前 300 条）不退化。最多 4 轮迭代，每轮记录 TTFT/TPOT 分位、主/宽松 SLO、吞吐、GPU 利用率、分解。
4. 回归：PP2×TP2/PP4 在 8B、24B 的 serial20 与冷 cache20 正确性（同拓扑普通实例为参照）；取消；200 条 time-scale=0.02 压测无泄漏；原 pdmux、原 PP、Chunked 路径不受影响。
5. 若修复有效，写好新版本（v5.1）的启动命令与 run_experiment 接入（新 label：muxpp_v51_pp2tp2、conv_muxpp_v51_pp2tp2、big_muxpp_v51_pp2tp2，以及若 PP4/PP2×DP2 代码路径也改变则相应 label），不跑正式。

## 产出
- 代码 commit；/root/workspace/muxpp_work/P12A_NOTES.md：A/B 结果、根因证据、修复、每轮结果、回归、v5.1 启动命令。
- 不跑正式实验；结束确认 GPU/进程清理。最后一条消息：根因、修复、v4/v5/v5.1 开发对比（8B 与 24B）、是否建议正式重跑。
