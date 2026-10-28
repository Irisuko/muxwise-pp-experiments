# 任务 P4b：MuxWise-PP v2 优化（只做开发验证，不跑正式实验）

## 背景与必读
- /root/workspace/muxpp_work/RESULTS.md（P3b 正式结果）、P2_NOTES.md、P1_NOTES.md、DESIGN.md。
- 正式日志：logs/p3/muxwise_pp/formal/pp4.log（含每 10s 的 "MuxWise-PP summary"）、gpu.csv。
- 环境：source /root/workspace/muxpp_work/env.sh；代码 /root/workspace/sglang，muxwise-pp 分支，继续 commit（不 push）。
- 注意：另一个任务 P4a（PD 修复）可能仍在占用 GPU。开始任何 GPU 测试前检查 `nvidia-smi --query-compute-apps=pid --format=csv,noheader` 为空且 /root/workspace/muxpp_work/logs/p4a.rc 已存在；否则只做 CPU 侧的代码与分析工作，并每 2 分钟检查一次，P4a 结束后再用 GPU。P4a 也可能在同一工作树中修改 PD 相关文件并提交：你在 P4a 结束前不要 git commit；提交时只 `git add` 你自己改动的文件，不要 stash/reset/checkout 别人的改动。

## 大脑的诊断（请先用日志复核，有不同结论就写出来）
P3b 正式运行中 MuxWise-PP：TTFT p50 21.9s（chunked 11.0s），TPOT p50 62ms（SLO 100ms，有余量），GPU 利用率约 70%（chunked 90–98%）。summary 显示：等待队列长期有 15–27 万 token，inflight=4 一直顶满，准入预算一直等于上限 16384，required=inf（56/64 次）。各 stage 几乎都停在分区 1（96 prefill / 32 decode SM）——这是配置中 prefill SM 最多的混合分区；而 decode 每个 micro-batch 只有 5–10 个请求。结论：prefill 算力是瓶颈，但 32 个 decode SM 大多空闲，相当于每张卡约 1/4 算力被浪费；另外可能存在流水线气泡/握手等待导致的空闲。

## 要做
1. **空闲时间剖析**（先做，决定优先级）：给每个 stage 的 prefill lane 加轻量计时统计（汇入现有 summary）：本 stage 计算时间、等待上游 hidden 的时间、等待 READY 握手/发送完成的时间、因分区切换同步的时间、队列为空的时间；decode lane 同理（计算 vs 等待 proxy/token）。用开发 trace 跑一次，给出各项占比，指出 GPU 70% 利用率的主要原因。
2. **分区表扩展**：新增偏向 prefill 的分区，例如 120/8、112/16、104/24（sm_89 约束：每份 ≥4 SM、步长 2），保留 96/32、80/48、64/64 供 decode 重负载用；评估每多一个分区多出的 CUDA graph 显存，必要时减少 graph 的 bs 档位。新建 configs/pdmux_pp_4090_v2.yml。
3. **ALP 策略修正**：确保在 TPOT 约束安全的前提下 required=inf 时能选到 prefill-SM 最大的分区；小 decode SM 分区的 T_d 预测此前没有样本，基于 SM 交互特征的外推不可靠——请加一个启动时的简短 profiling（每个分区跑几个代表性 decode bs 的 graph replay 与一次 prefill slice，总时长 ≤60s），用于初始化 RLS，或者用其他稳妥方法，保证新分区不因预测偏差被误选或从不被选。stage3 的 prefill MAPE 约 60%（其它 stage 约 1–2%），查明原因（末级 logits/sampling 是否计入、特征是否缺项）并修正。
4. **准入与在途**：把 --pdmux-pp-max-budget 上限、--pdmux-pp-max-inflight-prefill 作为可调项评估（例如 budget 上限 32768、inflight 4~6），注意激活显存（0.65 的 mem fraction 是 P1 为 16k uncached prefill 的激活定的；更大 batch 需验证不 OOM，包括 31k 单请求）。
5. 根据第 1 步剖析的结果修复其它主要空闲来源（如握手串行化、每片 slice 的 CPU 开销、slice 粒度 split_forward_token_budget 过小导致的启动开销等）。

## 开发验证规则（重要：避免在正式 trace 上调参）
- 用 bench/build_trace.py 生成**开发 trace**：--start-sec 1200 --duration-sec 240 --stretch 2.5 --max-total-len 32768，输出 bench/dev_toolagent_1200_240_x2.5.jsonl（报告其请求数、输入 token 速率、前缀复用率，与正式 trace 对比；若负载强度差异很大（>30%）可改用 1800s 起点，说明理由）。
- 所有调参只用开发 trace（可以只用前 300 条或全量，time-scale=1，--force-close）。**禁止**使用正式 trace（toolagent_600_240_x2.5.jsonl）做任何调参或测试。
- 先用 P3b 正式版配置（commit bd849d628 + pdmux_pp_4090.yml + 正式启动参数）在开发 trace 上跑一次作为 v1 基线，然后迭代 v2；迭代轮数不超过 4 轮，每轮记录配置与结果（TTFT/TPOT 分位、联合 SLO 达成率 TTFT≤5s&TPOT≤100ms、吞吐、GPU 利用率、分区分布）。另外在 v2 最终配置下跑一次 policy=static 作为对照，记录 ALP 是否有收益。
- 同时参考：同一开发 trace 上 chunked(DP4) 的结果，作为目标参照（按 P3b chunked 的启动命令跑一次即可）。

## 回归
- 正确性：P1 的串行正确性测试（20 条，temperature=0）与 TP1 一致；取消测试；200 条 time-scale=0.02 压测无泄漏；原 pdmux(pp=1) 与原 PP 回归。

## 产出
- 代码 commit；/root/workspace/muxpp_work/P4B_NOTES.md：剖析结论、改动、每轮开发结果表、最终推荐的 v2 正式启动命令（完整）。
- **不要**运行正式实验。结束确认 GPU 与服务进程清理。最后一条消息：简明总结（剖析结论、关键改动、开发 trace 上 v1/v2/chunked 的对比）。
