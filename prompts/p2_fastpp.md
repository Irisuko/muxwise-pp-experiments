# 任务 P2：把 FastPP 的调度思路有选择地移植进 MuxWise-PP

## 必读
- /root/workspace/muxpp_work/DESIGN.md（§3 是本任务核心）
- /root/workspace/muxpp_work/P1_NOTES.md（P1 的实现说明、接口、控制消息格式、已知问题）
- /root/workspace/muxpp_work/FASTPP_REPORT.md（FastPP 代码分析报告，含公式与 file:line）
- FastPP 源码（只读）：/root/ext/FastPP/python/sglang/srt/managers/{dynamic_chunk_controller.py, alp_scheduler.py, batch_rebalancer.py, schedule_policy.py, scheduler.py}
- 环境：source /root/workspace/muxpp_work/env.sh；代码在 /root/workspace/sglang 的 muxwise-pp 分支，继续在该分支上 commit（不 push）。


## P2-0（最高优先级，先于 FastPP 移植完成）：修复 P1 的两个性能问题
P1 审查结论（见 P1_NOTES.md“设计取舍”2、1）：
1. **Prefill 流水化**：P1 中整条 PP 同时最多一批 prefill 在途，各 stage 的 prefill 是串行接力的，每个 stage 约 (pp-1)/pp 时间 prefill 侧空闲，prefill 吞吐≈单卡。本负载以 prefill 为主（平均 7.2k 输入 token、约 2.6 req/s），这是 MuxWise-PP 最大的瓶颈。要求改为：最多 pp_size 批（参数化 --pdmux-pp-max-inflight-prefill，默认 pp_size）同时在途，每个 stage 同一时刻计算一批、收到的后续批在本 stage 排队（FIFO）；stage0 在本 stage 完成当前批的本地层后即可开始下一批（受 KV/准入预算约束）。保持 P1 的一致性协议（plan 顺序、batch_id、merge 决策都由 stage0 决定），注意：多批在途时 radix 插入/匹配的顺序语义（后一批的 prefix 匹配不能依赖前一批尚未 merge 的 KV，或者明确定义并在所有 stage 一致执行）、abort 在途批、内存预留按所有在途批计算。
2. **恢复 LRU 语义的 radix 驱逐**：P1 改成确定性 FIFO 驱逐。改为 stage0 按原 LRU 选择 victim（节点按路径/token key 标识），把驱逐指令放进控制消息/plan，下游按指令驱逐同一组节点（下游自身不做独立驱逐决策）；确保各 stage 树结构仍同构且命中长度一致。若实现风险过大，可以保留 FIFO 作为回退开关，但默认应为 LRU-by-stage0。
3. 完成 P2-0 后先重跑 P1 的正确性与 200 条压测（--time-scale 0.02）及取消测试，全部通过并 commit，再做下面的 FastPP 移植。另外用正式 trace 前 300 条、time-scale=1（真实到达速率，约 2 分钟）对比 P1 版本(commit 88d3eab9e) 与 P2-0 版本的 TTFT/吞吐，作为改进的证据记录在 P2_NOTES.md。

## 要实现（均需可通过参数独立开关，以便消融；默认值在 MuxWise-PP 模式下全部开启）
1. **(d) PP 感知的准入预留**：stage0 做 prefill 准入时，为所有 decode micro-batch 的后续 decode 预留 token（参照 FastPP schedule_policy.py:180-230, 296-301 的做法，pp_safety_margin = pp_size-1），避免 PP 下 KV 超卖导致 retract。
2. **(c) micro-batch 均衡**（不移植 pending 暂停）：
   - 合入策略：prefill 完成的请求分配给“负载最小”的 decode micro-batch，负载 = bs 与 Σ(当前上下文长度) 的加权（权重做成参数，默认按 decode 单步时间的近似模型：a*bs + b*Σctx，可由 P2-3 的 RLS 模型给出），一个 prefill batch 中多请求可拆分到不同 micro-batch。
   - 周期性再均衡：借鉴 FastPP 的滞回带（base_unit、margin=0.4*base_unit），4090/8B 场景下 base_unit 默认 8（参数化）；当 micro-batch 间负载差超过带宽时，从重的 micro-batch 迁移请求到轻的。迁移由 stage0 决定，经 P1 的控制消息在所有 stage 的同一逻辑步执行；需要 ScheduleBatch 的拆分/合并（参考 FastPP schedule_batch.py:1085-1130 的 split_batch，适配 0.5.3 的字段：sampling_info 及 penalizer、req_pool_indices、seq_lens(含 cpu)、output_ids、orig_seq_lens 等，以及 MuxWise 新增字段）。
   - 迁移只能在请求不处于“正在执行”状态时进行（只迁移已完成本轮结果处理、尚未开始下一步的 micro-batch 中的请求），务必保证 next_token_ids 与请求对应关系正确。
3. **(b) ALP 思想 → SLO 感知的 SM 分区与准入预算**：
   - 移植 FastPP alp_scheduler.py 的 RLS 在线线性模型（numpy，λ=0.9975，P0=1.25I，误差裁剪 ±0.4|pred|，warmup 16 个样本），放在 python/sglang/srt/multiplex/ 下新文件。
   - 两个预测器（每个 stage 本地维护，每个分区一组参数或把分区 SM 数作为特征——自行选择更稳的方式）：
     * decode 单步时间 T_d：特征 [bs, Σctx/1e4, 1]；样本来自 decode lane 每步的 CUDA event 计时；
     * prefill 每层时间 T_p：特征 [n_tokens/1e3, Σl_pre²/1e6, Σl_pre·l_past/1e6, 1]；样本来自 prefill 每片的 CUDA event 计时 / 层数。
     冷启动：可在启动阶段对每个分区做一次简短 profiling（几个合成 batch，耗时控制在 1 分钟内），或者直到样本数足够前回退到 MuxWise 原静态阈值策略（P1 的 policy 接口）。
   - 分区选择（替换静态阈值表，作为 policy 实现）：在满足 pp_size*T_d(idx) ≤ slo_tpot*coeff(默认1.0) 的分区中，选 prefill 吞吐能力 ≥ required_throughput 的最小 prefill-SM 分区；若无，取满足 TPOT 约束下 prefill-SM 最多者；若都不满足 TPOT，取 decode-SM 最多者（全 decode 分区仅在无 prefill 时使用）。分区切换有代价（需同步两条流），加入最小驻留时间/滞回（参数化，默认如 ≥ 若干 decode 步）防止抖动。
   - required_throughput（stage0 计算，随控制消息下发给其他 stage，使各 stage 使用相同的需求值）：严格沿用 FastPP：stale = 等待 ≥ fresh_eps*SLO_TTFT（fresh_eps=0.25）的请求；time_budget = SLO_TTFT - mean(stale 等待时间)；catch_up = stale_tokens/time_budget（time_budget ≤ SLO_TPOT 时视为 ∞）；required = catch_up + 近 window(10s) 内到达的输入 token 速率（扣除 radix 命中前缀后的有效 token）。
   - 准入预算：stage0 每次构建 prefill batch 时的 token 上限 = clamp(由 T_p 预测使该 batch 在单个 stage 的完成时间 ≤ 目标值的最大 token 数, min_budget, max_budget)，目标值与 SLO_TTFT/队列状态挂钩（队列积压时允许更大 batch 提升吞吐，空闲时小 batch 保证 TTFT）；单个超长请求（>预算）仍允许单独成批（不切 chunk，MuxWise 的按层切片保证不阻塞 decode）。请给出你的具体公式并写在文档里。
   - 新参数（server_args，带 help）：--pdmux-pp-policy {static,alp}、--pdmux-pp-slo-ttft (默认 5.0)、--pdmux-pp-slo-tpot (默认 0.1)、--pdmux-pp-rebalance/--no-...、--pdmux-pp-rebalance-unit、--pdmux-pp-reserve-decode 等（命名可调整，保持一致）。
4. 观测：每 N 秒（默认 10s）在 stage0 日志打印一行调度摘要：各 stage 当前分区、各 micro-batch bs/Σctx、waiting 队列长度与 token、required_throughput、准入预算、RLS 模型样本数与最近预测误差(MAPE)、迁移次数。方便实验后分析。

## 测试
- 单元测试（CPU 即可）：RLS 收敛、分区选择逻辑、再均衡决策、split/merge ScheduleBatch 前后字段一致性。
- 正确性：重复 P1 的输出比对测试（temperature=0，开启全部新特性，要求与 P1 同等的一致性），尤其要构造会触发迁移的场景（如把 rebalance unit 调小）。
- 压力：用 /root/workspace/muxpp_work/bench/bench_client.py 跑正式 trace 的前 300 条（--limit 300，time-scale 可设为 1 或适当加速），PP4+pdmux 全特性，无死锁无报错无泄漏；同时用相同前 300 条比较 static 与 alp 策略的 TTFT/TPOT（只作为健全性检查，记录即可，不要据此反复调参超过 2 轮）。
- 回归：原 pdmux(pp=1)、原 PP(无 pdmux) 仍工作。
- 测试日志放 /root/workspace/muxpp_work/logs/p2/；结束时 GPU 上不得留下进程。

## 产出
- 代码 commit 在 muxwise-pp 分支。
- /root/workspace/muxpp_work/P2_NOTES.md：移植了什么/没移植什么及理由、公式、参数表与默认值、测试结果、已知问题；并给出正式实验推荐的 MuxWise-PP 启动命令（4 卡 PP4）。
- 最后一条消息：简明总结。
