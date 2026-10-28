# 任务 P8a：MuxWise-PP 支持 TP（目标配置 PP2×TP2），开发验证，不跑正式

## 背景
- 并行矩阵结果（RESULTS.md、results/matrix_summary.csv）：MuxWise TP2×DP2 主 SLO 82.7%（TTFT p50 1.79s），MuxWise-PP PP2×DP2 40.2%（5.82s）、PP4 31.9%。大脑判断：PP 只提升吞吐、不缩短单请求 prefill 延迟，TP 让多卡并行算同一请求，在以 TTFT 为瓶颈的负载上更占优。MuxWise-PP 目前只支持 TP1，用户要求加 TP 支持，以测 PP2×TP2 并与 MuxWise TP2×DP2 正面对比。
- 必读：P1_NOTES.md、P2_NOTES.md、P5B_NOTES.md、P7A_NOTES.md；代码 /root/workspace/sglang（muxwise-pp，HEAD f91e8a18d），继续 commit，不 push。环境 source /root/workspace/muxpp_work/env.sh。
- 原 MuxWise（pp=1）已支持 TP：parallel_state 中有独立的 `pdmux_prefill_tp` 通信组（prefill 与 decode 的 TP allreduce 走不同 communicator，避免两条 stream 上的集合通信相互死锁），`set_pdmux_status` 切换；prefill 完成一致性通过 tp_cpu_group allreduce。请复用这些机制。
- 并行任务：另一个 Codex（P8b）只读日志写分析报告，不用 GPU、不改代码，但会编辑 RESULTS.md。你不要修改 RESULTS.md。GPU 4 张都归你用。

## 需要实现（保持 TP1 路径行为与性能不变）
1. 每个 PP stage 由 TP 组组成。决策者仍然只有 stage0 的 tp_rank0：准入 plan、MERGE、abort/retract、迁移、驱逐 journal、required/budget。控制包要可靠送达下游 stage 的所有 TP rank（例如下游 tp_rank0 接收后在 stage 内 TP cpu 组广播，或者按 tp_rank 一一配对发送），所有 TP rank 必须严格同序执行。
2. Prefill lane：独立 prefill PP 通信组需按 tp_rank 配对建立（与上游 pp_group 的构造方式一致）；hidden/residual 传输沿用上游 send/recv_tensor_dict 的 all_gather_group=attn_tp_group 分片 + 接收端 all-gather，或者直接整块发送（自行权衡带宽与正确性）；READY 握手、预接收槽、在途上限按 rank 正确工作。
3. Decode lane：AsyncDecodeTransport 按 tp_rank 配对；末级 token 只由一个 rank 回传 stage0，再经控制包下发给所有 stage 的所有 TP rank。
4. SM 分区：同一 stage 内的所有 TP rank 必须使用同一分区、在同一控制点切换（由 stage 内 tp_rank0 决定后广播，或者把分区决定放进控制包），避免集合通信两端 SM 配额不对称或切换时序不一致造成挂起。RLS 样本可只在 tp_rank0 收集。启动时分区校准（pp_profile_decode）需在 TP 组内同步执行。
5. MLP 分块（--pdmux-prefill-mlp-chunk-tokens）扩展到 TP：每个 TP rank 按相同的 token 切分执行，RowParallel 的 allreduce 次数随块数增加，需保证各 rank 一致；评估块大小对 TP2 的影响（allreduce 次数 vs 激活峰值）。
6. server_args：放开 MuxWise-PP 的 TP>1 限制（仅 Llama）；不支持的组合（如 DP attention）继续拒绝。

## 测试
- CPU 单测：控制包 TP 广播/重放、分区一致性、配对通信组构造。
- 正确性：PP2×TP2+pdmux 与普通 TP2（pp=1，非 pdmux，同 FlashInfer）对比 serial20（temperature=0，64 token）以及冷 cache 触发分块的 20 条；TP 会改变数值，以普通 TP2 为参照，报告首个分叉位置，要求与 P5b 同等水平的一致性。
- 取消测试、200 条 time-scale=0.02 压测（开发 trace）无泄漏；TP1 的 MuxWise-PP（PP4 v3 配置）、原 pdmux（pp=1，TP1 与 TP2）、原 PP 回归。
- 预检：PP2×TP2 用 31,663-token 长预热选择 mem fraction（参考 0.71，PP2×TP2 每卡只有 8 层的一半 TP 分片，可更高）；`--pdmux-pp-max-inflight-prefill` 按 PP 比例取 3；其余参数同 scripts/p5b_v3_serve.sh。用**开发 trace**前 300 条（time-scale=1，--warmup 2 --force-close --drain-timeout 300）跑一次健全性检查，只记录、不调参超过 1 轮。不得使用正式 trace。
- 写好 `muxpp_pp2tp2` 的 launch 脚本并接入 run_experiment.sh（含正式锁）。

## 产出
- 代码 commit；/root/workspace/muxpp_work/P8A_NOTES.md：设计（TP 下控制包/通信组/分区一致性）、测试结果、开发 trace 结果（与 MuxWise-PP PP2×DP2 开发结果 63.3% / 4.14s 对照）、推荐的正式启动命令。
- 不跑正式实验；结束确认 GPU/进程清理。最后一条消息：简明总结。
