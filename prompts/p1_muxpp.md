# 任务 P1：实现 MuxWise-PP（PD-Multiplexing + Pipeline Parallelism）

## 必读
1. /root/workspace/muxpp_work/DESIGN.md（总体设计，§2 是本任务核心，§3 是下一阶段，不在本任务范围，但请预留接口）。
2. /root/workspace/muxpp_work/ENV.md（环境：`source /root/workspace/muxpp_work/env.sh`）。
3. 代码：/root/workspace/sglang（git 仓库）。重点文件：
   - python/sglang/srt/multiplex/multiplexing.py、multiplex/pdmux_context.py（MuxWise）
   - python/sglang/srt/managers/scheduler.py（event_loop_pp、event_loop_pdmux 的调度入口、run_batch、get_new_batch_prefill）
   - python/sglang/srt/managers/tp_worker.py（forward_batch_generation、forward_batch_split_prefill）
   - python/sglang/srt/model_executor/model_runner.py、cuda_graph_runner.py
   - python/sglang/srt/models/llama.py（forward_split_prefill）
   - python/sglang/srt/distributed/parallel_state.py（pp_group、set_pdmux_status、pynccl 当前流）
   - python/sglang/srt/server_args.py（pdmux 与 pp_size 互斥的断言）
   - 可参考 /root/ext/FastPP（只读，SGLang 0.4.1 的 PP 实现）

## 工作方式
- 在 /root/workspace/sglang 中先 `git checkout -b muxwise-pp`（若已存在则切过去），每完成一个可工作的里程碑就本地 commit（不要 push）。
- 新代码尽量放在新文件 python/sglang/srt/multiplex/multiplexing_pp.py（SchedulerMultiplexPPMixin），对已有文件的修改保持最小且不破坏原 pdmux（pp=1）和原 PP（无 pdmux）路径。
- GPU：4×4090 都可用。不要留下僵尸进程；每次测试后确认 nvidia-smi 干净。测试日志写到 /root/workspace/muxpp_work/logs/p1/。
- 遇到设计文档未覆盖或不可行之处，自行做合理决定，并写进 /root/workspace/muxpp_work/P1_NOTES.md（含原因）。

## 需求
1. `--enable-pdmux` 与 `--pp-size N (N>1)` 可同时使用（Llama 系模型即可），启动后调度进入新的 event_loop_pdmux_pp。
2. 双 lane（见 DESIGN §2）：
   - decode lane：pp_size 个仅含 decode 的 micro-batch 轮转，在当前分区的 decode stream 上执行，沿用上游 PP 的 proxy 张量与 next_token_ids 传递机制；
   - prefill lane：每个 stage 用独立的 PP 通信组（新的 NCCL 组 + cpu/gloo 组）异步传递 prefill hidden 与元数据；在 prefill stream 上按层切片推进本 stage 的层；等待上游 prefill 数据时不得阻塞 decode lane（用非阻塞 irecv/isend + 轮询，或专用通信线程，自行选择稳妥方案）。
   - 注意 pynccl/NCCL 在不同 CUDA stream 上的使用（MuxWise 已为 pynccl 加了“使用当前流”接口），确保 prefill lane 的通信发射在 prefill stream 上，decode lane 的在 decode stream 上，且两组通信不会相互等待造成死锁。
3. 一致性（DESIGN §2.3）：stage0 唯一决策——prefill 准入 plan、prefill 完成后合入哪个 decode micro-batch、retract/abort 等；下游 stage 只执行。max_total_num_tokens 在 PP 组内取最小值（若上游未做）。
4. Radix cache：优先支持（stage0 决策 prefix 命中长度并写入 plan；下游 stage 维护同构的 radix tree，插入/驱逐严格按 plan 指令或用同样的确定性操作序列；确保各 stage 的 prefix 命中长度一致）。如果在合理时间内无法做到稳定正确，则 MuxWise-PP 模式下强制 disable radix cache，并在 P1_NOTES.md 中清楚写明原因与后续方案——但请认真尝试，因为 Tool&Agent 负载前缀复用很重，这对最终结果影响很大。
5. 模型侧：forward_split_prefill 支持 PP（层区间限定在本 stage [start_layer,end_layer)，首 stage embed，非首 stage 用收到的 hidden/residual，末 stage norm+logits+sample，非末 stage 返回 proxy 张量）。
6. SM 分区：每个 stage 独立按本地 decode bs 选择分区（先沿用 MuxWise 的静态阈值策略，接口抽象成一个可替换的 policy 对象，供 P2 的 ALP 策略接入）。为 4090 准备配置 /root/workspace/muxpp_work/configs/pdmux_pp_4090.yml（参考 P0 写的 pdmux_4090.yml；PP 模式每 stage 只有 8 层，split_forward_token_budget 需重新考虑，使每片 prefill 的时长适中）。
7. 预留 P2 接口：prefill 准入预算的 hook、合入 micro-batch 选择的 hook（默认选 bs 最小的 micro-batch）、micro-batch 间迁移请求的控制消息类型（可先实现消息框架）。

## 测试（必须全部通过并记录在 P1_NOTES.md）
a. 正确性：temperature=0，ignore_eos，max_new_tokens=64，20 条长度在 100~8000 token 的 prompt（含共享前缀的若干条），分别用 (1) TP1 普通模式、(2) PP4+pdmux 运行，对比输出 token：要求绝大多数完全一致（bf16 数值误差可能导致少量在若干 token 后分叉，统计首个分叉位置并报告）。
b. 并发压力：用 /root/workspace/muxpp_work/bench/ 下的压测工具（若 P3a 已完成；否则写个简单的并发脚本）对 PP4+pdmux 打 200 条 trace 请求（可用 --limit 200 与 --time-scale 加速），无死锁、无报错、全部完成，结束后 KV 池/req pool 无泄漏（检查服务端日志 / idle 时的 self-check 不报错）。
c. 回归：原 pdmux（pp=1）与原 PP（pp=2，无 pdmux）仍能正常启动与生成。
d. 记录 PP4+pdmux 的启动参数、显存占用、每 stage max_total_num_tokens、CUDA graph 分区数量。

## 产出
- 代码 commit 在 muxwise-pp 分支。
- /root/workspace/muxpp_work/P1_NOTES.md：实现说明（架构图/时序说明、关键文件与函数、控制消息格式）、设计偏离及原因、测试结果、已知问题。
- 最后一条消息：简明总结。
