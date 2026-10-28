# MuxWise-PP 设计文档（大脑 → 双手 的总体方案）

## 0. 目标与对比对象
在 /root/workspace/sglang（SGLang 0.5.3rc0 + MuxWise/PD-Multiplexing）上：
1. 让 MuxWise（`--enable-pdmux`）支持 Pipeline Parallelism（`--pp-size N`），即 **MuxWise-PP**。
2. 有选择地移植 FastPP（/root/ext/FastPP，OSDI'26）的调度思路。
3. 在 4×RTX4090 + Llama3.1-8B-Instruct 上用 Mooncake Tool&Agent trace 对比：
   MuxWise(DP4) / Chunked(DP4) / SGLang-PD(2P2D) / MuxWise-PP(PP4)。

## 1. 现有代码要点
- MuxWise 循环：`python/sglang/srt/multiplex/multiplexing.py::event_loop_pdmux`。
  - prefill 一次一个 split_prefill_batch（不切 chunk，`--chunked-prefill-size -1`），
    按层切片（`split_forward_token_budget // extend_num_tokens` 层/次）放在 prefill green-ctx stream 上；
  - decode（running_batch）在 decode green-ctx stream 上每轮跑一步；
  - SM 分区（stream_idx）按 decode batch size 查表（manual_divisions 的阈值）选择；
  - prefill 完成后（event query + tp allreduce 一致性）merge 进 running_batch。
- `ScheduleBatch` 新增 split_index / split_forward_count / split_prefill_finished 等字段；
  `tp_worker.forward_batch_split_prefill`、`model_runner.forward_split_prefill`、
  `LlamaForCausalLM.forward_split_prefill(split_interval=[start,end))` —— 目前**用全局层号 0..L 且假设本 rank 持有全部层**，
  embed/norm/logits 也假设单 rank。
- CudaGraphRunner 为每个 SM 分区录制一套 decode graph；`update_decode_attn_backend(stream_idx)`。
- 上游 PP：`scheduler.event_loop_pp`（pp_size 个 micro-batch 轮转、每个 stage 复制一份调度状态，靠
  `point_to_point_pyobj(recv_reqs)` 把新请求逐级下传，hidden 用 `pp_group.send/recv_tensor_dict`，
  最后一级把 next_token_ids 沿环回传，各 stage 用相同 token 做 process_batch_result，从而保持确定性）。
- server_args 中写死了 pdmux 要求 pp_size == 1（约 2748 行）。

## 2. MuxWise-PP 核心设计：双 lane 流水线
每个 PP stage 内部仍是 MuxWise：prefill 和 decode 在同一 GPU 的两个 SM 分区 stream 上并发。
跨 stage 分成两条相互独立的 lane：

### 2.1 Decode lane（同步、微批轮转，复用上游 event_loop_pp 的结构）
- pp_size 个 **只含 decode** 的 micro-batch（running_mbs[i]），按上游 event_loop_pp 方式轮转，
  在当前分区的 decode stream 上执行（CUDA graph 按 stream_idx 选择）。
- hidden/proxy 张量、next_token_ids 回传全部沿用上游 pp_group 的 send/recv（decode 专用通信组 = 原 pp_group）。

### 2.2 Prefill lane（异步、按层切片，独立通信组）
- 新建一个 **prefill 专用 PP 通信组**（与 pp_group 同成员、独立 NCCL communicator，以及一个 gloo/cpu 组传元数据），
  避免与 decode lane 的 send/recv 顺序互相阻塞/死锁。
- stage r 上同一时刻最多一个 split_prefill_batch 在计算（可选：允许 1 个在算 + 1 个已收到待算的排队）。
- stage r 执行本 stage 的层 [start_layer, end_layer)，按 `split_forward_token_budget` 切片，每轮循环推进一片，
  kernel 发射到 prefill stream，不阻塞 decode lane。
- 完成后（event.query() 为真）：非末级 → 在 prefill 组上 isend hidden_states+residual 给 r+1（异步，用 work 句柄检查完成）；
  末级 → norm + logits + sample 得到首 token。
- stage r>0 用非阻塞方式检查是否有来自 r-1 的 prefill 到达（先在 cpu 组上收元数据 plan，再 irecv 张量，未完成则继续 decode）。
  要求：等待 prefill 数据**绝不能阻塞 decode lane**。

### 2.3 一致性：stage0 是唯一决策者
上游 PP 靠“各 stage 用相同输入做相同决策”保持一致；双 lane 异步后各 stage 的时序不同，
（尤其 KV allocator 状态在不同时刻不同），必须改为 **stage0 决策、下游执行**：
- 准入（哪些请求组成 prefill batch、prefix 命中长度）：stage0 做，把 plan（rid 列表、fill_ids 长度、prefix 长度等）
  随 prefill lane 下传；下游按 plan 构建 batch 并分配 KV（下游不做 radix 匹配决策，或做了但必须与 plan 一致—推荐 PP 模式下
  下游 stage 的 radix cache 与 stage0 的结构严格同构：插入/驱逐也按 stage0 的指令）。
  * 若同构 radix 实现太复杂：MuxWise-PP v1 可以先 `--disable-radix-cache`（同时在报告里说明），v2 再做前缀复用。
    但 Tool&Agent 负载前缀复用很重，最终版本应尽量支持 radix cache（见 §5）。
- prefill 完成后合入哪个 decode micro-batch：末级把首 token 结果沿 prefill 组回传 stage0（last→0），
  stage0 决定 "在 mb_j 的下一次构建时 merge prefill batch X"，并把该指令放进 decode lane 每步下传的 pyobj 控制消息中
  （扩展上游 point_to_point_pyobj 传的内容：recv_reqs + control）。下游 stage 在处理 mb_j 的同一步执行同样的 merge。
- retract / 驱逐 / abort：同理由 stage0 决定并通过控制消息下发。各 stage 的 max_total_num_tokens 取 PP 组内最小值
  （确认上游是否已做；没有就加），保证 stage0 能代表所有 stage 做内存判断。
- finished 判断：各 stage 用相同 next_token_ids 在 process_batch_result 中自然一致（沿用上游）。

### 2.4 SM 分区与 stream 切换
- 分区切换需要 stage 内两条 stream 都 synchronize（沿用 MuxWise 做法）；各 stage 可**独立**选分区
  （分区只影响本地性能，不影响正确性），选择逻辑见 §3.2。
- decode CUDA graph：每个分区一套（沿用）。显存紧张时可减少分区数（4090 上建议 4~5 个分区）。

### 2.5 模型侧改动
- `forward_split_prefill` 支持 PP：层区间限定在 [start_layer, end_layer)；仅首 stage 做 embed，
  非首 stage 从收到的 proxy 张量取 hidden_states/residual；仅末 stage 做 norm+logits，非末 stage 返回 PPProxyTensors。
  至少支持 Llama（本实验模型），其他模型不需要。

## 3. 移植 FastPP 的哪些思路（有选择地）
FastPP 组成：(a) 基于 ZMQ 的 PP 传输；(b) 动态 chunk 大小（greedy / ALP：RLS 在线延迟模型 + catch-up 吞吐需求）；
(c) batch rebalancing（micro-batch 间均衡 + pending 暂停）；(d) PrefillAdder 为所有 micro-batch 预留 decode token + pp_safety_margin。

决定：
- (a) **不移植**，上游 p2p proxy 机制已覆盖。
- (d) **移植**：PP 模式下准入时为所有 micro-batch 的 decode 预留 token，加 pp_size-1 的安全余量。
- (c) **移植其“均衡”部分，不移植 pending 暂停**：
  1. prefill 完成的请求合入时，按 FastPP 的思路分配到负载最小的 micro-batch（按 bs 及 Σcontext tokens 加权），
     必要时把一个 prefill batch 的请求拆分到多个 micro-batch；
  2. 周期性 rebalancing：当 max(mb_bs) - min(mb_bs) 超过阈值（FastPP 以 128 为单位的滞回带思想，4090 上按比例缩小，
     例如单位 8，滞回 0.4）时，在 micro-batch 之间直接迁移 decode 请求（需要移植 `ScheduleBatch.split_batch` 到 0.5.3 字段）。
     迁移指令由 stage0 决定、经控制消息下发，所有 stage 在同一步执行。
     原因：PP 的吞吐受最慢 micro-batch 限制（气泡），而 MuxWise 的 decode 分区选择也依赖 bs。
  3. 不做 pending（暂停请求）——FastPP 中它绕开了内存记账/abort/retract，风险大；PP4 下 KV 充足，收益小。
- (b) **移植 ALP 的核心思想，但把“控制旋钮”从 chunk size 换成 MuxWise 的旋钮**：
  MuxWise 不切 chunk，它的旋钮是 (i) SM 分区 stream_idx、(ii) 每次 prefill batch 的准入 token 预算、(iii) 每片层数。
  - 移植 `alp_scheduler.py` 的 RLS 在线线性模型（numpy，λ=0.9975、误差裁剪 0.4），建两个预测器：
    * decode 步时间 T_d(stream_idx, bs, Σctx) ；
    * prefill 每层时间 T_p(stream_idx, Σl², Σl·l_past, n_tokens)。
    初值可用启动时一个简短的离线 profiling（或冷启动时用 MuxWise 原查表策略，等 RLS 有 ≥16 个样本后切换）。
  - 分区选择（替代 MuxWise 的静态 bs 阈值表）：在满足 `pp_size * T_d(idx) ≤ SLO_TPOT * coeff` 的分区中，
    选 prefill 吞吐能力 ≥ required_throughput 的**最小 prefill SM**（把多余 SM 给 decode）；若都不满足，选满足 TPOT 约束下
    prefill SM 最多的（Best-Effort）；都不满足 TPOT 时选 decode SM 最多的。
    required_throughput 直接沿用 FastPP 公式：
      stale = 等待 ≥ 0.25*SLO_TTFT 的请求；time_budget = SLO_TTFT - mean(stale 等待时间)；
      catch_up = stale_tokens / time_budget（budget ≤ SLO_TPOT 则 ∞）；required = catch_up + 近 10s 到达 token 速率。
  - 准入预算：prefill batch 准入 token 上限按 required_throughput 与预测时间动态取（上限 = 当前 stage KV 余量约束），
    避免一次准入过大导致后续请求 TTFT 被长 batch 阻塞（Tool&Agent 平均 7k token，长尾到 32k）。
  - 新增参数：`--pdmux-pp-slo-ttft`（默认 5.0s）、`--pdmux-pp-slo-tpot`（默认 0.1s）、`--pdmux-pp-policy {static,alp}`、
    `--enable-pdmux-pp-rebalance` 等，默认值写在 server_args，均可关闭以便做消融。
  - 各 stage 可以各自独立运行预测器（只影响本地分区），准入预算只在 stage0 用。

## 4. 实验设计
- 硬件：4×RTX4090 24GB（sm_89, 128 SM, PCIe, 无 NVLink）；模型 Llama3.1-8B-Instruct (bf16)。
- 负载：/root/ext/mooncake/FAST25-release/traces/toolagent_trace.jsonl
  取 timestamp ∈ [600000, 840000) ms，过滤 input_length + output_length > 32768（与“长度大于 32k”一致，记录被滤数量），
  到达时间 t' = (t - 600000) * 2.5 → 10 分钟；共 1551 条，平均输入约 7.25k，平均输出约 169。
  prompt 由 hash_ids 构造：块大小 512 token，每个 hash_id 映射到确定性的伪随机 token 序列（以 hash_id 为种子，
  避开特殊 token），最后一块截断到 input_length → 重现 trace 的前缀共享结构。用 /generate + input_ids，
  max_new_tokens = output_length，ignore_eos=True，stream=True，temperature=0。
- 系统配置（所有系统统一：--disable-overlap-schedule、相同 context-length=33792、radix cache 开启，除非某系统不支持—需记录）：
  1. Chunked：4 个 TP1 实例（DP4），`--chunked-prefill-size 2048 --enable-mixed-chunk`，前端 sglang-router（cache_aware）。
  2. MuxWise：4 个 TP1 实例，每个 `--enable-pdmux`（4090 配置），前端同上。
  3. SGLang-PD：2 prefill + 2 decode（TP1），KV 传输用 mooncake（无 RDMA 时用 TCP 或 NIXL，择能跑通者），sglang-router PD 模式。
  4. MuxWise-PP：单实例 `--pp-size 4 --enable-pdmux` + FastPP 移植特性。
- 指标：TTFT（mean/p50/p90/p99）、TPOT（mean/p50/p90/p99）、E2E、完成数、输入/输出吞吐、
  SLO 达成率（TTFT≤{2,5,10}s 与 TPOT≤{50,100,200}ms 的联合与分别达成率）、请求失败数；服务端记录 cache 命中率。
- 每个系统只跑一次，不补跑不重跑（启动后可发 1~2 个预热请求，不计入）。

## 5. 分阶段计划
- P0 环境（进行中）。
- P1 MuxWise-PP 基础功能：双 lane + stage0 决策 + 模型 PP split prefill；正确性测试（temperature=0 下与 TP1 普通模式输出逐 token 对比，
  多并发 + 长短混合 prompt 压力测试无死锁/无泄漏）。radix cache：尽力支持（stage0 决策、下游按 plan 同构操作）。
- P2 移植 FastPP：(d) 预留、(c) 均衡+迁移、(b) ALP/RLS 分区与准入；单元测试 + 小规模压测。
- P3 基准脚本 + 四个系统各跑一次 + 汇总表/图。
