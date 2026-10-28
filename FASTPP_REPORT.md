# FastPP 代码分析报告（/root/ext/FastPP，HEAD cb3ce5b，路径相对 python/sglang/srt/）

## 1. PP 事件循环
- `managers/scheduler.py:449-486` event_loop_normal；PP 下禁用 overlap。`PPCounter`(utils.py:299) 轮转 pp_size 个 micro-batch。
- 每 micro-batch 状态列表：_running_batch/_last_batch/_cur_batch/_being_chunked_req/_policy（scheduler.py:274-303），按 cur_pp 属性访问（:1763-1798）。
- 每个 rank 一份调度器副本；rank(pp0,tp0) 经 ZMQ PUB 广播 recv_reqs + available_size（:542-560），其余 rank 做相同决策。stage 间只传张量（tp_worker.py:166-226）；next_token_ids 由末级在 PP 组 broadcast。
- mixed chunk：每个 micro-batch 的 prefill chunk 与该 micro-batch 的 decode 合并（:1009-1021）；长 prompt 的 chunk 在连续 micro-batch 间交接（fetch_chunked_prefill :845-855）。
- PrefillAdder：为**所有** micro-batch 的 decode 请求预留 token，并加 `pp_safety_margin = pp_size-1`（schedule_policy.py:180-230, 296-301）。
- 不移植其传输层：上游 0.5.3 的 p2p proxy 张量已覆盖。

## 2. 动态 chunk（managers/dynamic_chunk_controller.py）
- 钩子：decide_chunk_size() 在 get_next_batch_to_run 前；pre_step/post_step 包住 forward（scheduler.py:458,470,477）。
- 节奏：ALP 每 pp_size*2 步、greedy 每 pp_size 步决策一次；只有 pp0/tp0 决策后 ZMQ 同步给其他 rank（:515-540）。
- 应用：`chunked_prefill_size = max_prefill_tokens = chunk_size`（:106-109）。
- 信号：TTFT = 首 token 时刻 - req 创建时刻；TPOT = 同一 micro-batch 槽连续两次结果的间隔；kv_free = (available + evictable - Σrunning)/max_total（:115-153）。均为窗口均值。
- Greedy（:189-256）：
  ```
  if kv_free < 0.10: next = ((min(running_req_num)+64)//128+1)*128
  elif restore_chunk: next = restore_chunk
  elif tpot > SLO_TPOT*1.08 or ttft < SLO_TTFT*0.1: next = max(cur-128, 128)
  elif ttft > SLO_TTFT*0.3 and tpot < SLO_TPOT*0.85: next = min(cur+128, 2048)
  ```
- ALP（:262-435）：
  ```
  FRESH_EPSILON = SLO_TTFT*0.25 ; stale = waits >= FRESH_EPSILON
  time_budget = SLO_TTFT - mean(stale_wait_times)
  catch_up = inf if time_budget <= SLO_TPOT else stale_tokens/time_budget
  required_throughput = catch_up + qps_tok_sum(近 10s 到达的输入 token / 10s)
  候选 chunk 128..2048 step 128 升序：
    if raw_pred*pp_size > SLO_TPOT*1.05: break
    capacity = max(chunk - avg_running_req, 0)/raw_pred
    if capacity >= required_throughput: return chunk   # Optimal
  否则返回最大安全 chunk（Best_Effort），都不安全返回最小 chunk
  ```
  预测特征（:738-964）：对每个候选模拟各 stage 的打包，取 {running_token_max, decode_ctx_sum, Σl_pre², Σl_pre·l_past} 的均值。
- ALP 模型（managers/alp_scheduler.py，纯 numpy）：`pred = T0[chunk] + θ·x`，
  `x = [ctx_sum/1e5, Σl_pre²/1024², Σl_pre·l_past/(1024·2048), 0.1]`；
  RLS：λ=0.9975，P0=1.25·I，`g = P x/(λ + xᵀP x)`，`θ += g·err`，`P = (P - g xᵀP)/λ`，误差裁剪 ±0.4·|pred|，θ 初值 0，warmup 16 次更新。
  T0[chunk] 由启动 profiling 得到（合成请求跑真实循环，减去 attention 时间），存 alp_wait_time_chunk_rls.json。

## 3. Batch rebalancing（managers/batch_rebalancer.py）
- 每个 micro-batch 轮次在准入前调用（scheduler.py:888-889）；在本 micro-batch 的 running_batch 与全局 pending_batch（暂停但保留 KV）之间移动 decode 请求。
  ```
  n = Σ_mb last_batch.size - 外来 chunked + pending.size
  has_kv_headroom = waiting_tok > 1024 and kv_free > 0.10
  auto: TPOT_FIRST if tpot >= 0.9*slo_tpot else E2E_FIRST
  base_th = 128*pp ; margin = 0.4*base_th ; th = (n//base_th)*base_th
  current_local_limit = th//pp + 128
  Case1 (th>0 and n < th+margin and not headroom): 保持 desired = th//pp（滞回带，micro-batch 大小为 128 的倍数）
  Case2 其他：desired = n//pp；不足则从 pending 取；超过 current_local_limit 则多余放回 pending
  ```
- 排序：TPOT_FIRST 从 running 尾部移出；E2E_FIRST 采用 LIFO 并从 running 头部移出。
- 依赖 ScheduleBatch.split_batch（schedule_batch.py:1085-1130），对 req_pool_indices/seq_lens/output_ids/sampling_info 切片。
- 缺陷：pending_batch 不计入 PrefillAdder 内存、abort、retract。
- 单测：/root/ext/FastPP/test/srt/test_batch_rebalancer.py

## 4. 参数默认值
--enable-dynamic-chunk、--dynamic-chunk-strategy alp|greedy、--dc-slo-ttft 2.0、--dc-slo-tpot 0.2、--dc-kv-free-threshold 0.10、
--dc-alp-chunk-min/max/step 128/2048/128、--dc-alp-fresh-epsilon 0.25、--dc-alp-slo-tpot-coeff 1.05、--dc-alp-window-sec 10、
--enable-batch-rebalancing、--br-waiting-tok-threshold 1024、--br-load-alpha 0.4、--br-base-unit 128、--br-mode auto|tpot_first|e2e_first。
基准 SLO：TTFT ≤ 2000ms、TPOT ≤ 200ms。

## 5. 可移植性
- alp_scheduler.py：可直接搬。
- greedy/ALP 决策：大部分可移植（映射 _last_batch→last_mbs 等），需新增 TTFT/TPOT 采集、到达 token 窗口、跨 stage 同步决策值。
- 启动 profiler：与 FastPP 循环耦合紧，建议重写为简短离线 profiling。
- BatchRebalancer：算法独立，需移植 split_batch 到 0.5.3 字段，pending 机制有记账缺陷（我们不移植 pending）。
