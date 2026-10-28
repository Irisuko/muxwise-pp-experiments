# P2：FastPP 调度思想移植（开发验证记录）

代码：`/root/workspace/sglang`，分支 `muxwise-pp`，本地提交、不 push。
所有测试证据：`logs/p2/`。这不是 DESIGN §4 的 P3 正式四系统实验。

## P2-0：流水化与 LRU

- prefill 同时最多 `--pdmux-pp-max-inflight-prefill` 批（默认 PP size）；各 stage 按 ADMIT 顺序执行 FIFO，每次只计算一个批次，已发往后级的批次保留 KV/请求状态直到统一 MERGE。
- stage0 统一决定 ADMIT、batch_id、MERGE、abort、retract。所有在途批次的输入 KV 已实际分配，未来输出预算额外计入预留。取消标记覆盖全部在途批次，继续排空通信后在统一结果处理点释放。
- radix 可见性：只有 MERGE/正常 decode 结果退休能插入；ADMIT 不插入。因此后一批不可能匹配未 merge 的前批 KV。所有 probe（包括未准入者）仍在各 stage 同序重放并断言命中长度。
- 默认 stage0 原生 LRU；每次 allocator evict 记录 `(请求数量, victim 完整 token 路径和 extra_key)` 到控制包。下游按同一调用顺序释放同组叶节点，校验路径、叶状态、锁和请求数量，不按本地时钟选择。`--pdmux-pp-radix-eviction fifo` 是显式回退，仍重放 stage0 的 victim。
- hidden 传输加入接收就绪握手：接收端分配并提交 recv 后用独立 Gloo tag 317 回传 batch_id，发送端再提交 NCCL send。避免未配对 send 与后续主线程设备同步形成环。
- 初版多批压力测试停滞；线程栈定位到 `ForwardBatch.init_new -> clamp_position -> Inductor _make_launchers -> torch.cuda.synchronize`。组合模式仅此辅助函数改为等价 eager 运算，避免首次动态 shape 编译的全设备同步。失败日志保留为 `p2-0-stress-200`、`p2-0-stress-debug`，不计为通过。

### P1 基线（88d3eab9e）

独立只读基线检出 `/root/workspace/sglang-p1`，使用相同启动配置、正式 trace 前300条、time-scale=1、warmup=2。到达跨度97.5秒。

| 版本 | 成功/失败 | 总时间(s) | TTFT mean/p50/p90/p99(s) | TPOT mean(s) | 输入tok/s | 输出tok/s |
|---|---|---|---|---|---|---|
| P1 | 300/0 | 241.603 | 68.873 / 64.596 / 122.271 / 126.813 | 0.06503 | 9085.15 | 185.88 |

| P2-0 (`c74f56ff1`) | 300/0 | 142.769 | 13.879 / 14.160 / 22.435 / 28.218 | 0.06226 | 15374.51 | 314.56 |

证据：`p1-baseline-300_summary.json`、对应 server/client 日志及逐请求 JSONL。

### P2-0 当前验收

- CPU：8 passed（FIFO 计算槽/保留 KV 所有者、异步 launch guard、abort、LRU 路径重放等）。
- 200 条、time-scale=0.02：`p2-0-stress-03` 200/0、101.000s；四级最终 free_kv=7339 + evictable=307972 = 315311，free_reqs=128，cached_tokens=572543，509 slices / 7062 decode steps。各级均达到4批在途。相对 P1 同配置压力轮177.829s减少43.2%。
- `p2-0-stress-02` 在190/200时出现Gloo peer断连，恰逢临时四进程定时栈输出；无 Python 业务断言或 OOM 记录，未确证根因。移除诊断输出后的完整轮通过，保留该失败记录。

P2-0 最终正确性 `p2-0-correctness-final`：20/20、1280/1280 token 与 TP1/P1 基准一致。取消 `p2-0-control-final`：16/16 abort，后续生成、unsupported HTTP400、flush HTTP200，退出 remaining requests=0。固定提交的真实到达率300条 `p2-0-real-300` 全通过；详见相应 summary 与逐请求日志。

## FastPP 移植范围与协议

已移植：(d) 所有 decode micro-batch + 所有未合入 prefill 的未来 token 预留；(c) 按请求均衡合入、带滞回的周期迁移；(b) RLS、catch-up 需求公式、SLO 分区选择和模型准入预算。未移植：ZMQ 传输、chunking、pending 暂停。现有两条 PP lane 继续使用；prefill 依然整请求按层切片；没有暂停 decode 请求但不记账的全局池。

预留采用保守完整剩余输出：`R = Σ_active [max(max_new_tokens - len(output_ids), 0) + pp_size - 1]`。active 包含所有 decode mb、在途 prefill、已处理首 token 但尚未进入目标槽的请求。输入 KV 已分配，不重复计算；新准入请求另需 `extend_input_len + max_new_tokens + pp_size - 1`。在锁住命中前缀后检查可用+可驱逐 KV，逐个准入更新 R。关闭 reserve 仅取消已有请求的未来预留，新批自身的输出仍检查。

完成 prefill 后 stage0 按 `L_j = a * bs_j + b * Σctx_j` 逐请求分配给最轻槽（相同负载选小编号），默认 a=1、b=0.0001：每一万上下文 token 约等于一个请求的负载。分配也计算尚待合入的请求与每槽容量。完整首 token 结果在唯一 MERGE 步处理并插入 radix；随后 split 成短暂的待合入批，每个目标槽在下次退休后的构建点接收，至多等待一次轮转。这不是 FastPP 的 pending 暂停；这些请求一直参与内存、abort 和 flush/idle 检查。

再均衡每32个逻辑控制步尝试。负载差超过 `1.4 * base_unit * a` 时选源/目标；广播 hold 槽列表，使这两个槽结束当前结果处理后不立即发起下一轮，其他槽继续运行。两槽 `in_flight is None` 时，stage0 才广播迁移 rid；各 stage 断言已退休后 split/merge。每次挑选最能缩小负载差的请求，直到进入 `0.4 * base_unit * a` 带、目标满，或移动不能改善绝对差。hold 仅用于这次通信轮转的安全退休，不长期暂停请求。

`ScheduleBatch.split_batch` 支持已退休的 text decode/已处理首 token 的 split prefill：保留 Req/KV 所有权及池引用；分别索引 req_pool_indices、seq_lens、CPU seq_lens cache、orig_seq_lens、output_ids；复制 sampling_info、penalizer orchestrator/各 penalizer 并重新绑定弱引用，再切片各历史张量。临时输入、attention/hidden、extend 与 MuxWise split 状态清空，由下一次 decode 重新构建。拒绝等待结果或仍持有 split_forward_batch 的批次。CPU 测试验证非连续拆分、重新合并、后续 penalty 施加及源批独立性。

## RLS、分区和准入公式

每 stage 本地两个 numpy RLS，共享该 stage 各分区样本。基础特征为：

- decode：`x_d = [bs, Σctx/1e4, 1]`。
- prefill：`x_p = [n_tokens/1e3, Σl_pre²/1e6, Σ(l_pre*l_past)/1e6, 1]`，目标值为每片 CUDA event 时间/实际层数。

为避免未访问分区永远没有16个样本，加入 SM 交互特征：`z = [x, x[:-1] * (SM_total/SM_lane - 1)]`，decode 5维、prefill 7维。相当于每个 workload 系数随 SM 数线性变化，并保留共享常数项。特征只作用于预测，不改变实际执行。

`pred=max(theta·z,1e-6)`；`g=Pz/(λ+zᵀPz)`；`theta += g*clip(y-pred,±0.4|pred|)`；`P=(P-gzᵀP)/λ`，λ=0.9975、P0=1.25I。没有启动合成 profiling；首个实测样本仅初始化截距，解决零预测导致相对裁剪无法学习的问题。每个模型达到16个样本前使用 P1 静态分区/预算。保留近128次相对预测误差的均值 MAPE。分区切换边界的零 SM lane 不进入训练。prefill 在全 decode 分区不发起计算，等待安全切换。

计时在 worker 内：decode 在收到 PP proxy、准备 ForwardBatch 后开始，包住本地 forward 和 sampling；prefill 在准备本片 ForwardBatch 后开始，包住本片 forward 和末级 sampling。两条 lane 各用 CUDA events，调度线程仅 query 已完成 event 后读取 elapsed_time，不为取样 synchronize。各 stage 的首/末层额外开销自然进入各自模型；分区切换仍同步本地两条流，未完成通信不强行同步。

需求只由 stage0 计算并随每个控制包广播：

```
stale = {r: now - arrival(r) >= 0.25 * SLO_TTFT}
time_budget = SLO_TTFT - mean(wait(stale))
catch_up = stale_tokens / time_budget
         = inf, if stale_tokens > 0 and time_budget <= SLO_TPOT
required = catch_up + effective_arriving_tokens(last 10s) / 10s
```

arrival 的有效 token 数由不修改 radix 的只读查询扣除当时命中前缀；等待请求使用到达/最近一次准入 probe 的有效 token 快照（与 FastPP 的 waiting prefix_indices 快照语义一致）。只读查询不切分节点、不刷新 LRU；会改变树结构的 probe 仍必须进入 plan 重放。

候选仅包含同时有 prefill/decode SM 的分区。满足 `pp_size*T_d <= slo_tpot*tpot_coeff` 的候选中，取 `n_tokens/(local_layers*T_p) >= required` 的最小 prefill-SM 者；无足够吞吐时取安全候选的最大 prefill-SM；无安全候选时取 decode-SM 最多者。有 prefill 但无 decode 时选全 prefill；无 prefill 时允许全 decode。默认最小驻留16个本地 decode 步、候选连续4次相同才切换；进出端点不受驻留限制，以免饿死新进入的 lane。

准入预算的具体模型：设 U=min(max_budget,max_prefill_tokens)，Q=等待有效 tokens，`pressure=min(Q/U,1)`，单 stage 时间目标 `τ=SLO_TTFT/(2*pp_size)*(1+pressure)`。因此空队列为半个理想 TTFT stage 份额，积压时增到完整份额。用当前本地 prefill（无则队首最多16个请求）的基础特征作形状代表，原有效 token 数 N。对候选 B：

```
x_p(B) = [B/1e3, original_square_feature*(B/N)^2,
          original_prefix_product_feature*(B/N), 1]
B_model = max {B in [min_budget,U], step=256: local_layers*T_p(B) <= τ}
budget = clamp(B_model, min_budget, U)
```

无可行点取 min_budget。实际 KV 限制由逐请求的准入检查再次收紧，不以预测替代内存检查。一个请求超过 budget 但 KV 足够时允许单独成批；不切 chunk。按层 slice budget 保持 P1 的8192 token-layer。

## 参数（仅组合模式消费；以下为默认值）

| 参数前缀 `--pdmux-pp-` | 默认 | 用途 |
|---|---:|---|
| max-inflight-prefill | pp_size | 同时未 merge 的 prefill 批数，设1消融流水化 |
| radix-eviction | lru | lru/fifo，均由 stage0 选择、下游重放 |
| policy | alp | static/alp；static 保留均衡和预留开关 |
| slo-ttft / slo-tpot | 5.0 / 0.1 | 秒 |
| tpot-coeff | 1.0 | TPOT 约束系数 |
| reserve-decode | true | 所有已准入请求的后续 decode 预留 |
| balance-merge | true | 按请求均衡合入，可独立关闭 |
| rebalance | true | 周期迁移，可独立关闭 |
| rebalance-unit / rebalance-interval | 8 / 32 | 等价请求负载单位 / 逻辑步 |
| load-bs-weight / load-ctx-weight | 1.0 / 0.0001 | 负载模型系数 |
| alp-admission | true | 模型准入，独立于 ALP 分区开关 |
| min-budget / max-budget | 1024 / 16384 | 模型 token 预算上下限 |
| min-residency-steps / hysteresis-steps | 16 / 4 | 分区驻留 / 连续候选计数 |
| fresh-eps / arrival-window | 0.25 / 10.0 | stale 判定比例 / 到达窗口秒数 |
| summary-interval | 10.0 | stage0 摘要周期秒数 |

布尔参数均支持 `--no-pdmux-pp-<name>`；所有参数在 `server_args.py` 提供 help 和边界校验。`--pdmux-pp-policy static --no-pdmux-pp-balance-merge --no-pdmux-pp-rebalance --no-pdmux-pp-alp-admission` 可保留仅 P2-0+安全预留。

摘要包含所有 stage 最近分区、RLS 两模型样本数/MAPE/分区样本数及单调时间戳，各 decode mb 的 bs/Σctx，waiting 数量/tokens、required、budget、在途批数、待合入请求数、累计迁移请求数。状态沿现有 CPU prefill 信箱转发到 stage0，不增加全组同步。MAPE 和预测均是在线诊断，不是 SLO 保证。

## 推荐正式实验启动命令（4×4090、PP4）

```bash
source /root/workspace/muxpp_work/env.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
  --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct \
  --tp 1 --pp-size 4 --port 30000 --disable-overlap-schedule \
  --enable-pdmux --chunked-prefill-size -1 \
  --pdmux-config-path /root/workspace/muxpp_work/configs/pdmux_pp_4090.yml \
  --context-length 33792 --mem-fraction-static 0.65 \
  --cuda-graph-max-bs 32 --max-running-requests 128 \
  --pdmux-pp-max-inflight-prefill 4 --pdmux-pp-radix-eviction lru \
  --pdmux-pp-policy alp --pdmux-pp-slo-ttft 5.0 --pdmux-pp-slo-tpot 0.1 \
  --pdmux-pp-reserve-decode --pdmux-pp-balance-merge --pdmux-pp-rebalance \
  --pdmux-pp-alp-admission --pdmux-pp-rebalance-unit 8 \
  --pdmux-pp-summary-interval 10 --watchdog-timeout 180 --log-level debug
```

保留所有对照系统相同的 context、无 overlap、dtype、greedy 请求与 trace；本轮未调换 CUDA/Torch/FlashInfer，也未修改 FastPP。消融只改对应参数；`static` 仍保留多批流水化、LRU、预留和均衡，用于隔离 ALP 策略影响。

## 已知边界

- 支持范围仍是 P1 的 Llama、TP1、普通 text/token 请求、Python 本地 radix；不支持 speculative/DP attention/外部分层缓存、logprobs/hidden/session/约束解码等 P1 已排除功能。split_batch 明确限定到这种退休批次，不宣称覆盖任意模型/调度模式。
- RLS 的 SM 特征是假设性近似，未做跨硬件校准；首末 stage 开销和缓存命中形状会影响 MAPE。CUDA event 预测本地计算，未直接建模 PCIe、Gloo、CPU 排队或整环气泡；TPOT 条件是调度目标，不是端到端保证。低采样或样本覆盖不足时按静态策略或依赖外推。
- 多批在途刻意不共享尚未 merge 的前缀：真实到达率300条中 P1 cache 加权命中44.13%，P2-0为40.22%。即使命中下降，P2-0输入吞吐9085→15375 tok/s（+69.2%），平均TTFT 68.87→13.88s（-79.8%）。这是完整P2-0组合改动的结果，不独立归因于LRU。
- 完整剩余输出预留偏保守；本轮未制造强制 retract 注入，也未实现跨 stage 故障恢复。取消在途 prefill 仍排空计算/通信再统一释放；避免释放尚在被 CUDA 访问的 KV。
- 回归的基准是任务指定的小规模开发验证，不将 static/ALP 的差异解释为正式性能结论；最终正式实验应保持固定参数一次运行。调度验证的第二轮修复零 SM 端点过渡样本污染；其后因 HTTP 连接复用问题补做同参数传输验证。全程未搜索权重/预算/SLO参数。

### 客户端连接复用诊断与补验

`p2-alp-300-final` 的299条完成，1条 `idx=3689`（input=6166、output=24）在67.5s到达后6.6ms报 `ServerDisconnectedError`，尚无HTTP状态/首token。服务端无异常、最终四级池检查通过；该轮stage0准入303条/2,188,877输入token（含4条启动/预热），而完整300条对照为304条/2,195,043，差值正好6166，说明该请求没有进入调度器。HTTP服务 keepalive=5s，aiohttp 默认连接复用窗口更长，符合过期空闲连接的复用竞争。

在 `/root/workspace/muxpp_work/bench/bench_client.py` 加入可选 `--force-close`（默认false、写入summary）；启用时每请求新建连接，**不增加重试、不重发失败请求**。`p2_verify.py --force-close` 可传递该开关。原基准客户端7项CPU测试仍通过。之后用相同调度参数、300条/time-scale=1 补做完整性验证，并让 static/ALP 最终比较双方都启用此选项；这不是调参，前述失败记录保留且不计为300条通过。

### 文件与控制消息索引

`multiplexing_pp.py` 管理 FIFO、KV/radix 重放、merge/hold/migrate 和统计；`pp_policy.py` 提供纯CPU的 RLS、分区、需求与均衡决策；`pp_timing.py` 延迟读取 CUDA events；`tp_worker.py` 在实际本地 forward 边界取样；`schedule_batch.py::split_batch` 转移退休批次状态；`server_args.py` 定义参数。

P1 控制消息扩展为：

```text
PPControl(step, mb_id, requests, plan, abort, retract,
  evictions=[(num_tokens, [(extra_key, full_token_path), ...]), ...],
  merge={batch_id, target_mb, next_token_ids, targets=[mb_for_each_original_req]},
  hold_mbs=[source_mb, target_mb], migrations=[MigrationPlan(rids, source_mb, target_mb)],
  required, waiting_tokens, admission_budget)
prefill mailbox: hidden / result / status(stage, partition, RLS statistics)
reverse READY: Gloo tensor(batch_id), tag=317
```

`target_mb` 是处理完整首token结果的控制槽；`targets` 才是逐请求的最终 decode 槽。先处理首token/缓存，再拆分，禁止提前改在途 batch 的索引。驱逐 journal 在该控制包的所有分配前绑定，之后检查重放消费完整。

后续 P3 建议四系统的基准客户端统一显式加 `--force-close`，保持HTTP连接策略一致。复现本轮全特性测试：

```bash
source /root/workspace/muxpp_work/env.sh
python /root/workspace/muxpp_work/scripts/p2_verify.py muxpp \
  --suite stress --limit 300 --time-scale 1 --force-close --tag UNIQUE_ALP
python /root/workspace/muxpp_work/scripts/p2_verify.py muxpp \
  --suite correctness --concurrency 20 --tag UNIQUE_CORRECTNESS \
  --extra --pdmux-pp-rebalance-unit 0.1 --pdmux-pp-rebalance-interval 8
```


### Greedy 输出一致性边界（没有隐藏复验分歧）

P2 全特性按 P1 原串行测试方法重跑（`p2-correctness-serial`）：20/20请求、1280/1280输出token与原TP1/P1基准完全一致。

20路并发、rebalance-unit=0.1 / interval=8 的第一次迁移测试 `p2-correctness-migration`：20/20完全一致，四级各13条相同迁移指令。最终代码的并发复验 `p2-correctness-final`：19/20完全一致，128-token prompt 在输出index=34（第35个token）分叉；四级各15条相同迁移指令，所有缓存/池检查通过。不能把这轮写成1280/1280通过。

进一步只读诊断（`diagnostics/p2_probe/sitecustomize.py`，仅设置对应PYTHONPATH和环境变量时启用，不在生产代码中）：对敏感请求在index=33/34验证 req_pool_idx 与张量索引、Req末token与input_ids、seq_len=prompt_len+已生成长度，并记录top logits。`p2-logits-probe` 的20路并发再次20/20完全一致；敏感请求单独prefill时，index=34 的 incomplete/just logits为18.25/18.0。

原 P1 commit 88d3eab9e 的20路并发对照也完成20/20；但用单个HTTP批请求固定把128/256/512-token三个prompt同时准入（`p1-fixed-batch-probe`），P1自身便在同一index=34从 `incomplete` 改选 `just`，两个logits并列18.125。该批为896 tokens，和发生分歧的P2 prefilling形状相同；从该分叉到index=50，两者也一致。故已有原P1在没有新迁移逻辑的条件下复现同一分叉，支持“默认BF16/FlashInfer对批形状敏感”的判断，不能承诺跨任意batch形状逐token恒等。迁移数据索引检查、CPU split/merge/penalizer检查均通过。

这是一项明确的精度边界：通过了P1相同串行方法的严格比对和能触发迁移的完全一致场景，但并发复验并非每次都与串行基准逐token相同。完整保留各轮输出和首分叉位置。正式性能配置仍保持与P1相同BF16/FlashInfer；未为追求位级一致性关闭radix或替换attention后端。


## 最终测试结果与证据

- CPU：18 passed，`p2-unit.log`；基准客户端原7项测试通过，`bench-unit.log`。Black、isort、compileall、git diff --check 通过。
- P1原串行正确性：全特性20/20、1280/1280 token严格一致；并发迁移的完整一致场景和一次19/20复验分歧详见上节，不能据此宣称跨所有批形状位级确定性。
- 最新代码只读logits诊断的20路并发：20/20、1280/1280一致，四级各8条相同迁移指令；原第一次迁移测试四级各13条。各级 ADMIT 序列和迁移命令逐项对比一致。
- P2完整trace压力：最终ALP、static均为300/300、0失败、0未发送；time-scale=1，2个warmup，到达跨度97.5s。每轮输入2,195,003、输出44,909 tokens，默认全部调度特性开启，仅policy不同；双方force-close=true。全程无scheduler异常/死锁，四级idle pool/cache自检通过。
- 取消：`p2-control` 的16条取消全部成功，取消后16-token生成成功，unsupported HTTP400、flush HTTP200，退出remaining requests=0。
- 原pdmux PP1、普通PP2各基础+6并发通过：`p2-pdmux-regression`、`p2-pp-regression`。
- 全消融 smoke 通过：`p2-ablation-off` 使用static、no-rebalance、no-balance-merge、no-reserve-decode、no-alp-admission、inflight=1、FIFO，基础+6并发及池自检成功。

| policy | 成功/失败 | 完成时间(s) | TTFT mean/p50/p90/p99(s) | TPOT mean/p50/p90/p99(ms) | 输入/输出tok/s |
|---|---|---|---|---|---|
| static | 300/0 | 141.072 | 11.375 / 10.846 / 18.693 / 22.335 | 67.68 / 64.10 / 90.84 / 133.57 | 15559.44 / 318.34 |
| alp | 300/0 | 144.809 | 15.358 / 14.964 / 23.629 / 26.855 | 65.97 / 62.67 / 86.06 / 121.84 | 15157.96 / 310.13 |

这个小样本中ALP没有显示优于static的稳定收益；按要求仅记录健全性结果，不继续调参。初轮（连接复用默认配置）的static=145.421s、ALP=143.950s；结果接近且方向会变化，不能据此给出正式性能结论。

完整验收汇总：`logs/p2/acceptance-summary.json`，生成器 `scripts/p2_collect.py`。每轮 `*-result.json` 含完整服务命令/启动耗时/GPU清理快照；`*_summary.json`、`*_requests.jsonl` 是原始基准输出。各轮分歧、失败与诊断记录均保留。

`p2-alp-300-transport` 最终各stage资源：`{'0': {'free_kv': 2964, 'evictable': 312347, 'free_reqs': 128}, '1': {'free_kv': 2964, 'evictable': 312347, 'free_reqs': 128}, '2': {'free_kv': 2964, 'evictable': 312347, 'free_reqs': 128}, '3': {'free_kv': 2964, 'evictable': 312347, 'free_reqs': 128}}`；准入批次数=105，迁移指令数=0。

`p2-static-300-transport` 最终各stage资源：`{'0': {'free_kv': 3031, 'evictable': 312280, 'free_reqs': 128}, '1': {'free_kv': 3031, 'evictable': 312280, 'free_reqs': 128}, '2': {'free_kv': 3031, 'evictable': 312280, 'free_reqs': 128}, '3': {'free_kv': 3031, 'evictable': 312280, 'free_reqs': 128}}`；准入批次数=109，迁移指令数=0。


## 最终提交与清理

- P2-0：`c74f56ff1`（在P2移植前通过正确性、200条压力、取消后提交）。
- FastPP调度移植：`bd849d628`。
- 分支 `muxwise-pp`，工作树干净；均为本地commit，未push。
- 最终 `nvidia-smi` 无compute process，无SGLang服务进程。证据：`logs/p2/nvidia-smi-final.txt`、`processes-final.json`、`git-status-final.txt`。
- 基准辅助脚本和本文位于用户指定的muxpp_work目录；可选force-close修改只影响显式启用该选项的客户端。
