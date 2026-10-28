# P8a：MuxWise-PP 的 TP 支持（开发验证）

代码分支 `muxwise-pp`；实现提交 `d8dfbfaba`（起点 `f91e8a18d`），仅本地提交，不 push。没有修改 RESULTS.md，没有使用正式 trace。指定正确性、取消、压力、原路径回归和唯一一轮开发300条检查均已完成。合成BF16张量的数值边界单独如实记录，见MLP探针部分。

## 设计与兼容性

- 全局唯一决策者为 stage0/tp0。它完成本控制步的请求处理、结果退休、MERGE、abort/retract、迁移、准入、required/budget 和所有 allocator 驱逐后，将完整 PPControl（含 eviction journal）通过 stage0 的 `tp_cpu_group` 广播给 TP 同伴。同伴只重放，随后每个 TP rank 沿自己的 PP 配对通道转发同一控制包。下游每个 rank 都校验 step/mb、decode batch size、退休结果 step/mb、prefix probe 和 journal 消费完整性。
- `parallel_state.pipeline_group_ranks` 提取上游已有的构组规则；PP2×TP2 的 TP 组为 `[0,1]`、`[2,3]`，decode PP 和独立 prefill PP 组均为 `[0,2]`、`[1,3]`。每个 stage 实际有16层，每个 rank 保存这16层的 TP2 权重/KV 分片；不能将本地层数写成8层。
- Prefill/decode 均按 TP rank 配对整块传送 hidden/residual。这样避免在后台通信线程增加 TP all-gather 或与前向的集合通信交错，代价是每个 TP rank 发送完整激活。每个下游 rank 有自己的两个预分配 prefill 接收槽、READY 握手和 decode RX/TX 槽；全局在途 prefill 上限为3，不乘 TP 数。
- 同一 stage 的 prefill FIFO 状态转换通过 `tp_cpu_group` 的 MIN allreduce 达成一致：所有 rank 都收到元数据、NCCL launch 完成、CUDA event 完成，才进入下一状态或下一层片。READY 和传输仍按 rank 独立执行；慢 rank 不会让快 rank 提前发起下一组 TP 集合通信。
- 每 stage 的 tp0 独立选择本 stage 的 SM 分区，通过 TP CPU 组广播；只有所有 TP rank 的本地 prefill 传输/计算都完成时才能切换。切换前同步两个旧 stream，并在 stage 内 barrier，保证同一控制点选用同一 SM 配额。不同 PP stage 可以选择不同分区。
- `pp_profile_decode` 在 TP 组内按同一分区/bs/context/shape 同步 replay，超时判断也归约，避免一个 rank 提前退出留下另一个 rank 的集合通信。RLS 可以在各 rank 本地采样，但只有 tp0 的模型参与分区决策；trace 文件和 stage summary 只由 tp0 写出。
- `set_pdmux_status` 保持原机制：prefill 前向使用独立 `pdmux_prefill_tp`，decode 使用原 TP communicator。TP>1 的 PP 强制使用已有 AsyncDecodeTransport 协议；末级仅 tp0 回传采样 token，stage0 放入后续控制包供所有 stage/rank 重放。prefill 首 token 同样只由末级 tp0 回传并由 MERGE 下发。
- Llama MLP 的 token 分块算法无需改变。所有 TP rank 使用相同批长和 chunk 边界，每块执行一次 RowParallel allreduce，仍走独立 prefill TP communicator。放开 server_args 的 TP1 限制，继续拒绝 DP attention；PP 运行时仍限制为 Llama/Python local RadixCache，原不支持请求类型和调度组合保持拒绝。
- TP1 保留配置选择的原 transport、原图、原接收槽和原 SM 策略，新增 TP helper 在 TP1 直接返回，不增加集合通信或 GPU 同步。pp=1 的原 pdmux 调度器没有修改。

## 测试与证据

日志目录：`logs/p8a/`。工具：`scripts/p8a_verify.py`、`p8a_regressions.sh`、`p8a_audit.py`、`p8a_mlp_gpu.py`、`p8a_runner.py`。

CPU：已有测试和新增测试共44项覆盖，包含真实四进程 Gloo 的 PP4/PP2×TP2 AsyncDecodeTransport、唯一 token 回传、RX/TX 槽复用、控制包 TP 广播/配对转发、准入 probe 重放、rank 完成时间不对称、分区广播和切换阻塞、通信组配对、TP1 无新增 collective、参数放开和 DP attention 拒绝。主测试日志 `unit-final.log` 为42项，新增参数/TP1测试及重复TP测试见 `unit-tp-extra.log`。启动脚本/正式锁测试2项通过，见 `runner-unit.log`。

正确性使用 BF16、FlashInfer、temperature=0、ignore_eos、64输出 token，普通TP2/PP1、非pdmux作参照：

| 测试 | 完全一致请求 | 同位 token 一致 | 首分叉位置 |
| --- | ---: | ---: | --- |
| serial20，保留 radix cache | 20/20 | 1280/1280 | 全部 null |
| cold20，每条 flush cache | 20/20 | 1280/1280 | 全部 null |

冷cache20中8条输入超过4096 token，实际触发MLP分块；热cache20本身不能证明分块覆盖。逐请求输出、输入及首分叉报告保存在 `*-result.json`、`*-prompts.json`、`audit.json`。普通TP2自身冷热cache并非完全相同：prompt index1（128 token）在输出index50分叉，index3（512 token）在输出index1分叉；所以各组使用同样cache条件的TP2参照，未隐藏数值边界。与P5b相同条件的严格一致性要求通过。

取消测试 `tp2pp2-cancel`：16/16 abort；后续16-token生成、unsupported HTTP400、flush HTTP200通过。四rank自检 `free_kv=363477 + evictable=27145 = 390622`，free_reqs=128，下游接收槽各2个全部回收。

压力测试 `tp2pp2-stress`：开发trace前200条，time-scale=0.02，warmup2、force-close、drain-timeout300；200完成/0失败/0未完成，66.164秒。四rank最终 `free_kv=5935 + evictable=384687 = 390622`，free_reqs=128，232个prefill slice、3472个decode step、653929 cached tokens完全一致，下游两槽全部回收。准入/分区日志在stage内逐项一致，无scheduler异常/OOM/泄漏。其前31,663-token随机长预热（8输出token）通过，服务端e2e 3.717s。

回归：PP4 v3/inflight6 的serial20与P5b `v3-correctness-result.json` 完全一致（20/20、1280/1280）；本次E2E中位数1.416s，对应历史1.407s，和无新增TP1通信/同步的代码检查一致，未见明显退化，但不是同机同期性能A/B。原pdmux PP1/TP1、PP1/TP2及普通PP2（无pdmux）各7条基础/并发请求全部通过。结果见 `*-regression-result.json`、`pp4-regression-comparison.json`。

## TP2 MLP块大小探针

独立合成探针使用真实 LlamaMLP 的 TP2 Column/RowParallel 层、BF16、31,663×4096 输入、14336 intermediate、固定随机权重，独立 prefill TP communicator，关闭 custom allreduce。各形状先预热2次，计时3次；使用全SM stream，不混入KV池和整模型开销。以下耗时取两个rank中的较慢者，峰值为每rank额外 allocated，包含输出/通信中间量，不是整卡显存。

| chunk tokens | RowParallel allreduce/层 | MLP耗时(ms) | 增量峰值(MiB) | max abs误差 | 超出旧容差元素占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 关闭 | 1 | 52.578 | 1546.893 | 0.00000000 | 0.000000% |
| 8192 | 4 | 53.881 | 648.000 | 0.01562500 | 0.011787% |
| 4096 | 8 | 54.084 | 448.000 | 0.02343750 | 0.006016% |
| 2048 | 16 | 54.781 | 348.000 | 0.03125000 | 0.067283% |

chunk4096相对关闭，峰值减少71.0%，耗时增加约2.9%，每层allreduce从1次变8次；2048再减少100MiB，但变为16次allreduce且数值误差更大。维持既定4096，不根据开发300条调参。8192/4096/2048两个TP rank的通信次数完全一致；每层总传输元素数量近似不变，次数增加带来额外集合通信发射开销。

首个探针沿用P5b的 `rtol=.02, atol=.002` 逐元素assert，在chunk8192失败（15287/129691648元素超界）；原日志和脚本保留为 `mlp-gpu.log`、`mlp-probe-initial.py`。同输入/权重/参数补做记录型探针，不放宽容差；三种分块均不是位级相同，旧容差也并非全部通过。误差源于BF16 GEMM形状变化及TP归约，不能将其写成浮点张量严格一致。完整模型的serial20/cold20输出token则与对应普通TP2参照全部一致。完整数据见 `mlp-tp0.json`、`mlp-tp1.json`、`mlp-gpu-recorded.log`。

## 显存预检与唯一一轮开发300条

mem fraction只尝试参考值0.71；长预热通过后停止选择，没有向上搜索容量。完整预检在干净提交 `d8dfbfaba` 上运行：就绪43.127s，每rank逻辑KV容量390622 token，8分区；31,663-token随机长预热3.739s，开发20条冒烟20/20（20.851s），flush后再长预热3.677s。两次均8输出token，无OOM。

`logs/p8a/dev300-mem071/dev300-command.json` 明确为开发trace前300条、time-scale=1、warmup2、force-close、drain-timeout300。独占标记 `logs/p8a/muxpp_pp2tp2.dev300-started.json`；仅此一轮，没有依据结果调参或重跑。

| 配置 | 完成/失败/未完成 | 主SLO | TTFT p50/p99(s) | TPOT p50/p99(ms) | 统计时长(s) |
| --- | --- | ---: | --- | --- | ---: |
| 本次MuxWise-PP PP2×TP2 | 300/0/0 | 68.00%（204/300） | 4.276 / 9.243 | 32.117 / 88.219 | 122.164 |
| P7a历史MuxWise-PP PP2×DP2开发 | 300/0/0 | 63.33% | 4.142 / 12.223 | 见P7a原始summary | 127.583 |

主SLO仍为TTFT≤5s且TPOT≤100ms。相对历史开发参照，SLO增加4.67个百分点，但TTFT p50增加0.134s（约3.2%），并未改善中位TTFT；p99较低。输出吞吐352.878 tok/s，输入吞吐17302.433 tok/s。此结果不证明优于原MuxWise TP2×DP2的正式结果，也不与不同trace的82.7%直接比较。GPU拓扑变化见下节；不作未经控制的因果结论。

结束时四rank一致：`free_kv=4531 + evictable=386091 = 390622`，free_reqs=128，下游free_receive_slots=2；379个prefill slice、11202个decode step、1071281 cached tokens一致（这些累计含启动/两次长预热/冒烟）。stage内准入/分区日志逐项一致，无scheduler异常。进程内allocator最大allocated 20283.29MiB；5秒NVML采样最大 24000MiB（包括启动/预检，不是严格连续峰值）。

原始请求和summary：`logs/p8a/dev300-mem071/dev300/`；完整命令、服务日志、GPU采样、预热输出、清理证据：`logs/p8a/dev300-mem071/`。

## 启动与正式锁

`launch_muxpp_pp2tp2.sh` 启动服务并管理进程清理；`run_experiment.sh muxpp_pp2tp2` 已接入新runner。参数直接从 `p5b_v3_serve.sh` 派生，仅改 TP2、PP2、inflight3，并显式使用 FlashInfer、关闭本机自定义 allreduce；mem fraction仍为0.71，MLP chunk4096。正式入口核验通过的开发门槛、干净且冻结的代码提交、完整命令和配置SHA256；全局GPU flock与独占 `.formal-started.json` 防止并发和失败后重跑，已有/归档结果也会拒绝重复运行。本任务不调用该入口。

当前GPU为4张RTX4090（24564MiB），TP组0+1在NUMA0、2+3在NUMA1；与P7a的GPU UUID及NUMA分布不同。精确记录见 `gpu-map.csv`、`topology.txt`。因此与历史PP2×DP2开发63.3% / TTFT p50 4.14s的比较只作为开发健全性参照，不能完全归因于并行配置。

后续正式推荐命令（准备完成后执行，本任务未执行）：

```bash
source /root/workspace/muxpp_work/env.sh
bash /root/workspace/muxpp_work/scripts/run_experiment.sh muxpp_pp2tp2 --tag formal-once
```

## 最终状态

`configs/p8a_selected.json` 保存已验证提交、配置SHA256、完整命令签名与通过的开发门槛。`scripts/p8a_runner.py commands` 可打印精确服务命令；`scripts/launch_muxpp_pp2tp2.sh` 只启动服务，`scripts/run_experiment.sh muxpp_pp2tp2` 为后续一次正式入口。

代码工作树干净，HEAD `d8dfbfaba84ba1257cc42a519c24fc6c71987ef7`，未push。11个GPU功能/正确性/回归结果均PASS；GPU计算进程为空，无服务、客户端、探针残留，30000及31000–31003测试端口均释放。新增正式运行数0、该label正式锁不存在；RESULTS.md未由本任务修改。最终证据：`logs/p8a/final-audit.json`、`nvidia-smi-final.txt`、`processes-final.txt`。
