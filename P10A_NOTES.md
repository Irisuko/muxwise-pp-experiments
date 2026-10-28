# P10a：Mistral-Small-24B 四卡场景准备

> P10a准备与中断恢复已完成：模型双副本校验、六组新GPU预检、serial20比较和六组开发150条结果均已归档。两组PP4限时冒烟未完成20/20，两组PP正确性存在分叉，详见下文。未运行正式实验。

## 范围、模型与存储

代码起点：`f6833f24ef0587278ae799696cfe17c3ade99048`，分支 `muxwise-pp`。所有实例固定使用四张 RTX 4090（TP4 / PP4 / PP2×TP2），不测试两卡实例。四卡全部无 P2P；TP 禁用 custom allreduce。拓扑与 UUID 见 `logs/p10a/topology.txt`、`gpu-map.csv`、`p2p.json`。

来源：[ModelScope mistralai/Mistral-Small-24B-Instruct-2501](https://modelscope.cn/models/mistralai/Mistral-Small-24B-Instruct-2501)。选择 HF 格式的 10 个 `model-*.safetensors` 分片及 config/tokenizer JSON，明确排除重复的 `consolidated.safetensors`、原生格式 `tekken.json` / `params.json`。未切换备用模型。

模型配置：40 层，hidden=5120，intermediate=32768，32 个 attention heads，8 个 KV heads，head_dim=128，BF16，vocab=131072，原生上下文32768，sliding_window=null。PP4 每 stage 10 层；PP2 每 stage 20 层。模型配置显式 head_dim 与 hidden/heads 不同；本代码已有正确的 `config.head_dim` 支持。

两卡容量下限（不启动两卡试验）：index 给出BF16权重47,144,806,400 bytes；单个32k序列KV为40×8×128×2(K/V)×2(bytes)×32768=5,368,709,120 bytes，合计52,513,515,520 bytes，高于两卡实际物理容量51,514,441,728 bytes，尚未计入工作区/激活/通信/graph。证明的是本任务32k BF16场景两卡不可容纳，并非声称权重文件本身超过两卡容量。证据：`logs/p10a/two-gpu-capacity-bound.json`。

指定持久路径：`/root/autodl-pvt-A40-hdd-1/models/Mistral-Small-24B-Instruct-2501`；运行路径：`/dev/shm/models/Mistral-Small-24B-Instruct-2501`。中断前，私有盘创建models报Permission denied（`storage-check.json`、`persist-attempt.log`），因此当时仅有shm副本。容器重启后shm被清空。恢复时根分区/临时盘空间仍不足以容纳47.16GB模型，重新下载至120GiB shm；**shm不是持久存储，容器重启会丢失**。

恢复后再次实测，指定私有盘现已可写：创建目录、24-byte写入、fsync、读回全部成功，测试文件已清理，见 `logs/p10a/resume-storage-write-probe.json`。随后已完成原定持久副本，并保留自动下载兜底。最终副本验收见本文收尾部分；不能把恢复前的权限错误或旧shm校验日志当作当前存储状态。

下载日志：`logs/p10a/download*.log`；提高并发时出现连接重置，随后将并发收敛到5个分片×8条 Range 流并断点续传。所有权重来自 ModelScope；hf-mirror 仅探测备用可用性，未用其权重替换已下载分片。

`p10a_model.py manifest` 检查远端清单大小、计算每文件 SHA256并对照ModelScope原始API提供的SHA256、用 safetensors 验证文件头/偏移/张量形状，并比对 index 的全部 tensor key→shard 映射。manifest 保存在 `configs/p10a_model_manifest.json`。启动先执行 `ensure()`：检查shm全部文件大小和safetensors/index；缺失或损坏时先尝试从完整私有盘副本补齐，以SHA256校验后原子替换。若私有盘也缺失/不可用，则自动调用ModelScope，按manifest的16个精确文件名下载（5文件×8 Range并发、最多3次命令重试），全部SHA256及safetensors/index通过后才启动服务。下载日志为 `logs/p10a/autodownload-*.log`，文件锁防止并行恢复；中断标记存在时强制完整SHA复核，避免未验证文件被下一次启动接受。`persist` 将 shm 副本写入持久盘，并重新完整校验。完整校验16/16文件通过，共47,162,178,592 bytes；363个BF16 tensor、23,572,403,200个参数。首次完整shm复核见 `logs/p10a/final-shm-sha-validation.log`（363）；该旧副本已随容器重启丢失，恢复后的校验另有独立日志。

## 最小代码适配与基线修复

- `99a756bfa`：MuxWise-PP 的 model_type 白名单从 llama 扩展为 llama/mistral。`MistralForCausalLM` 直接继承 `LlamaForCausalLM`，共享 forward_split_prefill、PP 层切分、权重加载、MLP 分块实现；没有复制或重写模型计算。原版 MuxWise（pdmux, pp=1）无对应白名单，已通过继承获得这些方法，无需改动。静态核验：`logs/p10a/model-static-audit.json`；六种拓扑均已通过31k长请求计算；具体冒烟结果见下表。
- `3285a4325`：修复上游 Chunked 的非正续块预算。开发 CPU 复现把剩余预算设为−69，原 `add_chunked_req` 直接将 extend_input_len 写为−69并加入执行列表；预算0也会提交零长批。P7b PP4 原日志有相同−69及 FlashInfer `qo_indptr` 负长度异常。复现证据：`logs/p10a/chunked-budget-reproduction.json`。
- 根因：PP 微批共享 KV pool，其他微批前进后，当前 chunk 的可用量减输出预留可变为非正；续块路径缺少普通请求已有的预算检查。修复在非正预算时保留 chunk、不生成 prefill 批，让 decode 释放容量；同时将已释放的 req_pool_idx 置 None，暂停期间仅缓存并释放前一 chunk 一次，防止重复释放请求槽。正预算路径保持原计算。
- 针对负预算、零预算、容量恢复后继续、连续暂停槽位守恒的3项 CPU 测试通过。原PP/MLP测试52项通过；修改后的合并回归结果见 `logs/p10a/final-unit.log`。客户端10项通过，见 `bench-unit.log`。

这里的负 token **已经在独立 CPU 调度复现中重现**，不冒充 P10a GPU trace 的实测复现；最终代码下的三组Chunked开发150条及预检均未出现负 `#new-token` 或FlashInfer异常；P10a没有用未修复代码重复GPU负预算试验。

## 负载

原始 Mooncake Tool&Agent：`/root/ext/mooncake/FAST25-release/traces/toolagent_trace.jsonl`。使用模型 tokenizer 重新生成；token IDs 0–999 全部为特殊/控制 token，均从采样词表排除，仅首位固定 BOS=1；普通采样范围1000–131071。保留原输入/输出长度与 hash 前缀对应关系。`build_trace.py` 原实现已满足该 tokenizer 要求，无需修改。

```bash
source /root/workspace/muxpp_work/env.sh
python /root/workspace/muxpp_work/bench/build_trace.py \
  --tokenizer /dev/shm/models/Mistral-Small-24B-Instruct-2501 \
  --start-sec 600 --duration-sec 80 --stretch 7.5 --max-total-len 32768 \
  --out /root/workspace/muxpp_work/bench/big_toolagent_600_80_x7.5.jsonl
# 开发：start-sec 1200；out 改为 dev_big_toolagent_1200_80_x7.5.jsonl
```

| 负载 | 窗口/过滤/保留请求 | 输入token | 输出token | 输入token/s | 输出token/s | 理论复用率 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 正式600–680s（仅生成/统计） | 570 / 15 / 555 | 3,997,588 | 97,918 | 6,662.647 | 163.197 | 45.9797% |
| 开发1200–1280s | 558 / 15 / 543 | 4,029,686 | 88,034 | 6,716.143 | 146.723 | 46.8078% |

速率分母均为600s（80×7.5），最后到达时间585s。理论复用率为首次出现后 hash 块 token 数/总输入，不等于实际 radix 命中率；两份 trace 均无补齐 token。全文件检查通过：长度精确、BOS=1、其余无特殊 token、总长≤32768。统计JSON在 `bench/*big*.stats.json`，生成日志在 `logs/p10a/{formal,dev}-build.log`。

## 启动、正式锁与预检方法

六个 `scripts/launch_big_*.sh` 均调用 `p10a_runner.py launch LABEL`，包含模型完整性检查和进程生命周期清理；打印展开命令用 `p10a_runner.py commands LABEL`。统一BF16、FlashInfer、CUDA graph bs=1/4/8/16/32、context32768、disable-overlap、FCFS、max-running128、cache report。TP>1禁用custom allreduce。MuxWise沿用 `configs/pdmux_4090.yml`；MuxWise-PP沿用v4全部开关、预算16384、MLP chunk4096、PP4/PP2在途6/3。Chunked chunk2048；仅TP4启用mixed chunk。

所有六个标签已接入 `run_experiment.sh`，但 `experiment` 模式无条件报 P10a FORMAL LOCK，且在模型检查/启动前退出。客户端另检查 `configs/p10a_formal.lock.json`，禁止直接回放正式big trace。锁验证见 `logs/p10a/formal-lock-check.json`。只有开发trace被 runner 白名单允许；未来正式任务需要显式更新入口和锁。

预检使用固定种子、模型普通词表生成31663-token输入、8-token输出，之后flush；冒烟使用开发前20条、time-scale0.2、warmup2、force-close、55s请求超时，总客户端运行上限60s。显存仅按这一步选择，开发前150条不参与性能调参。

正确性采用普通TP4（pp1，非pdmux，无chunk）作共同参考；20条100–8000-token输入，temperature0，ignore_eos，64输出，串行保留缓存，比较output_ids并记录各请求首个分叉（从0计数）。本版本正常 detokenizer 将 decode window IDs 放入 output_ids，不能直接作为64个生成token比较；首次采集因此被断言拒绝，保留在 reference-tp4-serial20-output-format。正确性实例统一加 --skip-tokenizer-init，输入仍由模型 tokenizer 生成；开发健全性实例不加该开关。

开发健全性固定前150条、time-scale1、warmup2、force-close、drain-timeout300；保留全部单次结果，异常仅排查启动/代码问题。预检与开发均先长请求预热并flush。

## PP4 开发故障与修复

GPU预检在微批由1增至2个请求时停滞。graph bs列表为1/4/8/16/32；CudaGraphRunner 的 PPProxyTensors 保留 padding 后的4行，AsyncDecodeTransport原先据此发送4行，但控制消息decode_bs=2，接收端只接收2行。NCCL两端count不一致。诊断事件证实stage0发送完成，而后级在graph执行前等待输入；较低mem=.74仍可复现，故不是仅降低KV可解的问题。提交 `03e2fe794` 修复：按实际microbatch大小裁剪hidden_states/residual再复制到TX slot，保留graph及v4策略。传输回归加入padding到4行的输出，涵盖PP4和PP2×TP2；全部55项回归通过。修复后mem=.76的GPU冒烟持续推进至18/20，余下两条在59.32s上限仍持续解码，协议停滞消失，未见OOM。并非20/20完成通过。

排查期间曾测试切换前stream.query保护、采集Python栈与graph前后事件；它们未解决根因，现已全部从生产代码移除，补丁与失败运行保存在logs/p10a。cuda-gdb附加因容器ptrace限制失败，不据此推断kernel状态。

## 最终启动配置与预检

显存配置在正确性/开发150条之前冻结于 `configs/p10a_frozen.json`；最终代码 `03e2fe794`。KV容量为四卡单实例可容纳的token位置数（各stage/rank持有对应分片，不乘以4）。

| 配置 | TP×PP | mem fraction | 实例KV容量 | 31k预热/s | 冒烟完成/失败/未完成 | 冒烟/s |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| big_muxwise_tp4 | 4×1 | 0.76 | 167,692 | 5.458 | 20/0/0 | 33.376 |
| big_chunked_tp4 | 4×1 | 0.90 | 252,288 | 5.635 | 20/0/0 | 33.497 |
| big_chunked_pp4 | 1×4 | 0.90 | 204,263 | 4.355 | 18/0/2 | 59.300 |
| big_chunked_pp2tp2 | 2×2 | 0.90 | 236,846 | 4.563 | 20/0/0 | 48.029 |
| big_muxpp_pp4 | 1×4 | 0.76 | 118,415 | 12.230 | 18/0/2 | 59.317 |
| big_muxpp_pp2tp2 | 2×2 | 0.78 | 162,858 | 7.876 | 20/0/0 | 50.644 |

两组PP4均为 **BOUNDED_SMOKE_INCOMPLETE**，不是20/20通过：一分钟内持续解码、无OOM/负token/协议停滞，余下两条长输出在客户端上限停止。未缩短trace输出，也未放宽一分钟限制。其余四组20/20通过。

显存选择记录：Chunked三组首个候选.90通过内存稳定性；MuxWise TP4的.84/.82/.80在长预热OOM、.78在冒烟OOM，选.76；MuxWise-PP PP4的.82/.80在长预热OOM、.78一次冒烟OOM并在其他运行遇到已修复的padding通信停滞，最终.76；PP2×TP2的.82/.80在长预热OOM，.78完整通过。诊断期间PP4还在.76/.74重现停滞，未把这些失败伪装成显存通过。所有尝试保留在logs/p10a/preflight-*和diagnostic-*。

六个启动入口：

```bash
source /root/workspace/muxpp_work/env.sh
/root/workspace/muxpp_work/scripts/launch_big_muxwise_tp4.sh
/root/workspace/muxpp_work/scripts/launch_big_chunked_tp4.sh
/root/workspace/muxpp_work/scripts/launch_big_chunked_pp4.sh
/root/workspace/muxpp_work/scripts/launch_big_chunked_pp2tp2.sh
/root/workspace/muxpp_work/scripts/launch_big_muxpp_pp4.sh
/root/workspace/muxpp_work/scripts/launch_big_muxpp_pp2tp2.sh
```

每条单独使用；脚本先检查/修复shm模型副本，再启动并管理退出清理。以下是它们的最终展开命令（同时保存于 `logs/p10a/final-commands.sh`）；直接执行展开命令前也应运行 `python scripts/p10a_model.py ensure`。

```bash
source /root/workspace/muxpp_work/env.sh

# big_chunked_tp4
env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/root/workspace/sglang/python /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /dev/shm/models/Mistral-Small-24B-Instruct-2501 --tp 4 --pp-size 1 --dtype bfloat16 --port 30000 --disable-overlap-schedule --context-length 32768 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --attention-backend flashinfer --mem-fraction-static 0.9 --cuda-graph-max-bs 32 --cuda-graph-bs 1 4 8 16 32 --disable-custom-all-reduce --chunked-prefill-size 2048 --enable-mixed-chunk

# big_chunked_pp4
env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/root/workspace/sglang/python /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /dev/shm/models/Mistral-Small-24B-Instruct-2501 --tp 1 --pp-size 4 --dtype bfloat16 --port 30000 --disable-overlap-schedule --context-length 32768 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --attention-backend flashinfer --mem-fraction-static 0.9 --cuda-graph-max-bs 32 --cuda-graph-bs 1 4 8 16 32 --chunked-prefill-size 2048

# big_chunked_pp2tp2
env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/root/workspace/sglang/python /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /dev/shm/models/Mistral-Small-24B-Instruct-2501 --tp 2 --pp-size 2 --dtype bfloat16 --port 30000 --disable-overlap-schedule --context-length 32768 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --attention-backend flashinfer --mem-fraction-static 0.9 --cuda-graph-max-bs 32 --cuda-graph-bs 1 4 8 16 32 --disable-custom-all-reduce --chunked-prefill-size 2048

# big_muxwise_tp4
env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/root/workspace/sglang/python PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /dev/shm/models/Mistral-Small-24B-Instruct-2501 --tp 4 --pp-size 1 --dtype bfloat16 --port 30000 --disable-overlap-schedule --context-length 32768 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --attention-backend flashinfer --mem-fraction-static 0.76 --cuda-graph-max-bs 32 --cuda-graph-bs 1 4 8 16 32 --disable-custom-all-reduce --enable-pdmux --chunked-prefill-size -1 --pdmux-config-path /root/workspace/muxpp_work/configs/pdmux_4090.yml

# big_muxpp_pp4
env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/root/workspace/sglang/python /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /dev/shm/models/Mistral-Small-24B-Instruct-2501 --tp 1 --pp-size 4 --dtype bfloat16 --port 30000 --disable-overlap-schedule --context-length 32768 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --enable-pdmux --chunked-prefill-size -1 --pdmux-config-path /root/workspace/muxpp_work/configs/pdmux_pp_4090_v4.yml --mem-fraction-static 0.76 --cuda-graph-max-bs 32 --cuda-graph-bs 1 4 8 16 32 --max-prefill-tokens 16384 --pdmux-pp-max-budget 16384 --pdmux-pp-max-inflight-prefill 6 --pdmux-pp-radix-eviction lru --pdmux-pp-policy alp --pdmux-pp-slo-ttft 5.0 --pdmux-pp-slo-tpot 0.1 --pdmux-pp-tpot-coeff 0.75 --pdmux-pp-reserve-decode --pdmux-pp-balance-merge --pdmux-pp-rebalance --pdmux-pp-alp-admission --pdmux-pp-rebalance-unit 8 --pdmux-pp-summary-interval 10 --pdmux-prefill-mlp-chunk-tokens 4096 --attention-backend flashinfer --pdmux-pp-latency-partition --no-pdmux-pp-runnable-partition --no-pdmux-pp-decode-aware-slices --no-pdmux-pp-adaptive-inflight --no-pdmux-pp-cpu-merge-tokens --no-pdmux-pp-lazy-admission-budget

# big_muxpp_pp2tp2
env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/root/workspace/sglang/python /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /dev/shm/models/Mistral-Small-24B-Instruct-2501 --tp 2 --pp-size 2 --dtype bfloat16 --port 30000 --disable-overlap-schedule --context-length 32768 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --enable-pdmux --chunked-prefill-size -1 --pdmux-config-path /root/workspace/muxpp_work/configs/pdmux_pp_4090_v4.yml --mem-fraction-static 0.78 --cuda-graph-max-bs 32 --cuda-graph-bs 1 4 8 16 32 --max-prefill-tokens 16384 --pdmux-pp-max-budget 16384 --pdmux-pp-max-inflight-prefill 3 --pdmux-pp-radix-eviction lru --pdmux-pp-policy alp --pdmux-pp-slo-ttft 5.0 --pdmux-pp-slo-tpot 0.1 --pdmux-pp-tpot-coeff 0.75 --pdmux-pp-reserve-decode --pdmux-pp-balance-merge --pdmux-pp-rebalance --pdmux-pp-alp-admission --pdmux-pp-rebalance-unit 8 --pdmux-pp-summary-interval 10 --pdmux-prefill-mlp-chunk-tokens 4096 --disable-custom-all-reduce --attention-backend flashinfer --pdmux-pp-latency-partition --no-pdmux-pp-runnable-partition --no-pdmux-pp-decode-aware-slices --no-pdmux-pp-adaptive-inflight --no-pdmux-pp-cpu-merge-tokens --no-pdmux-pp-lazy-admission-budget
```

## 正确性：serial20

共同参考：`logs/p10a/reference-tp4-serial20`，普通TP4/PP1、非pdmux、greedy、64输出。三组测试及参考全部在干净的 `03e2fe794` 上完成。输入完全相同，每条输出均为64 token，串行保留缓存。这里“测试执行成功”不等于“逐token一致”。

| 对比普通TP4 | 完全一致请求 | 同位token一致数 | 按请求顺序遇到的首个分叉（均从0计） |
| --- | ---: | ---: | --- |
| big_muxwise_tp4 | 20/20 | 1280/1280 | 无 |
| big_muxpp_pp4 | 9/20 | 883/1280 | 请求0，输出位置34（参考token 1321 → 实际 1799） |
| big_muxpp_pp2tp2 | 9/20 | 922/1280 | 请求0，输出位置8（参考token 1531 → 实际 17231） |

MuxWise TP4达到逐token一致；两种MuxWise-PP未达到逐token一致。本任务按要求记录分叉，没有将跨TP规模的BF16结果差异直接归因于舍入，也未做逐层logit溯源或据此宣称数值等价。每条请求的首个输出分叉如下，`—`表示64个token全部一致：

| 请求index | 输入长度 | MuxWise TP4 | MuxWise-PP PP4 | MuxWise-PP PP2×TP2 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 100 | — | 34 | 8 |
| 1 | 128 | — | 2 | 25 |
| 2 | 256 | — | 0 | — |
| 3 | 512 | — | 16 | 16 |
| 4 | 768 | — | 63 | 63 |
| 5 | 1000 | — | — | — |
| 6 | 1500 | — | — | — |
| 7 | 2000 | — | 33 | — |
| 8 | 2500 | — | — | — |
| 9 | 3000 | — | 25 | — |
| 10 | 3500 | — | — | 55 |
| 11 | 4000 | — | — | — |
| 12 | 4500 | — | — | 35 |
| 13 | 5000 | — | — | — |
| 14 | 5500 | — | — | 52 |
| 15 | 6000 | — | 28 | 26 |
| 16 | 6500 | — | — | 19 |
| 17 | 7000 | — | 38 | — |
| 18 | 7500 | — | 27 | 1 |
| 19 | 8000 | — | 26 | 26 |

完整prompt、64-token输出和响应保存在 `logs/p10a/correctness-*/serial20*.json`；比较汇总为 `logs/p10a/summary.json` 的correctness字段，`correctness-progress.json` 已刷新为完整结果。

## 开发健全性：六组各一次150条

后台队列在会话网络中断后继续运行，于2026-10-22 04:34:28 UTC（本地12:34:28）结束。六组均为 **150完成 / 0失败 / 0未完成**，没有scheduler异常、CUDA异常或负new-token；使用 `03e2fe794`，各run的source.patch均为空。运行命令严格为开发trace前150条、time-scale1、warmup2、force-close、drain-timeout300，见 `logs/p10a/development-command-audit.json`。恢复时只读取这些结果，不重跑150条或用它们调参。

| 配置 | 耗时/s | 输入token/s | 输出token/s | 实测复用率 | 主SLO通过 |
| --- | ---: | ---: | ---: | ---: | ---: |
| big_muxwise_tp4 | 194.573 | 5563.275 | 116.568 | 47.538% | 60/150 |
| big_chunked_tp4 | 197.838 | 5471.446 | 114.644 | 47.585% | 26/150 |
| big_chunked_pp4 | 233.854 | 4628.786 | 96.988 | 45.410% | 6/150 |
| big_chunked_pp2tp2 | 204.542 | 5292.124 | 110.887 | 47.585% | 32/150 |
| big_muxpp_pp4 | 296.053 | 3656.311 | 76.611 | 37.568% | 1/150 |
| big_muxpp_pp2tp2 | 202.049 | 5357.417 | 112.255 | 44.233% | 50/150 |

耗时和吞吐包括最终排空；实测复用率为cached_tokens/总输入token。主SLO为每请求TTFT≤5s且TPOT≤100ms。这些是指定开发窗口的单次记录，不是正式实验结果。

| 配置 | TTFT p50/s | TTFT p90/s | TPOT p50/ms | TPOT p90/ms | TPOT p99/ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| big_muxwise_tp4 | 4.803 | 14.708 | 28.969 | 138.032 | 213.248 |
| big_chunked_tp4 | 3.548 | 9.142 | 309.046 | 326.210 | 340.804 |
| big_chunked_pp4 | 3.151 | 8.089 | 215.564 | 2899.336 | 3850.893 |
| big_chunked_pp2tp2 | 2.947 | 6.888 | 344.741 | 4140.721 | 5619.740 |
| big_muxpp_pp4 | 60.640 | 77.009 | 70.553 | 100.346 | 437.935 |
| big_muxpp_pp2tp2 | 7.599 | 13.137 | 57.785 | 76.870 | 90.758 |

PP4异常核对：修复后的 `big_muxpp_pp4` 完成150/150，TPOT p50=70.553ms、p90=100.346ms、p99=437.935ms，max=574.911ms；不能说尾延迟已满足目标。更突出的是TTFT p50=60.640s、p90=77.009s，max=107.363s，主SLO仅1/150。日志显示准入受到inflight、KV和decode reserve约束；末次累计统计分别记录109.755s、66.310s、26.318s（累计含预热/排空，不是各请求可相加的延迟分解）。证据在 `logs/p10a/muxpp-pp4-admission-review.json`。未据这些数值修改v4策略、budget、inflight或mem。

修复后的PP4限时冒烟为 `preflight-big_muxpp_pp4-m0.76-proxyfix`。该run记录的commit仍为修复前HEAD，但保存的source.patch与 `git diff 3285a4325 03e2fe794` **逐字节相同**，因此确实测试了最终修复；随后serial20及150条均直接使用已提交的03e2fe794。旧失败冒烟、诊断试验和最终成功完成开发运行有独立目录，未覆盖或混算，见 `logs/p10a/resume-audit.json`。

## 中断恢复与最终验收

当前GPU分配的UUID与首次验证不同；仍为四张RTX4090、24564MiB/卡、GPU0/1与GPU2/3分别同NUMA，所有P2P为false。首次环境保存在gpu-map.csv/topology.txt/p2p.json；恢复环境见 `logs/p10a/resume-gpu-map.csv`、`resume-topology.txt`、`resume-p2p.json`。为确认冻结配置在当前分配下可启动，下载恢复后仅补六组31k预热和20条限时冒烟；原单次开发150条保持不变。

恢复后的六组预检于2026-10-22 05:40–05:51 UTC完成，全部直接使用干净的 `03e2fe794`（source.patch为空），mem fraction不变，KV容量与首次结果完全一致。shm已经完整校验，预检与私有盘复制并行；本次仅检验启动、内存稳定性与持续输出，不把该阶段耗时当作性能对比数据。

| 配置 | mem fraction | 当前实例KV容量 | 31k预热/s | 冒烟完成/失败/未完成 | 冒烟/s |
| --- | ---: | ---: | ---: | --- | ---: |
| big_muxwise_tp4 | 0.76 | 167,692 | 5.499 | 20/0/0 | 32.855 |
| big_chunked_tp4 | 0.90 | 252,288 | 5.532 | 20/0/0 | 33.675 |
| big_chunked_pp4 | 0.90 | 204,263 | 4.349 | 18/0/2 | 59.299 |
| big_chunked_pp2tp2 | 0.90 | 236,846 | 4.562 | 20/0/0 | 48.152 |
| big_muxpp_pp4 | 0.76 | 118,415 | 12.187 | 18/0/2 | 59.311 |
| big_muxpp_pp2tp2 | 0.78 | 162,858 | 7.883 | 20/0/0 | 50.729 |

四组为PASS；两组PP4仍为BOUNDED_SMOKE_INCOMPLETE，原runner保留严格20/20断言的非零退出记录，没有改成成功。Chunked PP4未完成的两条分别观察到739/882和739/984 token，最后输出均在59.256s（截止59.299s）；MuxWise-PP PP4为513/882和718/984，最后输出59.259s、59.272s（截止59.311s）。六组均无scheduler异常、CUDA错误、OOM或负new-token。完整审查为 `logs/p10a/resume-preflight-review.json`，原始结果为 `resume-preflight-*/`；修复后的PP4已在本次新GPU分配上再次验证持续推进。

本次实际冷恢复通过启动检查的下载兜底完成：2026-10-22 05:04:18 UTC开始，05:34:22 UTC完成下载与完整校验，约30分钟。证据为 `logs/p10a/autodownload-20261022T050418Z-3901.log`、`model-recovery-20261022T050418Z.json`、`resume-model-ensure.log`；后续ensure对中断标记再次完整校验并清除标记，见 `resume-ensure-verified.log`。

指定私有盘副本于05:35 UTC开始复制，05:59 UTC前完成：16/16文件共47,162,178,592 bytes，逐文件SHA256及最后整套SHA256/safetensors/index复核全部通过，363个张量完整；持久目录同时保存 `p10a_manifest.json`。日志为 `logs/p10a/resume-persist.log`，双副本验收记录为 `resume-storage-verified.json`。当前模型同时保留在指定私有盘与shm；shm的易失性和私有盘曾出现的权限限制仍需注意，启动脚本可从完整私有盘恢复，私有盘不可用时自动重新下载并校验。

为并行完成慢速私有盘I/O与GPU复检，旧顺序队列在等待复制子进程时暂停；复制子进程退出0、独立六组复检队列退出0后，主动终止旧队列（exit137）以避免重复运行。该编排记录在 `logs/p10a/resume-pipeline.json`，各次运行日志均保留。

最终审计：`logs/p10a/resume-final-code-audit.json`确认34个基线文件未变、冻结配置未变、无新增formal-started标记；六个正式入口及直接客户端正式trace锁验证见 `resume-formal-lock-check.json`。代码最终为干净的 `03e2fe79488971af6e611af09d576f9eb61342b0`，三个修复提交仅在本地，未push。

2026-10-22 05:59:31 UTC最终清理确认：四卡GPU利用率均0%、各仅1MiB基础占用，无GPU计算进程，无P10a下载/复制/运行队列或SGLang服务残留；30000和31000–31003端口均空闲。shm完整模型保留供后续使用，证据为 `logs/p10a/resume-final-cleanup.json`。本任务未运行正式实验。
