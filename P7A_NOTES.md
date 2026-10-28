# P7a 并行配置矩阵准备与预检

本任务只执行启动、合成长请求预热、开发trace冒烟及指定的一次开发300条健全性检查。没有运行任何新正式实验。
既有四个正式运行仅作只读截断重算，原始请求、summary和正式锁的SHA256未变；当前主报告为 [RESULTS.md](RESULTS.md)。

## 环境与边界

- 环境：`source /root/workspace/muxpp_work/env.sh`；模型 `/root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct`，BF16。
- 主仓库 `/root/workspace/sglang` 固定 `f91e8a18da502106a07bb67c67dbf351a933bfd9`，分支muxwise-pp；没有修改推理/调度源码。
- PD固定 `/root/workspace/sglang-p4a` 的r2提交 `19261937a524bc67216d8b925a8ea5e50bfda0f3`。进入任务时该worktree在P5a提交 `2374302bb75443ee572927d35a309def4777c06d`，已保留为git引用 `p7a-preserved-p5a` 后切回冻结r2。没有把P5a插桩或候选传输参数带入本次PD配置。
- 统一 `--disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --enable-cache-report`。除明确记录的限制，采用FlashInfer和CUDA graph。
- 普通多实例采用wheel sglang-router/cache_aware；PD采用冻结worktree的mini_lb PD/random模式，r2全部Mooncake TCP环境变量与参数保持，详见下方精确命令。
- 每个新TP实例显式关闭自定义allreduce，使用NCCL；实际日志含 `via SHM/direct/direct`。这是无P2P的硬件条件，不作为不支持原因。

## GPU拓扑和映射

[nvidia-smi topo -m](logs/p7a/topology.txt) · [NVML UUID与PCI映射](logs/p7a/gpu-uuid.csv) · [CUDA UUID/P2P核验](logs/p7a/cuda-device-map.json)

| nvidia-smi / CUDA GPU | UUID | NUMA |
| --- | --- | --- |
| 0 | GPU-52777e54-80e7-9357-a595-86ce2678536f | 0 |
| 1 | GPU-5230ed6b-99d1-5f7f-b746-707c95ab50ce | 1 |
| 2 | GPU-19fac1b0-72dc-17f3-8aa1-132c6bee8183 | 1 |
| 3 | GPU-5e84f85f-d7eb-bd99-4492-85dac7c020a2 | 1 |

所有卡对 CUDA P2P=false。双卡实例分组为GPU1+2（NUMA1）与GPU0+3（跨NUMA，四卡中只有一张在NUMA0，因此无法避免）。单个四卡实例顺序为1,2,0,3；TP2×PP2的第一个TP组在NUMA1。TP1四实例按GPU0/1/2/3分配；PD P实例在前、D实例在后。启动命令使用UUID，避免枚举顺序歧义。

## 配置矩阵

已有最终正式运行映射：chunked→Chunked DP4；muxwise_r2→MuxWise DP4；sglang_pd_r2→PD 2P2D(TP1)；muxwise_pp_v3→MuxWise-PP PP4。新配置仅有预检状态，不能当成正式性能结果。

| 新label | TP×PP×实例数 | mem fraction | 每实例逻辑KV容量(token) | 结论 | 说明 |
| --- | --- | --- | --- | --- | --- |
| chunked_tp2dp2 | 2×1×2 | 0.88 | worker0: 207118; worker1: 207062 | 通过 | FlashInfer + graph32；mixed chunk开启 |
| chunked_pp2dp2 | 1×2×2 | 0.88 | worker0: 191374; worker1: 191318 | 通过 | PP不支持mixed chunk，已关闭；FlashInfer + graph32 |
| chunked_tp4 | 4×1×1 | 0.88 | worker0: 536476 | 通过 | FlashInfer + graph32；mixed chunk开启 |
| chunked_pp4 | 1×4×1 | 0.88 | worker0: 489244 | 通过 | PP不支持mixed chunk，已关闭；FlashInfer + graph32 |
| chunked_tp2pp2 | 2×2×1 | 0.88 | worker0: 520522 | 通过 | PP不支持mixed chunk，已关闭；FlashInfer + graph32 |
| muxwise_tp2dp2 | 2×1×2 | 0.76 | worker0: 161473; worker1: 161375 | 通过 | 0.826/0.80长预热OOM；0.78保留缓存复查OOM；0.76通过；FlashInfer + graph32，无MLP分块 |
| muxwise_tp4 | 4×1×1 | 0.8 | worker0: 475809 | 通过 | 0.826长预热OOM；0.80通过；FlashInfer + graph32，无MLP分块 |
| pd_1p1d_tp2 | 2×1×2 | 0.88 | worker0: 207118; worker1: 207062 | 通过 | P由源码关闭graph，D保留graph32；Mooncake TCP |
| pd_3p1d | 1×1×4 | 0.9 | worker0: 46986; worker1: 46986; worker2: 46986; worker3: 46986 | 通过 | 0.88冒烟18/1/1；0.90提高KV容量后20/0/0；P graph自动关闭 |
| pd_1p3d | 1×1×4 | 0.88 | worker0: 43196; worker1: 43197; worker2: 43196; worker3: 43196 | 通过 | P由源码关闭graph，D保留graph32；Mooncake TCP |
| muxpp_pp2dp2 | 1×2×2 | 0.71 | worker0: 126501; worker1: 126411 | 通过 | inflight=3，其余v3参数保持；开发300条全部完成，未调参 |

容量来自健康就绪后的get_server_info.max_total_num_tokens；TP/PP各rank存储同一逻辑KV的分片，不能乘rank数。DP副本则分别列出，不把PP stage容量相加。

支持范围和例外：

- Chunked的PP2DP2、PP4、TP2PP2关闭mixed chunk；冻结源码server_args.check_server_args明确断言PP不能与mixed chunk并用。其它Chunked参数仍为chunk2048。
- 原MuxWise的PP组合不适用；TP2/TP4不启用MLP token分块，因为该功能只支持TP1。显存选择仅改变mem fraction，若有其它例外在表中记录。
- PD prefill由既有源码自动关闭CUDA graph，decode保留graph；decode的radix也由源码强制关闭（chunk cache）。这些是沿用r2的条件。
- MuxWise-PP只支持TP1；TP2DP2、TP4、TP2PP2均记为不支持。PP1DP4回到原MuxWise路径，不作为独立MuxWise-PP配置。新增PP2DP2为两个PP2实例，inflight=3（6/4×2），其余v3参数保持。

## 显存选择过程

遵循P3b SELECTION方法：Chunked/PD从0.88、MuxWise从已有0.826、MuxWise-PP从已有0.71开始；只在启动/长预热/≤1分钟冒烟失败后依据显存或KV准入证据调整（MuxWise下调，PD3P1D尝试增加KV容量）。通过参考值即可停止，不搜索极限，不用正式trace调参。MuxWise另在冒烟后清cache并复查长请求，防止完整kernel加载后的显存变化遗漏。

长预热：固定随机普通ID，31,663输入token、8输出token，每个普通模型实例分别执行；PD用与mini_lb相同的bootstrap协议显式覆盖所有P/D配对，避免随机路由遗漏实例。预热后清cache。
预检脚本复核：早期MuxWise复查在冒烟后保留缓存直接发第二次长请求（TP4 mem0.80通过，TP2DP2 mem0.78在worker1 OOM）。后续协议v2补齐P3b的二次长预热前flush步骤，写入long-warmup-after-smoke-cache-flush.json；旧日志未覆盖，显存选择保留对已观察OOM的余量。
冒烟只读 `bench/dev_toolagent_1200_240_x2.5.jsonl` 前20条，`--time-scale 0.2 --warmup 2 --force-close`；为满足≤1分钟，客户端timeout与drain-timeout初轮为50秒，后续对齐P3b为55秒（实际值逐候选记录在smoke-command.json中），外层硬上限60秒。正式与开发300的排空均为300秒。

| 候选日志 | mem | backend / graph | 长预热通过数 | 冒烟完成/未完/失败；秒 | 冒烟后长预热数 | 结论/证据 |
| --- | --- | --- | --- | --- | --- | --- |
| [chunked_tp2dp2/mem088](logs/p7a/chunked_tp2dp2/mem088) | 0.88 | flashinfer / max-bs32 | 2 | 20/0/0; 18.470 | 0 | 通过 |
| [chunked_pp2dp2/mem088](logs/p7a/chunked_pp2dp2/mem088) | 0.88 | flashinfer / max-bs32 | 2 | 20/0/0; 28.059 | 0 | 通过 |
| [chunked_tp4/mem088](logs/p7a/chunked_tp4/mem088) | 0.88 | flashinfer / max-bs32 | 1 | 20/0/0; 15.700 | 0 | 通过 |
| [chunked_pp4/mem088](logs/p7a/chunked_pp4/mem088) | 0.88 | flashinfer / max-bs32 | 1 | 20/0/0; 30.939 | 0 | 通过 |
| [chunked_tp2pp2/mem088](logs/p7a/chunked_tp2pp2/mem088) | 0.88 | flashinfer / max-bs32 | 1 | 20/0/0; 20.635 | 0 | 通过 |
| [muxwise_tp2dp2/mem076](logs/p7a/muxwise_tp2dp2/mem076) | 0.76 | flashinfer / max-bs32 | 2 | 20/0/0; 17.809 | 2 | 通过 |
| [muxwise_tp2dp2/mem078](logs/p7a/muxwise_tp2dp2/mem078) | 0.78 | flashinfer / max-bs32 | 2 | 20/0/0; 17.321 | 1 | CUDA out of memory. Tried to allocate 248.00 MiB. GPU 0 has a total capacity of 23.52 GiB of which 95.88 MiB is free. |
| [muxwise_tp2dp2/mem080](logs/p7a/muxwise_tp2dp2/mem080) | 0.8 | flashinfer / max-bs32 | 0 | 未进入/未完成 | 0 | CUDA out of memory. Tried to allocate 248.00 MiB. GPU 0 has a total capacity of 23.52 GiB of which 191.88 MiB is free. |
| [muxwise_tp2dp2/mem0826](logs/p7a/muxwise_tp2dp2/mem0826) | 0.826 | flashinfer / max-bs32 | 0 | 未进入/未完成 | 0 | CUDA out of memory. Tried to allocate 434.00 MiB. GPU 0 has a total capacity of 23.52 GiB of which 251.88 MiB is free. |
| [muxwise_tp4/mem080](logs/p7a/muxwise_tp4/mem080) | 0.8 | flashinfer / max-bs32 | 1 | 20/0/0; 14.801 | 1 | 通过 |
| [muxwise_tp4/mem0826](logs/p7a/muxwise_tp4/mem0826) | 0.826 | flashinfer / max-bs32 | 0 | 未进入/未完成 | 0 | CUDA out of memory. Tried to allocate 248.00 MiB. GPU 3 has a total capacity of 23.52 GiB of which 163.81 MiB is free. |
| [pd_1p1d_tp2/mem088](logs/p7a/pd_1p1d_tp2/mem088) | 0.88 | flashinfer / max-bs32 | 1 | 20/0/0; 26.318 | 0 | 通过 |
| [pd_3p1d/mem088](logs/p7a/pd_3p1d/mem088) | 0.88 | flashinfer / max-bs32 | 3 | 18/1/1; 53.001 | 0 | AssertionError('smoke gate failed') |
| [pd_3p1d/mem090](logs/p7a/pd_3p1d/mem090) | 0.9 | flashinfer / max-bs32 | 3 | 20/0/0; 48.854 | 0 | 通过 |
| [pd_1p3d/mem088](logs/p7a/pd_1p3d/mem088) | 0.88 | flashinfer / max-bs32 | 3 | 20/0/0; 44.538 | 0 | 通过 |
| [muxpp_pp2dp2/mem071](logs/p7a/muxpp_pp2dp2/mem071) | 0.71 | flashinfer / max-bs32 | 2 | 20/0/0; 28.768 | 0 | 通过 |

## 每个配置的最终启动命令

以下命令为已保存候选的实际argv/env；`launch_<label>.sh`由统一runner启动整个集群并负责清理。每个脚本默认采用 `configs/p7a_selected.json` 的选定参数。候选中不支持的配置，启动入口会给出原因并拒绝启动。
每个配置已经接入 `scripts/run_experiment.sh <label>`，本任务没有调用该入口。入口校验冻结源码、通过的预检门槛及完整命令指纹；全局GPU flock防止矩阵并发占卡，正式开始前用排他创建 `<label>.formal-started.json` 防止重复正式运行（失败也不重试）。

### chunked_tp2dp2

启动脚本：[launch_chunked_tp2dp2.sh](scripts/launch_chunked_tp2dp2.sh)；选择门槛：[mem088](logs/p7a/chunked_tp2dp2/mem088)。

```bash
source /root/workspace/muxpp_work/env.sh
env CUDA_VISIBLE_DEVICES=GPU-5230ed6b-99d1-5f7f-b746-707c95ab50ce,GPU-19fac1b0-72dc-17f3-8aa1-132c6bee8183 PYTHONPATH=/root/workspace/sglang/python NCCL_DEBUG=INFO /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --tp 2 --pp-size 1 --dtype bfloat16 --disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --port 31000 --mem-fraction-static 0.88 --attention-backend flashinfer --cuda-graph-max-bs 32 --disable-custom-all-reduce --chunked-prefill-size 2048 --enable-mixed-chunk
env CUDA_VISIBLE_DEVICES=GPU-52777e54-80e7-9357-a595-86ce2678536f,GPU-5e84f85f-d7eb-bd99-4492-85dac7c020a2 PYTHONPATH=/root/workspace/sglang/python NCCL_DEBUG=INFO /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --tp 2 --pp-size 1 --dtype bfloat16 --disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --port 31001 --mem-fraction-static 0.88 --attention-backend flashinfer --cuda-graph-max-bs 32 --disable-custom-all-reduce --chunked-prefill-size 2048 --enable-mixed-chunk
env /root/venvs/sgl/bin/python -m sglang_router.launch_router --host 127.0.0.1 --port 30000 --policy cache_aware --request-timeout-secs 3600 --max-concurrent-requests 4096 --log-level info --worker-urls http://127.0.0.1:31000 http://127.0.0.1:31001
```

### chunked_pp2dp2

启动脚本：[launch_chunked_pp2dp2.sh](scripts/launch_chunked_pp2dp2.sh)；选择门槛：[mem088](logs/p7a/chunked_pp2dp2/mem088)。

```bash
source /root/workspace/muxpp_work/env.sh
env CUDA_VISIBLE_DEVICES=GPU-5230ed6b-99d1-5f7f-b746-707c95ab50ce,GPU-19fac1b0-72dc-17f3-8aa1-132c6bee8183 PYTHONPATH=/root/workspace/sglang/python NCCL_DEBUG=INFO /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --tp 1 --pp-size 2 --dtype bfloat16 --disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --port 31000 --mem-fraction-static 0.88 --attention-backend flashinfer --cuda-graph-max-bs 32 --chunked-prefill-size 2048
env CUDA_VISIBLE_DEVICES=GPU-52777e54-80e7-9357-a595-86ce2678536f,GPU-5e84f85f-d7eb-bd99-4492-85dac7c020a2 PYTHONPATH=/root/workspace/sglang/python NCCL_DEBUG=INFO /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --tp 1 --pp-size 2 --dtype bfloat16 --disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --port 31001 --mem-fraction-static 0.88 --attention-backend flashinfer --cuda-graph-max-bs 32 --chunked-prefill-size 2048
env /root/venvs/sgl/bin/python -m sglang_router.launch_router --host 127.0.0.1 --port 30000 --policy cache_aware --request-timeout-secs 3600 --max-concurrent-requests 4096 --log-level info --worker-urls http://127.0.0.1:31000 http://127.0.0.1:31001
```

### chunked_tp4

启动脚本：[launch_chunked_tp4.sh](scripts/launch_chunked_tp4.sh)；选择门槛：[mem088](logs/p7a/chunked_tp4/mem088)。

```bash
source /root/workspace/muxpp_work/env.sh
env CUDA_VISIBLE_DEVICES=GPU-5230ed6b-99d1-5f7f-b746-707c95ab50ce,GPU-19fac1b0-72dc-17f3-8aa1-132c6bee8183,GPU-52777e54-80e7-9357-a595-86ce2678536f,GPU-5e84f85f-d7eb-bd99-4492-85dac7c020a2 PYTHONPATH=/root/workspace/sglang/python NCCL_DEBUG=INFO /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --tp 4 --pp-size 1 --dtype bfloat16 --disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --port 30000 --mem-fraction-static 0.88 --attention-backend flashinfer --cuda-graph-max-bs 32 --disable-custom-all-reduce --chunked-prefill-size 2048 --enable-mixed-chunk
```

### chunked_pp4

启动脚本：[launch_chunked_pp4.sh](scripts/launch_chunked_pp4.sh)；选择门槛：[mem088](logs/p7a/chunked_pp4/mem088)。

```bash
source /root/workspace/muxpp_work/env.sh
env CUDA_VISIBLE_DEVICES=GPU-5230ed6b-99d1-5f7f-b746-707c95ab50ce,GPU-19fac1b0-72dc-17f3-8aa1-132c6bee8183,GPU-52777e54-80e7-9357-a595-86ce2678536f,GPU-5e84f85f-d7eb-bd99-4492-85dac7c020a2 PYTHONPATH=/root/workspace/sglang/python NCCL_DEBUG=INFO /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --tp 1 --pp-size 4 --dtype bfloat16 --disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --port 30000 --mem-fraction-static 0.88 --attention-backend flashinfer --cuda-graph-max-bs 32 --chunked-prefill-size 2048
```

### chunked_tp2pp2

启动脚本：[launch_chunked_tp2pp2.sh](scripts/launch_chunked_tp2pp2.sh)；选择门槛：[mem088](logs/p7a/chunked_tp2pp2/mem088)。

```bash
source /root/workspace/muxpp_work/env.sh
env CUDA_VISIBLE_DEVICES=GPU-5230ed6b-99d1-5f7f-b746-707c95ab50ce,GPU-19fac1b0-72dc-17f3-8aa1-132c6bee8183,GPU-52777e54-80e7-9357-a595-86ce2678536f,GPU-5e84f85f-d7eb-bd99-4492-85dac7c020a2 PYTHONPATH=/root/workspace/sglang/python NCCL_DEBUG=INFO /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --tp 2 --pp-size 2 --dtype bfloat16 --disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --port 30000 --mem-fraction-static 0.88 --attention-backend flashinfer --cuda-graph-max-bs 32 --disable-custom-all-reduce --chunked-prefill-size 2048
```

### muxwise_tp2dp2

启动脚本：[launch_muxwise_tp2dp2.sh](scripts/launch_muxwise_tp2dp2.sh)；选择门槛：[mem076](logs/p7a/muxwise_tp2dp2/mem076)。

```bash
source /root/workspace/muxpp_work/env.sh
env CUDA_VISIBLE_DEVICES=GPU-5230ed6b-99d1-5f7f-b746-707c95ab50ce,GPU-19fac1b0-72dc-17f3-8aa1-132c6bee8183 PYTHONPATH=/root/workspace/sglang/python NCCL_DEBUG=INFO PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --tp 2 --pp-size 1 --dtype bfloat16 --disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --port 31000 --mem-fraction-static 0.76 --attention-backend flashinfer --cuda-graph-max-bs 32 --disable-custom-all-reduce --enable-pdmux --chunked-prefill-size -1 --pdmux-config-path /root/workspace/muxpp_work/configs/pdmux_4090.yml
env CUDA_VISIBLE_DEVICES=GPU-52777e54-80e7-9357-a595-86ce2678536f,GPU-5e84f85f-d7eb-bd99-4492-85dac7c020a2 PYTHONPATH=/root/workspace/sglang/python NCCL_DEBUG=INFO PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --tp 2 --pp-size 1 --dtype bfloat16 --disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --port 31001 --mem-fraction-static 0.76 --attention-backend flashinfer --cuda-graph-max-bs 32 --disable-custom-all-reduce --enable-pdmux --chunked-prefill-size -1 --pdmux-config-path /root/workspace/muxpp_work/configs/pdmux_4090.yml
env /root/venvs/sgl/bin/python -m sglang_router.launch_router --host 127.0.0.1 --port 30000 --policy cache_aware --request-timeout-secs 3600 --max-concurrent-requests 4096 --log-level info --worker-urls http://127.0.0.1:31000 http://127.0.0.1:31001
```

### muxwise_tp4

启动脚本：[launch_muxwise_tp4.sh](scripts/launch_muxwise_tp4.sh)；选择门槛：[mem080](logs/p7a/muxwise_tp4/mem080)。

```bash
source /root/workspace/muxpp_work/env.sh
env CUDA_VISIBLE_DEVICES=GPU-5230ed6b-99d1-5f7f-b746-707c95ab50ce,GPU-19fac1b0-72dc-17f3-8aa1-132c6bee8183,GPU-52777e54-80e7-9357-a595-86ce2678536f,GPU-5e84f85f-d7eb-bd99-4492-85dac7c020a2 PYTHONPATH=/root/workspace/sglang/python NCCL_DEBUG=INFO PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --tp 4 --pp-size 1 --dtype bfloat16 --disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --port 30000 --mem-fraction-static 0.8 --attention-backend flashinfer --cuda-graph-max-bs 32 --disable-custom-all-reduce --enable-pdmux --chunked-prefill-size -1 --pdmux-config-path /root/workspace/muxpp_work/configs/pdmux_4090.yml
```

### pd_1p1d_tp2

启动脚本：[launch_pd_1p1d_tp2.sh](scripts/launch_pd_1p1d_tp2.sh)；选择门槛：[mem088](logs/p7a/pd_1p1d_tp2/mem088)。

```bash
source /root/workspace/muxpp_work/env.sh
env CUDA_VISIBLE_DEVICES=GPU-5230ed6b-99d1-5f7f-b746-707c95ab50ce,GPU-19fac1b0-72dc-17f3-8aa1-132c6bee8183 PYTHONPATH=/root/workspace/sglang-p4a/python NCCL_DEBUG=INFO MC_FORCE_TCP=1 MC_TCP_ENABLE_CONNECTION_POOL=1 MC_TCP_LANES_PER_PEER=8 MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER=65535 MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER=65535 MC_TCP_ADMISSION_TIMEOUT_MS=60000 MC_TCP_SLICE_SIZE=1048576 MC_TRANSFER_TIMEOUT=300 MC_TCP_STATUS_TIMEOUT_SEC=120 MC_TCP_PROGRESS_TIMEOUT_SEC=120 SGLANG_DISAGGREGATION_QUEUE_SIZE=4 SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=4 SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600 GLOG_v=1 /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --tp 2 --pp-size 1 --dtype bfloat16 --disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --port 31000 --mem-fraction-static 0.88 --attention-backend flashinfer --cuda-graph-max-bs 32 --disable-custom-all-reduce --disaggregation-mode prefill --disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port 8998 --chunked-prefill-size 4096
env CUDA_VISIBLE_DEVICES=GPU-52777e54-80e7-9357-a595-86ce2678536f,GPU-5e84f85f-d7eb-bd99-4492-85dac7c020a2 PYTHONPATH=/root/workspace/sglang-p4a/python NCCL_DEBUG=INFO MC_FORCE_TCP=1 MC_TCP_ENABLE_CONNECTION_POOL=1 MC_TCP_LANES_PER_PEER=8 MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER=65535 MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER=65535 MC_TCP_ADMISSION_TIMEOUT_MS=60000 MC_TCP_SLICE_SIZE=1048576 MC_TRANSFER_TIMEOUT=300 MC_TCP_STATUS_TIMEOUT_SEC=120 MC_TCP_PROGRESS_TIMEOUT_SEC=120 SGLANG_DISAGGREGATION_QUEUE_SIZE=4 SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=4 SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600 GLOG_v=1 /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --tp 2 --pp-size 1 --dtype bfloat16 --disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --port 31001 --mem-fraction-static 0.88 --attention-backend flashinfer --cuda-graph-max-bs 32 --disable-custom-all-reduce --disaggregation-mode decode --disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port 8999
env PYTHONPATH=/root/workspace/sglang-p4a/sgl-router/py_src:/root/workspace/sglang-p4a/python /root/venvs/sgl/bin/python -m sglang_router.launch_router --host 127.0.0.1 --port 30000 --policy random --request-timeout-secs 3600 --max-concurrent-requests 4096 --log-level info --mini-lb --pd-disaggregation --prefill http://127.0.0.1:31000 8998 --decode http://127.0.0.1:31001
```

### pd_3p1d

启动脚本：[launch_pd_3p1d.sh](scripts/launch_pd_3p1d.sh)；选择门槛：[mem090](logs/p7a/pd_3p1d/mem090)。

```bash
source /root/workspace/muxpp_work/env.sh
env CUDA_VISIBLE_DEVICES=GPU-52777e54-80e7-9357-a595-86ce2678536f PYTHONPATH=/root/workspace/sglang-p4a/python NCCL_DEBUG=INFO MC_FORCE_TCP=1 MC_TCP_ENABLE_CONNECTION_POOL=1 MC_TCP_LANES_PER_PEER=8 MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER=65535 MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER=65535 MC_TCP_ADMISSION_TIMEOUT_MS=60000 MC_TCP_SLICE_SIZE=1048576 MC_TRANSFER_TIMEOUT=300 MC_TCP_STATUS_TIMEOUT_SEC=120 MC_TCP_PROGRESS_TIMEOUT_SEC=120 SGLANG_DISAGGREGATION_QUEUE_SIZE=4 SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=4 SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600 GLOG_v=1 /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --tp 1 --pp-size 1 --dtype bfloat16 --disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --port 31000 --mem-fraction-static 0.9 --attention-backend flashinfer --cuda-graph-max-bs 32 --disaggregation-mode prefill --disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port 8998 --chunked-prefill-size 4096
env CUDA_VISIBLE_DEVICES=GPU-5230ed6b-99d1-5f7f-b746-707c95ab50ce PYTHONPATH=/root/workspace/sglang-p4a/python NCCL_DEBUG=INFO MC_FORCE_TCP=1 MC_TCP_ENABLE_CONNECTION_POOL=1 MC_TCP_LANES_PER_PEER=8 MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER=65535 MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER=65535 MC_TCP_ADMISSION_TIMEOUT_MS=60000 MC_TCP_SLICE_SIZE=1048576 MC_TRANSFER_TIMEOUT=300 MC_TCP_STATUS_TIMEOUT_SEC=120 MC_TCP_PROGRESS_TIMEOUT_SEC=120 SGLANG_DISAGGREGATION_QUEUE_SIZE=4 SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=4 SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600 GLOG_v=1 /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --tp 1 --pp-size 1 --dtype bfloat16 --disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --port 31001 --mem-fraction-static 0.9 --attention-backend flashinfer --cuda-graph-max-bs 32 --disaggregation-mode prefill --disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port 8999 --chunked-prefill-size 4096
env CUDA_VISIBLE_DEVICES=GPU-19fac1b0-72dc-17f3-8aa1-132c6bee8183 PYTHONPATH=/root/workspace/sglang-p4a/python NCCL_DEBUG=INFO MC_FORCE_TCP=1 MC_TCP_ENABLE_CONNECTION_POOL=1 MC_TCP_LANES_PER_PEER=8 MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER=65535 MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER=65535 MC_TCP_ADMISSION_TIMEOUT_MS=60000 MC_TCP_SLICE_SIZE=1048576 MC_TRANSFER_TIMEOUT=300 MC_TCP_STATUS_TIMEOUT_SEC=120 MC_TCP_PROGRESS_TIMEOUT_SEC=120 SGLANG_DISAGGREGATION_QUEUE_SIZE=4 SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=4 SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600 GLOG_v=1 /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --tp 1 --pp-size 1 --dtype bfloat16 --disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --port 31002 --mem-fraction-static 0.9 --attention-backend flashinfer --cuda-graph-max-bs 32 --disaggregation-mode prefill --disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port 9000 --chunked-prefill-size 4096
env CUDA_VISIBLE_DEVICES=GPU-5e84f85f-d7eb-bd99-4492-85dac7c020a2 PYTHONPATH=/root/workspace/sglang-p4a/python NCCL_DEBUG=INFO MC_FORCE_TCP=1 MC_TCP_ENABLE_CONNECTION_POOL=1 MC_TCP_LANES_PER_PEER=8 MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER=65535 MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER=65535 MC_TCP_ADMISSION_TIMEOUT_MS=60000 MC_TCP_SLICE_SIZE=1048576 MC_TRANSFER_TIMEOUT=300 MC_TCP_STATUS_TIMEOUT_SEC=120 MC_TCP_PROGRESS_TIMEOUT_SEC=120 SGLANG_DISAGGREGATION_QUEUE_SIZE=4 SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=4 SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600 GLOG_v=1 /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --tp 1 --pp-size 1 --dtype bfloat16 --disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --port 31003 --mem-fraction-static 0.9 --attention-backend flashinfer --cuda-graph-max-bs 32 --disaggregation-mode decode --disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port 9001
env PYTHONPATH=/root/workspace/sglang-p4a/sgl-router/py_src:/root/workspace/sglang-p4a/python /root/venvs/sgl/bin/python -m sglang_router.launch_router --host 127.0.0.1 --port 30000 --policy random --request-timeout-secs 3600 --max-concurrent-requests 4096 --log-level info --mini-lb --pd-disaggregation --prefill http://127.0.0.1:31000 8998 --prefill http://127.0.0.1:31001 8999 --prefill http://127.0.0.1:31002 9000 --decode http://127.0.0.1:31003
```

### pd_1p3d

启动脚本：[launch_pd_1p3d.sh](scripts/launch_pd_1p3d.sh)；选择门槛：[mem088](logs/p7a/pd_1p3d/mem088)。

```bash
source /root/workspace/muxpp_work/env.sh
env CUDA_VISIBLE_DEVICES=GPU-52777e54-80e7-9357-a595-86ce2678536f PYTHONPATH=/root/workspace/sglang-p4a/python NCCL_DEBUG=INFO MC_FORCE_TCP=1 MC_TCP_ENABLE_CONNECTION_POOL=1 MC_TCP_LANES_PER_PEER=8 MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER=65535 MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER=65535 MC_TCP_ADMISSION_TIMEOUT_MS=60000 MC_TCP_SLICE_SIZE=1048576 MC_TRANSFER_TIMEOUT=300 MC_TCP_STATUS_TIMEOUT_SEC=120 MC_TCP_PROGRESS_TIMEOUT_SEC=120 SGLANG_DISAGGREGATION_QUEUE_SIZE=4 SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=4 SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600 GLOG_v=1 /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --tp 1 --pp-size 1 --dtype bfloat16 --disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --port 31000 --mem-fraction-static 0.88 --attention-backend flashinfer --cuda-graph-max-bs 32 --disaggregation-mode prefill --disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port 8998 --chunked-prefill-size 4096
env CUDA_VISIBLE_DEVICES=GPU-5230ed6b-99d1-5f7f-b746-707c95ab50ce PYTHONPATH=/root/workspace/sglang-p4a/python NCCL_DEBUG=INFO MC_FORCE_TCP=1 MC_TCP_ENABLE_CONNECTION_POOL=1 MC_TCP_LANES_PER_PEER=8 MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER=65535 MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER=65535 MC_TCP_ADMISSION_TIMEOUT_MS=60000 MC_TCP_SLICE_SIZE=1048576 MC_TRANSFER_TIMEOUT=300 MC_TCP_STATUS_TIMEOUT_SEC=120 MC_TCP_PROGRESS_TIMEOUT_SEC=120 SGLANG_DISAGGREGATION_QUEUE_SIZE=4 SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=4 SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600 GLOG_v=1 /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --tp 1 --pp-size 1 --dtype bfloat16 --disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --port 31001 --mem-fraction-static 0.88 --attention-backend flashinfer --cuda-graph-max-bs 32 --disaggregation-mode decode --disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port 8999
env CUDA_VISIBLE_DEVICES=GPU-19fac1b0-72dc-17f3-8aa1-132c6bee8183 PYTHONPATH=/root/workspace/sglang-p4a/python NCCL_DEBUG=INFO MC_FORCE_TCP=1 MC_TCP_ENABLE_CONNECTION_POOL=1 MC_TCP_LANES_PER_PEER=8 MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER=65535 MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER=65535 MC_TCP_ADMISSION_TIMEOUT_MS=60000 MC_TCP_SLICE_SIZE=1048576 MC_TRANSFER_TIMEOUT=300 MC_TCP_STATUS_TIMEOUT_SEC=120 MC_TCP_PROGRESS_TIMEOUT_SEC=120 SGLANG_DISAGGREGATION_QUEUE_SIZE=4 SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=4 SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600 GLOG_v=1 /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --tp 1 --pp-size 1 --dtype bfloat16 --disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --port 31002 --mem-fraction-static 0.88 --attention-backend flashinfer --cuda-graph-max-bs 32 --disaggregation-mode decode --disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port 9000
env CUDA_VISIBLE_DEVICES=GPU-5e84f85f-d7eb-bd99-4492-85dac7c020a2 PYTHONPATH=/root/workspace/sglang-p4a/python NCCL_DEBUG=INFO MC_FORCE_TCP=1 MC_TCP_ENABLE_CONNECTION_POOL=1 MC_TCP_LANES_PER_PEER=8 MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER=65535 MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER=65535 MC_TCP_ADMISSION_TIMEOUT_MS=60000 MC_TCP_SLICE_SIZE=1048576 MC_TRANSFER_TIMEOUT=300 MC_TCP_STATUS_TIMEOUT_SEC=120 MC_TCP_PROGRESS_TIMEOUT_SEC=120 SGLANG_DISAGGREGATION_QUEUE_SIZE=4 SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=4 SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600 GLOG_v=1 /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --tp 1 --pp-size 1 --dtype bfloat16 --disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --port 31003 --mem-fraction-static 0.88 --attention-backend flashinfer --cuda-graph-max-bs 32 --disaggregation-mode decode --disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port 9001
env PYTHONPATH=/root/workspace/sglang-p4a/sgl-router/py_src:/root/workspace/sglang-p4a/python /root/venvs/sgl/bin/python -m sglang_router.launch_router --host 127.0.0.1 --port 30000 --policy random --request-timeout-secs 3600 --max-concurrent-requests 4096 --log-level info --mini-lb --pd-disaggregation --prefill http://127.0.0.1:31000 8998 --decode http://127.0.0.1:31001 --decode http://127.0.0.1:31002 --decode http://127.0.0.1:31003
```

### muxpp_pp2dp2

启动脚本：[launch_muxpp_pp2dp2.sh](scripts/launch_muxpp_pp2dp2.sh)；选择门槛：[mem071](logs/p7a/muxpp_pp2dp2/mem071)。

```bash
source /root/workspace/muxpp_work/env.sh
env CUDA_VISIBLE_DEVICES=GPU-5230ed6b-99d1-5f7f-b746-707c95ab50ce,GPU-19fac1b0-72dc-17f3-8aa1-132c6bee8183 PYTHONPATH=/root/workspace/sglang/python NCCL_DEBUG=INFO /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --tp 1 --pp-size 2 --dtype bfloat16 --port 31000 --disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --enable-pdmux --chunked-prefill-size -1 --pdmux-config-path /root/workspace/muxpp_work/configs/pdmux_pp_4090_v3.yml --mem-fraction-static 0.71 --cuda-graph-max-bs 32 --cuda-graph-bs 1 4 8 16 32 --max-prefill-tokens 16384 --pdmux-pp-max-budget 16384 --pdmux-pp-max-inflight-prefill 3 --pdmux-pp-radix-eviction lru --pdmux-pp-policy alp --pdmux-pp-slo-ttft 5.0 --pdmux-pp-slo-tpot 0.1 --pdmux-pp-tpot-coeff 0.75 --pdmux-pp-reserve-decode --pdmux-pp-balance-merge --pdmux-pp-rebalance --pdmux-pp-alp-admission --pdmux-pp-rebalance-unit 8 --pdmux-pp-summary-interval 10 --pdmux-prefill-mlp-chunk-tokens 4096
env CUDA_VISIBLE_DEVICES=GPU-52777e54-80e7-9357-a595-86ce2678536f,GPU-5e84f85f-d7eb-bd99-4492-85dac7c020a2 PYTHONPATH=/root/workspace/sglang/python NCCL_DEBUG=INFO /root/venvs/sgl/bin/python -m sglang.launch_server --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --tp 1 --pp-size 2 --dtype bfloat16 --port 31001 --disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --watchdog-timeout 180 --log-level debug --enable-cache-report --enable-pdmux --chunked-prefill-size -1 --pdmux-config-path /root/workspace/muxpp_work/configs/pdmux_pp_4090_v3.yml --mem-fraction-static 0.71 --cuda-graph-max-bs 32 --cuda-graph-bs 1 4 8 16 32 --max-prefill-tokens 16384 --pdmux-pp-max-budget 16384 --pdmux-pp-max-inflight-prefill 3 --pdmux-pp-radix-eviction lru --pdmux-pp-policy alp --pdmux-pp-slo-ttft 5.0 --pdmux-pp-slo-tpot 0.1 --pdmux-pp-tpot-coeff 0.75 --pdmux-pp-reserve-decode --pdmux-pp-balance-merge --pdmux-pp-rebalance --pdmux-pp-alp-admission --pdmux-pp-rebalance-unit 8 --pdmux-pp-summary-interval 10 --pdmux-prefill-mlp-chunk-tokens 4096
env /root/venvs/sgl/bin/python -m sglang_router.launch_router --host 127.0.0.1 --port 30000 --policy cache_aware --request-timeout-secs 3600 --max-concurrent-requests 4096 --log-level info --worker-urls http://127.0.0.1:31000 http://127.0.0.1:31001
```

## MuxWise-PP PP2DP2 开发健全性检查

只执行一次，独立排他标记记录于 `logs/p7a/muxpp_pp2dp2.dev300-started.json`。使用开发trace前300条，time-scale=1、warmup=2、force-close、drain-timeout=300；只记录，没有据此调参。
日志：[dev300](logs/p7a/muxpp_pp2dp2/dev300)。复用P3函数产生的router-formal.log及早期“formal replay”重置提示仅为历史命名，实际客户端命令明确使用开发trace。
完成/未完成/失败=300/0/0；未发送=0；实际统计时长=127.583s，截止时刻=405.004s。
主SLO=63.333%；TTFT p50/p99=4.142/12.223s；TPOT样本数=300。这些仅为开发健全性记录。

## 工具、统计、归档与验证

- `bench/bench_client.py --drain-timeout SEC`默认300秒，按最后实际发送计时；取消归unfinished，保留已观测首token和部分输出。summary新增completed/unfinished/failed/cutoff_time。
- `bench/metrics.py`供客户端和summarize共用；SLO分母为全计划请求，TTFT/E2E含+∞，JSON/表显示`>cutoff`；TPOT仅已完成且输出>1并报告样本数；吞吐按截止前已交付token与截断时长计算。输入只计已经交付首token的请求，包含缓存，每请求一次。详细定义见 [bench/README.md](bench/README.md)。
- `scripts/p7a_recompute.py`产生四个`*_cut300_summary.json`，原始文件不改；`summary.*`、`final_summary.*`与所有当前图只含四个最终版本并采用新口径。事后截止为最后实际发送+300秒。
- 被替代版本的原始结果、日志及旧报告已移动到`archive/superseded/`，没有删除；归档manifest包含逐文件SHA256并在移动后复核。P*_NOTES.md保留为实现文档，其历史当前结论不维护。当前报告不引用被替代结果。
- [13项CPU单测日志](logs/p7a/unit-tests.log)：mock验证实际最后发送deadline、取消关闭连接、保留部分输出、failed与unfinished区分、提前完成不空等、全计划分母、扩展实数分位、离线一致性及边界时刻；配置测试覆盖11项卡数/参数、v3参数保持与正式锁阻止客户端启动。

## 最终审计与清理

[完整审计](logs/p7a/final-audit.json)。
原始正式文件校验：True；新增正式运行数：0。
GPU计算进程：无；残留服务/客户端进程：无；占用测试服务端口：无。
主仓库与PD冻结worktree均通过提交与干净状态核验。
