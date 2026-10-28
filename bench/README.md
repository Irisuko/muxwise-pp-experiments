# Tool&Agent trace 回放压测（P3a）

这些工具使用 CPU 构造 token、发送 HTTP 请求和绘图，不加载模型，也不导入 torch/CUDA。与 `../DESIGN.md` §4 的负载窗口和 `/generate` 请求格式一致。P3a使用aiohttp mock自测；后续P3b四系统正式实验已完成，启动配置、异常和结果见[RESULTS.md](../RESULTS.md)。

## 文件

- `build_trace.py`：从原始 trace 构造确定性 token prompt，输出 JSONL 和 `.stats.json`。
- `bench_client.py`：按绝对到达时间开环回放，保存请求明细和统计。
- `summarize.py`：多个运行的 Markdown/CSV 对比及PNG/SVG图。
- `summarize.py --matrix`：P7b固定15个最终配置，生成完整矩阵、各系统最佳配置及PNG/SVG图；严格要求历史四个输入为`_cut300`。
- `mock_server.py`：CPU aiohttp SSE mock，可模拟逐 token、合包和字节分片响应。
- `selftest.py`：两组各 200 条的完整 mock 回放，自动清理本地服务。
- `test_bench.py`：流式解析、异常处理、指标口径和 prompt 构造测试。
- `common.py`、`requirements.txt`：公共统计函数和依赖。
- `toolagent_600_240_x2.5.jsonl`、`toolagent_600_240_x2.5.stats.json`：已生成的正式负载。
- `selftest_results/`：mock 请求明细、summary、自测报告和正式 trace 校验记录。
- `../results/summary.md`、`summary.csv`、`*.png`：当前只含四个最终版本、300秒排空口径；mock原始自测保留在`selftest_results/`。

## 环境

Python 3.9+，依赖 `numpy`、`aiohttp`、`tokenizers`、`matplotlib`。构造 trace 只读取本地 `tokenizer.json` 和 tokenizer 配置，不下载权重，也不需要 transformers/torch。

P3a在 `/root/miniconda3/bin/python` 中开发、安装缺失依赖并自测；当时未运行真实后端。P3b客户端与推理服务使用`/root/venvs/sgl`，绘图使用base Python，并在venv安装了`sglang-router==0.1.9`，详见正式报告。

```bash
cd /root/workspace/muxpp_work
/root/miniconda3/bin/python -m pip install -r bench/requirements.txt
```

已有相应依赖时，以下命令中的 `python` 可以直接替换为 `/root/venvs/sgl/bin/python`。

## 生成正式 trace

```bash
python bench/build_trace.py \
  --source /root/ext/mooncake/FAST25-release/traces/toolagent_trace.jsonl \
  --start-sec 600 --duration-sec 240 --stretch 2.5 \
  --max-total-len 32768 --block-size 512 \
  --tokenizer /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct \
  --out bench/toolagent_600_240_x2.5.jsonl
```

窗口为 `[600000, 840000)` ms；仅过滤总长度 **大于** 32768 的请求。`arrival` 单位为秒：`(timestamp - 600000) / 1000 * 2.5`。按 timestamp、原始行号稳定排序；`idx` 是原始 trace 的零基行号，因此经过过滤后不连续。

每个非负整数 hash 对应 `numpy.Generator(PCG64(hash_id))`，其他类型通过稳定的 BLAKE2b-128 转换为 seed；不使用 Python 的随机化 `hash()`。从排序后的普通 token ID 集合采样，排除 tokenizer 的全部 added token（包括非特殊的 added token）与特殊 token；Llama3 额外排除所有 `id >= 128000`。块依次拼接并截断；不足时使用带 `padding` 命名空间和原始行号的独立 seed 补齐。将首 token **替换**为 BOS 128000，不插入新 token，不移动块边界。

紧凑 JSONL 的每行包含 `idx, arrival, input_len, output_len, input_ids, hash_ids`。正式文件约 66 MiB。客户端在开始计时之前一次加载、验证并预编码 HTTP body，避免计时阶段重复处理大 prompt 列表。

本次生成统计：

| 指标 | 结果 |
| --- | ---: |
| 窗口内请求 / 过滤 / 保留 | 1603 / 52 / 1551 |
| 规划窗口 / 最后到达 | 600 s / 599.9975 s |
| 输入总 token / 输出总 token | 11,245,637 / 261,776 |
| 输入 mean / p50 / p90 / p99 / max | 7250.57 / 6235 / 14655 / 29009.5 / 31663 |
| 输出 mean / p50 / p90 / p99 / max | 168.78 / 29 / 482 / 802.5 / 1987 |
| hash 块不足而补齐的请求 | 0 |
| 普通词表大小 / 被排除 token | 128000 / 256 |
| 首次出现口径理论复用率 | 52.54365% |

理论复用率 = 后续出现的 hash 块覆盖的有效 token 数 / 全部输入 token 数。截断块仅计实际使用长度，padding 不可复用；首次出现状态从选定窗口开始。这是按 hash 首次出现计算的估计，未考虑必须连续的前缀、首次块仅部分出现、缓存驱逐、DP 路由、并发请求尚未完成等因素，不能当成实际 radix cache 命中率。实际命中以服务端记录及响应中的 `cached_tokens` 为准。

## 正式开环回放

对已经启动的 server 或 router 使用相同命令，仅改变地址与标签：

```bash
python bench/bench_client.py \
  --trace bench/toolagent_600_240_x2.5.jsonl \
  --base-url http://127.0.0.1:30000 \
  --label chunked_dp4 --out-dir results --warmup 2 --force-close --drain-timeout 300
```

`--base-url` 支持带路径前缀的地址，也允许直接以 `/generate` 结尾。默认总超时和 socket read 超时均为 3600 秒，aiohttp 总连接数和单 host 连接数均不设上限。每个请求发送：

```json
{"input_ids":[128000,123,456],"sampling_params":{"max_new_tokens":29,"ignore_eos":true,"temperature":0},"stream":true}
```

计时从预热全部成功后开始。预热 N 次，每次使用首条 prompt 的前 16 个 token、生成 4 个 token，不计入结果；预热失败则在正式回放前退出。回放使用 monotonic clock 的绝对 deadline 独立创建任务，不等待已有请求完成、不使用并发 semaphore，也不重试失败请求。记录 `send_lag = sent_at - arrival`，用于检查客户端是否及时发送。结果序列化和磁盘写入在回放结束之后进行。

默认每 30 秒打印已发、成功完成、失败、在途和最近 30 秒已结束请求的 TTFT p50。`--progress-interval` 可调整打印频率。`--limit N` 只跑前 N 条；`--time-scale 0.05` 把 arrival 再乘以 0.05（20 倍加速，不改变输入/输出长度）。正式运行使用默认 `--time-scale 1`，不带 limit。

同标签文件已存在时拒绝覆盖；`--overwrite` 可显式覆盖调试结果。遵循设计文档，每个真实系统只运行一次，不自动补跑或重跑。Ctrl-C 会取消在途请求、保存已发请求的部分结果，记录 `interrupted` 和未发送数量，退出码 130；普通失败保存结果后退出码 1；只有完成或到时取消（unfinished）时为 0。

服务端的 context length、radix cache、overlap、并行配置等按设计文档统一设置，客户端不修改服务端配置。请将这些配置与正式实验结果一起保存。

## 请求明细与统计口径（P7a）

`--drain-timeout SEC` 默认 300 秒。从最后一个请求的实际 `sent_at` 起计时，最多再等 SEC 秒；若所有请求先结束则立即结束。到时取消全部在途任务并关闭 HTTP 连接；取消归为 `unfinished`，HTTP/协议错误仍归为 `failed`。每个请求的已观测首 token、`chunks`（SSE 时间戳及累计 token 数）、部分输出和错误保留。无重试。正式启动入口显式使用 300 秒；预检冒烟为满足≤1分钟使用 50/55 秒超时（以候选命令记录为准）。

summary schema v2 新增 `completed / unfinished / failed / cutoff_time`；兼容保留 `*_requests` 字段。`cutoff_time` 是相对预热后回放原点的秒数，等于最后实际发送时间加排空上限；不是固定的单请求超时。`actual_duration_sec` 为最后终态时间，`duration_sec = min(actual_duration_sec, cutoff_time)`。

- SLO 分母是全部计划请求。unfinished、failed、未发送和指标未定义均算未达标，TTFT 单项 SLO 也要求请求成功完成。
- TTFT 与 E2E 分位数均以全部计划请求为样本。截止前未收到首 token 的 TTFT 取 +∞；未成功完成的 E2E 取 +∞。已收到首 token 的 unfinished/failed 请求保留真实 TTFT。未发请求两个指标均取 +∞。采用与 NumPy 默认相同的线性分位插值，在扩展实数上计算；落在 +∞ 的统计显示为字符串 `>cutoff`，JSON 不写非标准 Infinity/NaN。
- TPOT = (E2E − TTFT)/(输出 token − 1)，只计算截止前成功完成且输出>1的请求。分位表及 JSON `count` 标注样本数。单 token 输出不参与 TPOT 分位且不达成 TPOT/联合 SLO。
- 输出吞吐 = 截止前收到的 SSE 累计输出 token（包括 unfinished/failed 的部分输出）/duration_sec。输入吞吐 = 截止前已经交付首 token 的请求输入量/duration_sec（含缓存 token，每请求计一次）。排队中未交付首 token 的输入、服务端未被观测的部分 prefill 不计入；这是客户端可观测的输入口径，不等同于 GPU 实际 prefill 工作量。
- `chunks[].at <= cutoff_time` 的事件保留，事件恰好在截止时刻算入；合包输出全部按该事件接收时间计数，不推测包内 token 时间。完成时刻恰好等于截止时刻算完成。
- ITL 是相邻两个增加累计 token 数的 SSE 事件间隔；合包时无法恢复真实逐 token ITL。汇总 ITL 只取已完成请求。
- cache 比例只取截止前完成且提供 `meta_info.cached_tokens` 的响应；缺失为 null。PD decode 响应不代表 prefill 实际缓存命中。

`bench_client.py` 和 `summarize.py` 共同使用 `metrics.py`。后者从原始明细重算，不能直接采信旧 summary 的成功子集分位数。TTFT/E2E CDF 的分母为全部计划请求，截止后未观测的尾部不会补成有限值；TPOT CDF 分母为有效已完成样本。

已有四个最终运行的截断只读重算：

```bash
python scripts/p7a_recompute.py
python bench/summarize.py results/chunked_cut300_summary.json results/muxwise_r2_cut300_summary.json results/sglang_pd_r2_cut300_summary.json results/muxwise_pp_v3_cut300_summary.json --out-dir results --final-labels chunked muxwise_r2 sglang_pd_r2 muxwise_pp_v3
```

输出 `<label>_cut300_summary.json`，保留原始 requests、summary、正式锁；重算前后 SHA256 核对。截止 = 原始最后 `sent_at` + 300 秒，完成晚于截止归 unfinished；首 token 晚于截止 TTFT 取 +∞。这是一份历史观测的截断投影，并未重新运行服务。

## 汇总多个系统

P7b正式矩阵使用以下命令，只读取已保存的正式数据，不启动服务：

```bash
python bench/summarize.py --matrix --results-dir results --out-dir results
```

该模式自动选择4个历史`_cut300_summary.json`及11个新配置summary，并从请求明细重算。
输出`matrix_summary.md/csv`（15行）、`best_summary.md/csv`（4行）、
`matrix_slo_heatmap.png/svg`、`best_ttft_cdf.png/svg`、`best_tpot_cdf.png/svg`、
`best_main_slo.png/svg`及最佳配置E2E/时间线/全部SLO图。
最佳配置按主SLO（TTFT≤5s且TPOT≤100ms）最高选择，同分取TTFT p50最低。
矩阵带系统、并行配置、正式日志路径、每实例KV容量和mem fraction；容量读取正式`run.json`。
CSV延迟单位为秒、比例为0–1，TPOT样本数为`latencies_sec.tpot.count`。
不支持的组合只在说明和热力图中标记，不伪造结果行。

通用多运行模式仍支持显式输入：

```bash
python bench/summarize.py \
  results/chunked_cut300_summary.json \
  results/muxwise_r2_cut300_summary.json \
  results/sglang_pd_r2_cut300_summary.json \
  results/muxwise_pp_v3_cut300_summary.json \
  --out-dir results
```

不要同时加载同一 label 的原始及 cut300 summary。也支持带引号的 glob，例如 `'results/*_summary.json'`。同目录需要相应 `{label}_requests.jsonl`，才能画 CDF 和散点图；复制结果目录后优先使用 summary 同目录的明细文件。

输出 `summary.md`、`summary.csv`、`ttft_cdf.png`、`tpot_cdf.png`、`ttft_over_time.png`、`slo_attainment.png`。CSV 展开全部 summary 字段，保留秒和 0–1 比例；Markdown/图表将 TPOT 转为毫秒、SLO 转为百分比。散点横轴为乘过 time-scale 的计划到达时间。比较时使用相同 trace、请求数、time-scale 和服务端实验配置。

## CPU 自测与结果

一键运行（内部绑定 127.0.0.1 的动态端口，完成后关闭 mock）：

```bash
/root/miniconda3/bin/python -m unittest discover -s bench -p 'test_*.py' -v
/root/miniconda3/bin/python bench/selftest.py
```

也可以手动启动 mock，并在另一终端发送 200 条请求：

```bash
python bench/mock_server.py --host 127.0.0.1 --port 18080
```

```bash
python bench/bench_client.py \
  --trace bench/toolagent_600_240_x2.5.jsonl \
  --base-url http://127.0.0.1:18080 \
  --label mock_manual --out-dir bench/selftest_results \
  --warmup 2 --time-scale 0.05 --limit 200
python bench/summarize.py bench/selftest_results/mock_manual_summary.json --out-dir results
```

mock 默认 prefill sleep 为 `0.01 + input_len * 0.000002` 秒，随后每 token 间隔约 1ms。可通过 `--token-delay` 调整，通过 `--tokens-per-chunk 2 --fragment-bytes 7 --metadata-only --omit-cache` 测试合包、分片、初始零 token 事件及缺失 cache 字段。`GET /stats` 提供请求数量和并发峰值。

本次实际验证：7 项 unittest 全部通过（包含多种错误流子用例）；正式 trace 1551 行全部核对源数据、长度、时间、BOS、普通 token 范围及每个 hash 块的独立 PCG64 重建。

| mock 配置 | 成功 / 失败 | 时长 | 峰值在途 | TTFT p50 | TPOT p50 | 发送延迟 p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1ms/token | 200 / 0 | 4.123s | 36 | 27.99ms | 1.369ms | 4.00ms |
| 2ms/token | 200 / 0 | 4.789s | 42 | 28.65ms | 2.503ms | 4.48ms |

两次均使用 `--time-scale 0.05 --limit 200 --warmup 2`，计划最后到达 3.375s，各输出 30,659 token。服务端各收到 202 条请求，统计中各为 200 条，验证预热排除；也验证请求在前序请求完成前发出、每个 token/ITL 数量、TPOT 公式、cache 计数和全部汇总产物。时间会随 CPU 负载变化，不作为 GPU 后端性能指标。
