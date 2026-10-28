# 任务 P3a：编写 Tool&Agent trace 回放压测工具（不使用 GPU）

先阅读 /root/workspace/muxpp_work/DESIGN.md 的 §4（实验设计）。注意：另一个 Codex 进程正在搭建 /root/venvs/sgl 环境并占用 GPU 做冒烟测试，你**不要使用 GPU、不要启动推理服务、不要修改 /root/venvs/sgl 和 /root/workspace/sglang**。你的代码放在 /root/workspace/muxpp_work/bench/ 下。可以用 base 环境 python（/root/miniconda3）开发和测试；最终脚本需同时能在 /root/venvs/sgl 中运行（只依赖 aiohttp、numpy、transformers/tokenizers 等常见包；缺的包可 pip 装在 base 环境中用于测试）。

## 1. build_trace.py
- 输入：/root/ext/mooncake/FAST25-release/traces/toolagent_trace.jsonl（字段 timestamp(ms), input_length, output_length, hash_ids）。
- 参数：--start-sec 600 --duration-sec 240 --stretch 2.5 --max-total-len 32768 --block-size 512 --tokenizer /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct --out /root/workspace/muxpp_work/bench/toolagent_600_240_x2.5.jsonl
- 取 start ≤ timestamp < start+duration；过滤 input_length + output_length > max_total_len 的请求（打印过滤数）；arrival = (timestamp - start*1000)/1000 * stretch 秒。
- 构造 prompt token：每个 hash_id 对应一个长度 block_size 的确定性伪随机 token 块（numpy Generator，seed 由 hash_id 决定），token 从词表中采样，排除所有特殊 token / added tokens（用 tokenizer 判断，Llama3 的 id ≥ 128000 均为特殊 token）；请求的 token 序列 = 依次拼接各 hash 块，截断/补齐到 input_length（若 hash 块总长不足 input_length，用以请求序号为种子的随机 token 补齐——并打印出现该情况的次数）。第一个 token 用 BOS(128000)。
- 输出每行：{"idx", "arrival", "input_len", "output_len", "input_ids": [...], "hash_ids"}（input_ids 可能很大，总文件几百 MB 可接受；或者另存 .npy/.npz，自行权衡，但要保证加载快）。
- 打印统计：请求数、总时长、输入长度均值/中位/p90/p99/max、输出长度统计、理论前缀复用率（按 hash 块首次出现计算的可复用 token 比例）。

## 2. bench_client.py
- 按 arrival 时间（相对开始时刻）开环发送请求（asyncio + aiohttp），**不因前面请求未完成而推迟**。
- 后端：SGLang 原生 /generate 接口，payload：{"input_ids": [...], "sampling_params": {"max_new_tokens": output_len, "ignore_eos": true, "temperature": 0}, "stream": true}。base-url 参数化（指向 server 或 router）。
- 流式解析（SGLang /generate 的 SSE 格式 "data: {...}"，以 "data: [DONE]" 结束；用 meta_info.completion_tokens 计算 token 数），记录每请求：发送时间、首 token 时间（TTFT）、每个 chunk 到达时间、完成时间、输出 token 数、cached_tokens（meta_info 中若有）、错误信息。TPOT = (E2E - TTFT) / (output_tokens - 1)（output_tokens>1 时）。另外记录 ITL 列表。
- aiohttp 超时设置足够长（例如 3600s），连接数不限。
- 参数：--trace、--base-url、--label、--out-dir、--warmup N（开始前发 N 个短请求预热，不计入）、--limit / --time-scale（调试用，可加速或只跑前 N 条）。
- 结束后写：{out_dir}/{label}_requests.jsonl（逐请求明细）与 {label}_summary.json：完成数/失败数、总时长、输入/输出 token 吞吐、请求吞吐、TTFT/TPOT/E2E/ITL 的 mean/p50/p90/p95/p99/max、SLO 达成率（TTFT≤2/5/10s，TPOT≤50/100/200ms，联合达成：TTFT≤5s&TPOT≤100ms 等几个组合）、平均 cached_tokens 比例。
- 运行中每 30s 打印一行进度（已发/已完成/在途/最近 TTFT p50）。

## 3. summarize.py
- 读取多个 {label}_summary.json，输出 markdown 对比表（/root/workspace/muxpp_work/results/summary.md）以及 csv；并用 matplotlib 画：TTFT CDF、TPOT CDF、随时间的 TTFT 散点（按到达时间）、SLO 达成率柱状图，存 results/*.png。

## 4. 自测
- 写一个简易 mock server（aiohttp，模拟 /generate 流式接口，按 prompt 长度 sleep 一段时间再逐 token 流式返回），用 --time-scale 0.05 和 --limit 200 跑通 bench_client 与 summarize。
- 运行 build_trace.py 生成正式 trace 文件，检查请求数应为 1551 左右（允许你复核：若计算口径不同请在报告中说明）。

## 5. 产出
- /root/workspace/muxpp_work/bench/README.md：用法说明。
- 最后一条消息：简要总结生成的文件、trace 统计、自测结果。
