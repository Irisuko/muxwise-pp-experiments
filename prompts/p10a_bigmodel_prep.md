# 任务 P10a：大模型场景准备（Mistral-Small-24B，必须跨 4 卡），不跑正式

## 背景
- 结论回顾（RESULTS.md）：在 Llama-8B 上 TP2 就够用，PP 的优势出不来；TP4 在 4090（无 P2P，allreduce 走主机内存）上所有系统都崩。PP 真正有优势的场景是模型必须跨 3–4 卡。用户同意用大模型再比一次。
- 模型：Mistral-Small-24B-Instruct-2501（bf16 约 48GB，两张 24GB 卡放不下，只能用 4 卡实例：TP4 / PP4 / PP2×TP2；原生 32k 上下文；SGLang 中 MistralForCausalLM 继承 LlamaForCausalLM）。若该模型不可得或在本代码基上无法运行，退而用 CodeLlama-34b-Instruct-hf（Llama 架构），并在报告中说明。
- 环境 source /root/workspace/muxpp_work/env.sh；代码 /root/workspace/sglang（muxwise-pp，HEAD f6833f24e），可以 commit（不 push）。GPU 全归你用。

## 1. 下载与存放
- 优先 ModelScope（`modelscope download --model mistralai/Mistral-Small-24B-Instruct-2501 --local_dir ...`，名称以实际搜索为准），备选 hf-mirror.com；只下载 safetensors 与 tokenizer/config 文件（不要 consolidated.* 的重复权重）。
- 持久存放：/root/autodl-pvt-A40-hdd-1/models/Mistral-Small-24B-Instruct-2501（用户私有盘）；运行时复制到 /dev/shm/models/Mistral-Small-24B-Instruct-2501 从内存盘加载（启动脚本先检查 shm 中的副本完整性，缺失则从私有盘复制）。校验文件完整（大小/sha 或 safetensors 可加载）。
- 下载是长时间任务，日志写 logs/p10a/。

## 2. 代码适配（最小改动）
- 让 MuxWise（pdmux，pp=1）与 MuxWise-PP 接受 MistralForCausalLM（目前可能只按 Llama 架构名放行）；确认 forward_split_prefill、PP 层切分、MLP 分块、split prefill 的 TP 支持在该模型上可用。保持对 Llama 的行为不变。
- Chunked（上游）在 PP 下曾因 KV 压力出现 `#new-token` 为负导致 FlashInfer 崩溃（P7b chunked_pp4/tp2pp2）。大模型下 KV 更紧，若开发阶段复现，定位并最小修复上游 bug（作为基线的代码修复，写清根因与修改），否则记录未复现。

## 3. 负载
- 仍用 Mooncake Tool&Agent（与之前相同的取窗起点 600s、过滤总长 >32k），但大模型算力约为 8B 的 3 倍（按参数量 24B/8B），负载需等比例降低以保持可比的“相对压力”，同时正式运行仍为约 10 分钟：取 --start-sec 600 --duration-sec 80 --stretch 7.5（80s×7.5=600s）。用该模型的 tokenizer 重新构建 token（build_trace.py --tokenizer 指向模型目录，排除所有特殊/控制 token，BOS 用该 tokenizer 的 bos）。
  - 正式：bench/big_toolagent_600_80_x7.5.jsonl（只生成和统计，本任务不得使用）
  - 开发：--start-sec 1200 --duration-sec 80 --stretch 7.5 → bench/dev_big_toolagent_1200_80_x7.5.jsonl
  报告两者的请求数、输入/输出 token 速率、复用率。

## 4. 配置（每个 4 卡单实例）
- `big_muxwise_tp4`：原版 MuxWise（pdmux，pp=1，TP4；分区配置可沿用 configs/pdmux_4090.yml 或按需调整，记录）。
- `big_chunked_tp4`、`big_chunked_pp4`、`big_chunked_pp2tp2`：chunked-prefill 2048（PP 下按代码要求关闭 mixed chunk）。
- `big_muxpp_pp4`、`big_muxpp_pp2tp2`：MuxWise-PP v4（参数同 P9A_NOTES 中 v4 的 pp4 / pp2tp2 命令；inflight 按 pp 比例 6/3），分区校准会在启动时自动进行。
- 统一：bf16、FlashInfer + CUDA graph（例外须记录原因）、--context-length 32768（或模型上限，记录）、--disable-overlap-schedule、--schedule-policy fcfs、--max-running-requests 128、--enable-cache-report。
- 每个配置写 launch 脚本并接入 run_experiment.sh（正式锁）。

## 5. 预检
- 每个配置：用约 31k token 的长请求预热 + ≤1 分钟冒烟（开发 trace，--limit 20 --time-scale 0.2）选 mem fraction（取能稳定通过的较高值），记录每实例 KV 容量。
- 正确性：MuxWise-PP PP4 / PP2×TP2 各与普通 TP4（pp=1，非 pdmux）对比 serial20（temperature=0，64 token），报告首个分叉位置；MuxWise TP4 同样对比一次。
- 开发健全性：6 个配置各用开发 trace 前 150 条（time-scale=1，--warmup 2 --force-close --drain-timeout 300）跑一次，只记录、不调参（若某配置明显异常——例如崩溃——先排查并修复启动/代码层问题，记录）。

## 产出
- /root/workspace/muxpp_work/P10A_NOTES.md：模型来源与校验、代码改动、负载统计、6 个配置的最终启动命令/mem fraction/KV 容量、正确性与开发健全性结果。
- 不跑正式实验；结束确认 GPU/进程清理（/dev/shm 中的模型副本保留供正式使用）。最后一条消息：简明总结。
