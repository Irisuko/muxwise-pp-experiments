# 任务 P10b：大模型场景下 MuxWise-PP 的正确性复核与显存/调度修复（开发验证，不跑正式）

## 必读
- /root/workspace/muxpp_work/P10A_NOTES.md（Mistral-Small-24B 准备、6 个配置、正确性与开发 150 条结果）、P9A_NOTES.md（v4）。代码 /root/workspace/sglang（muxwise-pp，HEAD 03e2fe794），可 commit 不 push。环境 source env.sh；模型 /dev/shm/models/Mistral-Small-24B-Instruct-2501（缺失时 python scripts/p10a_model.py ensure，从私有盘副本恢复）。GPU 全归你。
- 只用开发 trace bench/dev_big_toolagent_1200_80_x7.5.jsonl，禁止使用正式 trace。

## 大脑审查发现的问题
1. 正确性参照不当：MuxWise-PP PP4 / PP2×TP2 与普通 TP4 比只有 9/20 一致，其中请求 2 在输出位置 0、请求 18 在位置 1 就分叉。TP4 与 PP4 的归约顺序不同，不能作为判断依据。
2. big_muxpp_pp4 开发 150 条 TTFT p50 60.6s、主 SLO 1/150；其 KV 容量仅 118,415（mem 0.76，0.78 以上 OOM），而 big_chunked_pp4 为 204,263（mem 0.90）。准入阻塞记录为 inflight / KV / decode reserve。MuxWise-PP PP2×TP2 为 162,858（0.78）。

## 要做
1. **正确性复核**：用同并行方式的普通实例作参照——普通 PP4（pp4，TP1，非 pdmux，不加 chunked prefill 之外的差异，尽量与 MuxWise-PP 同 attention backend/graph 设置）对比 big_muxpp_pp4；普通 PP2×TP2 对比 big_muxpp_pp2tp2；serial20（与 P10A 相同的 20 条输入，temperature=0，64 token，--skip-tokenizer-init）。若仍有早期（位置 0–2）分叉，必须定位原因（逐层对比 hidden/logits，确认 split prefill 的层区间、MLP 分块、PP proxy 截断修复、TP 归约、Mistral 与 Llama 实现差异等），这是阻断项，修复后重测。另外同时报告冷 cache 20 条（触发 MLP 分块）的对比。
2. **显存剖析与降低开销**：剖析 MuxWise-PP 在该模型上每个 stage 的显存构成（权重、KV、各分区 CUDA graph 及其 workspace、FlashInfer workspace、预接收槽与 TX/RX 槽、prefill 激活峰值（整请求 31k 不切 chunk 时的 attention/MLP/中间张量）、allocator 碎片），找出为何只能用 0.76。可选方法：减少分区数量或 graph bs 档位（评估对调度的影响）、按需分配/复用接收槽、降低 MLP 分块大小或把 attention 前后的大张量也按 token 分块、对整请求超长 prefill 采用分段 split（仍不切 chunk 语义，只降低峰值）、expandable_segments 等。目标：MuxWise-PP PP4 的 KV 容量接近同配置 Chunked（≥180k），PP2×TP2 相应提高；同时保证 31k 长预热不 OOM。
3. **PP4 调度复核**：KV 修复后，在开发 trace 前 150 条上重测 PP4 与 PP2×TP2；如 PP4 仍明显异常（TTFT 远高于 PP2×TP2），用 P9a 的 TTFT 分解与准入归因定位并修复（例如 decode reserve 过于保守、inflight 上限与 KV 不匹配、分区校准在该模型上的结果等）。最多 4 轮迭代，每轮记录；保持 FCFS 与 KV 安全预留。
4. 所有改动保持 8B Llama 路径行为不变：回归 Llama-8B 的 PP2×TP2 v4 serial20 与开发 300 条（Tool&Agent）一次，确认未退化。
5. 更新 6 个大模型配置的 launch 脚本（若 MuxWise-PP 的 mem fraction 或参数改变，需重新做 31k 长预热 + 冒烟选择；MuxWise TP4 与 Chunked 不变）。

## 产出
- 代码 commit；/root/workspace/muxpp_work/P10B_NOTES.md：正确性复核结果（含参照与首分叉）、显存剖析表与改动、每轮开发结果、最终 6 个配置的启动命令与 KV 容量。
- 不跑正式实验；结束确认 GPU/进程清理。最后一条消息：简明总结。
