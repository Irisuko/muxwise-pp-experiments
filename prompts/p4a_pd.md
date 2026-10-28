# 任务 P4a：修复 SGLang-PD 的 KV 传输并正式重跑一次（用户已批准这一次重跑）

## 背景
- 先读 /root/workspace/muxpp_work/RESULTS.md（尤其 sglang_pd 部分）、logs/p3/SELECTION.md、logs/p3/sglang_pd/ 下的日志、scripts/launch_sglang_pd.sh、scripts/run_experiment.sh、scripts/p3_runner.py。
- 首次正式运行（label sglang_pd）：Mooncake TCP（MC_FORCE_TCP=1）在正式负载下大量 `Failed to batch transfer data` → KVTransferError，1523/1551 失败。该结果保留，不得删除或覆盖。
- 硬件事实：4×4090 之间**不支持 P2P**（nvidia-smi topo -p2p r 全为 CNS），GPU0/1/2 在 NUMA0，GPU3 在 NUMA1；无 RDMA 网卡。Llama3.1-8B 的 KV 约 128KB/token，单个平均请求（7.2k token）约 0.9GB，负载平均需要约 2.5GB/s 的 P→D 传输，突发更高。
- 环境：source /root/workspace/muxpp_work/env.sh；代码 /root/workspace/sglang（muxwise-pp 分支）。不要改动 MuxWise-PP 相关代码；如必须改 PD 相关代码/参数才能稳定，最小修改并在报告中写明（commit 到 muxwise-pp 分支）。

## 要求
1. 先定位根因：阅读 python/sglang/srt/disaggregation/mooncake/ 与 mooncake transfer engine 的 TCP 路径，结合首轮日志判断失败原因（超时？并发连接/缓冲上限？单次传输尺寸？GPU 内存不可直接 TCP 发送需中转？）。记录证据。
2. 候选修复（按顺序尝试，择第一个稳定方案）：
   a. Mooncake TCP 参数调整（超时、并发、分片大小、相关 MC_* / SGLANG_DISAGGREGATION_* 环境变量，以及 sglang 的 --disaggregation-* 参数）；
   b. NIXL 后端（pip install nixl 到 /root/venvs/sgl；UCX 在无 P2P 时走 cuda_copy/host 中转 + tcp/shm，同机可用 shm；设置合适的 UCX_TLS）；
   c. 其他 sglang 支持的同机传输后端（如有）。
   P/D 布局可以调整以减少跨 NUMA（例如 P=GPU0,1、D=GPU2,3 或其他，说明理由），仍保持 2P2D、TP1。
3. 验证门槛（全部通过才能进入正式）：20 条短请求与 TP1 普通模式输出一致；2 条 31,663-token 长请求成功；**用正式 trace 前 300 条、time-scale=1 的预检压测 0 失败**（这是开发预检，结果单独存放在 logs/p4a/，不计入正式结果）。若某方案预检仍有失败，换下一方案。
4. 正式运行：label 用 `sglang_pd_r2`（不要覆盖 sglang_pd），其余设置与 P3b 完全一致（统一参数、--warmup 2 --force-close、日志与 GPU 采样、结束清理）。只跑一次。
5. 更新 RESULTS.md：新增 sglang_pd_r2 的完整命令、传输方案、根因分析、结果；汇总表中保留 sglang_pd（首轮失败）一行并注明，新增 r2 一行；重新运行 summarize.py（包含 r2）。不要修改其它系统的结果数据。
6. 结束确认 GPU 和服务进程已清理。最后一条消息：根因、最终方案、r2 结果要点。
