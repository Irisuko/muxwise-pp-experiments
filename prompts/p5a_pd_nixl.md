# 任务 P5a：SGLang-PD 传输带宽改进评估（必要时正式跑 sglang_pd_r3）

## 背景
- 先读 /root/workspace/muxpp_work/RESULTS.md 的 P4a 部分、logs/p4a/STATE.md、logs/p4a/ 下 a-mooncake-* 与 b-nixl-host 记录。
- r2（Mooncake TCP 调参版）可靠性已解决，但性能受传输限制：输入吞吐仅 5054 tok/s（≈0.64 GB/s KV），prefill GPU0/1 平均利用率仅约 20%，TTFT p50 797s。大脑判断：瓶颈是 Mooncake TCP 回环路径（GPU→host→TCP→host→GPU，1MiB 分片），不是 PD 架构本身的 prefill 算力。同机进程间用共享内存/CUDA IPC 的主机中转理论上可达数 GB/s。
- 硬件：4×4090 不支持 P2P（topo -p2p 全 CNS），GPU0/1/2 在 NUMA0、GPU3 在 NUMA1，无 RDMA。
- 环境：source /root/workspace/muxpp_work/env.sh；PD 使用隔离 worktree /root/workspace/sglang-p4a（不要改 /root/workspace/sglang 工作树——另一个任务 P5b 正在那里开发；如需改 PD 代码/参数，在 sglang-p4a 上 commit）。
- 用户规则：每个版本正式实验只跑一次；代码/配置修复改进后可以作为新版本再正式跑一次（新 label，旧结果保留）。

## 要做
1. 传输微基准：在 2P2D 布局下，对 Mooncake-tuned（r2 配置）与 NIXL（已安装，见 install-nixl.log；尝试 UCX_TLS 组合，如 cuda_copy,cuda_ipc,sm,self 或 cuda_copy,sm,tcp 等，记录哪些可用）测量单请求 KV 传输耗时/有效带宽（例如 4k、16k、31k token 的请求，从 prefill 完成到 decode 收齐的时间，可从服务端日志或计时插桩获得）。若 sglang 0.5.3 的 NIXL 后端有需要修正的 bug，可在 sglang-p4a 上最小修复。也可评估其它可行的同机方案（例如 Mooncake 其它协议若可用于无 RDMA 同机），按投入产出决定。
2. 选出带宽最好且稳定的方案后，用**开发 trace**（/root/workspace/muxpp_work/bench/dev_toolagent_1200_240_x2.5.jsonl，前 300 条，time-scale=1，--warmup 2 --force-close）对比 r2 配置与新方案（两者各一次，结果放 logs/p5a/）。禁止用正式 trace 做任何调参/预检。
3. 正确性门槛：沿用 P4a 的 text20 固定用例与普通 TP1 一致（20/20），2 条 31,663-token 长请求成功，开发 300 条 0 失败。
4. 决策：只有当新方案在开发 300 条上相对 r2 有实质改进（输入吞吐 ≥1.5 倍，或 TTFT p50 降低 ≥50%）且通过全部门槛时，才以 label `sglang_pd_r3` 正式运行一次（正式 trace，设置与 P3b/P4a 一致，--warmup 2 --force-close，日志/GPU 采样/清理同前，使用 scripts/run_experiment.sh 的既有机制并防止重复运行）。否则不跑正式，写明原因。
5. 更新 RESULTS.md（新增 P5a 小节：微基准表、开发对比、决策、若有 r3 则加入汇总表，保留 sglang_pd 与 sglang_pd_r2 行），重新运行 summarize.py。不要修改其它系统的结果。
6. GPU 使用完后创建标记文件 /root/workspace/muxpp_work/logs/p5a.gpu_done（另一个任务以此判断 GPU 可用）；结束时确认 GPU 与进程清理。最后一条消息：微基准结论、开发对比、是否跑了 r3 及结果要点。
