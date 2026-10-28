# 任务 P3b：四个系统的启动脚本 + 正式实验（每个系统只跑一次）

## 必读
- /root/workspace/muxpp_work/DESIGN.md §4（实验设计）
- /root/workspace/muxpp_work/P2_NOTES.md（MuxWise-PP 推荐启动命令）、P1_NOTES.md、ENV.md
- /root/workspace/muxpp_work/bench/README.md（压测工具；正式 trace：bench/toolagent_600_240_x2.5.jsonl，1551 条，10 分钟）
- 环境：source /root/workspace/muxpp_work/env.sh；代码在 /root/workspace/sglang 的 muxwise-pp 分支（所有系统都用这份代码运行，保证同一代码基）。

## 统一设置
- 模型 /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct，bf16，4×4090 全部使用。
- 所有实例：--disable-overlap-schedule、--context-length 33792、radix cache 开启（若某系统不支持须记录）、--schedule-policy 统一使用默认（fcfs 或 lpm 二选一，全部一致，写明）、--mem-fraction-static 取该配置下能稳定启动且通过预热的较高值（记录选择过程，不要为此跑正式负载）。
- 前端：DP 型系统使用 sglang-router（pip install sglang-router 到 /root/venvs/sgl，版本与 sglang 0.5.3 兼容），策略 cache_aware；PD 使用 sglang-router 的 PD 模式（或 sglang 自带 mini_lb，择能稳定跑通者，记录）。
- 压测客户端在 base 或 venv 中运行均可，命令：bench_client.py --trace <正式trace> --base-url <入口> --label <名字> --out-dir /root/workspace/muxpp_work/results --warmup 2 --force-close（四个系统统一使用 --force-close，原因见 P2_NOTES“客户端连接复用诊断”）。
- 每个系统运行前确保 GPU 干净；运行时保存所有服务端日志到 /root/workspace/muxpp_work/logs/p3/<label>/；同时每 5s 采样 nvidia-smi（显存、利用率）到该目录。运行结束后关闭所有进程。

- SLO 指标口径与 MuxWise-PP 的调度目标一致：主口径 TTFT≤5s、TPOT≤100ms（summary 中其他阈值照常给出）。
- muxwise_pp 的 --mem-fraction-static 0.65 是 P1 因激活 OOM 定下的；DP 型单卡实例同样需要为 32k 长 prompt 的激活留出空间（尤其 muxwise 的 --chunked-prefill-size -1 会整请求 prefill），选择 mem fraction 时请用一条 ~31k token 的请求做预热验证不 OOM。

## 四个系统（label）
1. `chunked`：4 个 TP1 实例（每卡一个），--chunked-prefill-size 2048 --enable-mixed-chunk，router cache_aware。
2. `muxwise`：4 个 TP1 实例，--enable-pdmux --pdmux-config-path configs/pdmux_4090.yml --chunked-prefill-size -1，router cache_aware。
3. `sglang_pd`：2 prefill（GPU0,1）+ 2 decode（GPU2,3），TP1，--disaggregation-mode prefill/decode，传输后端优先 mooncake（4090 无 RDMA：尝试 mooncake 的 TCP 协议，如 MC 相关环境变量 / --disaggregation-ib-device 设置；不行就用 nixl（pip install nixl，UCX 走 cuda_ipc/tcp），记录最终方案），prefill 侧 chunked-prefill-size 保持默认或 4096（记录），decode 侧默认。先用 20 条短请求验证 KV 传输正确（输出与普通模式一致）再跑正式负载。
4. `muxwise_pp`：单实例 --pp-size 4 + pdmux + P2 全部特性（按 P2_NOTES 推荐命令）。

## 流程
1. 为每个系统写启动脚本 /root/workspace/muxpp_work/scripts/launch_<label>.sh 和统一运行脚本 run_experiment.sh <label>（启动→等待健康→预热→压测→收集日志→关闭）。
2. 每个系统先做 ≤1 分钟的冒烟（--limit 20 --time-scale 0.2），通过后再跑正式实验。冒烟结果不计入。
3. 正式实验每个系统**只跑一次**，不补跑、不重跑；即便结果很差（例如大量超时或 SLO 不达标）也如实记录。只有当系统根本无法完成（启动失败/崩溃导致无法产出数据）时，修复启动问题后允许再启动一次，并在报告中明确说明。
4. 顺序：chunked → muxwise → sglang_pd → muxwise_pp。
5. 全部完成后运行 summarize.py 生成 results/summary.md、csv、png（先删除 results/ 里 mock 数据产生的旧文件）。

## 产出
- /root/workspace/muxpp_work/RESULTS.md：各系统精确启动命令、实际配置（mem fraction、每实例 max_total_num_tokens、router 策略、PD 传输后端）、汇总对比表（TTFT/TPOT/E2E 各分位、吞吐、SLO 达成率、完成/失败数、cache 命中率）、图链接、异常与观察（如 MuxWise-PP 调度摘要日志中的分区分布、迁移次数、RLS 误差），以及客观分析（不要夸大，数据说话）。
- 最后一条消息：简明总结结果表。
