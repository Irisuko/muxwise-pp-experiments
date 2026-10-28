P11b 被中断（容器重启），尚未产出任何正式结果。请从头完成原任务（/root/workspace/muxpp_work/prompts/p11b_formal.md），注意：
1. /dev/shm 又被清空，先用 scripts/p10a_model.py ensure 从私有盘副本恢复 Mistral-24B（私有盘副本 44G 完好）。
2. GPU 状态：GPU0–2 空闲；GPU3 上有一个容器外进程常驻约 454MiB 显存、间歇性短时 100% 利用率（形态与宿主机的 GPU keepalive 脚本一致：GPU 空闲时跑短 matmul 突发）。处理规则：开始每个 label 前检查；若该外部占用保持 ≤1GB 显存，视为可接受，在每个运行的元数据和报告中如实记录（GPU3 采样中的显存/利用率基线）；mem fraction 计算时确保不因这 454MiB 导致 OOM（若某配置因此 OOM，记录并停止该配置，不要私自改参数）。若外部占用超过 1GB 或出现持续高利用率，立即停止 GPU 工作并在最后报告。
3. 正式锁：此前没有任何 v5/sat label 的正式锁或结果，均可正常运行一次。
完成后按原任务产出报告与最后一条消息。
