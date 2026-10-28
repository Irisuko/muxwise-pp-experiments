P11b 因用户更换 GPU 被大脑手动停止，现已换卡，请从头重新完成原任务（/root/workspace/muxpp_work/prompts/p11b_formal.md）。大脑已核实的状态：
1. 4 张 4090 显存均 1MiB、利用率 0%，连续 60 秒稳定，无容器外占用（此前 GPU3 的 454MiB 保活进程已不存在）。新 UUID：GPU0 3fe63c80、GPU1 ffc7795e、GPU2 52777e54、GPU3 65206bc3；拓扑变化：现在 4 张卡都在 NUMA0（互为 NODE），与此前所有正式运行不同，须在报告中注明。
2. 容器内没有残留的 sglang/客户端/runner 进程。/dev/shm 已清空，先用 scripts/p10a_model.py ensure 从私有盘（44G 完好）恢复 Mistral-24B。
3. 中断时 muxpp_v5_pp4 的正式客户端刚启动、无任何完整结果；其正式锁已移到 archive/aborted_p11b/（含说明）。这属于外部中断，不是重跑，该 label 以及其余 12 个 label 都可以正常各跑一次；报告中注明此次中断。results/ 中目前没有任何 v5/sat 的锁或结果。
4. 若运行中再出现容器外 GPU 占用（容器内无对应进程但有显存/利用率），立即停止 GPU 工作并在最后报告。
完成后按原任务产出报告与最后一条消息。
