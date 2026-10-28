GPU 外部占用已解除：大脑刚确认 4 张卡显存均为 1MiB、利用率 0%，连续 45 秒稳定。请从中断处继续完成 P10b（原任务见 /root/workspace/muxpp_work/prompts/p10b_bigmodel_fix.md，续跑步骤见 P10B_NOTES.md 末尾）：
1. 先确认 /dev/shm 中的模型副本完整（缺失则 scripts/p10a_model.py ensure）。
2. 按 P10B_NOTES 中列出的步骤续跑 R2 候选（新标签，不覆盖旧证据）、PP4 调度复核（KV 提升后 TTFT 仍 49s，需用 TTFT 分解与准入归因定位并修复，最多再迭代 3 轮）、Llama-8B 回归、6 个大模型配置最终启动脚本。
3. 若运行中再次发现 GPU 被容器外进程占用（nvidia-smi 显示显存/利用率但容器内无对应进程），立即停止 GPU 工作、在 P10B_NOTES.md 记录，并在最后一条消息中说明，不要等待超过 10 分钟。
4. 不跑正式实验。完成后更新 P10B_NOTES.md，最后一条消息给出简明总结（正确性、KV 容量、PP4 与 PP2×TP2 开发结果、Llama 回归、6 个配置的最终命令是否就绪）。
