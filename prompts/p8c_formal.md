# 任务 P8c：正式运行 muxpp_pp2tp2（只跑一次）并更新最终报告

- 必读：/root/workspace/muxpp_work/P8A_NOTES.md、RESULTS.md、ANALYSIS_MUXWISE_TP2DP2.md（末尾有对 PP2×TP2 的预测）、bench/README.md。环境 source /root/workspace/muxpp_work/env.sh；不得修改源码与选定参数（HEAD d8dfbfaba，configs/p8a_selected.json）。
- 运行：`bash /root/workspace/muxpp_work/scripts/run_experiment.sh muxpp_pp2tp2 --tag formal-once`（正式 trace，--warmup 2 --force-close --drain-timeout 300，流程与 P7b 一致）。只跑一次；只有在完全无法产出数据（启动失败/崩溃）时才可以在修复启动层面问题后再启动一次，并注明。
- 记录运行时 GPU UUID/NUMA 映射（P8a 发现与 P7a/P7b 时不同），在报告中注明“与矩阵其它配置运行于不同日期，GPU 物理映射可能不同”。
- 更新 RESULTS.md：矩阵增加 MuxWise-PP PP2×TP2 一行（16 行），“不支持”说明改为 MuxWise-PP 支持 TP（TP4/TP2×DP2 这类无 PP 的组合不适用）；若 PP2×TP2 成为 MuxWise-PP 最佳配置，则更新各系统最佳配置对比表与图；新增小节对照 ANALYSIS_MUXWISE_TP2DP2.md 的预测逐条检验（命中/未命中）；保持客观。重新生成 results/matrix_summary.*、best_summary.* 与相关图。
- 结束确认 GPU/进程清理。最后一条消息：PP2×TP2 正式结果、与 MuxWise TP2×DP2 及 MuxWise-PP 其它配置的对比、预测检验结论。
