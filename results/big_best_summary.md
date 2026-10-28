# Mistral-Small-24B各系统最佳配置

| 系统 | 配置 | label | 完成/未完/失败 | TTFT p50/p90/p99(s) | TPOT p50/p99(ms); n | E2E p50(s) | 输入/输出 tok/s | 主 SLO | TTFT≤2s | 10s/100ms SLO | cache命中率† | 每实例KV(token) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MuxWise | TP4 | big_muxwise_tp4 | 555/0/0 | 8.867/24.412/34.635 | 29.603/299.279; n=552 | 13.458 | 6555.383/160.569 | 19.640% | 7.568% | 48.108% | 41.062% | server(model)=167692 |
| Chunked | PP2×TP2 | big_chunked_pp2tp2 | 555/0/0 | 4.996/19.158/31.508 | 125.176/6759.185; n=552 | 15.335 | 6485.989/158.870 | 13.694% | 19.820% | 25.225% | 39.022% | server(model)=236846 |
| MuxWise-PP | PP2×TP2 v5 | big_muxpp_v5_pp2tp2 | 555/0/0 | 10.989/24.274/37.068 | 63.423/99.155; n=552 | 17.691 | 6460.320/158.241 | 19.459% | 5.225% | 44.685% | 38.272% | server(model)=203279 |

主SLO：TTFT≤5s且TPOT≤100ms。每配置一次测量，无置信区间。SLO/TTFT/E2E分母为全部555条计划请求；TPOT仅取已完成且输出>1的样本。吞吐包含排空，输入包含缓存。KV为每实例逻辑容量，不乘TP/PP rank数。† PD cache仅为decode响应。

## 运行配置与GPU采样

| label | mem fraction | GPU均值% | 样本数 | 发送延迟p99/max(ms) | 正式日志 |
| --- | --- | --- | --- | --- | --- |
| big_muxwise_tp4 | {'server': 0.76} | 94.169 | 480 | 13.471/13.932 | /root/workspace/muxpp_work/logs/p10c/big_muxwise_tp4 |
| big_chunked_pp2tp2 | {'server': 0.9} | 84.419 | 484 | 18.581/19.076 | /root/workspace/muxpp_work/logs/p10c/big_chunked_pp2tp2 |
| big_muxpp_v5_pp2tp2 | {'server': 0.82} | 94.143 | 488 | 13.893/22.392 | /root/workspace/muxpp_work/logs/p11b/big_muxpp_v5_pp2tp2 |

GPU每5秒采样，仅取客户端测量窗口（包括排空，不含长预热/冒烟）；GPU利用率包含通信等待，不能等同于有效计算效率。

## 完整延迟分位

| 系统 | 指标 | 样本数 | mean | p50 | p90 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| big_muxwise_tp4 | TTFT (s) | 555 | 11.120 | 8.867 | 24.412 | 29.029 | 34.635 | 44.628 |
| big_muxwise_tp4 | TPOT (ms) | 552 | 46.955 | 29.603 | 120.750 | 160.347 | 299.279 | 308.446 |
| big_muxwise_tp4 | E2E (s) | 555 | 15.934 | 13.458 | 31.157 | 37.748 | 50.642 | 81.812 |
| big_chunked_pp2tp2 | TTFT (s) | 555 | 7.539 | 4.996 | 19.158 | 25.767 | 31.508 | 38.169 |
| big_chunked_pp2tp2 | TPOT (ms) | 552 | 716.462 | 125.176 | 2130.685 | 4485.196 | 6759.185 | 8655.521 |
| big_chunked_pp2tp2 | E2E (s) | 555 | 26.161 | 15.335 | 58.716 | 72.760 | 95.228 | 220.889 |
| big_muxpp_v5_pp2tp2 | TTFT (s) | 555 | 12.702 | 10.989 | 24.274 | 31.296 | 37.068 | 45.156 |
| big_muxpp_v5_pp2tp2 | TPOT (ms) | 552 | 63.389 | 63.423 | 77.600 | 85.840 | 99.155 | 102.882 |
| big_muxpp_v5_pp2tp2 | E2E (s) | 555 | 22.704 | 17.691 | 44.838 | 56.589 | 74.954 | 145.366 |

## 全部 SLO 阈值

| 系统 | ttft_le_2s | ttft_le_5s | ttft_le_10s | tpot_le_50ms | tpot_le_100ms | tpot_le_200ms | ttft_le_2s_and_tpot_le_50ms | ttft_le_2s_and_tpot_le_100ms | ttft_le_2s_and_tpot_le_200ms | ttft_le_5s_and_tpot_le_50ms | ttft_le_5s_and_tpot_le_100ms | ttft_le_5s_and_tpot_le_200ms | ttft_le_10s_and_tpot_le_50ms | ttft_le_10s_and_tpot_le_100ms | ttft_le_10s_and_tpot_le_200ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| big_muxwise_tp4 | 7.568% | 25.225% | 56.937% | 85.225% | 87.207% | 97.117% | 6.126% | 7.207% | 7.568% | 18.198% | 19.640% | 24.144% | 46.667% | 48.108% | 55.495% |
| big_chunked_pp2tp2 | 19.820% | 50.450% | 76.577% | 10.811% | 36.036% | 59.459% | 1.081% | 3.243% | 6.847% | 4.324% | 13.694% | 22.703% | 7.027% | 25.225% | 39.279% |
| big_muxpp_v5_pp2tp2 | 5.225% | 19.640% | 45.586% | 13.874% | 98.919% | 99.459% | 0.180% | 5.225% | 5.225% | 1.622% | 19.459% | 19.459% | 6.126% | 44.685% | 45.225% |

TTFT/E2E 分位以全部计划请求为样本；缺少首token/未成功完成的值取+∞，显示为 >cutoff。TPOT 仅采用截止前成功完成且输出>1的请求，上表列出样本数。SLO 分母为全部计划请求，unfinished/failed/未发均不达标。吞吐计入截止前交付的全部输出（含部分输出）；输入只计截止前已收到首token的请求。分母为min(实际时长,截止时刻)。

ITL 是两个增加 completion_tokens 的 SSE 事件之间的观测间隔；合包时无法还原真实逐 token ITL。缓存使用成功响应的输入token加权比例；字段缺失显示为 —。PD decode响应的缓存字段不代表prefill实际命中。CSV 保留完整指标（延迟单位为秒，比例范围 0–1）。每版本仅一次正式测量，无置信区间。

## 运行条件

| 系统 | time-scale | 请求数 | 预热 | 中断 | trace |
| --- | --- | --- | --- | --- | --- |
| big_muxwise_tp4 | 1.0 | 555 | 2 | False | /root/workspace/muxpp_work/bench/big_toolagent_600_80_x7.5.jsonl |
| big_chunked_pp2tp2 | 1.0 | 555 | 2 | False | /root/workspace/muxpp_work/bench/big_toolagent_600_80_x7.5.jsonl |
| big_muxpp_v5_pp2tp2 | 1.0 | 555 | 2 | False | /root/workspace/muxpp_work/bench/big_toolagent_600_80_x7.5.jsonl |

比较要求相同 trace、time-scale 和请求数量；不同版本/系统的源码、后端、KV容量和配置差异必须结合主报告解释。

![TTFT CDF](big_best_ttft_cdf.png)

![TPOT CDF](big_best_tpot_cdf.png)

![E2E CDF](big_best_e2e_cdf.png)

![TTFT timeline](big_best_ttft_over_time.png)

![SLO attainment](big_best_slo_attainment.png)
