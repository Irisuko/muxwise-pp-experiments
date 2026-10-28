# Tool&Agent v3 / v4（历史对照，不是同日消融）

| 系统 | 配置 | label | 完成/未完/失败 | TTFT p50/p90/p99(s) | TPOT p50/p99(ms); n | E2E p50(s) | 输入/输出 tok/s | 主 SLO | TTFT≤2s | 10s/100ms SLO | cache命中率† | 每实例KV(token) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MuxWise-PP | PP2×TP2 | muxpp_pp2tp2 | 1551/0/0 | 4.742/8.453/11.795 | 34.787/85.038; n=1543 | 7.237 | 18270.858/425.309 | 55.513% | 7.028% | 95.035% | 43.271% | pp2tp2(model)=390622 |
| MuxWise-PP | PP4 | muxwise_pp_v3 | 1551/0/0 | 6.150/9.763/13.365 | 33.940/90.305; n=1543 | 8.624 | 18090.880/421.120 | 31.915% | 4.191% | 89.555% | 42.880% | pp4(model)=359498 |
| MuxWise-PP | PP2DP2 | muxpp_pp2dp2 | 1551/0/0 | 5.817/19.423/31.574 | 39.622/77.895; n=1543 | 10.967 | 18111.926/421.610 | 40.232% | 12.250% | 71.051% | 37.659% | worker0(model)=126501; worker1(model)=126411 |
| MuxWise-PP | PP2×TP2 v4 | muxpp_v4_pp2tp2 | 1551/0/0 | 3.654/7.368/10.646 | 46.810/83.826; n=1543 | 6.495 | 18218.426/424.089 | 70.084% | 18.375% | 97.614% | 43.326% | worker0(model)=390622 |
| MuxWise-PP | PP4 v4 | muxpp_v4_pp4 | 1551/0/0 | 3.524/6.664/10.328 | 50.101/86.186; n=1543 | 6.277 | 18075.790/420.768 | 72.792% | 14.442% | 97.872% | 42.935% | worker0(model)=359498 |
| MuxWise-PP | PP2DP2 v4 | muxpp_v4_pp2dp2 | 1551/0/0 | 4.459/16.468/25.862 | 45.654/77.645; n=1543 | 10.389 | 18089.650/421.091 | 55.384% | 26.692% | 74.919% | 38.210% | worker0(model)=126501; worker1(model)=126501 |

主SLO：TTFT≤5s且TPOT≤100ms。每配置一次测量，无置信区间。SLO/TTFT/E2E分母为全部1551条计划请求；TPOT仅取已完成且输出>1的样本。吞吐包含排空，输入包含缓存。KV为每实例逻辑容量，不乘TP/PP rank数。† PD cache仅为decode响应。

## 运行配置与GPU采样

| label | mem fraction | GPU均值% | 样本数 | 发送延迟p99/max(ms) | 正式日志 |
| --- | --- | --- | --- | --- | --- |
| muxpp_pp2tp2 | {'pp2tp2': 0.71} | 96.048 | 484 | 29.112/31.428 | /root/workspace/muxpp_work/archive/superseded/p9b/logs/p8a/formal-once |
| muxwise_pp_v3 | {'pp4': 0.71} | 96.061 | 488 | 8.672/30.767 | /root/workspace/muxpp_work/archive/superseded/p9b/logs/p6/muxwise_pp_v3-formal |
| muxpp_pp2dp2 | {'worker0': 0.71, 'worker1': 0.71} | 90.561 | 488 | 24.604/26.369 | /root/workspace/muxpp_work/archive/superseded/p9b/logs/p7a/muxpp_pp2dp2/p7b-formal |
| muxpp_v4_pp2tp2 | {'worker0': 0.71} | 92.119 | 488 | 24.266/31.761 | /root/workspace/muxpp_work/logs/p9b/muxpp_v4_pp2tp2 |
| muxpp_v4_pp4 | {'worker0': 0.71} | 91.887 | 488 | 23.483/26.805 | /root/workspace/muxpp_work/logs/p9b/muxpp_v4_pp4 |
| muxpp_v4_pp2dp2 | {'worker0': 0.71, 'worker1': 0.71} | 88.819 | 492 | 27.888/29.422 | /root/workspace/muxpp_work/logs/p9b/muxpp_v4_pp2dp2 |

GPU每5秒采样，仅取客户端测量窗口（包括排空，不含长预热/冒烟）；GPU利用率包含通信等待，不能等同于有效计算效率。

## 完整延迟分位

| 系统 | 指标 | 样本数 | mean | p50 | p90 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| muxpp_pp2tp2 | TTFT (s) | 1551 | 5.096 | 4.742 | 8.453 | 9.890 | 11.795 | 16.059 |
| muxpp_pp2tp2 | TPOT (ms) | 1543 | 37.809 | 34.787 | 60.524 | 69.441 | 85.038 | 124.941 |
| muxpp_pp2tp2 | E2E (s) | 1551 | 10.518 | 7.237 | 23.072 | 27.209 | 36.373 | 79.902 |
| muxwise_pp_v3 | TTFT (s) | 1551 | 6.354 | 6.150 | 9.763 | 10.940 | 13.365 | 18.274 |
| muxwise_pp_v3 | TPOT (ms) | 1543 | 38.365 | 33.940 | 56.121 | 63.390 | 90.305 | 213.447 |
| muxwise_pp_v3 | E2E (s) | 1551 | 12.095 | 8.624 | 24.552 | 28.848 | 38.766 | 91.276 |
| muxpp_pp2dp2 | TTFT (s) | 1551 | 8.400 | 5.817 | 19.423 | 24.274 | 31.574 | 36.700 |
| muxpp_pp2dp2 | TPOT (ms) | 1543 | 41.499 | 39.622 | 57.180 | 63.280 | 77.895 | 229.825 |
| muxpp_pp2dp2 | E2E (s) | 1551 | 14.642 | 10.967 | 30.611 | 36.333 | 51.403 | 103.209 |
| muxpp_v4_pp2tp2 | TTFT (s) | 1551 | 4.068 | 3.654 | 7.368 | 8.518 | 10.646 | 15.185 |
| muxpp_v4_pp2tp2 | TPOT (ms) | 1543 | 46.730 | 46.810 | 62.869 | 72.246 | 83.826 | 123.459 |
| muxpp_v4_pp2tp2 | E2E (s) | 1551 | 10.623 | 6.495 | 24.794 | 29.008 | 39.955 | 91.436 |
| muxpp_v4_pp4 | TTFT (s) | 1551 | 3.884 | 3.524 | 6.664 | 7.810 | 10.328 | 14.962 |
| muxpp_v4_pp4 | TPOT (ms) | 1543 | 49.803 | 50.101 | 62.938 | 70.884 | 86.186 | 182.697 |
| muxpp_v4_pp4 | E2E (s) | 1551 | 11.222 | 6.277 | 26.554 | 30.391 | 42.393 | 101.000 |
| muxpp_v4_pp2dp2 | TTFT (s) | 1551 | 6.721 | 4.459 | 16.468 | 20.469 | 25.862 | 36.454 |
| muxpp_v4_pp2dp2 | TPOT (ms) | 1543 | 45.796 | 45.654 | 59.625 | 65.369 | 77.645 | 95.935 |
| muxpp_v4_pp2dp2 | E2E (s) | 1551 | 13.583 | 10.389 | 30.335 | 37.286 | 48.681 | 90.531 |

## 全部 SLO 阈值

| 系统 | ttft_le_2s | ttft_le_5s | ttft_le_10s | tpot_le_50ms | tpot_le_100ms | tpot_le_200ms | ttft_le_2s_and_tpot_le_50ms | ttft_le_2s_and_tpot_le_100ms | ttft_le_2s_and_tpot_le_200ms | ttft_le_5s_and_tpot_le_50ms | ttft_le_5s_and_tpot_le_100ms | ttft_le_5s_and_tpot_le_200ms | ttft_le_10s_and_tpot_le_50ms | ttft_le_10s_and_tpot_le_100ms | ttft_le_10s_and_tpot_le_200ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| muxpp_pp2tp2 | 7.028% | 56.093% | 96.003% | 78.208% | 99.033% | 99.484% | 5.738% | 6.963% | 7.028% | 43.585% | 55.513% | 55.706% | 74.919% | 95.035% | 95.487% |
| muxwise_pp_v3 | 4.191% | 32.495% | 90.716% | 84.913% | 98.711% | 99.355% | 3.611% | 4.126% | 4.191% | 25.725% | 31.915% | 32.302% | 78.143% | 89.555% | 90.200% |
| muxpp_pp2dp2 | 12.250% | 40.554% | 71.373% | 74.726% | 99.355% | 99.420% | 11.734% | 12.250% | 12.250% | 33.785% | 40.232% | 40.297% | 57.769% | 71.051% | 71.115% |
| muxpp_v4_pp2tp2 | 18.375% | 70.471% | 98.517% | 63.121% | 99.097% | 99.484% | 8.575% | 18.311% | 18.311% | 43.069% | 70.084% | 70.084% | 61.702% | 97.614% | 98.001% |
| muxpp_v4_pp4 | 14.442% | 73.630% | 98.839% | 49.452% | 99.033% | 99.484% | 4.900% | 14.313% | 14.378% | 35.590% | 72.792% | 73.243% | 48.678% | 97.872% | 98.324% |
| muxpp_v4_pp2dp2 | 26.692% | 55.577% | 75.306% | 62.476% | 99.484% | 99.484% | 12.315% | 26.628% | 26.628% | 32.237% | 55.384% | 55.384% | 46.099% | 74.919% | 74.919% |

TTFT/E2E 分位以全部计划请求为样本；缺少首token/未成功完成的值取+∞，显示为 >cutoff。TPOT 仅采用截止前成功完成且输出>1的请求，上表列出样本数。SLO 分母为全部计划请求，unfinished/failed/未发均不达标。吞吐计入截止前交付的全部输出（含部分输出）；输入只计截止前已收到首token的请求。分母为min(实际时长,截止时刻)。

ITL 是两个增加 completion_tokens 的 SSE 事件之间的观测间隔；合包时无法还原真实逐 token ITL。缓存使用成功响应的输入token加权比例；字段缺失显示为 —。PD decode响应的缓存字段不代表prefill实际命中。CSV 保留完整指标（延迟单位为秒，比例范围 0–1）。每版本仅一次正式测量，无置信区间。

## 运行条件

| 系统 | time-scale | 请求数 | 预热 | 中断 | trace |
| --- | --- | --- | --- | --- | --- |
| muxpp_pp2tp2 | 1.0 | 1551 | 2 | False | /root/workspace/muxpp_work/bench/toolagent_600_240_x2.5.jsonl |
| muxwise_pp_v3 | 1.0 | 1551 | 2 | False | /root/workspace/muxpp_work/bench/toolagent_600_240_x2.5.jsonl |
| muxpp_pp2dp2 | 1.0 | 1551 | 2 | False | /root/workspace/muxpp_work/bench/toolagent_600_240_x2.5.jsonl |
| muxpp_v4_pp2tp2 | 1.0 | 1551 | 2 | False | /root/workspace/muxpp_work/bench/toolagent_600_240_x2.5.jsonl |
| muxpp_v4_pp4 | 1.0 | 1551 | 2 | False | /root/workspace/muxpp_work/bench/toolagent_600_240_x2.5.jsonl |
| muxpp_v4_pp2dp2 | 1.0 | 1551 | 2 | False | /root/workspace/muxpp_work/bench/toolagent_600_240_x2.5.jsonl |

比较要求相同 trace、time-scale 和请求数量；不同版本/系统的源码、后端、KV容量和配置差异必须结合主报告解释。

![TTFT CDF](p9b_v3_v4_ttft_cdf.png)

![TPOT CDF](p9b_v3_v4_tpot_cdf.png)

![E2E CDF](p9b_v3_v4_e2e_cdf.png)

![TTFT timeline](p9b_v3_v4_ttft_over_time.png)

![SLO attainment](p9b_v3_v4_slo_attainment.png)
