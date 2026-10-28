# Tool&Agent trace 回放对比

| 系统 | 完成 | 未完成 | 失败 | 时长(s) | 输入 tok/s | 输出 tok/s | req/s | TTFT 均值(s) | TTFT p50(s) | TTFT p90(s) | TTFT p99(s) | TPOT 均值(ms) | TPOT p50(ms) | TPOT p90(ms) | TPOT p99(ms) | E2E p99(s) | 联合 SLO 5s/100ms | 输入token加权缓存比例 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sat_muxwise_tp2dp2 | 1088 | 463 | 0 | 330.013 | 24242.785 | 516.892 | 3.297 | >cutoff | 220.923 | >cutoff | >cutoff | 31.249 | 24.598 | 37.350 | 118.066 | >cutoff | 1.418% | 46.160% |
| sat_chunked_pp2dp2 | 1159 | 390 | 2 | 330.004 | 24703.146 | 485.280 | 3.512 | >cutoff | 172.261 | >cutoff | >cutoff | 205.659 | 79.377 | 212.063 | 4790.238 | >cutoff | 0.580% | 46.314% |
| sat_muxpp_v5_pp4 | 1211 | 340 | 0 | 330.012 | 27119.113 | 592.793 | 3.670 | >cutoff | 196.363 | >cutoff | >cutoff | 53.926 | 52.050 | 65.022 | 90.960 | >cutoff | 1.289% | 41.448% |
| sat_big_muxwise_tp4 | 293 | 262 | 0 | 329.255 | 6636.127 | 133.079 | 0.890 | >cutoff | 299.731 | >cutoff | >cutoff | 48.039 | 31.269 | 138.368 | 172.638 | >cutoff | 0.901% | 45.029% |
| sat_big_chunked_pp2tp2 | 318 | 237 | 0 | 329.256 | 7612.011 | 141.950 | 0.966 | >cutoff | 263.337 | >cutoff | >cutoff | 870.195 | 158.891 | 1105.653 | 15242.762 | >cutoff | 0.000% | 41.399% |
| sat_big_muxpp_v5_pp2tp2 | 371 | 184 | 0 | 329.255 | 8299.671 | 175.499 | 1.127 | >cutoff | 237.874 | >cutoff | >cutoff | 70.799 | 66.130 | 88.112 | 121.852 | >cutoff | 0.180% | 42.623% |

## 完整延迟分位

| 系统 | 指标 | 样本数 | mean | p50 | p90 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sat_muxwise_tp2dp2 | TTFT (s) | 1551 | >cutoff | 220.923 | >cutoff | >cutoff | >cutoff | >cutoff |
| sat_muxwise_tp2dp2 | TPOT (ms) | 1082 | 31.249 | 24.598 | 37.350 | 90.297 | 118.066 | 144.386 |
| sat_muxwise_tp2dp2 | E2E (s) | 1551 | >cutoff | 222.077 | >cutoff | >cutoff | >cutoff | >cutoff |
| sat_chunked_pp2dp2 | TTFT (s) | 1551 | >cutoff | 172.261 | >cutoff | >cutoff | >cutoff | >cutoff |
| sat_chunked_pp2dp2 | TPOT (ms) | 1152 | 205.659 | 79.377 | 212.063 | 505.143 | 4790.238 | 6371.538 |
| sat_chunked_pp2dp2 | E2E (s) | 1551 | >cutoff | 177.523 | >cutoff | >cutoff | >cutoff | >cutoff |
| sat_muxpp_v5_pp4 | TTFT (s) | 1551 | >cutoff | 196.363 | >cutoff | >cutoff | >cutoff | >cutoff |
| sat_muxpp_v5_pp4 | TPOT (ms) | 1205 | 53.926 | 52.050 | 65.022 | 74.827 | 90.960 | 133.630 |
| sat_muxpp_v5_pp4 | E2E (s) | 1551 | >cutoff | 201.351 | >cutoff | >cutoff | >cutoff | >cutoff |
| sat_big_muxwise_tp4 | TTFT (s) | 555 | >cutoff | 299.731 | >cutoff | >cutoff | >cutoff | >cutoff |
| sat_big_muxwise_tp4 | TPOT (ms) | 293 | 48.039 | 31.269 | 138.368 | 148.144 | 172.638 | 225.158 |
| sat_big_muxwise_tp4 | E2E (s) | 555 | >cutoff | 306.217 | >cutoff | >cutoff | >cutoff | >cutoff |
| sat_big_chunked_pp2tp2 | TTFT (s) | 555 | >cutoff | 263.337 | >cutoff | >cutoff | >cutoff | >cutoff |
| sat_big_chunked_pp2tp2 | TPOT (ms) | 317 | 870.195 | 158.891 | 1105.653 | 4012.802 | 15242.762 | 15653.321 |
| sat_big_chunked_pp2tp2 | E2E (s) | 555 | >cutoff | 281.284 | >cutoff | >cutoff | >cutoff | >cutoff |
| sat_big_muxpp_v5_pp2tp2 | TTFT (s) | 555 | >cutoff | 237.874 | >cutoff | >cutoff | >cutoff | >cutoff |
| sat_big_muxpp_v5_pp2tp2 | TPOT (ms) | 370 | 70.799 | 66.130 | 88.112 | 96.029 | 121.852 | 132.374 |
| sat_big_muxpp_v5_pp2tp2 | E2E (s) | 555 | >cutoff | 246.396 | >cutoff | >cutoff | >cutoff | >cutoff |

## 全部 SLO 阈值

| 系统 | ttft_le_2s | ttft_le_5s | ttft_le_10s | tpot_le_50ms | tpot_le_100ms | tpot_le_200ms | ttft_le_2s_and_tpot_le_50ms | ttft_le_2s_and_tpot_le_100ms | ttft_le_2s_and_tpot_le_200ms | ttft_le_5s_and_tpot_le_50ms | ttft_le_5s_and_tpot_le_100ms | ttft_le_5s_and_tpot_le_200ms | ttft_le_10s_and_tpot_le_50ms | ttft_le_10s_and_tpot_le_100ms | ttft_le_10s_and_tpot_le_200ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sat_muxwise_tp2dp2 | 0.580% | 1.418% | 2.837% | 64.410% | 66.925% | 69.761% | 0.580% | 0.580% | 0.580% | 1.418% | 1.418% | 1.418% | 2.837% | 2.837% | 2.837% |
| sat_chunked_pp2dp2 | 1.225% | 3.482% | 4.900% | 14.571% | 50.806% | 66.086% | 0.000% | 0.000% | 0.258% | 0.000% | 0.580% | 1.032% | 0.000% | 1.161% | 1.612% |
| sat_muxpp_v5_pp4 | 0.193% | 1.289% | 3.095% | 31.077% | 77.240% | 77.692% | 0.064% | 0.193% | 0.193% | 0.322% | 1.289% | 1.289% | 0.387% | 3.095% | 3.095% |
| sat_big_muxwise_tp4 | 0.180% | 2.162% | 2.342% | 44.685% | 45.946% | 52.432% | 0.000% | 0.180% | 0.180% | 0.721% | 0.901% | 2.162% | 0.901% | 1.081% | 2.342% |
| sat_big_chunked_pp2tp2 | 0.721% | 2.162% | 4.865% | 5.405% | 15.315% | 39.820% | 0.000% | 0.000% | 0.180% | 0.000% | 0.000% | 0.541% | 0.000% | 0.000% | 0.721% |
| sat_big_muxpp_v5_pp2tp2 | 0.000% | 0.541% | 2.162% | 0.180% | 64.685% | 66.667% | 0.000% | 0.000% | 0.000% | 0.000% | 0.180% | 0.541% | 0.000% | 1.081% | 2.162% |

TTFT/E2E 分位以全部计划请求为样本；缺少首token/未成功完成的值取+∞，显示为 >cutoff。TPOT 仅采用截止前成功完成且输出>1的请求，上表列出样本数。SLO 分母为全部计划请求，unfinished/failed/未发均不达标。吞吐计入截止前交付的全部输出（含部分输出）；输入只计截止前已收到首token的请求。分母为min(实际时长,截止时刻)。

ITL 是两个增加 completion_tokens 的 SSE 事件之间的观测间隔；合包时无法还原真实逐 token ITL。缓存使用成功响应的输入token加权比例；字段缺失显示为 —。PD decode响应的缓存字段不代表prefill实际命中。CSV 保留完整指标（延迟单位为秒，比例范围 0–1）。每版本仅一次正式测量，无置信区间。

## 运行条件

| 系统 | time-scale | 请求数 | 预热 | 中断 | trace |
| --- | --- | --- | --- | --- | --- |
| sat_muxwise_tp2dp2 | 0.05 | 1551 | 2 | False | /root/workspace/muxpp_work/bench/toolagent_600_240_x2.5.jsonl |
| sat_chunked_pp2dp2 | 0.05 | 1551 | 2 | False | /root/workspace/muxpp_work/bench/toolagent_600_240_x2.5.jsonl |
| sat_muxpp_v5_pp4 | 0.05 | 1551 | 2 | False | /root/workspace/muxpp_work/bench/toolagent_600_240_x2.5.jsonl |
| sat_big_muxwise_tp4 | 0.05 | 555 | 2 | False | /root/workspace/muxpp_work/bench/big_toolagent_600_80_x7.5.jsonl |
| sat_big_chunked_pp2tp2 | 0.05 | 555 | 2 | False | /root/workspace/muxpp_work/bench/big_toolagent_600_80_x7.5.jsonl |
| sat_big_muxpp_v5_pp2tp2 | 0.05 | 555 | 2 | False | /root/workspace/muxpp_work/bench/big_toolagent_600_80_x7.5.jsonl |

比较要求相同 trace、time-scale 和请求数量；不同版本/系统的源码、后端、KV容量和配置差异必须结合主报告解释。

![TTFT CDF](sat_ttft_cdf.png)

![TPOT CDF](sat_tpot_cdf.png)

![E2E CDF](sat_e2e_cdf.png)

![TTFT timeline](sat_ttft_over_time.png)

![SLO attainment](sat_slo_attainment.png)
