#!/usr/bin/env bash
# Reviewed v2 candidate only; this script was NOT executed for a formal experiment.
set -euo pipefail
source /root/workspace/muxpp_work/env.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m sglang.launch_server \
  --model-path /root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct \
  --tp 1 --pp-size 4 --dtype bfloat16 --port 30000 \
  --disable-overlap-schedule --context-length 33792 \
  --schedule-policy fcfs --max-running-requests 128 \
  --watchdog-timeout 180 --log-level debug --enable-cache-report \
  --enable-pdmux --chunked-prefill-size -1 \
  --pdmux-config-path /root/workspace/muxpp_work/configs/pdmux_pp_4090_v2.yml \
  --mem-fraction-static 0.63 --cuda-graph-max-bs 32 --cuda-graph-bs 1 4 8 16 32 \
  --max-prefill-tokens 16384 --pdmux-pp-max-budget 16384 \
  --pdmux-pp-max-inflight-prefill 6 --pdmux-pp-radix-eviction lru \
  --pdmux-pp-policy alp --pdmux-pp-slo-ttft 5.0 --pdmux-pp-slo-tpot 0.1 \
  --pdmux-pp-tpot-coeff 0.75 --pdmux-pp-reserve-decode \
  --pdmux-pp-balance-merge --pdmux-pp-rebalance --pdmux-pp-alp-admission \
  --pdmux-pp-rebalance-unit 8 --pdmux-pp-summary-interval 10
