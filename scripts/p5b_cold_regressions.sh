#!/usr/bin/env bash
set -euo pipefail
source /root/workspace/muxpp_work/env.sh
work=/root/workspace/muxpp_work
flags=(--pdmux-config-path "$work/configs/pdmux_pp_4090_v3.yml" --max-prefill-tokens 16384 --pdmux-pp-max-budget 16384 --pdmux-pp-max-inflight-prefill 6 --mem-fraction-static 0.71 --cuda-graph-bs 1 4 8 16 32 --pdmux-pp-tpot-coeff 0.75 --pdmux-prefill-mlp-chunk-tokens 4096)
policy=(--dtype bfloat16 --schedule-policy fcfs --enable-cache-report --pdmux-pp-radix-eviction lru --pdmux-pp-policy alp --pdmux-pp-slo-ttft 5.0 --pdmux-pp-slo-tpot 0.1 --pdmux-pp-reserve-decode --pdmux-pp-balance-merge --pdmux-pp-rebalance --pdmux-pp-alp-admission --pdmux-pp-rebalance-unit 8 --pdmux-pp-summary-interval 10)
python "$work/scripts/p5b_verify.py" normal --suite correctness --cold-cache --tag tp1-cold-correctness > "$work/logs/p5b/tp1-cold-correctness-driver.log" 2>&1
python "$work/scripts/p5b_verify.py" muxpp --suite correctness --cold-cache --tag v3-cold-correctness --extra "${flags[@]}" "${policy[@]}" > "$work/logs/p5b/v3-cold-correctness-driver.log" 2>&1
python "$work/scripts/p5b_verify.py" pdmux --suite correctness --cold-cache --tag pdmux-chunk-cold-correctness --extra --pdmux-prefill-mlp-chunk-tokens 4096 > "$work/logs/p5b/pdmux-chunk-cold-correctness-driver.log" 2>&1
python "$work/scripts/p5b_check_regressions.py" > "$work/logs/p5b/regression-audit.log" 2>&1
