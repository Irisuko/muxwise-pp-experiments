#!/usr/bin/env bash
set -euo pipefail
source /root/workspace/muxpp_work/env.sh
work=/root/workspace/muxpp_work
flags=(--pdmux-config-path "$work/configs/pdmux_pp_4090_v2.yml" --max-prefill-tokens 16384 --pdmux-pp-max-budget 16384 --pdmux-pp-max-inflight-prefill 6 --mem-fraction-static 0.63 --cuda-graph-bs 1 4 8 16 32 --pdmux-pp-tpot-coeff 0.75)
policy=(--dtype bfloat16 --schedule-policy fcfs --enable-cache-report --pdmux-pp-radix-eviction lru --pdmux-pp-policy alp --pdmux-pp-slo-ttft 5.0 --pdmux-pp-slo-tpot 0.1 --pdmux-pp-reserve-decode --pdmux-pp-balance-merge --pdmux-pp-rebalance --pdmux-pp-alp-admission --pdmux-pp-rebalance-unit 8 --pdmux-pp-summary-interval 10)
python "$work/scripts/p4b_runner.py" muxwise_pp --tag v2-static --extra "${flags[@]}" --pdmux-pp-policy static > "$work/logs/p4b/v2-static-driver.log" 2>&1
python "$work/scripts/p4b_analyze.py" "$work/logs/p4b/v2-static"
python "$work/scripts/p4b_verify.py" normal --suite correctness --tag tp1-correctness > "$work/logs/p4b/tp1-correctness-driver.log" 2>&1
python "$work/scripts/p4b_verify.py" muxpp --suite correctness --tag v2-correctness --extra "${flags[@]}" "${policy[@]}" > "$work/logs/p4b/v2-correctness-driver.log" 2>&1
python "$work/scripts/p4b_verify.py" muxpp --suite control --tag v2-control --extra "${flags[@]}" "${policy[@]}" > "$work/logs/p4b/v2-control-driver.log" 2>&1
python "$work/scripts/p4b_verify.py" muxpp --suite stress --tag v2-stress --limit 200 --time-scale 0.02 --force-close --extra "${flags[@]}" "${policy[@]}" > "$work/logs/p4b/v2-stress-driver.log" 2>&1
python "$work/scripts/p4b_verify.py" pdmux --suite smoke --tag original-pdmux > "$work/logs/p4b/original-pdmux-driver.log" 2>&1
python "$work/scripts/p4b_verify.py" pp --suite smoke --tag original-pp > "$work/logs/p4b/original-pp-driver.log" 2>&1
