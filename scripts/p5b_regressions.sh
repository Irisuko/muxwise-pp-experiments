#!/usr/bin/env bash
set -euo pipefail
source /root/workspace/muxpp_work/env.sh
work=/root/workspace/muxpp_work
flags=(--pdmux-config-path "$work/configs/pdmux_pp_4090_v3.yml" --max-prefill-tokens 16384 --pdmux-pp-max-budget 16384 --pdmux-pp-max-inflight-prefill 6 --mem-fraction-static 0.71 --cuda-graph-bs 1 4 8 16 32 --pdmux-pp-tpot-coeff 0.75 --pdmux-prefill-mlp-chunk-tokens 4096)
policy=(--dtype bfloat16 --schedule-policy fcfs --enable-cache-report --pdmux-pp-radix-eviction lru --pdmux-pp-policy alp --pdmux-pp-slo-ttft 5.0 --pdmux-pp-slo-tpot 0.1 --pdmux-pp-reserve-decode --pdmux-pp-balance-merge --pdmux-pp-rebalance --pdmux-pp-alp-admission --pdmux-pp-rebalance-unit 8 --pdmux-pp-summary-interval 10)
python "$work/scripts/p5b_mlp_gpu.py" > "$work/logs/p5b/mlp-gpu-driver.log" 2>&1
python "$work/scripts/p5b_verify.py" normal --suite correctness --tag tp1-correctness > "$work/logs/p5b/tp1-correctness-driver.log" 2>&1
SGLANG_PDMUX_PP_ROUND_TRACE="$work/logs/p5b/v3-serial-rounds.jsonl" python "$work/scripts/p5b_verify.py" muxpp --suite correctness --tag v3-correctness --extra "${flags[@]}" "${policy[@]}" > "$work/logs/p5b/v3-correctness-driver.log" 2>&1
python "$work/scripts/p5b_verify.py" muxpp --suite control --tag v3-control --extra "${flags[@]}" "${policy[@]}" > "$work/logs/p5b/v3-control-driver.log" 2>&1
python "$work/scripts/p5b_verify.py" muxpp --suite stress --tag v3-stress --limit 200 --time-scale 0.02 --force-close --extra "${flags[@]}" "${policy[@]}" > "$work/logs/p5b/v3-stress-driver.log" 2>&1
python "$work/scripts/p5b_verify.py" pdmux --suite smoke --tag original-pdmux > "$work/logs/p5b/original-pdmux-driver.log" 2>&1
python "$work/scripts/p5b_verify.py" pp --suite smoke --tag original-pp > "$work/logs/p5b/original-pp-driver.log" 2>&1
python "$work/scripts/p5b_verify.py" pdmux --suite correctness --tag pdmux-chunk-correctness --extra --pdmux-prefill-mlp-chunk-tokens 4096 > "$work/logs/p5b/pdmux-chunk-correctness-driver.log" 2>&1
# Same serial20 correctness corpus at frozen v2 provides an exact bs/ctx decode comparison.
P5B_SOURCE=/root/workspace/sglang-p5b-baseline SGLANG_PDMUX_PP_ROUND_TRACE="$work/logs/p5b/v2-serial-rounds.jsonl" python "$work/scripts/p5b_verify.py" muxpp --suite correctness --tag v2-paired-correctness --extra --pdmux-config-path "$work/configs/pdmux_pp_4090_v2.yml" --max-prefill-tokens 16384 --pdmux-pp-max-budget 16384 --pdmux-pp-max-inflight-prefill 6 --mem-fraction-static 0.63 --cuda-graph-bs 1 4 8 16 32 --pdmux-pp-tpot-coeff 0.75 "${policy[@]}" > "$work/logs/p5b/v2-paired-correctness-driver.log" 2>&1
python "$work/scripts/p5b_check_regressions.py" > "$work/logs/p5b/regression-audit.log" 2>&1
