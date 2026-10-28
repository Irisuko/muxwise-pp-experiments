#!/usr/bin/env bash
set -euo pipefail
source /root/workspace/muxpp_work/env.sh
work=/root/workspace/muxpp_work
v3=(--pdmux-config-path "$work/configs/pdmux_pp_4090_v3.yml" --max-prefill-tokens 16384 --pdmux-pp-max-budget 16384 --pdmux-pp-max-inflight-prefill 3 --mem-fraction-static 0.71 --cuda-graph-bs 1 4 8 16 32 --pdmux-pp-tpot-coeff 0.75 --pdmux-prefill-mlp-chunk-tokens 4096 --dtype bfloat16 --enable-cache-report)
run() {
  tag="$1"
  shift
  if [[ -f "$work/logs/p8a/$tag-result.json" ]]; then
    python -c "import json; assert json.load(open('$work/logs/p8a/$tag-result.json'))['status'] == 'PASS'"
    return
  fi
  python "$work/scripts/p8a_verify.py" --tag "$tag" "$@" > "$work/logs/p8a/$tag-driver.log" 2>&1
}
run tp2-reference normal --tp 2 --suite correctness
run tp2-reference-cold normal --tp 2 --suite correctness --cold-cache
run tp2pp2-serial muxpp --tp 2 --pp 2 --suite correctness --extra "${v3[@]}"
run tp2pp2-cold muxpp --tp 2 --pp 2 --suite correctness --cold-cache --extra "${v3[@]}"
run tp2pp2-cancel muxpp --tp 2 --pp 2 --suite control --extra "${v3[@]}"
run tp2pp2-stress muxpp --tp 2 --pp 2 --suite stress --force-close --long-warmup --extra "${v3[@]}"
run pp4-v3-regression muxpp --tp 1 --pp 4 --suite correctness --extra "${v3[@]}" --pdmux-pp-max-inflight-prefill 6
run pdmux-tp1-regression pdmux --tp 1
run pdmux-tp2-regression pdmux --tp 2 --extra --mem-fraction-static 0.76
run pp2-regression pp --tp 1 --pp 2
