#!/usr/bin/env bash
set -euo pipefail
source /root/workspace/muxpp_work/env.sh
python /root/workspace/muxpp_work/scripts/p5b_gate.py
export P5B_SOURCE=/root/workspace/sglang-p5b-baseline
python /root/workspace/muxpp_work/scripts/p5b_runner.py muxwise_pp --tag r1-v2-instrumented --extra --pdmux-config-path /root/workspace/muxpp_work/configs/pdmux_pp_4090_v2.yml --max-prefill-tokens 16384 --pdmux-pp-max-budget 16384 --pdmux-pp-max-inflight-prefill 6 --mem-fraction-static 0.63 --cuda-graph-bs 1 4 8 16 32 --pdmux-pp-tpot-coeff 0.75
python /root/workspace/muxpp_work/scripts/p4b_analyze.py /root/workspace/muxpp_work/logs/p5b/r1-v2-instrumented
