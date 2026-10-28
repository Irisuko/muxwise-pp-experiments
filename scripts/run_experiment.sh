#!/usr/bin/env bash
set -euo pipefail
source /root/workspace/muxpp_work/env.sh
case "${1:-}" in
  muxpp_v51_pp2tp2|conv_muxpp_v51_pp2tp2|big_muxpp_v51_pp2tp2)
    exec python /root/workspace/muxpp_work/scripts/p12a_experiment.py experiment "$@"
    ;;
  big_muxwise_tp4|big_chunked_tp4|big_chunked_pp4|big_chunked_pp2tp2|big_muxpp_pp4|big_muxpp_pp2tp2)
    exec python /root/workspace/muxpp_work/scripts/p10a_runner.py experiment "$@"
    ;;
  conv_muxwise_tp2dp2|conv_muxwise_tp4|conv_chunked_pp2dp2|conv_chunked_tp2dp2|conv_muxpp_v4_pp2tp2|conv_muxpp_v4_pp4|muxpp_v4_pp2tp2|muxpp_v4_pp4|muxpp_v4_pp2dp2)
    exec python /root/workspace/muxpp_work/scripts/p9b_runner.py experiment "$@"
    ;;
  muxpp_pp2tp2)
    exec python /root/workspace/muxpp_work/scripts/p8a_runner.py experiment "$@"
    ;;
  muxwise|sglang_pd|muxwise_pp)
    echo "Superseded label: formal replay disabled; use the final version." >&2
    exit 2
    ;;
  chunked_tp2dp2|chunked_pp2dp2|chunked_tp4|chunked_pp4|chunked_tp2pp2|muxwise_tp2dp2|muxwise_tp4|pd_1p1d_tp2|pd_3p1d|pd_1p3d|muxpp_pp2dp2)
    exec python /root/workspace/muxpp_work/scripts/p7a_runner.py experiment "$@"
    ;;
esac
if [[ "${1:-}" == "muxwise_pp_v3" || "${1:-}" == "muxwise_r2" ]]; then
  exec python /root/workspace/muxpp_work/scripts/p6_runner.py experiment "$@"
fi
if [[ "${1:-}" == "sglang_pd_r3" ]]; then
  shift
  exec python /root/workspace/muxpp_work/scripts/p5a_runner.py experiment "$@"
fi
if [[ "${1:-}" == "sglang_pd_r2" ]]; then
  shift
  exec python /root/workspace/muxpp_work/scripts/p4a_runner.py experiment "$@"
fi
exec python /root/workspace/muxpp_work/scripts/p3_runner.py experiment "$@"
