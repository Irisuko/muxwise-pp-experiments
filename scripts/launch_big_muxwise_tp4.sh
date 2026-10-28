#!/usr/bin/env bash
set -euo pipefail
source /root/workspace/muxpp_work/env.sh
exec python /root/workspace/muxpp_work/scripts/p10b_serve.py launch big_muxwise_tp4 "$@"
