#!/usr/bin/env bash
set -euo pipefail
source /root/workspace/muxpp_work/env.sh
exec python /root/workspace/muxpp_work/scripts/p7a_runner.py launch muxpp_pp2dp2 "$@"
