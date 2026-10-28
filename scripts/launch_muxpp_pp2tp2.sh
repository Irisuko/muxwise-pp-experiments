#!/usr/bin/env bash
set -euo pipefail
source /root/workspace/muxpp_work/env.sh
exec python /root/workspace/muxpp_work/scripts/p8a_runner.py launch muxpp_pp2tp2 "$@"
