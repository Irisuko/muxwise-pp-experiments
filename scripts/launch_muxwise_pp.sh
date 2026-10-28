#!/usr/bin/env bash
set -euo pipefail
source /root/workspace/muxpp_work/env.sh
exec python /root/workspace/muxpp_work/scripts/p3_runner.py launch muxwise_pp "$@"
