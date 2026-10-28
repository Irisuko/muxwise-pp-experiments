#!/usr/bin/env bash
set -euo pipefail
source /root/workspace/muxpp_work/env.sh
python /root/workspace/muxpp_work/scripts/p4b_gate.py
python /root/workspace/muxpp_work/scripts/p4b_runner.py muxwise_pp --tag v1 --baseline > /root/workspace/muxpp_work/logs/p4b/v1-driver.log 2>&1
python /root/workspace/muxpp_work/scripts/p4b_runner.py muxwise_pp --tag v1-profile --profile-v1 > /root/workspace/muxpp_work/logs/p4b/v1-profile-driver.log 2>&1
python /root/workspace/muxpp_work/scripts/p4b_runner.py chunked --tag chunked > /root/workspace/muxpp_work/logs/p4b/chunked-driver.log 2>&1
