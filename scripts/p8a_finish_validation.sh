#!/usr/bin/env bash
set -euo pipefail
source /root/workspace/muxpp_work/env.sh
work=/root/workspace/muxpp_work
while pgrep -f '^bash /root/workspace/muxpp_work/scripts/p8a_regressions.sh$' >/dev/null; do sleep 2; done
python - <<'PY'
import json
from pathlib import Path
root=Path('/root/workspace/muxpp_work/logs/p8a')
for name in ['tp2-reference','tp2-reference-cold','tp2pp2-serial','tp2pp2-cold','tp2pp2-cancel','tp2pp2-stress','pp4-v3-regression','pdmux-tp1-regression','pdmux-tp2-regression','pp2-regression']:
    assert json.loads((root/f'{name}-result.json').read_text())['status']=='PASS',name
PY
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 "$work/scripts/p8a_mlp_gpu.py" > "$work/logs/p8a/mlp-gpu.log" 2>&1
python "$work/scripts/p8a_runner.py" sanity muxpp_pp2tp2 --tag dev300-mem071 > "$work/logs/p8a/dev300-driver.log" 2>&1
