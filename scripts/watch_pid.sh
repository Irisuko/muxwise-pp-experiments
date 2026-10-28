#!/bin/bash
# usage: watch_pid.sh <pid> <phase>   e.g. watch_pid.sh 22410 p2
PID=$1; PH=$2; W=/root/workspace/muxpp_work
START=$(stat -c %Y /proc/$PID 2>/dev/null)
tail --pid="$PID" -f /dev/null
echo "== $PH (pid $PID) exited at $(date '+%F %T')"
LAST=$W/logs/${PH}_last.md; LOG=${LOG_OVERRIDE:-$W/logs/${PH}_codex.log}
if [ -s "$LAST" ] && [ "$(stat -c %Y "$LAST")" -ge "${START:-0}" ]; then
  echo "RESULT: final message written (normal completion)"; else
  echo "RESULT: NO fresh final message -> likely abnormal exit"; fi
grep -nE "^(ERROR|Error)|stream error|rate limit|turn aborted|panicked" "$LOG" | tail -5
echo "== log tail"; tail -c 1500 "$LOG"; echo
echo "== final message"; cat "$LAST" 2>/dev/null | head -40
echo "== git"; git -C /root/workspace/sglang log --oneline -6; git -C /root/workspace/sglang status --short | head
echo "== gpu"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
