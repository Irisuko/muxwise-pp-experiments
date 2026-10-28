#!/bin/bash
# Wait for P12a's Codex PID to exit, then launch P12b and record its PID and exit code.
W=/root/workspace/muxpp_work
tail --pid="$(cat $W/logs/p12a.pid)" -f /dev/null
for i in $(seq 1 60); do [ -f $W/logs/p12a.rc ] && break; sleep 5; done
cd /root/workspace
codex exec -m gpt-6-astra -c model_reasoning_effort="xhigh" --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check -C /root/workspace -o $W/logs/p12b_last.md - < $W/prompts/p12b_baselines_newtopo.md > $W/logs/p12b_codex.log 2>&1 &
echo $! > $W/logs/p12b.pid
wait $!
echo $? > $W/logs/p12b.rc
