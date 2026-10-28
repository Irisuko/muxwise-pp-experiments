#!/usr/bin/env python3
"""Summarize existing phase stamps and scheduler samples in the replay window."""
import datetime
import json
from pathlib import Path
import re
import statistics
import sys


def analyze(p):
    s = json.loads((p/"dev300/dev300_summary.json").read_text())
    start, end = s["run_started_unix"], s["run_started_unix"] + s["duration_sec"]
    events, batches, pref, paths = {}, [], [], []
    pat = re.compile(r"P5A_KV event=(\w+) ns=(\d+) room=(\d+) tokens=(\d+) start=(\d+) end=(\d+) last=(\d+)")
    for f in p.glob("worker*.log"):
        for line_no, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            if "intra-node cfg" in line:
                paths.append(f"{f.name}:{line_no}: {line}")
            try:
                t = datetime.datetime.strptime(line[1:20], "%Y-%m-%d %H:%M:%S").timestamp()
            except ValueError:
                continue
            if not (start <= t <= end+1):
                continue
            m = pat.search(line)
            if m:
                k, ns, room, n, st, en, last = m.groups()
                events.setdefault(room, []).append((k, int(ns), int(last)))
            m = re.search(r"Decode batch. #running-req: (\d+), #token: (\d+), token usage: ([.\d]+), pre-allocated usage: ([.\d]+)", line)
            if m:
                batches.append(list(map(float, m.groups())))
            m = re.search(r"#unbootstrapped-req: (\d+)", line)
            if m:
                pref.append(int(m[1]))
    spans = []
    for room, ev in events.items():
        a = [ns for k,ns,last in ev if k == "send"]
        b = [ns for k,ns,last in ev if k == "send" and last]
        r = [ns for k,ns,last in ev if k == "received"]
        if room != "0" and a and b and r:
            spans.append(((r[-1]-a[0])/1e9, (r[-1]-b[-1])/1e9))
    result = {
        "rooms":len(spans),
        "full_span_p50_s":statistics.median(a for a,b in spans),
        "residual_p50_s":statistics.median(b for a,b in spans),
        "decode_log_samples":len(batches),
        "running_reqs_median":statistics.median(x[0] for x in batches),
        "token_usage_median":statistics.median(x[2] for x in batches),
        "preallocated_usage_median":statistics.median(x[3] for x in batches),
        "prefill_unbootstrapped_max":max(pref),
        "note":"Log-window sample statistics, not time-weighted or causal attribution; wall timestamps have 1-second granularity.",
    }
    (p/"phase-analysis.json").write_text(json.dumps(result, indent=2)+"\n")
    if paths:
        (p/"ucx-path-evidence.txt").write_text("\n".join(paths)+"\n")
    print(p.name, result)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        analyze(Path(arg))
