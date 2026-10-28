#!/usr/bin/env python3
"""Read-only evidence checks, except writing the audit result."""
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

ROOT = Path("/root/workspace/muxpp_work")
LOG = ROOT/"logs/p5a"


def read(p):
    return json.loads(p.read_text())


def main():
    checks = []
    def check(name, ok, detail=None):
        checks.append({"name":name,"pass":bool(ok),"detail":detail})
    for filename, relative in [("original-sha256.json",False),("../p4a/diagnosis/original-sha256.json",True),
                               ("input-code-sha256.json",False)]:
        hashes = read(LOG/filename)
        bad = []
        for s,h in hashes.items():
            p = ROOT/s if relative else Path(s)
            if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != h:
                bad.append(s)
        check(f"hashes:{filename}",not bad,{"count":len(hashes),"changed":bad})
    source = Path("/root/workspace/sglang-p4a")
    status = subprocess.check_output(["git","-C",str(source),"status","--porcelain"],text=True)
    check("isolated source clean",not status.strip(),status)
    diff = subprocess.check_output(["git","-C",str(source),"diff","--name-only","19261937a524bc67216d8b925a8ea5e50bfda0f3","HEAD"],text=True).splitlines()
    expected = {
        "benchmark/pd_4090/p5a_profiles.json","benchmark/pd_4090/p5a_runner.snapshot.py",
        "python/sglang/srt/disaggregation/p5a_timing.py","python/sglang/srt/disaggregation/prefill.py",
        "python/sglang/srt/disaggregation/decode.py"}
    check("only P5a source/config/logging changes",set(diff)==expected,diff)
    check("committed runner matches live runner",
          (source/"benchmark/pd_4090/p5a_runner.snapshot.py").read_bytes()==(ROOT/"scripts/p5a_runner.py").read_bytes())
    for name in ["mooncake-r2","nixl-host","nixl-host-1m","nixl-ipc-host-1m","mooncake-8m"]:
        d = LOG/f"micro-{name}"
        s = read(d/"micro-summary.json")
        points = {(r["prefill_gpu"],r["decode_gpu"],r["tokens"]) for r in s if not r["warmup"]}
        expect = {(p,d,n) for p in [0,1] for d in [2,3] for n in [4096,16384,31663]}
        check(f"micro all paths:{name}",read(d/"run.json")["status"]=="PASS" and points==expect)
        check(f"micro timing:{name}",all(r["prefill_complete_to_received_s"]>0 and
              r["first_send_to_received_s"]>=r["prefill_complete_to_received_s"] and
              r["kv_bytes"]==r["tokens"]*131072 for r in s))
    for name in ["nixl-ipc-sm","nixl-sm-1m"]:
        d = LOG/f"micro-{name}"
        check(f"failed micro retained:{name}",read(d/"run.json")["status"]=="FAIL" and (d/"failure-evidence.json").exists())
    runs = []
    with (ROOT/"bench/dev_toolagent_1200_240_x2.5.jsonl").open() as f:
        dev = [json.loads(line) for line in itertools.islice(f,300)]
    expected_requests = {(r["idx"],r["input_len"],r["output_len"],r["arrival"]) for r in dev}
    for name in ["mooncake-r2","nixl-ipc-host-1m"]:
        d = LOG/f"dev-{name}"
        m = read(d/"run.json")
        s = m["dev300_summary"]
        runs.append(m)
        cmd = read(d/"dev300-command.json")
        check(f"development only:{name}",s["trace"]==str(ROOT/"bench/dev_toolagent_1200_240_x2.5.jsonl")
              and s["planned_requests"]==300 and s["time_scale"]==1 and s["warmup"]==2
              and "--force-close" in cmd and (LOG/f"development-{name}.started.json").exists(),cmd)
        check(f"correctness:{name}",read(d/"correctness-comparison.json")["passed"]==20
              and len(read(d/"long-warmup.json"))==2 and s["completed_requests"]==300
              and s["failed_requests"]==s["unsent_requests"]==0)
        raw = [json.loads(line) for line in (d/"dev300/dev300_requests.jsonl").read_text().splitlines()]
        actual_requests = {(r["idx"],r["input_len"],r["output_len"],r["trace_arrival"]) for r in raw}
        check(f"raw requests match first300 development:{name}",len(raw)==300 and actual_requests==expected_requests
              and sum(r["success"] for r in raw)==s["completed_requests"])
    decision = read(LOG/"decision.json")
    check("same frozen development code",runs[0]["git_commit"]==runs[1]["git_commit"]==decision["source_commit"])
    baseline, candidate = (r["dev300_summary"] for r in runs)
    ratio = candidate["throughput"]["input_tokens_per_sec"]/baseline["throughput"]["input_tokens_per_sec"]
    ttft = candidate["latencies_sec"]["ttft"]["p50"]/baseline["latencies_sec"]["ttft"]["p50"]
    allowed = ratio>=1.5 or ttft<=0.5
    check("decision follows thresholds",decision["performance_pass"]==allowed
          and abs(decision["throughput_ratio"]-ratio)<1e-10 and abs(decision["ttft_p50_ratio"]-ttft)<1e-10)
    marker = ROOT/"results/sglang_pd_r3.formal-started.json"
    if decision["formal_allowed"]:
        check("qualified candidate formal executed",marker.exists() and (ROOT/"results/sglang_pd_r3_summary.json").exists())
        d = Path(read(marker)["log_dir"])
        s = read(ROOT/"results/sglang_pd_r3_summary.json")
        rows = [json.loads(l) for l in (ROOT/"results/sglang_pd_r3_requests.jsonl").read_text().splitlines()]
        check("formal single recorded run",len(list(LOG.glob("*/formal-command.json")))==1 and s["planned_requests"]==1551
              and s["sent_requests"]==len(rows) and s["warmup"]==2 and s["time_scale"]==1)
        cmd = read(d/"formal-command.json")
        check("formal original trace/force-close",s["trace"]==str(ROOT/"bench/toolagent_600_240_x2.5.jsonl")
              and "--force-close" in cmd and "--limit" not in cmd,cmd)
        check("formal gated frozen code",read(d/"run.json")["git_commit"]==decision["source_commit"])
    else:
        check("unqualified candidate no formal",not marker.exists() and not list((ROOT/"results").glob("sglang_pd_r3_*")))
    release = read(LOG/"cleanup.json")
    check("GPU clean at release",not release["compute_apps"].strip())
    check("ports clean at release",release["listening_ports"]==[])
    check("own processes cleaned at release",release["own_live_pids"]==[])
    check("GPU handoff marker exists",(ROOT/"logs/p5a.gpu_done").exists())
    report = (ROOT/"RESULTS.md").read_text()
    check("P4a original content preserved",report.startswith((LOG/"RESULTS.before.md").read_text()) if not decision["formal_run"]
          else "## P4a：" in report)
    check("old rows and P5a retained","| sglang_pd_r2 |" in report and "| sglang_pd（首轮失败，保留） |" in report and "## P5a：" in report)
    labels = [read(p)["label"] for p in (ROOT/"results").glob("*_summary.json")]
    summary = (ROOT/"results/summary.md").read_text()
    check("summarize includes all formal labels",all(l in summary for l in labels),labels)
    result = {"pass":all(c["pass"] for c in checks),"passed":sum(c["pass"] for c in checks),
              "total":len(checks),"checks":checks}
    (LOG/"audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
