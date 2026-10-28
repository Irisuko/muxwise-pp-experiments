"""Read-only final source, development-scope, and process cleanup audit."""
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

import psutil

root = Path("/root/workspace/muxpp_work")
source = Path("/root/workspace/sglang")
log = root / "logs/p5b"


def command(args):
    return subprocess.check_output(args, text=True).strip()


head = command(["git", "-C", str(source), "rev-parse", "HEAD"])
dirty = command(["git", "-C", str(source), "status", "--porcelain=v1"])
assert head == "f91e8a18da502106a07bb67c67dbf351a933bfd9" and not dirty
compute = command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"])
assert not compute, compute
tasks = []
task_scripts = {
    "p5b_gate.py", "p5b_runner.py", "p5b_verify.py", "p5b_baseline.sh",
    "p5b_regressions.sh", "p5b_cold_regressions.sh", "p5b_transport_gpu.py",
    "p5b_mlp_gpu.py", "p5b_mlp_gpu_warmed.py",
}
for p in psutil.process_iter(["pid", "name", "cmdline", "status"]):
    args = p.info["cmdline"] or []
    if p.info["status"] == psutil.STATUS_ZOMBIE:
        continue
    if (
        any(Path(a).name in task_scripts for a in args)
        or "sglang.launch_server" in args
        or "sglang_router.launch_router" in args
        or p.info["name"].startswith("sglang::")
        or (p.info["name"] == "nvidia-smi" and any("--loop" in a for a in args))
    ):
        tasks.append(p.info)
assert not tasks, tasks
ports = [
    dict(pid=c.pid, address=list(c.laddr))
    for c in psutil.net_connections(kind="tcp")
    if c.status == psutil.CONN_LISTEN and c.laddr.port in {30000, 30001, 30002, 30003, 31000}
]
assert not ports, ports
tags = ["r1-v2-instrumented", "r2-async", "r3-capacity", "r4-inflight8"]
performance = []
for tag in tags:
    r = json.loads((log / tag / "run.json").read_text())
    s = r["summary"]
    assert r["status"] == "PASS"
    assert s["trace"] == str(root / "bench/dev_toolagent_1200_240_x2.5.jsonl")
    assert s["limit"] == 300 and s["time_scale"] == 1 and s["warmup"] == 2 and s["force_close"]
    assert s["completed_requests"] == 300 and s["failed_requests"] == 0
    performance.append(dict(tag=tag, source_commit=r["commit"], trace=s["trace"], completed=300))
assert sorted(p.name for p in log.glob("r[1-4]-*") if p.is_dir()) == sorted(tags)
selected = json.loads((log / "r3-capacity/run.json").read_text())
for rel, sha in selected["source_sha256"].items():
    path = source / "python/sglang/srt" / rel
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha, rel
assert (source / "configs/pdmux_pp_4090_v3.yml").read_bytes() == (root / "configs/pdmux_pp_4090_v3.yml").read_bytes()
regression = json.loads((log / "regression-audit.json").read_text())
assert regression["v3-cold-correctness"]["status"] == "PASS"
assert regression["pdmux-chunk-cold-correctness"]["status"] == "PASS"
assert regression["token_equality"]["mismatches"] == 0
out = dict(
    status="PASS",
    at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_commit=head,
    source_dirty=dirty,
    selected_source_hashes_match=True,
    compute_pids=compute,
    task_processes=tasks,
    listeners=ports,
    gpu=command(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv"]),
    performance_rounds=performance,
    formal_trace_replayed=False,
    notes_sha256=hashlib.sha256((root / "P5B_NOTES.md").read_bytes()).hexdigest(),
)
(log / "cleanup-final.json").write_text(json.dumps(out, indent=2))
(log / "STATE.md").write_text(
    "# P5b complete\n\n"
    f"Final local commit `{head}`; clean main tree; no push.\n\n"
    "Four development rounds complete. Selected R3: mem=.71, inflight=6, chunk=4096, async decode.\n\n"
    "Correctness, cold-cache chunk coverage, cancellation, stress200, original PP/pdmux passed. "
    "GPU/process/port cleanup verified. See ../../P5B_NOTES.md and cleanup-final.json.\n"
)
print(json.dumps(out, indent=2))
