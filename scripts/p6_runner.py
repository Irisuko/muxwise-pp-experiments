#!/usr/bin/env python3
"""P6 lifecycle only; immutable inference source and exclusive formal markers."""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import random
import shlex
import shutil
import signal
import subprocess
import time

import p3_runner as base

ROOT = base.ROOT
SOURCE = pathlib.Path("/root/workspace/sglang")
COMMIT = "f91e8a18da502106a07bb67c67dbf351a933bfd9"
DEV_TRACE = ROOT / "bench/dev_toolagent_1200_240_x2.5.jsonl"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_gate():
    commit = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(SOURCE), "status", "--porcelain"], text=True)
    assert commit == COMMIT and not dirty, (commit, dirty)
    forbidden = [k for k in os.environ if k.startswith(("P3_", "SGLANG_PDMUX_PP_"))]
    assert not forbidden, f"Unexpected experiment overrides: {forbidden}"


def make_commands(args):
    if args.label == "muxwise_pp_v3":
        # Read the selected script itself, preserving its exact server argv.
        script = (ROOT / "scripts/p5b_v3_serve.sh").read_text()
        argv = shlex.split(script.split("exec python ", 1)[1].replace("\\\n", " "))
        return [("pp4", [base.PYTHON] + argv, {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}, 30000)]
    jobs = []
    common = [
        base.PYTHON, "-m", "sglang.launch_server", "--model-path", base.MODEL,
        "--tp", "1", "--dtype", "bfloat16", "--disable-overlap-schedule",
        "--context-length", "33792", "--schedule-policy", "fcfs",
        "--cuda-graph-max-bs", str(args.graph_bs), "--max-running-requests", "128",
        "--watchdog-timeout", "180", "--log-level", "debug", "--enable-cache-report",
        "--mem-fraction-static", str(args.mem), "--enable-pdmux",
        "--chunked-prefill-size", "-1", "--pdmux-config-path",
        str(ROOT / "configs/pdmux_4090.yml"), "--pdmux-prefill-mlp-chunk-tokens", "4096",
    ]
    if args.attention:
        common += ["--attention-backend", args.attention]
    if args.no_graph:
        common += ["--disable-cuda-graph"]
    for i in range(4):
        jobs.append((f"worker{i}", common + ["--port", str(31000+i)],
                     {"CUDA_VISIBLE_DEVICES": str(i), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}, 31000+i))
    router = [
        base.PYTHON, "-m", "sglang_router.launch_router", "--host", "127.0.0.1",
        "--port", "30000", "--policy", "cache_aware", "--request-timeout-secs", "3600",
        "--max-concurrent-requests", "4096", "--log-level", "info", "--worker-urls",
    ] + [f"http://127.0.0.1:{31000+i}" for i in range(4)]
    jobs.append(("router", router, {}, 30000))
    return jobs


class Cluster(base.Cluster):
    def reset_router(self):
        if self.label != "muxwise_pp_v3":
            super().reset_router()


def warmup(cluster, name="long-warmup.json"):
    rng = random.Random(730031)
    ids = [128000] + [rng.randrange(128000) for _ in range(31662)]
    urls = [base.BASE] if cluster.label == "muxwise_pp_v3" else [
        f"http://127.0.0.1:{31000+i}" for i in range(4)]
    outputs = []
    for url in urls:
        result = base.generate(url, ids)
        result["url"] = url
        outputs.append(result)
        base.save(cluster.path / name, outputs)
        print(f"long warmup {url}: {result['elapsed_s']:.2f}s / 31663 tokens", flush=True)
    base.flush(cluster)


def sanity(cluster, sanity_round):
    label = "muxwise_r2_dev300" + ("_adjust1" if sanity_round else "")
    cluster.meta["dev_label"] = label
    out = cluster.path / "dev300"
    cmd = [
        base.PYTHON, str(ROOT / "bench/bench_client.py"), "--trace", str(DEV_TRACE),
        "--base-url", base.BASE, "--label", label, "--out-dir", str(out),
        "--limit", "300", "--time-scale", "1", "--warmup", "2", "--force-close",
    ]
    base.save(cluster.path / "dev-command.json", cmd)
    print("RUN " + shlex.join(cmd), flush=True)
    with (cluster.path / "dev-client.log").open("w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
    summary = json.loads((out / f"{label}_summary.json").read_text())
    cluster.meta["dev_summary"] = summary
    base.save(cluster.path / "run.json", cluster.meta)
    assert summary["completed_requests"] == 300 and summary["failed_requests"] == 0 and summary["unsent_requests"] == 0, summary
    print(f"Development sanity passed: 300/300, {summary['duration_sec']:.2f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["preflight", "experiment"])
    parser.add_argument("label", choices=["muxwise_pp_v3", "muxwise_r2"])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--mem", type=float, default=0.88)
    parser.add_argument("--graph-bs", type=int, default=32)
    parser.add_argument("--attention", default="")
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--gate", type=pathlib.Path)
    parser.add_argument("--sanity-round", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    source_gate()
    marker = ROOT / "results" / f"{args.label}.formal-started.json"
    if args.mode == "experiment":
        assert not marker.exists(), f"Formal launch already recorded; replay forbidden: {marker}"
        assert not (ROOT / "results" / f"{args.label}_requests.jsonl").exists()
        assert not (ROOT / "results" / f"{args.label}_summary.json").exists()
    jobs = make_commands(args)
    if args.mode == "experiment" and args.label == "muxwise_r2":
        assert args.gate, "MuxWise r2 requires a successful selection preflight"
        gate = json.loads((args.gate / "run.json").read_text())
        assert gate["status"] == "PASS" and gate["mode"] == "preflight"
        expected = [{"name": n, "argv": c, "env": e, "port": p} for n,c,e,p in jobs]
        assert json.loads((args.gate / "commands.json").read_text()) == expected
        assert gate["commit"] == COMMIT
        previous = list((ROOT / "logs/p6").glob("*/dev-command.json"))
        assert len(previous) == args.sanity_round, "Only initial sanity and one adjustment are allowed"
        if args.sanity_round:
            prior_cmd = json.loads(previous[0].read_text())
            prior_label = prior_cmd[prior_cmd.index("--label")+1]
            prior_out = pathlib.Path(prior_cmd[prior_cmd.index("--out-dir")+1])
            prior_summary = json.loads((prior_out / f"{prior_label}_summary.json").read_text())
            assert prior_summary["failed_requests"] > 0, "Adjustment is only for a failed sanity check"
    base.commands = lambda _: jobs
    path = ROOT / "logs/p6" / args.tag
    assert not path.exists(), path
    cluster = Cluster(args.label, path)
    cluster.meta.update(mode=args.mode, commit=COMMIT, source=str(SOURCE),
                        sanity_round=args.sanity_round,
                        selection_gate=str(args.gate) if args.gate else None,
                        entry_command=[base.PYTHON] + __import__("sys").argv)
    def term(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, term)
    try:
        cluster.start()
        for name, cmd, _, _ in jobs:
            if "--pdmux-config-path" in cmd:
                cfg = pathlib.Path(cmd[cmd.index("--pdmux-config-path")+1])
                shutil.copy2(cfg, path / f"{name}-config.yml")
        cluster.meta["trace_sha256"] = digest(base.TRACE)
        cluster.meta["dev_trace_sha256"] = digest(DEV_TRACE)
        cluster.meta["serve_script_sha256"] = digest(ROOT / "scripts/p5b_v3_serve.sh")
        base.save(path / "run.json", cluster.meta)
        warmup(cluster)
        base.bench(cluster, smoke=True)
        base.flush(cluster)
        if args.label == "muxwise_r2":
            warmup(cluster, "long-warmup-after-smoke.json")
        if args.mode == "experiment":
            if args.label == "muxwise_r2":
                cluster.reset_router()
                sanity(cluster, args.sanity_round)
            base.flush(cluster)
            cluster.reset_router()
            source_gate()
            base.bench(cluster)
            # Allow final idle checks and summary output without any new request.
            time.sleep(12)
        cluster.meta["status"] = "PASS"
    except BaseException as exc:
        cluster.meta.update(status="FAIL", error=repr(exc))
        raise
    finally:
        cluster.close()


if __name__ == "__main__":
    main()
