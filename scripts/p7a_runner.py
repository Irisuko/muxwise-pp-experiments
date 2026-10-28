#!/usr/bin/env python3
"""P7 matrix lifecycle; preflight reads development trace only."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import signal
import subprocess
import time

import p3_runner as base

ROOT = base.ROOT
DEV = ROOT / "bench/dev_toolagent_1200_240_x2.5.jsonl"
MAIN = Path("/root/workspace/sglang")
PD = Path("/root/workspace/sglang-p4a")
COMMITS = {MAIN: "f91e8a18da502106a07bb67c67dbf351a933bfd9",
           PD: "19261937a524bc67216d8b925a8ea5e50bfda0f3"}
# family, TP, PP, number of model replicas (P/D split separately).
MATRIX = {
    "chunked_tp2dp2": ("chunked", 2, 1, 2),
    "chunked_pp2dp2": ("chunked", 1, 2, 2),
    "chunked_tp4": ("chunked", 4, 1, 1),
    "chunked_pp4": ("chunked", 1, 4, 1),
    "chunked_tp2pp2": ("chunked", 2, 2, 1),
    "muxwise_tp2dp2": ("muxwise", 2, 1, 2),
    "muxwise_tp4": ("muxwise", 4, 1, 1),
    "pd_1p1d_tp2": ("pd", 2, 1, 2),
    "pd_3p1d": ("pd", 1, 1, 4),
    "pd_1p3d": ("pd", 1, 1, 4),
    "muxpp_pp2dp2": ("muxpp", 1, 2, 2),
}
DEFAULT_MEM = {"chunked": .88, "muxwise": .826, "pd": .88, "muxpp": .71}
UUIDS = [
    "GPU-52777e54-80e7-9357-a595-86ce2678536f",
    "GPU-5230ed6b-99d1-5f7f-b746-707c95ab50ce",
    "GPU-19fac1b0-72dc-17f3-8aa1-132c6bee8183",
    "GPU-5e84f85f-d7eb-bd99-4492-85dac7c020a2",
]


def source_gate(source):
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
    assert head == COMMITS[source] and not dirty, (source, head, dirty)
    forbidden = [k for k in os.environ if k.startswith(("P3_", "SGLANG_PDMUX_PP_"))]
    assert not forbidden, forbidden
    return head


def make_commands(args):
    family, tp, pp, replicas = MATRIX[args.label]
    source = PD if family == "pd" else MAIN
    groups = [[1, 2], [0, 3]] if replicas == 2 else [[0], [1], [2], [3]] if replicas == 4 else [[1, 2, 0, 3]]
    prefill_count = 3 if args.label == "pd_3p1d" else 1
    jobs = []
    for i, group in enumerate(groups):
        port = 30000 if replicas == 1 else 31000 + i
        cmd = [base.PYTHON, "-m", "sglang.launch_server", "--model-path", base.MODEL,
               "--tp", str(tp), "--pp-size", str(pp), "--dtype", "bfloat16",
               "--disable-overlap-schedule", "--context-length", "33792",
               "--schedule-policy", "fcfs", "--max-running-requests", "128",
               "--watchdog-timeout", "180", "--log-level", "debug", "--enable-cache-report",
               "--port", str(port), "--mem-fraction-static", str(args.mem),
               "--attention-backend", args.attention, "--cuda-graph-max-bs", str(args.graph_bs)]
        env = {"CUDA_VISIBLE_DEVICES": ",".join(UUIDS[g] for g in group),
               "PYTHONPATH": str(source / "python"), "NCCL_DEBUG": "INFO"}
        if args.no_graph:
            cmd += ["--disable-cuda-graph"]
        if tp > 1:
            cmd += ["--disable-custom-all-reduce"]
        if family == "chunked":
            cmd += ["--chunked-prefill-size", "2048"]
            if pp == 1:
                cmd += ["--enable-mixed-chunk"]
        elif family == "muxwise":
            cmd += ["--enable-pdmux", "--chunked-prefill-size", "-1",
                    "--pdmux-config-path", str(ROOT / "configs/pdmux_4090.yml")]
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        elif family == "muxpp":
            # Exact selected PP4 arguments, with only topology, memory and inflight scaling.
            script = (ROOT / "scripts/p5b_v3_serve.sh").read_text()
            cmd = [base.PYTHON] + shlex.split(script.split("exec python ", 1)[1].replace("\\\n", " "))
            for key, val in (("--pp-size", 2), ("--port", port), ("--mem-fraction-static", args.mem),
                             ("--pdmux-pp-max-inflight-prefill", 3)):
                cmd[cmd.index(key) + 1] = str(val)
            assert args.attention == "flashinfer" and args.graph_bs == 32 and not args.no_graph
        else:
            mode = "prefill" if i < prefill_count else "decode"
            cmd += ["--disaggregation-mode", mode, "--disaggregation-transfer-backend", "mooncake",
                    "--disaggregation-bootstrap-port", str(8998 + i)]
            if mode == "prefill":
                cmd += ["--chunked-prefill-size", "4096"]
            profile = json.loads((PD / "benchmark/pd_4090/p4a_profiles.json").read_text())["profiles"]["mooncake-tuned"]
            env.update(profile["env"])
        jobs.append((f"worker{i}", cmd, env, port))
    if replicas > 1:
        cmd = [base.PYTHON, "-m", "sglang_router.launch_router", "--host", "127.0.0.1",
               "--port", "30000", "--policy", "random" if family == "pd" else "cache_aware",
               "--request-timeout-secs", "3600", "--max-concurrent-requests", "4096", "--log-level", "info"]
        env = {}
        if family == "pd":
            cmd += ["--mini-lb", "--pd-disaggregation"]
            for i in range(prefill_count):
                cmd += ["--prefill", f"http://127.0.0.1:{31000+i}", str(8998+i)]
            for i in range(prefill_count, replicas):
                cmd += ["--decode", f"http://127.0.0.1:{31000+i}"]
            env["PYTHONPATH"] = str(PD / "sgl-router/py_src") + ":" + str(PD / "python")
        else:
            cmd += ["--worker-urls"] + [f"http://127.0.0.1:{31000+i}" for i in range(replicas)]
        jobs.append(("router", cmd, env, 30000))
    return jobs


class Cluster(base.Cluster):
    def reset_router(self):
        if base.commands(self.label)[-1][0] == "router":
            super().reset_router()

    def close(self):
        try:
            super().close()
        except RuntimeError as exc:
            if not str(exc).startswith("GPU is not clean:"):
                raise
            # NVML can briefly retain a dead CUDA PID after the process is reaped.
            for _ in range(30):
                time.sleep(1)
                try:
                    base.clean_gpu()
                    (self.path/"gpu-cleanup.txt").write_text(base.gpu())
                    return
                except RuntimeError:
                    pass
            raise


def warmup(cluster, name="long-warmup.json"):
    family, _, _, replicas = MATRIX[cluster.label]
    rng = random.Random(730031)
    ids = [128000] + [rng.randrange(128000) for _ in range(31662)]
    results = []
    if family == "pd":
        # Explicit paired requests use the identical native PD protocol as mini_lb.
        # Cover each P and D instance once without random routing hiding an unwarmed worker.
        from concurrent.futures import ThreadPoolExecutor
        pcount = 3 if cluster.label == "pd_3p1d" else 1
        pairs = [(p, d) for p in range(pcount) for d in range(pcount, replicas)]
        for p, d in pairs:
            ids[1] = (ids[1] + 1) % 128000
            body = {"input_ids": ids, "sampling_params": {"temperature": 0, "max_new_tokens": 8, "ignore_eos": True},
                    "bootstrap_host": "127.0.0.1", "bootstrap_port": 8998+p,
                    "bootstrap_room": random.SystemRandom().randrange(1, 2**60)}
            started = time.monotonic()
            with ThreadPoolExecutor(2) as pool:
                futures = [pool.submit(base.post, f"http://127.0.0.1:{31000+i}/generate", json=body, timeout=180)
                           for i in (p, d)]
                responses = [f.result() for f in futures]
            for response in responses:
                response.raise_for_status()
            result = responses[1].json()
            assert result["meta_info"]["completion_tokens"] == 8, result
            results.append({"pair": [p, d], "input_len": len(ids), "response": result,
                            "elapsed_s": time.monotonic()-started})
            base.save(cluster.path/name, results)
            print(f"long warmup P{p}/D{d}: {results[-1]['elapsed_s']:.2f}s", flush=True)
            base.flush(cluster)
    else:
        for worker, _, _, port in base.commands(cluster.label):
            if worker == "router":
                continue
            result = base.generate(f"http://127.0.0.1:{port}", ids)
            result["worker"] = worker
            results.append(result)
            base.save(cluster.path/name, results)
            print(f"long warmup {worker}: {result['elapsed_s']:.2f}s", flush=True)
        base.flush(cluster)


def replay(cluster, stage):
    formal = stage == "formal"
    limit = 20 if stage == "smoke" else 300
    trace = base.TRACE if formal else DEV
    out = ROOT / "results" if formal else cluster.path / stage
    label = cluster.label if formal else cluster.label + "_" + stage
    cmd = [base.PYTHON, str(ROOT / "bench/bench_client.py"), "--trace", str(trace),
           "--base-url", base.BASE, "--label", label, "--out-dir", str(out),
           "--warmup", "2", "--force-close", "--drain-timeout", "55" if stage == "smoke" else "300"]
    if not formal:
        cmd += ["--limit", str(limit), "--time-scale", "0.2" if stage == "smoke" else "1"]
    if stage == "smoke":
        cmd += ["--timeout", "55"]
    if formal:
        with (out/f"{label}.formal-started.json").open("x") as f:
            json.dump({"command": cmd, "log_dir": str(cluster.path),
                       "started": datetime.datetime.now(datetime.timezone.utc).isoformat()}, f, indent=2)
    elif stage == "dev300":
        with (ROOT/"logs/p7a"/f"{cluster.label}.dev300-started.json").open("x") as f:
            json.dump({"command": cmd, "log_dir": str(cluster.path),
                       "started": datetime.datetime.now(datetime.timezone.utc).isoformat()}, f, indent=2)
    base.save(cluster.path / f"{stage}-command.json", cmd)
    print("RUN " + shlex.join(cmd), flush=True)
    with (cluster.path / f"{stage}-client.log").open("x") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            rc = proc.wait(timeout=60 if stage == "smoke" else 1100)
        except BaseException:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
    summary = json.loads((out / f"{label}_summary.json").read_text())
    cluster.meta[stage + "_summary"] = summary
    cluster.meta[stage + "_returncode"] = rc
    base.save(cluster.path/"run.json", cluster.meta)
    print(f"{stage}: {summary['completed']}/{summary['unfinished']}/{summary['failed']} duration={summary['duration_sec']:.2f}s", flush=True)
    if stage == "smoke":
        assert rc == 0 and summary["completed"] == 20 and summary["duration_sec"] <= 60, "smoke gate failed"
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["launch", "preflight", "sanity", "experiment", "commands"])
    parser.add_argument("label", choices=MATRIX)
    parser.add_argument("--tag")
    parser.add_argument("--mem", type=float)
    parser.add_argument("--attention", default="flashinfer")
    parser.add_argument("--graph-bs", type=int, default=32)
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--gate", type=Path)
    args = parser.parse_args()
    selected_file = ROOT / "configs/p7a_selected.json"
    selected = json.loads(selected_file.read_text()).get(args.label, {}) if selected_file.exists() else {}
    if args.mem is None:
        args.mem = selected.get("mem", DEFAULT_MEM[MATRIX[args.label][0]])
        for key in ("attention", "graph_bs", "no_graph"):
            if key in selected:
                setattr(args, key, selected[key])
    if args.mode in ("launch", "experiment") and selected.get("status") == "UNSUPPORTED":
        raise RuntimeError("Unsupported frozen-source configuration: " + selected["reason"])
    source = PD if MATRIX[args.label][0] == "pd" else MAIN
    commit = source_gate(source)
    jobs = make_commands(args)
    serial = [{"name": n, "argv": c, "env": e, "port": p} for n, c, e, p in jobs]
    if args.mode == "commands":
        print("\n".join(shlex.join(["env"]+[k+"="+v for k,v in e.items()]+c) for n,c,e,p in jobs))
        return
    if args.mode == "sanity":
        assert args.label == "muxpp_pp2dp2"
    # Global advisory lock prevents races between matrix launches.
    lock = (ROOT / "logs/p7a/gpu.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.mode == "experiment":
        assert selected.get("status") == "PASS", "No supported selected preflight"
        args.gate = args.gate or Path(selected["gate"])
        gate = json.loads((args.gate/"run.json").read_text())
        assert gate["status"] == "PASS" and gate["mode"] == "preflight"
        assert gate["commit"] == commit
        assert json.loads((args.gate/"commands.json").read_text()) == serial
        assert not (ROOT/"results"/f"{args.label}.formal-started.json").exists(), "Formal replay forbidden"
        assert not list((ROOT/"results").glob(f"{args.label}_*.json*")), "Formal data exists"
    tag = args.tag or args.mode + "-" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    path = ROOT / "logs/p7a" / args.label / tag
    path.parent.mkdir(parents=True, exist_ok=True)
    base.commands = lambda _: jobs
    cluster = Cluster(args.label, path)
    cluster.meta.update(mode=args.mode, commit=commit, source=str(source),
                        preflight_protocol=2,
                        configuration=vars(args), dev_trace=str(DEV),
                        dev_trace_sha256=hashlib.sha256(DEV.read_bytes()).hexdigest())
    def term(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, term)
    try:
        cluster.start()
        if args.mode == "launch":
            signal.pause()
        else:
            warmup(cluster)
            replay(cluster, "smoke")
            if MATRIX[args.label][0] == "muxwise":
                base.flush(cluster)
                base.save(cluster.path/"long-warmup-after-smoke-cache-flush.json",
                          {"flushed": True, "before": "long-warmup-after-smoke.json"})
                warmup(cluster, "long-warmup-after-smoke.json")
            if args.mode in ("sanity", "experiment"):
                base.flush(cluster)
                cluster.reset_router()
                replay(cluster, "dev300" if args.mode == "sanity" else "formal")
            cluster.meta["status"] = "PASS"
    except BaseException as exc:
        cluster.meta.update(status="FAIL", error=repr(exc))
        raise
    finally:
        cluster.close()


if __name__ == "__main__":
    main()
