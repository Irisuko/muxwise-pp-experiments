#!/usr/bin/env python3
"""Read-only analysis of P6 formal artifacts; no inference or replay."""
import ast
import collections
import datetime
import json
import pathlib
import re
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from p4b_analyze import analyze
from p5b_analyze import audit

ROOT = pathlib.Path("/root/workspace/muxpp_work")
LABELS = ["muxwise_pp_v3", "muxwise_r2"]


def inspect(label):
    marker = json.loads((ROOT / f"results/{label}.formal-started.json").read_text())
    path = pathlib.Path(marker["log_dir"])
    # Keep a byte-identical local summary for the existing diagnostic readers.
    shutil.copy2(ROOT / f"results/{label}_summary.json", path / f"{label}_summary.json")
    result = audit(path) if label == "muxwise_pp_v3" else analyze(path)
    summary = result["summary"]
    start = summary["run_started_unix"]
    end = start + summary["duration_sec"]
    jobs = json.loads((path / "commands.json").read_text())
    rows = [json.loads(l) for l in (ROOT / f"results/{label}_requests.jsonl").read_text().splitlines()]
    first_request = min(rows, key=lambda r: r["arrival"])
    boundary_recovered = []
    boundary_warmups_excluded = []
    cache = collections.Counter()
    idle = {}
    capacities = {}
    backends = {}
    error_evidence = []
    retract = []
    migrate = []
    decode_graph_logs = collections.Counter()
    for job in jobs:
        if job["name"] == "router":
            continue
        text = (path / (job["name"] + ".log")).read_text()
        capacities[job["name"]] = list(dict.fromkeys(map(int, re.findall(r"max_total_num_tokens=(\d+)", text))))
        if "--attention-backend" in job["argv"]:
            backends[job["name"]] = job["argv"][job["argv"].index("--attention-backend")+1]
        elif "Use flashinfer backend by default." in text:
            backends[job["name"]] = "flashinfer (default; verified in log)"
        else:
            backends[job["name"]] = "unresolved"
        for number, line in enumerate(text.splitlines(), 1):
            try:
                stamp = datetime.datetime.strptime(line[1:20], "%Y-%m-%d %H:%M:%S").timestamp()
            except ValueError:
                stamp = None
            if "idle self-check PASS" in line and stamp is not None and stamp >= start:
                m = re.search(r"stage=(\d+)", line)
                if m:
                    idle[m[1]] = {"line": number, "text": line}
            if re.search(r"OutOfMemoryError|CUDA error:|Traceback \(most recent call last\)|AssertionError|scheduler.*exception", line, re.I):
                error_evidence.append({"file": job["name"]+".log", "line": number, "text": line[:2000]})
            in_window = stamp is not None and start <= stamp <= end+1
            # Logs have whole-second timestamps. The first actual formal admit
            # can share that second with the client's two 16-token warmups.
            # Recover only the unique one-request batch matching the first
            # formal input length, and audit the final request/token totals.
            if label == "muxwise_pp_v3" and stamp is not None and int(start) == stamp and "PP0]" in line:
                m = re.search(r"admit batch=\d+ requests=(\d+) tokens=(\d+) cached=(\d+)", line)
                if m and int(m[1]) == 1 and int(m[2])+int(m[3]) == first_request["input_len"]:
                    in_window = True
                    boundary_recovered.append({"line":number,"text":line})
            elif label == "muxwise_r2" and stamp is not None and int(start) == stamp:
                m = re.search(r"#new-token: (\d+), #cached-token: (\d+)", line)
                if m:
                    # Both client warmups have exactly 16 input tokens. No
                    # formal request in the first scheduled second is that short.
                    assert min(r["input_len"] for r in rows if r["arrival"] < 1) > 16
                    item = {"file":job["name"]+".log","line":number,"text":line}
                    if int(m[1])+int(m[2]) == 16:
                        in_window = False
                        boundary_warmups_excluded.append(item)
                    else:
                        in_window = True
                        boundary_recovered.append(item)
            if not in_window:
                continue
            match_graph = re.search(r"Decode batch\..*cuda graph: (True|False)", line)
            if match_graph:
                decode_graph_logs[match_graph[1]] += 1
            if label == "muxwise_pp_v3":
                if "PP0]" not in line:
                    continue
                match = re.search(r"admit batch=\d+ requests=(\d+) tokens=(\d+) cached=(\d+)", line)
                if match:
                    cache.update(batches=1, requests=int(match[1]), new_tokens=int(match[2]), cached_tokens=int(match[3]))
                if "MuxWise-PP migrate step=" in line:
                    migrate.append({"line": number, "text": line})
            else:
                match = re.search(r"#new-token: (\d+), #cached-token: (\d+)", line)
                if match:
                    cache.update(batches=1, new_tokens=int(match[1]), cached_tokens=int(match[2]))
            if re.search(r"retract", line, re.I):
                retract.append({"file": job["name"]+".log", "line": number, "text": line[:2000]})
    result.update(
        capacity_by_worker=capacities,
        effective_attention_backend=backends,
        final_idle=idle,
        server_error_evidence=error_evidence,
        formal_retraction_evidence=retract,
        formal_migration_evidence=migrate,
        response_errors=dict(collections.Counter(r["error"] for r in rows if not r["success"])),
        server_cache=dict(cache),
        cache_boundary_recovered=boundary_recovered,
        cache_boundary_warmups_excluded=boundary_warmups_excluded,
        decode_graph_log_counts=dict(decode_graph_logs),
    )
    if label == "muxwise_pp_v3":
        assert len(boundary_recovered) <= 1
        assert cache["requests"] == summary["sent_requests"]
        assert cache["new_tokens"]+cache["cached_tokens"] == sum(r["input_len"] for r in rows)
    if cache["new_tokens"] + cache["cached_tokens"]:
        result["server_cache"]["ratio"] = cache["cached_tokens"] / (cache["new_tokens"] + cache["cached_tokens"])
    (path / "analysis.json").write_text(json.dumps(result, indent=2))
    if label == "muxwise_pp_v3":
        plots(result, path)
    return result


def plots(result, path):
    s = result["summary"]
    rows = result["allocator_series"]
    x = [v["at"] - s["run_started_unix"] for v in rows]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].stackplot(x, *[[v[k]/1000 for v in rows] for k in ["protected", "evictable", "free"]],
                      labels=["Protected", "Evictable", "Free"], colors=["#bf4a47", "#dfb35b", "#6ba89e"])
    axes[0].set(ylabel="Allocator KV tokens (k)", title="MuxWise-PP v3: actual allocator occupancy")
    axes[0].legend(ncols=3)
    axes[1].plot(x, [v["waiting"] for v in rows], label="Waiting requests", color="#364b80")
    axes[1].plot(x, [v["inflight"] for v in rows], label="Inflight prefill batches", color="#b05d2f")
    axes[1].set(xlabel="Seconds from formal replay origin", ylabel="Count")
    axes[1].legend()
    for ax in axes:
        ax.axvline(s["scheduled_duration_sec"], color="#555555", linestyle="--", linewidth=1)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path / "allocator.png", dpi=160)
    fig.savefig(path / "allocator.svg")
    plt.close(fig)
    reasons = ["inflight", "kv", "decode_reserve", "budget", "max_running_requests", "other"]
    d = result["admission_arrival"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, title in zip(axes, ["blocked", "wall"], ["Zero-admission decisions", "Estimated blocked wall time (s)"]):
        vals = [d[key].get(k, 0) for k in reasons]
        ax.bar(reasons, vals, color="#426b98")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        for i, v in enumerate(vals):
            ax.text(i, v, str(int(v)) if key == "blocked" else f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path / "admission_histogram.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    for label in sys.argv[1:] or LABELS:
        r = inspect(label)
        print(json.dumps({"label": label, "cache": r["server_cache"], "errors": r["response_errors"],
                          "server_errors": len(r["server_error_evidence"])}))
