#!/usr/bin/env python3
"""Read-only consistency, single-run, preservation and cleanup audit for P6."""
import collections
import csv
import datetime
import hashlib
import json
import math
import pathlib
import re
import shlex
import statistics
import subprocess
import sys

import psutil

ROOT = pathlib.Path("/root/workspace/muxpp_work")
sys.path.insert(0, str(ROOT / "bench"))
from bench_client import summarize_records

FINAL = ["chunked", "muxwise_r2", "sglang_pd_r2", "muxwise_pp_v3"]
ALL = ["chunked", "muxwise", "sglang_pd", "muxwise_pp", "sglang_pd_r2", "muxwise_pp_v3", "muxwise_r2"]
NEW = ["muxwise_pp_v3", "muxwise_r2"]
COMMIT = "f91e8a18da502106a07bb67c67dbf351a933bfd9"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args):
    return subprocess.check_output(args, text=True).strip()


def equal(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
    return a == b


def main():
    checks = []
    def check(name, value):
        checks.append({"name": name, "passed": bool(value)})
    before = read(ROOT / "logs/p6/before.json")
    check("all_original_raw_results_and_markers_unchanged", all(sha(ROOT / p) == h for p,h in before["sha256"].items()))
    trace = ROOT / "bench/toolagent_600_240_x2.5.jsonl"
    records = {r["idx"]: r for r in map(json.loads, trace.open())}
    check("trace_unchanged_from_P3", sha(trace) == (ROOT / "logs/p3/setup/trace-sha256.txt").read_text().strip())
    check("exactly_seven_formal_markers", sorted(p.name for p in (ROOT / "results").glob("*.formal-started.json")) == sorted(f"{l}.formal-started.json" for l in ALL))
    previous_closed = ""
    runs = {}
    for label in NEW:
        marker = read(ROOT / f"results/{label}.formal-started.json")
        path = pathlib.Path(marker["log_dir"])
        summary = read(ROOT / f"results/{label}_summary.json")
        rows = list(map(json.loads, (ROOT / f"results/{label}_requests.jsonl").open()))
        meta = read(path / "run.json")
        jobs = read(path / "commands.json")
        check(label+":single_formal_command", sum(read(p)[read(p).index("--label")+1] == label
              for p in (ROOT/"logs/p6").glob("*/formal-command.json")) == 1)
        check(label+":marker_matches_command", marker["command"] == read(path/"formal-command.json"))
        check(label+":source_fixed_clean", meta["commit"] == COMMIT)
        check(label+":formal_options", summary["trace"] == str(trace) and summary["time_scale"] == 1 and
              summary["limit"] is None and summary["warmup"] == 2 and summary["force_close"] and
              summary["timeout_sec"] == 3600 and not summary["interrupted"])
        check(label+":trace_once_per_request", len(rows) == len({r["idx"] for r in rows}) == 1551
              and set(r["idx"] for r in rows) == set(records)
              and all(all(r[k] == records[r["idx"]][k] for k in ["arrival","input_len","output_len"]) for r in rows))
        recomputed = summarize_records(rows, 1551)
        check(label+":all_metrics_recomputed", all(equal(v, summary[k]) for k,v in recomputed.items()))
        check(label+":sequential_lifecycles", meta["started_utc"] > previous_closed)
        previous_closed = meta["finished_utc"]
        check(label+":four_gpus", {int(g) for j in jobs if j["name"] != "router" for g in j["env"]["CUDA_VISIBLE_DEVICES"].split(",")} == {0,1,2,3})
        for name, info in meta["server_info"].items():
            expected = {"model_path": "/root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct",
                        "dtype": "bfloat16", "context_length": 33792, "disable_overlap_schedule": True,
                        "schedule_policy": "fcfs", "tp_size": 1, "pp_size": 4 if label == "muxwise_pp_v3" else 1,
                        "max_running_requests": 128, "pdmux_prefill_mlp_chunk_tokens": 4096,
                        "chunked_prefill_size": -1}
            check(label+":"+name+":common_configuration", all(info.get(k) == v for k,v in expected.items()))
        smoke = read(path/"smoke"/f"{label}_smoke_summary.json")
        check(label+":smoke_under_one_minute", smoke["completed_requests"] == 20 and smoke["failed_requests"] == 0 and smoke["duration_sec"] <= 60)
        warm = read(path/"long-warmup.json")
        check(label+":long_warmup", len(warm) == (1 if label == "muxwise_pp_v3" else 4) and
              all(r["input_len"] == 31663 and r["response"]["meta_info"]["completion_tokens"] == 8 for r in warm))
        ts = collections.defaultdict(list)
        for fields in csv.reader((path/"gpu.csv").open()):
            try:
                at = datetime.datetime.strptime(fields[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
            except ValueError:
                continue
            ts[int(fields[1])].append(at)
        intervals = {k:[b-a for a,b in zip(v,v[1:])] for k,v in ts.items()}
        check(label+":five_second_gpu_sampling", set(ts) == {0,1,2,3} and all(4.9 < statistics.median(v) < 5.5 for v in intervals.values()))
        if label == "muxwise_pp_v3":
            script = (ROOT/"scripts/p5b_v3_serve.sh").read_text()
            expected = ["/root/venvs/sgl/bin/python"] + shlex.split(script.split("exec python ",1)[1].replace("\\\n"," "))
            check(label+":exact_selected_R3_argv", jobs[0]["argv"] == expected and jobs[0]["env"] == {"CUDA_VISIBLE_DEVICES":"0,1,2,3"})
            analysis = read(path/"analysis.json")
            check(label+":admission_totals_match_requests", analysis["server_cache"]["requests"] == 1551 and
                  analysis["server_cache"]["new_tokens"]+analysis["server_cache"]["cached_tokens"] == summary["input_tokens"] and
                  analysis["server_cache"]["cached_tokens"] == summary["cached_tokens"]["total"])
            check(label+":allocator_invariants", all(r["free"]+r["used"] == r["capacity"] == 359498 and
                  r["protected"]+r["evictable"] == r["used"] for r in analysis["allocator_series"]))
            check(label+":all_stage_idle_checks", set(analysis["final_idle"]) == {"0","1","2","3"})
            closed = []
            for stage, item in analysis["final_idle"].items():
                fields = {k:int(v) for k,v in re.findall(r"(free_kv|evictable|free_reqs|free_receive_slots)=(\d+)", item["text"])}
                closed.append(fields["free_kv"]+fields["evictable"] == 359498 and fields["free_reqs"] == 128
                              and fields["free_receive_slots"] == (0 if stage == "0" else 2))
            check(label+":all_idle_resource_balances_closed", len(closed) == 4 and all(closed))
        else:
            analysis = read(path/"analysis.json")
            check(label+":cache_boundary_accounting", len(analysis["cache_boundary_recovered"]) == 8 and
                  len(analysis["cache_boundary_warmups_excluded"]) == 1 and
                  analysis["server_cache"]["cached_tokens"] == summary["cached_tokens"]["total"])
            gate = pathlib.Path(meta["selection_gate"])
            check(label+":selected_preflight_passed", read(gate/"run.json")["status"] == "PASS"
                  and read(gate/"commands.json") == jobs)
            dev = read(path/"dev300"/f"{meta.get('dev_label','muxwise_r2_dev300')}_summary.json")
            check(label+":dev300_passed", dev["completed_requests"] == 300 and dev["failed_requests"] == 0
                  and dev["time_scale"] == 1 and dev["warmup"] == 2 and dev["force_close"]
                  and dev["trace"] == str(ROOT/"bench/dev_toolagent_1200_240_x2.5.jsonl"))
            dev_commands = list((ROOT/"logs/p6").glob("*/dev-command.json"))
            check(label+":at_most_one_sanity_adjustment", len(dev_commands) == 1+meta.get("sanity_round",0) <= 2)
            if meta.get("sanity_round",0):
                initial = read(ROOT/"logs/p6/muxwise_r2-formal/dev300/muxwise_r2_dev300_summary.json")
                check(label+":initial_failed_sanity_preserved", initial["completed_requests"] == 160 and initial["failed_requests"] == 140
                      and not (ROOT/"logs/p6/muxwise_r2-formal/formal-command.json").exists())
        runs[label] = {"log_dir":str(path), "completed":summary["completed_requests"], "failed":summary["failed_requests"],
                       "sha256":{p.name:sha(p) for p in [ROOT/f"results/{label}_summary.json",ROOT/f"results/{label}_requests.jsonl"]},
                       "sampling_median":{k:statistics.median(v) for k,v in intervals.items()},
                       "sampling_max":{k:max(v) for k,v in intervals.items()}}
    with (ROOT/"results/summary.csv").open() as f:
        check("all_labels_in_summary", {r["label"] for r in csv.DictReader(f)} == set(ALL))
    with (ROOT/"results/final_summary.csv").open() as f:
        check("exact_final_four_in_summary", [r["label"] for r in csv.DictReader(f)] == FINAL)
    check("final_plots_exist", all((ROOT/f"results/final_{x}.png").exists() for x in
          ["ttft_cdf","tpot_cdf","e2e_cdf","ttft_over_time","slo_attainment"]))
    commit = command(["git","-C","/root/workspace/sglang","rev-parse","HEAD"])
    dirty = command(["git","-C","/root/workspace/sglang","status","--porcelain"])
    branch = command(["git","-C","/root/workspace/sglang","branch","--show-current"])
    check("inference_source_unchanged_and_clean", commit == COMMIT and not dirty and branch == "muxwise-pp")
    gpu = command(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader"])
    check("no_GPU_compute_processes", not gpu)
    servers = []
    for p in psutil.process_iter(["pid","cmdline"]):
        args = p.info["cmdline"] or []
        if any(a in ["sglang.launch_server","sglang_router.launch_router"] or a.startswith("sglang::")
               or a.endswith("/bench_client.py") for a in args):
            servers.append(p.info)
    check("no_servers_routers_clients", not servers)
    ports = sorted({c.laddr.port for c in psutil.net_connections(kind="inet") if c.status == "LISTEN"
                    and c.laddr.port in {30000,31000,31001,31002,31003}})
    check("experiment_ports_released", not ports)
    (ROOT/"logs/p6/final-gpu.txt").write_text(command(["nvidia-smi"])+"\n")
    result = {"at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "status":"PASS" if all(c["passed"] for c in checks) else "FAIL", "checks":checks,"runs":runs,
              "remaining_gpu_processes":gpu,"remaining_server_processes":servers,"remaining_ports":ports,
              "source_commit":commit,"source_status":dirty}
    (ROOT/"logs/p6/final-audit.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({"status":result["status"],"checks":len(checks),"failed":[c["name"] for c in checks if not c["passed"]]}))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
