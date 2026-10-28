#!/usr/bin/env python3
"""Read-only final checks of frozen commands, measurements and GPU cleanup."""
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import psutil
from p7a_runner import COMMITS, MATRIX, ROOT, source_gate

sys.path.insert(0, str(ROOT / "bench"))
from summarize import HISTORICAL, MAIN_SLO, load_runs, prepare_matrix, select_best


def read(path):
    return json.loads(path.read_text())


def main():
    out = ROOT / "logs/p7b"
    for manifest in ("logs/p7a/original-final-sha256.json", "logs/p7b/frozen-input-sha256.json"):
        for relative, digest in read(ROOT / manifest).items():
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
    selected = read(ROOT / "configs/p7a_selected.json")
    paths = [ROOT / "results" / f"{label}_cut300_summary.json" for label in HISTORICAL]
    paths += [ROOT / "results" / f"{label}_summary.json" for label in MATRIX]
    runs = prepare_matrix(load_runs(list(map(str, paths))), ROOT / "results")
    assert {p.name.removesuffix(".formal-started.json") for p in
            (ROOT / "results").glob("*.formal-started.json")} == {s["label"] for s,_ in runs}
    checks = {}
    for summary, records in runs:
        label = summary["label"]
        raw = read(ROOT / "results" / f"{label}{'_cut300' if label in HISTORICAL else ''}_summary.json")
        for key in ("completed", "unfinished", "failed", "latencies_sec", "slo", "throughput", "cached_tokens"):
            assert summary[key] == raw[key], (label, key)
        assert len(records) == summary["planned_requests"] == 1551
        assert len({r["idx"] for r in records}) == 1551
        assert summary["unsent_requests"] == 0
        assert sum(summary[k] for k in ("completed", "unfinished", "failed")) == 1551
        assert summary["latencies_sec"]["ttft"]["count"] == 1551
        assert summary["latencies_sec"]["e2e"]["count"] == 1551
        assert all(v["total"] == 1551 for v in summary["slo"].values())
        assert abs(summary["cutoff_time"] - max(r["sent_at"] for r in records) - 300) < 1e-7
        assert not summary["interrupted"]
        if label in HISTORICAL:
            continue
        path = Path(summary["matrix"]["log_dir"])
        gate = Path(selected[label]["gate"])
        assert read(path / "commands.json") == read(gate / "commands.json"), label
        marker = read(ROOT / "results" / f"{label}.formal-started.json")
        cmd = read(path / "formal-command.json")
        assert cmd == marker["command"]
        assert cmd[cmd.index("--warmup")+1] == "2"
        assert cmd[cmd.index("--drain-timeout")+1] == "300"
        assert "--force-close" in cmd and "--limit" not in cmd and "--time-scale" not in cmd
        assert Path(cmd[cmd.index("--trace")+1]) == ROOT / "bench/toolagent_600_240_x2.5.jsonl"
        meta = read(path / "run.json")
        warm = read(path / "long-warmup.json")
        assert all(w["input_len"] == 31663 and w["response"]["meta_info"]["completion_tokens"] == 8 for w in warm)
        expected_warm = 3 if label in ("pd_3p1d", "pd_1p3d") else (1 if label == "pd_1p1d_tp2" else MATRIX[label][3])
        assert len(warm) == expected_warm
        if label.startswith("muxwise_"):
            assert read(path / "long-warmup-after-smoke-cache-flush.json")["flushed"]
            assert len(read(path / "long-warmup-after-smoke.json")) == expected_warm
        smoke = meta["smoke_summary"]
        assert (smoke["completed"], smoke["unfinished"], smoke["failed"]) == (20, 0, 0)
        assert smoke["duration_sec"] <= 60
        if MATRIX[label][3] > 1:
            assert (path / "router-formal.log").exists()
        log = (out / f"{label}-runner.log").read_text()
        assert log.count("Cache flushed") >= 2
        assert len(list((ROOT / "logs/p7a" / label).glob("*/formal-command.json"))) == 1
        assert (path / "gpu-cleanup.txt").exists()
        with (path / "gpu.csv").open() as f:
            samples = list(csv.DictReader(f, skipinitialspace=True))
        assert {r["index"] for r in samples} == {"0", "1", "2", "3"}
        stamps = [datetime.datetime.strptime(r["timestamp"], "%Y/%m/%d %H:%M:%S.%f")
                  for r in samples if r["index"] == "0"]
        gaps = [(b-a).total_seconds() for a,b in zip(stamps, stamps[1:])]
        assert len(stamps) > 100
        checks[label] = {
            "one_formal_run": True, "frozen_commands_match": True,
            "completed": summary["completed"], "unfinished": summary["unfinished"], "failed": summary["failed"],
            "main_slo": summary["slo"][MAIN_SLO],
            "smoke_duration_sec": smoke["duration_sec"], "long_warmup_requests": len(warm),
            "sample_cycles": len(stamps), "sampling_interval_sec_min": min(gaps),
            "sampling_interval_sec_max": max(gaps),
            "max_send_lag_sec": summary["latencies_sec"]["send_lag"]["max"],
            "run_status": meta["status"], "formal_returncode": meta["formal_returncode"],
        }
    matrix_csv = list(csv.DictReader((ROOT / "results/matrix_summary.csv").open()))
    best_csv = list(csv.DictReader((ROOT / "results/best_summary.csv").open()))
    assert len(matrix_csv) == 15 and len(best_csv) == 4
    assert [r["label"] for r in matrix_csv] == [s["label"] for s,_ in runs]
    assert [r["label"] for r in best_csv] == [s["label"] for s,_ in select_best(runs)]
    progress = (out / "progress.md").read_text().splitlines()
    assert sum(line.startswith("| "+label+" |") for label in MATRIX for line in progress) == 11
    commits = {str(p): source_gate(p) for p in COMMITS}
    residual = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            argv, name = proc.info["cmdline"] or [], proc.info["name"] or ""
            if proc.pid != os.getpid() and (
                name.startswith("sglang::") or
                any(a in ("sglang.launch_server", "sglang_router.launch_router") for a in argv) or
                (any(a.endswith("/bench_client.py") for a in argv) and "--base-url" in argv)):
                residual.append(proc.info)
        except psutil.NoSuchProcess:
            pass
    gpu = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                   "--format=csv,noheader"], text=True).strip()
    occupied = []
    for port in (30000, 31000, 31001, 31002, 31003, 8998, 8999, 9000, 9001):
        with socket.socket() as sock:
            sock.settimeout(.2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                occupied.append(port)
    assert not gpu and not residual and not occupied, (gpu, residual, occupied)
    formal_hashes = {}
    for summary, _ in runs:
        for suffix in ("_requests.jsonl", "_summary.json", ".formal-started.json"):
            artifact = ROOT / "results" / (summary["label"] + suffix)
            formal_hashes[str(artifact.relative_to(ROOT))] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (out / "formal-artifacts-sha256.json").write_text(json.dumps(formal_hashes, indent=2)+"\n")
    tests = (out / "unit-tests.log").read_text()
    assert "Ran 14 tests" in tests and tests.rstrip().endswith("OK")
    result = {
        "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "old_formal_artifacts_unchanged": True, "frozen_inputs_unchanged": True,
        "source_commits": commits, "sources_clean": True, "new_formal_runs": len(checks),
        "restarts": 0, "checks": checks, "matrix_rows": 15, "best_rows": 4,
        "configurations_with_request_failures": [s["label"] for s,_ in runs if s["failed"]],
        "formal_artifact_hashes": "logs/p7b/formal-artifacts-sha256.json",
        "unit_tests": 14,
        "gpu_compute_processes": gpu, "residual_processes": residual, "occupied_ports": occupied,
    }
    (out / "final-audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    (out / "final-gpu.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
