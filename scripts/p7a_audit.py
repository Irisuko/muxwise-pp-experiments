#!/usr/bin/env python3
"""Final read-only checks, then save the P7a audit artifact."""
import argparse
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
from p7a_runner import COMMITS, MATRIX, ROOT, make_commands, source_gate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    labels = ["chunked", "muxwise_r2", "sglang_pd_r2", "muxwise_pp_v3"]
    hashes = json.loads((ROOT/"logs/p7a/original-final-sha256.json").read_text())
    assert all(sha(ROOT/p) == h for p,h in hashes.items())
    new_formals = list((ROOT/"results").glob("*.formal-started.json"))
    assert {p.name.removesuffix(".formal-started.json") for p in new_formals} == set(labels)
    assert not list((ROOT/"logs/p7a").glob("*/*/formal-command.json"))
    for old in ("muxwise", "sglang_pd", "muxwise_pp"):
        assert not (ROOT/"results"/f"{old}_summary.json").exists()
        assert (ROOT/"archive/superseded/results"/f"{old}_summary.json").exists()
    selected = json.loads((ROOT/"configs/p7a_selected.json").read_text())
    assert set(selected) == set(MATRIX)
    for label, cfg in selected.items():
        assert cfg["status"] in ("PASS", "UNSUPPORTED")
        path = Path(cfg["gate"])
        meta = json.loads((path/"run.json").read_text())
        args = argparse.Namespace(label=label, mem=cfg["mem"],
                                  attention=cfg["attention"], graph_bs=cfg["graph_bs"],
                                  no_graph=cfg["no_graph"])
        expected = [{"name":n, "argv":c, "env":e, "port":p} for n,c,e,p in make_commands(args)]
        assert expected == json.loads((path/"commands.json").read_text()), label
        if cfg["status"] == "PASS":
            assert meta["status"] == "PASS" and meta["mode"] == "preflight", label
            assert meta["smoke_summary"]["completed"] == 20
            assert meta["smoke_summary"]["duration_sec"] <= 60
            assert (path/"long-warmup.json").exists()
        else:
            assert cfg["reason"]
        assert (ROOT/"scripts"/f"launch_{label}.sh").exists()
    dev_commands = []
    for path in sorted((ROOT/"logs/p7a").glob("*/*/*-command.json")):
        command = json.loads(path.read_text())
        assert Path(command[command.index("--trace")+1]) == ROOT/"bench/dev_toolagent_1200_240_x2.5.jsonl"
        assert command[command.index("--limit")+1] in ("20", "300")
        if command[command.index("--limit")+1] == "300":
            assert command[command.index("--drain-timeout")+1] == "300"
        dev_commands.append(str(path.relative_to(ROOT)))
    sanity_marker = json.loads((ROOT/"logs/p7a/muxpp_pp2dp2.dev300-started.json").read_text())
    sanity_runs = list((ROOT/"logs/p7a").glob("*/*/dev300-command.json"))
    assert len(sanity_runs) == 1
    assert Path(sanity_marker["log_dir"]) == sanity_runs[0].parent
    for csv_path in (ROOT/"results/summary.csv", ROOT/"results/final_summary.csv"):
        rows = list(csv.DictReader(csv_path.open()))
        assert [r["label"] for r in rows] == labels
        assert all(r["drain_timeout_sec"] == "300" for r in rows)
    for label in labels:
        summary = json.loads((ROOT/"results"/f"{label}_cut300_summary.json").read_text())
        assert summary["completed"]+summary["unfinished"]+summary["failed"]+summary["unsent_requests"] == 1551
        assert summary["latencies_sec"]["ttft"]["count"] == 1551
        assert summary["latencies_sec"]["e2e"]["count"] == 1551
        assert summary["slo"]["ttft_le_5s_and_tpot_le_100ms"]["total"] == 1551
    manifest = json.loads((ROOT/"archive/superseded/manifest.json").read_text())
    for entry in manifest:
        for relative, digest in entry["sha256"].items():
            assert sha(ROOT/"archive/superseded"/relative) == digest, relative
    commits = {str(p): source_gate(p) for p in COMMITS}
    residual = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if process.pid == os.getpid():
                continue
            argv = process.info["cmdline"] or []
            name = process.info["name"] or ""
            if (name.startswith("sglang::") or
                any(a in ("sglang.launch_server", "sglang_router.launch_router") for a in argv) or
                (any(a.endswith("/bench_client.py") for a in argv) and "--base-url" in argv)):
                residual.append(process.info)
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
    tests = (ROOT/"logs/p7a/unit-tests.log").read_text()
    assert "Ran 13 tests" in tests and tests.rstrip().endswith("OK")
    result = {
        "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "original_final_hashes_unchanged": True, "new_formal_runs": 0,
        "source_commits": commits, "source_clean": True,
        "selected_status": {k:v["status"] for k,v in selected.items()},
        "development_commands": dev_commands, "sanity_runs": 1,
        "archived_files_verified": sum(len(e["sha256"]) for e in manifest),
        "gpu_compute_processes": gpu, "residual_processes": residual, "occupied_ports": occupied,
        "unit_tests": 13,
    }
    (ROOT/"logs/p7a/final-audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    (ROOT/"logs/p7a/final-gpu.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
