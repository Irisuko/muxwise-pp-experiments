#!/usr/bin/env python3
"""Collect formal-window server evidence without instrumentation or replay."""
import collections
import datetime
import json
from pathlib import Path
import re
import numpy as np

ROOT = Path("/root/workspace/muxpp_work")


def main():
    result = {}
    for marker_path in sorted((ROOT / "results").glob("*.formal-started.json")):
        label = marker_path.name.removesuffix(".formal-started.json")
        marker = json.loads(marker_path.read_text())
        path = Path(marker["log_dir"])
        summary_file = ROOT / "results" / f"{label}{'_cut300' if label in ('chunked', 'muxwise_r2', 'sglang_pd_r2', 'muxwise_pp_v3') else ''}_summary.json"
        if not summary_file.exists():
            continue
        summary = json.loads(summary_file.read_text())
        start = summary["run_started_unix"]
        stop = start + summary["duration_sec"]
        records = [json.loads(line) for line in (ROOT / "results" / f"{label}_requests.jsonl").read_text().splitlines()]
        errors = collections.Counter(r.get("error") for r in records if not r["success"])
        evidence = []
        shm = []
        pd_decode = collections.defaultdict(list)
        pd_prefill = collections.defaultdict(list)
        # Scheduler timestamps are local system time (UTC+8 on this machine).
        for log in sorted(path.glob("*.log")):
            if log.name.endswith("client.log"):
                continue
            stamp = None
            for number, line in enumerate(log.read_text(errors="replace").splitlines(), 1):
                m = re.match(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
                if m:
                    stamp = datetime.datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S").timestamp()
                if "via SHM/" in line and len(shm) < 8:
                    shm.append({"file": str(log.relative_to(ROOT)), "line": number, "text": line})
                if stamp is None or not int(start) <= stamp <= stop + 1:
                    continue
                decode = re.search(r"Decode batch\. #running-req: (\d+).*?token usage: ([\d.]+), pre-allocated usage: ([\d.]+)", line)
                if decode and start <= stamp and stamp + 1 <= stop:
                    pd_decode[log.name].append([int(decode[1]), float(decode[2]), float(decode[3])])
                prefill = re.search(r"#unbootstrapped-req: (\d+).*?#transferring-req: (\d+)", line)
                if prefill and start <= stamp and stamp + 1 <= stop:
                    pd_prefill[log.name].append([int(prefill[1]), int(prefill[2])])
                if re.search(r"Scheduler hit an exception|#new-token: -|^RuntimeError:|^torch.OutOfMemoryError:|^AssertionError:|Received sigquit|Watchdog timeout", line):
                    evidence.append({"file": str(log.relative_to(ROOT)), "line": number,
                                     "at_sec_approx": stamp-start, "text": line[:2000]})
        pressure = {}
        for name, samples in pd_decode.items():
            a = np.asarray(samples)
            pressure[name] = {"log_samples": len(samples),
                              "running_requests_p50": float(np.median(a[:, 0])),
                              "token_usage_p50": float(np.median(a[:, 1])),
                              "preallocated_usage_p50": float(np.median(a[:, 2])),
                              "preallocated_usage_p99": float(np.percentile(a[:, 2], 99)),
                              "preallocated_usage_max": float(a[:, 2].max())}
        for name, samples in pd_prefill.items():
            a = np.asarray(samples)
            pressure[name] = {"log_samples": len(samples),
                              "unbootstrapped_requests_max": int(a[:, 0].max()),
                              "transferring_requests_max": int(a[:, 1].max())}
        result[label] = {"completed": summary["completed"], "unfinished": summary["unfinished"],
                         "failed": summary["failed"], "response_errors": dict(errors),
                         "formal_error_evidence": evidence, "nccl_shm_evidence": shm,
                         "pd_pressure_log_samples": pressure}
    out = ROOT / "logs/p7b/analysis.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    print(out)


if __name__ == "__main__":
    main()
