#!/usr/bin/env python3
"""Run the frozen P7a selections once in order and append durable progress."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

import p7a_runner as runner

ROOT = runner.ROOT
OUT = ROOT / "logs/p7b"


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    OUT.mkdir(exist_ok=True)
    # This batch is intentionally not resumable by blind rerunning.
    with (OUT / "batch-started.json").open("x") as f:
        json.dump({"started": now(), "labels": list(runner.MATRIX)}, f, indent=2)
    protected = [ROOT / "configs/p7a_selected.json", ROOT / "scripts/p7a_runner.py",
                 ROOT / "scripts/p3_runner.py", ROOT / "scripts/run_experiment.sh",
                 ROOT / "bench/bench_client.py", ROOT / "bench/metrics.py", runner.base.TRACE]
    runner.base.save(OUT / "frozen-input-sha256.json",
                     {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in protected})
    progress = OUT / "progress.md"
    with progress.open("x") as f:
        f.write("# P7b 正式运行进度（UTC）\n\n"
                "| label | 开始 | 结束 | 完成/未完成/失败 | 主 SLO | 运行状态 |\n"
                "| --- | --- | --- | --- | --- | --- |\n")
    for label in runner.MATRIX:
        runner.base.clean_gpu()
        started = now()
        path = ROOT / "logs/p7a" / label / "p7b-formal"
        (OUT / label).symlink_to(path, target_is_directory=True)
        cmd = ["bash", str(ROOT / "scripts/run_experiment.sh"), label, "--tag", "p7b-formal"]
        runner.base.save(OUT / "current.json", {"label": label, "started": started, "command": cmd})
        print(f"START {label} {started}", flush=True)
        with (OUT / f"{label}-runner.log").open("x") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
        ended = now()
        summary_file = ROOT / "results" / f"{label}_summary.json"
        counts, slo = "无正式数据", "—"
        if summary_file.exists():
            summary = json.loads(summary_file.read_text())
            counts = "/".join(str(summary[k]) for k in ("completed", "unfinished", "failed"))
            slo = f'{summary["slo"]["ttft_le_5s_and_tpot_le_100ms"]["rate"]:.6%}'
        status = "结束" if rc == 0 else f"运行异常 rc={rc}"
        with progress.open("a") as f:
            f.write(f"| {label} | {started} | {ended} | {counts} | {slo} | {status} |\n")
            f.flush()
        print(f"END {label}: {counts}, SLO={slo}, {status} {ended}", flush=True)
        # Stop before taking another configuration's GPUs if cleanup failed.
        runner.base.clean_gpu()
    runner.base.save(OUT / "batch-finished.json", {"finished": now()})
    runner.base.save(OUT / "current.json", {"status": "finished", "finished": now()})


if __name__ == "__main__":
    main()
