#!/usr/bin/env python3
"""Run two 200-request CPU mock replays and produce comparison artifacts."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

from aiohttp import ClientSession, web

from common import write_json
from mock_server import create_app

HERE = Path(__file__).resolve().parent


async def command(*args):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    process = await asyncio.create_subprocess_exec(sys.executable, *map(str, args),
                                                  env=environment, stdout=asyncio.subprocess.PIPE,
                                                  stderr=asyncio.subprocess.STDOUT)
    output, _ = await process.communicate()
    print(output.decode(), end="", flush=True)
    if process.returncode:
        raise RuntimeError(f"command exited with {process.returncode}: {args}")


async def main_async(args):
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"mock_only": True, "time_scale": 0.05, "limit": 200, "runs": []}
    summaries = []
    for label, delay in (("mock_1ms", 0.001), ("mock_2ms", 0.002)):
        runner = web.AppRunner(create_app(token_delay=delay))
        await runner.setup()
        try:
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            base = f"http://127.0.0.1:{runner.addresses[0][1]}"
            await command(HERE / "bench_client.py", "--trace", args.trace, "--base-url", base,
                          "--label", label, "--out-dir", output, "--warmup", "2", "--limit", "200",
                          "--time-scale", "0.05", "--progress-interval", "1", "--overwrite")
            async with ClientSession() as session:
                async with session.get(base + "/stats") as response:
                    server_stats = await response.json()
        finally:
            await runner.cleanup()
        summary_path = output / f"{label}_summary.json"
        summary = json.loads(summary_path.read_text())
        records = [json.loads(line) for line in (output / f"{label}_requests.jsonl").read_text().splitlines()]
        assert len(records) == 200
        assert summary["completed_requests"] == 200 and summary["failed_requests"] == 0
        assert server_stats["requests"] == 202, "warmup must be sent but not measured"
        assert server_stats["max_active"] > 1, "replay must be concurrent/open-loop"
        assert all(r["output_tokens"] == r["output_len"] for r in records)
        assert all(len(r["itl"]) == r["output_tokens"] - 1 for r in records)
        assert all(r["cached_tokens"] == r["input_len"] // 4 for r in records)
        assert all(r["sent_at"] >= r["arrival"] for r in records)
        assert max(r["send_lag"] for r in records) < 1.0, "severe client scheduling lag"
        assert any(b["sent_at"] < a["finished_at"] for a, b in zip(records, records[1:]))
        for record in records:
            if record["output_tokens"] > 1:
                expected = (record["e2e"] - record["ttft"]) / (record["output_tokens"] - 1)
                assert abs(record["tpot"] - expected) < 1e-12
        report["runs"].append({"label": label, "completed": 200, "failed": 0,
                                "server": server_stats, "duration_sec": summary["duration_sec"],
                                "output_tokens": summary["output_tokens"],
                                "ttft_p50_sec": summary["latencies_sec"]["ttft"]["p50"],
                                "tpot_p50_sec": summary["latencies_sec"]["tpot"]["p50"],
                                "send_lag_p99_sec": summary["latencies_sec"]["send_lag"]["p99"]})
        summaries.append(summary_path)
    await command(HERE / "summarize.py", *summaries, "--out-dir", args.report_dir)
    for name in ("summary.md", "summary.csv", "ttft_cdf.png", "tpot_cdf.png", "ttft_over_time.png", "slo_attainment.png"):
        assert (Path(args.report_dir) / name).stat().st_size > 0
    report["passed"] = True
    write_json(output / "selftest_report.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default=str(HERE / "toolagent_600_240_x2.5.jsonl"))
    parser.add_argument("--out-dir", default=str(HERE / "selftest_results"))
    parser.add_argument("--report-dir", default=str(HERE.parent / "results"))
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
