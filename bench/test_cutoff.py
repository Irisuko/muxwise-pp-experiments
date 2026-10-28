"""Drain cancellation, extended quantiles, partial delivery, and offline parity."""
import argparse
import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from aiohttp import web

from bench_client import run
from metrics import censored_distribution, cutoff_records, summarize_records
from summarize import load_runs, write_tables


class DrainTests(unittest.IsolatedAsyncioTestCase):
    async def test_last_actual_send_deadline_and_closed_connections(self):
        closed = set()

        async def handler(request):
            body = await request.json()
            kind = body["input_ids"][0]
            if kind == 3:
                return web.Response(status=503)
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            if kind != 4:
                await response.write(b'data: {"meta_info":{"completion_tokens":1}}\n\n')
            if kind == 1:
                await response.write(b'data: {"meta_info":{"completion_tokens":2}}\n\ndata: [DONE]\n\n')
                return response
            # A peer close must be observable, not merely task cancellation in the client.
            transport = request.transport
            for _ in range(100):
                if transport.is_closing():
                    closed.add(kind)
                    break
                await asyncio.sleep(.01)
            return response

        app = web.Application()
        app.router.add_post("/generate", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                trace = root / "trace.jsonl"
                trace.write_text("".join(json.dumps({
                    "idx": i, "arrival": i * .04, "input_ids": [i + 1],
                    "input_len": 1, "output_len": 2,
                }) + "\n" for i in range(4)))
                args = argparse.Namespace(trace=str(trace), base_url=f"http://127.0.0.1:{runner.addresses[0][1]}",
                    label="mock", out_dir=str(root), warmup=0, limit=None, time_scale=1,
                    timeout=10, drain_timeout=.15, progress_interval=10, overwrite=False, force_close=True)
                rc = await run(args)
                self.assertEqual(rc, 1)  # Real HTTP failure, not the two unfinished requests.
                summary = json.loads((root/"mock_summary.json").read_text())
                records = [json.loads(line) for line in (root/"mock_requests.jsonl").read_text().splitlines()]
                self.assertEqual((summary["completed"], summary["unfinished"], summary["failed"]), (1, 2, 1))
                self.assertAlmostEqual(summary["cutoff_time"], max(r["sent_at"] for r in records)+.15)
                self.assertGreater(summary["duration_sec"], .26)
                self.assertLess(summary["duration_sec"], .40)
                self.assertEqual(summary["output_tokens"], 3)
                self.assertEqual(summary["input_tokens"], 2)
                self.assertEqual(summary["latencies_sec"]["ttft"]["count"], 4)
                self.assertEqual(summary["latencies_sec"]["ttft"]["p90"], ">cutoff")
                self.assertEqual(summary["latencies_sec"]["e2e"]["p50"], ">cutoff")
                self.assertEqual(summary["latencies_sec"]["tpot"]["count"], 1)
                self.assertEqual(summary["slo"]["ttft_le_5s"]["passed"], 1)
                partial = next(r for r in records if r["idx"] == 1)
                self.assertEqual(partial["status"], "unfinished")
                self.assertIsNotNone(partial["first_token_at"])
                self.assertEqual(partial["output_tokens"], 1)
                runs = load_runs([str(root/"mock_summary.json")])
                self.assertEqual(runs[0][0]["latencies_sec"], summary["latencies_sec"])
                self.assertEqual(runs[0][0]["throughput"], summary["throughput"])
                write_tables(runs, root)
                self.assertIn(">cutoff", (root/"summary.md").read_text())
                for _ in range(30):
                    if closed == {2, 4}:
                        break
                    await asyncio.sleep(.01)
                self.assertEqual(closed, {2, 4})
                # All done before the deadline: do not pad throughput duration to cutoff.
                trace.write_text(json.dumps({"idx": 0, "arrival": 0, "input_ids": [1],
                                             "input_len": 1, "output_len": 2}) + "\n")
                args.label, args.drain_timeout = "fast", 5
                self.assertEqual(await run(args), 0)
                fast = json.loads((root/"fast_summary.json").read_text())
                self.assertEqual((fast["completed"], fast["unfinished"], fast["failed"]), (1, 0, 0))
                self.assertLess(fast["duration_sec"], 1)
                self.assertGreaterEqual(fast["cutoff_time"], 5)
                self.assertEqual(fast["duration_sec"], fast["actual_duration_sec"])
        finally:
            await runner.cleanup()


class CutoffMetricTests(unittest.TestCase):
    def record(self, sent, first, finished, chunks, success=True):
        output = chunks[-1][1] if chunks else 0
        return dict(sent_at=sent, first_token_at=first, finished_at=finished,
                    ttft=None if first is None else first-sent,
                    e2e=finished-sent, tpot=(finished-first)/(output-1) if first is not None and output>1 else None,
                    input_len=10, output_tokens=output, cached_tokens=None,
                    success=success, send_lag=0, itl=[],
                    chunks=[dict(at=t, completion_tokens=n) for t,n in chunks])

    def test_retrospective_boundaries_partial_tokens_failures_and_unsent(self):
        records = [
            self.record(0, 1, 5, [(1, 1), (5, 3)]),  # DONE exactly at cutoff.
            self.record(1, 2, 6, [(2, 1), (4, 3), (6, 7)]),  # Partial: retain three.
            self.record(2, 6, 7, [(6, 1), (7, 2)]),  # First token after cutoff.
            self.record(3, 4, 4.5, [(4, 1)], False),  # Failed partial output still delivered.
        ]
        before = json.dumps(records)
        summary = summarize_records(records, 5, 5)
        self.assertEqual(json.dumps(records), before)
        self.assertEqual((summary["completed"], summary["unfinished"], summary["failed"], summary["unsent_requests"]), (1, 2, 1, 1))
        self.assertEqual(summary["duration_sec"], 5)
        self.assertEqual(summary["output_tokens"], 7)
        self.assertEqual(summary["input_tokens"], 30)
        self.assertEqual(summary["throughput"]["output_tokens_per_sec"], 1.4)
        self.assertEqual(summary["slo"]["ttft_le_5s"]["rate"], .2)
        self.assertEqual(summary["latencies_sec"]["ttft"]["p50"], 1)
        self.assertEqual(summary["latencies_sec"]["e2e"]["p50"], ">cutoff")
        self.assertEqual(summary["latencies_sec"]["tpot"]["count"], 1)
        self.assertIsNone(cutoff_records(records, 5)[2]["first_token_at"])
        json.dumps(summary, allow_nan=False)

    def test_extended_linear_quantiles(self):
        self.assertEqual(censored_distribution([1, 3])["p50"], 2)
        self.assertEqual(censored_distribution([1, 3, float("inf")])["p50"], 3)
        self.assertEqual(censored_distribution([1, float("inf")])["p50"], ">cutoff")
        self.assertEqual(censored_distribution([float("inf")])["max"], ">cutoff")
        self.assertIsNone(censored_distribution([])["p50"])


if __name__ == "__main__":
    unittest.main()
