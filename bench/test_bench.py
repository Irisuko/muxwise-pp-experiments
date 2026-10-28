"""Correctness tests: run with python -m unittest discover -s bench -v."""

import argparse
import asyncio
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import time
import unittest

import aiohttp
from aiohttp import web
import numpy as np
from tokenizers import AddedToken, Tokenizer, models

from bench_client import encode_payload, issue_request, sse_data, summarize_records
from build_trace import build, load_vocabulary
from mock_server import create_app


class ByteContent:
    def __init__(self, chunks):
        self.chunks = chunks

    async def iter_any(self):
        for chunk in self.chunks:
            yield chunk


class SSETests(unittest.IsolatedAsyncioTestCase):
    async def test_fragmentation_multiline_unicode_and_done(self):
        data = ': heartbeat\r\n\r\nevent: message\r\ndata: {"text":"中文",\r\ndata: "n":2}\r\n\r\ndata: [DONE]'.encode()
        for size in (1, 3, 11, len(data)):
            events = [item async for item in sse_data(ByteContent([data[i:i + size] for i in range(0, len(data), size)]))]
            self.assertEqual(json.loads(events[0]), {"text": "中文", "n": 2})
            self.assertEqual(events[1], "[DONE]")

    async def test_event_size_limit(self):
        with self.assertRaisesRegex(ValueError, "maximum size"):
            _ = [item async for item in sse_data(ByteContent([b"data: " + b"x" * 100]), max_event_bytes=32)]


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runners = []
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5),
                                             connector=aiohttp.TCPConnector(limit=0))

    async def asyncTearDown(self):
        await self.session.close()
        for runner in self.runners:
            await runner.cleanup()

    async def serve(self, app):
        runner = web.AppRunner(app)
        await runner.setup()
        self.runners.append(runner)
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        return f"http://127.0.0.1:{runner.addresses[0][1]}/generate"

    async def fetch(self, url, output_len=5):
        request = {"idx": 1, "arrival": 0, "input_len": 16, "output_len": output_len,
                   "payload": encode_payload([128000] + [10] * 15, output_len)}
        return await issue_request(self.session, url, request, asyncio.get_running_loop().time(), time.time())

    async def test_batched_stream_metadata_and_missing_cache(self):
        url = await self.serve(create_app(prefill_base=0.03, token_delay=0.002, tokens_per_chunk=2,
                                          fragment_bytes=7, metadata_only=True, omit_cache=True))
        result = await self.fetch(url)
        self.assertTrue(result["success"], result["error"])
        self.assertGreaterEqual(result["ttft"], 0.03)
        self.assertEqual([c["token_delta"] for c in result["chunks"]], [0, 2, 2, 1])
        self.assertEqual(result["output_tokens"], 5)
        self.assertEqual(result["itl_token_deltas"], [2, 1])
        self.assertEqual(len(result["itl"]), 2)
        self.assertIsNone(result["cached_tokens"])
        self.assertAlmostEqual(result["tpot"], (result["e2e"] - result["ttft"]) / 4)

    async def test_single_token_tpot_is_undefined(self):
        result = await self.fetch(await self.serve(create_app()), 1)
        self.assertTrue(result["success"])
        self.assertIsNone(result["tpot"])
        self.assertEqual(result["itl"], [])

    async def test_http_truncation_server_error_and_length_mismatch(self):
        cases = [
            (503, "overloaded", "HTTP 503"),
            (200, 'data: {"meta_info":{"completion_tokens":1}}\n\n', "before data: [DONE]"),
            (200, 'data: {"error":"failed"}\n\ndata: [DONE]\n\n', "server error"),
            (200, 'data: {"meta_info":{"completion_tokens":1}}\n\ndata: [DONE]\n\n', "length mismatch"),
            (200, 'data: {bad json}\n\n', "JSONDecodeError"),
            (200, 'data: {"meta_info":{"completion_tokens":2}}\n\ndata: {"meta_info":{"completion_tokens":1}}\n\n', "non-monotonic"),
        ]
        for status, body, error in cases:
            async def handler(request, status=status, body=body):
                return web.Response(status=status, text=body, content_type="text/event-stream")
            app = web.Application()
            app.router.add_post("/generate", handler)
            result = await self.fetch(await self.serve(app))
            self.assertFalse(result["success"])
            self.assertIn(error, result["error"])


class SummaryTests(unittest.TestCase):
    def test_slo_failure_denominator_and_cache_missing(self):
        common = {"finished_at": 10, "input_len": 100, "send_lag": 0.01, "cached_tokens": None, "itl": [0.1]}
        records = [
            dict(common, success=True, output_tokens=3, ttft=2, tpot=0.1, e2e=2.2, cached_tokens=50),
            dict(common, success=True, output_tokens=1, ttft=1, tpot=None, e2e=1),
            dict(common, success=False, output_tokens=1, ttft=1, tpot=None, e2e=1),
        ]
        summary = summarize_records(records, 4)
        self.assertEqual(summary["slo"]["ttft_le_5s"]["rate"], 0.5)
        self.assertEqual(summary["slo"]["ttft_le_5s_and_tpot_le_100ms"]["rate"], 0.25)
        self.assertEqual(summary["latencies_sec"]["tpot"]["count"], 1)
        self.assertEqual(summary["throughput"]["output_tokens_per_sec"], 0.5)
        self.assertEqual(summary["cached_tokens"]["mean_ratio"], 0.5)
        self.assertEqual(summary["unsent_requests"], 1)
        empty = summarize_records([])
        self.assertIsNone(empty["latencies_sec"]["ttft"]["p99"])
        self.assertIsNone(empty["cached_tokens"]["mean_ratio"])
        json.dumps(empty, allow_nan=False)


class TraceTests(unittest.TestCase):
    def test_determinism_boundaries_padding_and_added_token_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "a": 1, "b": 2, "[BOS]": 3}, unk_token="[UNK]"))
            tokenizer.add_special_tokens([AddedToken("[UNK]", special=True), AddedToken("[BOS]", special=True)])
            tokenizer.add_tokens([AddedToken("ordinary_added")])
            tokenizer.save(str(root / "tokenizer.json"))
            (root / "tokenizer_config.json").write_text(json.dumps({"bos_token": "[BOS]", "unk_token": "[UNK]"}))
            vocab, bos, _ = load_vocabulary(root)
            np.testing.assert_array_equal(vocab, [1, 2])
            self.assertEqual(bos, 3)
            source = root / "raw.jsonl"
            rows = [
                {"timestamp": 999, "input_length": 8, "output_length": 2, "hash_ids": [1]},
                {"timestamp": 1000, "input_length": 8, "output_length": 2, "hash_ids": [1]},
                {"timestamp": 1100, "input_length": 10, "output_length": 2, "hash_ids": [1]},
                {"timestamp": 1200, "input_length": 15, "output_length": 2, "hash_ids": [1]},
                {"timestamp": 2000, "input_length": 8, "output_length": 2, "hash_ids": [1]},
            ]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            args = argparse.Namespace(source=str(source), tokenizer=str(root), start_sec=1, duration_sec=1,
                                      stretch=2.5, max_total_len=12, block_size=8, out=str(root / "out.jsonl"))
            with redirect_stdout(io.StringIO()):
                stats = build(args)
                first = Path(args.out).read_bytes()
                build(args)
            self.assertEqual(first, Path(args.out).read_bytes())
            self.assertEqual(stats["requests"], 2)
            self.assertEqual(stats["filtered_requests"], 1)
            self.assertEqual(stats["padded_requests"], 1)
            self.assertEqual(stats["padded_tokens"], 2)
            generated = [json.loads(line) for line in first.splitlines()]
            self.assertEqual([r["arrival"] for r in generated], [0, 0.25])
            self.assertEqual(generated[0]["input_ids"], generated[1]["input_ids"][:8])
            self.assertEqual(generated[0]["input_ids"][0], 3)
            self.assertTrue(all(t in (1, 2) for row in generated for t in row["input_ids"][1:]))
            self.assertAlmostEqual(stats["theoretical_prefix_reuse_ratio"], 8 / 18)


if __name__ == "__main__":
    unittest.main()
