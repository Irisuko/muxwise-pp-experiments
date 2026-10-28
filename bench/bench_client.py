#!/usr/bin/env python3
"""Open-loop, CPU-only SSE replay client for SGLang's native /generate API."""

import argparse
import asyncio
from collections import deque
from contextlib import suppress
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import aiohttp

from common import distribution, nonnegative_float, positive_float, write_json
from metrics import summarize_records


def encode_payload(input_ids, output_len):
    return json.dumps({"input_ids": input_ids,
                       "sampling_params": {"max_new_tokens": output_len, "ignore_eos": True,
                                           "temperature": 0}, "stream": True},
                      separators=(",", ":")).encode("utf-8")


def load_trace(path, limit=None, time_scale=1.0):
    """Pre-encode bodies before the clock starts; retain no large Python int lists."""
    lock = Path(__file__).resolve().parent.parent / "configs/p10a_formal.lock.json"
    if lock.exists() and Path(path).resolve() == Path(json.loads(lock.read_text())["trace"]).resolve():
        raise RuntimeError("P10a FORMAL LOCK: this trace is generation/statistics only")
    requests, warmup_ids, previous, seen = [], None, -1.0, set()
    with Path(path).open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            arrival = float(row["arrival"]) * time_scale
            idx = row["idx"]
            if not math.isfinite(arrival) or arrival < 0 or arrival < previous:
                raise ValueError(f"line {line_number}: arrivals must be finite, nonnegative and sorted")
            if idx in seen:
                raise ValueError(f"duplicate idx: {idx}")
            if row["input_len"] <= 0 or len(row["input_ids"]) != row["input_len"] or row["output_len"] < 0:
                raise ValueError(f"line {line_number}: invalid token lengths")
            if any(type(i) is not int or i < 0 for i in row["input_ids"]):
                raise ValueError(f"line {line_number}: invalid input token ID")
            previous = arrival
            seen.add(idx)
            if warmup_ids is None:
                warmup_ids = row["input_ids"][:16]
            requests.append({"idx": idx, "arrival": arrival, "trace_arrival": float(row["arrival"]),
                             "input_len": row["input_len"], "output_len": row["output_len"],
                             "payload": encode_payload(row["input_ids"], row["output_len"])})
            if limit is not None and len(requests) >= limit:
                break
    if not requests:
        raise ValueError("empty trace selection")
    return requests, warmup_ids


async def sse_data(content, max_event_bytes=16 * 1024 * 1024):
    """Parse SSE framing independently of TCP/HTTP chunk boundaries, including CRLF."""
    buffer = b""
    fields = []
    event_bytes = 0
    async for part in content.iter_any():
        buffer += part
        while b"\n" in buffer:
            raw, buffer = buffer.split(b"\n", 1)
            raw = raw.rstrip(b"\r")
            if not raw:
                if fields:
                    yield b"\n".join(fields).decode("utf-8")
                fields, event_bytes = [], 0
            elif raw.startswith(b"data:"):
                value = raw[5:]
                if value.startswith(b" "):
                    value = value[1:]
                fields.append(value)
                event_bytes += len(value)
            if event_bytes > max_event_bytes:
                raise ValueError("SSE event exceeds maximum size")
        if len(buffer) + event_bytes > max_event_bytes:
            raise ValueError("unterminated SSE event exceeds maximum size")
    # Accept a final data line without a trailing newline; the caller still requires [DONE].
    if buffer.rstrip(b"\r").startswith(b"data:"):
        fields.append(buffer.rstrip(b"\r")[5:].removeprefix(b" "))
    if fields:
        yield b"\n".join(fields).decode("utf-8")


async def issue_request(session, url, request, origin, wall_origin, state=None):
    loop = asyncio.get_running_loop()
    sent = loop.time() - origin
    result = {key: value for key, value in request.items() if key != "payload"}
    result.update({"sent_at": sent, "sent_unix": wall_origin + sent,
                   "send_lag": sent - request["arrival"], "first_token_at": None,
                   "finished_at": None, "ttft": None, "tpot": None, "e2e": None,
                   "output_tokens": 0, "cached_tokens": None, "chunks": [],
                   "itl": [], "itl_token_deltas": [], "itl_per_token_estimates": [],
                   "http_status": None, "done_received": False, "success": False, "error": None})
    if state is not None:
        state["sent"] += 1
        state["last_sent_at"] = sent
        state["sent_event"].set()
    previous_token_time = None
    try:
        async with session.post(url, data=request["payload"],
                                headers={"Content-Type": "application/json", "Accept": "text/event-stream"}) as response:
            result["http_status"] = response.status
            if response.status != 200:
                body = (await response.content.read(4096)).decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {response.status}: {body}")
            async for data in sse_data(response.content):
                now = loop.time() - origin
                if data.strip() == "[DONE]":
                    result["done_received"] = True
                    result["finished_at"] = now
                    break
                event = json.loads(data)
                if not isinstance(event, dict):
                    raise ValueError("expected one JSON object per SSE event")
                if event.get("error"):
                    raise RuntimeError(f"server error: {event['error']}")
                meta = event.get("meta_info") or {}
                if meta.get("id") is not None:
                    result["rid"] = meta["id"]
                count = meta.get("completion_tokens", result["output_tokens"])
                if type(count) is not int or count < result["output_tokens"]:
                    raise ValueError(f"invalid/non-monotonic completion_tokens: {count}")
                delta = count - result["output_tokens"]
                result["chunks"].append({"at": now, "completion_tokens": count, "token_delta": delta})
                if delta:
                    if previous_token_time is None:
                        result["first_token_at"] = now
                        result["ttft"] = now - sent
                    else:
                        interval = now - previous_token_time
                        result["itl"].append(interval)
                        result["itl_token_deltas"].append(delta)
                        result["itl_per_token_estimates"].append(interval / delta)
                    previous_token_time = now
                result["output_tokens"] = count
                if meta.get("cached_tokens") is not None:
                    cached = meta["cached_tokens"]
                    if type(cached) is not int or not 0 <= cached <= request["input_len"]:
                        raise ValueError(f"invalid cached_tokens: {cached}")
                    result["cached_tokens"] = cached
                finish_reason = meta.get("finish_reason")
                if isinstance(finish_reason, dict) and finish_reason.get("type") in ("abort", "error"):
                    raise RuntimeError(f"server finish_reason: {finish_reason}")
            if not result["done_received"]:
                raise RuntimeError("stream ended before data: [DONE]")
            if result["output_tokens"] != request["output_len"]:
                raise RuntimeError(f"output length mismatch: expected {request['output_len']}, "
                                   f"received {result['output_tokens']}")
            result["success"] = True
    except asyncio.CancelledError:
        result["status"] = "unfinished"
        result["error"] = "CancelledError: " + (state.get("cancel_reason", "benchmark interrupted") if state else "benchmark interrupted")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {str(exc)[:4096]}"
    finally:
        if result["finished_at"] is None:
            result["finished_at"] = loop.time() - origin
        result.setdefault("status", "completed" if result["success"] else "failed")
        result["e2e"] = result["finished_at"] - sent
        if result["ttft"] is not None and result["output_tokens"] > 1:
            result["tpot"] = (result["e2e"] - result["ttft"]) / (result["output_tokens"] - 1)
        if state is not None:
            state["finished"] += 1
            state["successful"] += int(result["success"])
            state["unfinished"] += int(result["status"] == "unfinished")
            if result["ttft"] is not None:
                state["recent_ttft"].append((result["finished_at"], result["ttft"]))
    return result


async def report_progress(state, origin, interval):
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(interval)
        now = loop.time() - origin
        recent = state["recent_ttft"]
        while recent and recent[0][0] < now - 30:
            recent.popleft()
        p50 = distribution(x[1] for x in recent)["p50"]
        p50_text = f"{p50:.3f}s" if p50 is not None else "n/a"
        print(f"[{now:8.1f}s] sent={state['sent']} completed={state['successful']} "
              f"failed={state['finished'] - state['successful'] - state['unfinished']} "
              f"inflight={state['sent'] - state['finished']} recent_TTFT_p50={p50_text}", flush=True)


async def run(args):
    requests, warmup_ids = load_trace(args.trace, args.limit, args.time_scale)
    base = args.base_url.rstrip("/")
    url = base if base.endswith("/generate") else base + "/generate"
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    details_path = output / f"{args.label}_requests.jsonl"
    summary_path = output / f"{args.label}_summary.json"
    if not args.overwrite and (details_path.exists() or summary_path.exists()):
        raise FileExistsError("label already exists; use a new --label or explicit --overwrite")
    timeout = aiohttp.ClientTimeout(total=args.timeout, sock_read=args.timeout)
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0, force_close=getattr(args, "force_close", False))
    interrupted = False
    cutoff_time = None
    drain_timeout = getattr(args, "drain_timeout", 300.0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector,
                                     read_bufsize=1024 * 1024) as session:
        loop = asyncio.get_running_loop()
        for i in range(args.warmup):
            warmup = {"idx": -i - 1, "arrival": 0, "trace_arrival": 0, "input_len": len(warmup_ids),
                      "output_len": 4, "payload": encode_payload(warmup_ids, 4)}
            result = await issue_request(session, url, warmup, loop.time(), time.time())
            if not result["success"]:
                raise RuntimeError(f"warmup {i + 1} failed: {result['error']}")
        print(f"Loaded {len(requests)} requests; warmup={args.warmup}; target={url}; "
              f"last arrival={requests[-1]['arrival']:.3f}s", flush=True)
        wall_origin, origin = time.time(), loop.time()
        state = {"sent": 0, "finished": 0, "successful": 0, "unfinished": 0,
                 "recent_ttft": deque(), "sent_event": asyncio.Event()}
        progress = asyncio.create_task(report_progress(state, origin, args.progress_interval))
        tasks = []
        try:
            for i, request in enumerate(requests):
                delay = origin + request["arrival"] - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                # No semaphore, connector cap, or await on preceding responses.
                tasks.append(asyncio.create_task(issue_request(session, url, request, origin, wall_origin, state)))
                if i % 128 == 127:
                    await asyncio.sleep(0)
            # Wait for all tasks to enter issue_request, so the deadline uses actual send time.
            while state["sent"] < len(tasks):
                state["sent_event"].clear()
                await state["sent_event"].wait()
            cutoff_time = state["last_sent_at"] + drain_timeout
            _, pending = await asyncio.wait(tasks, timeout=max(0, origin + cutoff_time - loop.time()))
            state["cancel_reason"] = "drain timeout"
            for task in pending:
                task.cancel()
            records = await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            interrupted = True
            for task in tasks:
                task.cancel()
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
            records = [r for r in outcomes if isinstance(r, dict)]
        finally:
            progress.cancel()
            with suppress(asyncio.CancelledError):
                await progress
    # Serialize after replay so disk writes cannot delay scheduled arrivals.
    for record in records:
        if cutoff_time is not None and record["finished_at"] > cutoff_time:
            record["status"] = "unfinished"
            record["success"] = False
    temporary = details_path.with_name(details_path.name + ".tmp")
    with temporary.open("w") as destination:
        for record in records:
            destination.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
    temporary.replace(details_path)
    summary = summarize_records(records, len(requests), cutoff_time)
    summary.update({"schema_version": 2, "drain_timeout_sec": drain_timeout, "label": args.label, "base_url": args.base_url,
                    "trace": str(Path(args.trace).resolve()), "request_details_file": str(details_path.resolve()),
                    "run_started_at": datetime.fromtimestamp(wall_origin, timezone.utc).isoformat(),
                    "run_started_unix": wall_origin, "time_scale": args.time_scale, "force_close": getattr(args, "force_close", False), "limit": args.limit,
                    "warmup": args.warmup, "timeout_sec": args.timeout, "interrupted": interrupted,
                    "scheduled_duration_sec": requests[-1]["arrival"]})
    write_json(summary_path, summary)
    print(f"Finished: completed={summary['completed_requests']} unfinished={summary['unfinished_requests']} failed={summary['failed_requests']} "
          f"unsent={summary['unsent_requests']} duration={summary['duration_sec']:.3f}s\n"
          f"Details: {details_path}\nSummary: {summary_path}", flush=True)
    return 130 if interrupted else (1 if summary["failed_requests"] else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "results"))
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--time-scale", type=positive_float, default=1.0,
                        help="multiply trace arrival seconds; 0.05 runs arrivals 20x faster")
    parser.add_argument("--drain-timeout", type=nonnegative_float, default=300.0,
                        help="cancel unfinished requests SEC after the last actual send (default: 300)")
    parser.add_argument("--timeout", type=positive_float, default=3600.0)
    parser.add_argument("--progress-interval", type=positive_float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force-close", action="store_true",
                        help="Use a fresh HTTP connection per request to avoid server keepalive expiry races; no retries.")
    args = parser.parse_args()
    if args.warmup < 0 or (args.limit is not None and args.limit <= 0):
        parser.error("warmup must be >= 0 and limit must be > 0")
    if not args.label or Path(args.label).name != args.label or args.label in (".", ".."):
        parser.error("label must be a simple filename component")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
