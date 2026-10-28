#!/usr/bin/env python3
"""Local aiohttp SSE mock. No model, torch, CUDA, or GPU access."""

import argparse
import asyncio
import json

from aiohttp import web


def create_app(prefill_base=0.01, prefill_per_token=0.000002, token_delay=0.001,
               tokens_per_chunk=1, fragment_bytes=0, metadata_only=False, omit_cache=False):
    app = web.Application(client_max_size=32 * 1024 * 1024)
    state = {"requests": 0, "active": 0, "max_active": 0}

    async def generate(request):
        body = await request.json()
        if not body.get("stream") or not isinstance(body.get("input_ids"), list):
            raise web.HTTPBadRequest(text="expected input_ids and stream=true")
        params = body.get("sampling_params", {})
        output_len = params.get("max_new_tokens")
        if type(output_len) is not int or output_len < 0:
            raise web.HTTPBadRequest(text="invalid max_new_tokens")
        prompt_len = len(body["input_ids"])
        state["requests"] += 1
        state["active"] += 1
        state["max_active"] = max(state["max_active"], state["active"])
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})

        async def write(data):
            if fragment_bytes:
                for offset in range(0, len(data), fragment_bytes):
                    await response.write(data[offset:offset + fragment_bytes])
                    await asyncio.sleep(0)
            else:
                await response.write(data)

        async def event(count):
            meta = {"completion_tokens": count, "prompt_tokens": prompt_len}
            if not omit_cache:
                meta["cached_tokens"] = prompt_len // 4
            data = json.dumps({"text": "mock", "meta_info": meta}, separators=(",", ":"))
            await write(("data: " + data + "\r\n\r\n").encode())

        try:
            await response.prepare(request)
            if metadata_only:
                await write(b": heartbeat\r\n\r\n")
                await event(0)
            await asyncio.sleep(prefill_base + prompt_len * prefill_per_token)
            count = 0
            while count < output_len:
                if count:
                    await asyncio.sleep(token_delay * min(tokens_per_chunk, output_len - count))
                count = min(output_len, count + tokens_per_chunk)
                await event(count)
            if not output_len:
                await event(0)
            await write(b"data: [DONE]\r\n\r\n")
            await response.write_eof()
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            state["active"] -= 1
        return response

    async def stats(request):
        return web.json_response(state)

    app.router.add_post("/generate", generate)
    app.router.add_get("/stats", stats)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--prefill-base", type=float, default=0.01)
    parser.add_argument("--prefill-per-token", type=float, default=0.000002)
    parser.add_argument("--token-delay", type=float, default=0.001)
    parser.add_argument("--tokens-per-chunk", type=int, default=1)
    parser.add_argument("--fragment-bytes", type=int, default=0)
    parser.add_argument("--metadata-only", action="store_true", help="send an initial zero-token metadata event")
    parser.add_argument("--omit-cache", action="store_true")
    args = parser.parse_args()
    if min(args.prefill_base, args.prefill_per_token, args.token_delay, args.fragment_bytes) < 0 or args.tokens_per_chunk < 1:
        parser.error("delays/fragment-bytes must be >= 0 and tokens-per-chunk must be >= 1")
    app = create_app(args.prefill_base, args.prefill_per_token, args.token_delay,
                     args.tokens_per_chunk, args.fragment_bytes, args.metadata_only, args.omit_cache)
    web.run_app(app, host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    main()
