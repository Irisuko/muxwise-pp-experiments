#!/usr/bin/env python3
"""Build reproducible token-ID prompts without loading a model or using torch."""

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from common import distribution, positive_float, write_json

DEFAULT_SOURCE = "/root/ext/mooncake/FAST25-release/traces/toolagent_trace.jsonl"
DEFAULT_TOKENIZER = "/root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct"


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def seed_for(value, namespace="hash"):
    if namespace == "hash" and isinstance(value, int) and value >= 0:
        return value
    raw = (namespace + ":" + canonical(value)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=16).digest(), "little")


def load_vocabulary(path):
    path = Path(path)
    tokenizer_path = path / "tokenizer.json" if path.is_dir() else path
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    # Includes non-special added tokens, which must also never be sampled.
    excluded = set(tokenizer.get_added_tokens_decoder())
    config_path = tokenizer_path.parent / "tokenizer_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    excluded.update(int(i) for i in config.get("added_tokens_decoder", {}))
    for key, value in config.items():
        if key.endswith("_token"):
            token = value.get("content") if isinstance(value, dict) else value
            if isinstance(token, str):
                token_id = tokenizer.token_to_id(token)
                if token_id is not None:
                    excluded.add(token_id)
    special_path = tokenizer_path.parent / "special_tokens_map.json"
    if special_path.exists():
        for value in json.loads(special_path.read_text()).values():
            for item in value if isinstance(value, list) else [value]:
                token = item.get("content") if isinstance(item, dict) else item
                token_id = tokenizer.token_to_id(token) if isinstance(token, str) else None
                if token_id is not None:
                    excluded.add(token_id)
    llama3 = tokenizer.token_to_id("<|begin_of_text|>") == 128000
    bos = config.get("bos_token", "<|begin_of_text|>")
    bos = bos.get("content") if isinstance(bos, dict) else bos
    bos_id = 128000 if llama3 else tokenizer.token_to_id(bos)
    if bos_id is None:
        raise ValueError("tokenizer must define a BOS token")
    excluded.add(bos_id)
    ids = sorted(set(tokenizer.get_vocab(with_added_tokens=False).values()) - excluded)
    if llama3:
        ids = [i for i in ids if i < 128000]
    if not ids:
        raise ValueError("no ordinary tokens available for sampling")
    return np.asarray(ids, dtype=np.int32), bos_id, len(excluded)


def build(args):
    if args.start_sec < 0 or args.block_size <= 0 or args.max_total_len <= 0:
        raise ValueError("start-sec must be >= 0; block-size and max-total-len must be > 0")
    vocab, bos, excluded_count = load_vocabulary(args.tokenizer)
    start_ms = args.start_sec * 1000
    end_ms = (args.start_sec + args.duration_sec) * 1000
    selected, window_count, filtered = [], 0, 0
    with Path(args.source).open() as source:
        for idx, line in enumerate(source):
            if not line.strip():
                continue
            row = json.loads(line)
            if not start_ms <= row["timestamp"] < end_ms:
                continue
            window_count += 1
            if row["input_length"] + row["output_length"] > args.max_total_len:
                filtered += 1
                continue
            if row["input_length"] < 1 or row["output_length"] < 0:
                raise ValueError(f"invalid lengths on source row {idx}")
            selected.append((idx, row))
    selected.sort(key=lambda x: (x[1]["timestamp"], x[0]))

    @lru_cache(maxsize=4096)
    def block(key):
        rng = np.random.Generator(np.random.PCG64(seed_for(json.loads(key))))
        return rng.choice(vocab, size=args.block_size)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    seen = set()
    reusable_tokens = hash_tokens = padded_requests = padded_tokens = 0
    total_input = total_output = 0
    with temporary.open("w") as destination:
        for idx, row in selected:
            length = row["input_length"]
            ids = np.empty(length, dtype=np.int32)
            offset = 0
            for hash_id in row["hash_ids"]:
                if offset == length:
                    break
                key = canonical(hash_id)
                count = min(args.block_size, length - offset)
                ids[offset:offset + count] = block(key)[:count]
                hash_tokens += count
                if key in seen:
                    reusable_tokens += count
                seen.add(key)
                offset += count
            if offset < length:
                padded_requests += 1
                padded_tokens += length - offset
                rng = np.random.Generator(np.random.PCG64(seed_for(idx, "padding")))
                ids[offset:] = rng.choice(vocab, size=length - offset)
            ids[0] = bos  # Replace, do not prepend: preserve exact length/block boundaries.
            result = {"idx": idx, "arrival": (row["timestamp"] - start_ms) / 1000 * args.stretch,
                      "input_len": length, "output_len": row["output_length"],
                      "input_ids": ids.tolist(), "hash_ids": row["hash_ids"]}
            destination.write(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
            total_input += length
            total_output += row["output_length"]
    temporary.replace(output)
    stats = {
        "schema_version": 1, "source": str(Path(args.source).resolve()),
        "trace": str(output.resolve()), "tokenizer": str(Path(args.tokenizer).resolve()),
        "start_sec": args.start_sec, "source_duration_sec": args.duration_sec,
        "stretch": args.stretch, "duration_sec": args.duration_sec * args.stretch,
        "last_arrival_sec": (selected[-1][1]["timestamp"] - start_ms) / 1000 * args.stretch if selected else None,
        "window_requests": window_count, "filtered_requests": filtered, "requests": len(selected),
        "max_total_len": args.max_total_len, "block_size": args.block_size,
        "padded_requests": padded_requests, "padded_tokens": padded_tokens,
        "ordinary_vocab_size": len(vocab), "excluded_token_count": excluded_count, "bos_token_id": bos,
        "input_length": distribution(row["input_length"] for _, row in selected),
        "output_length": distribution(row["output_length"] for _, row in selected),
        "total_input_tokens": total_input, "total_output_tokens": total_output,
        "unique_hash_blocks": len(seen), "hash_covered_tokens": hash_tokens,
        "theoretical_reusable_tokens": reusable_tokens,
        "theoretical_prefix_reuse_ratio": reusable_tokens / total_input if total_input else None,
        "reuse_definition": "Tokens in hash blocks after each hash's first occurrence / all input tokens. "
                            "Truncated blocks count only used tokens; padding is not reusable. "
                            "This first-occurrence estimate is not a measured radix-cache hit rate; "
                            "it ignores eviction, concurrent completion, contiguous-prefix and partial-block constraints.",
        "seed_definition": "PCG64(hash_id) for nonnegative integer hashes; otherwise BLAKE2b-128 of "
                           "hash:canonical_JSON. Padding uses BLAKE2b-128 of padding:source_row_index.",
    }
    write_json(output.with_suffix(".stats.json"), stats)
    print(json.dumps(stats, indent=2, ensure_ascii=False, allow_nan=False))
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--start-sec", type=float, default=600)
    parser.add_argument("--duration-sec", type=positive_float, default=240)
    parser.add_argument("--stretch", type=positive_float, default=2.5)
    parser.add_argument("--max-total-len", type=int, default=32768)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--out", default=str(Path(__file__).parent / "toolagent_600_240_x2.5.jsonl"))
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
