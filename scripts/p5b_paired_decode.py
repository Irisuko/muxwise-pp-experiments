"""Compare exact serial20 bs/context/partition shapes; no new GPU workload."""
import json
import statistics
from collections import defaultdict
from pathlib import Path

root = Path("/root/workspace/muxpp_work/logs/p5b")


def samples(path):
    groups = defaultdict(list)
    previous = None
    for line in path.read_text().splitlines():
        row = json.loads(line)
        # Exclude warmup, empty or mixed partitions, and the first interval after
        # a partition transition. Serial20 has no simultaneous prefill/decode.
        if (
            row["bs"] == 1
            and row["ctx"] >= 100
            and row["partition"] == 7
            and previous
            and previous["partition"] == 7
            and previous["ctx"] + 1 == row["ctx"]
        ):
            groups[(row["bs"], row["ctx"], row["partition"])].append(
                row["seconds"] * 1000
            )
        previous = row
    return groups


a = samples(root / "v2-serial-rounds.jsonl")
b = samples(root / "v3-serial-rounds.jsonl")
pairs = [
    dict(
        bs=k[0],
        ctx=k[1],
        partition=k[2],
        baseline_ms=a[k][0],
        v3_ms=b[k][0],
        reduction_pct=100 * (1 - b[k][0] / a[k][0]),
    )
    for k in sorted(a.keys() & b.keys())
    if len(a[k]) == len(b[k]) == 1
]
assert len(pairs) >= 1000, len(pairs)
base = [p["baseline_ms"] for p in pairs]
v3 = [p["v3_ms"] for p in pairs]
out = dict(
    metric="stage0 same-slot launch-to-launch wall milliseconds",
    scope="serial20 correctness corpus, unique exact bs=1/context/partition=7 pairs",
    pairs=len(pairs),
    baseline_median_ms=statistics.median(base),
    v3_median_ms=statistics.median(v3),
    median_reduction_pct=100 * (1 - statistics.median(v3) / statistics.median(base)),
    baseline_mean_ms=statistics.mean(base),
    v3_mean_ms=statistics.mean(v3),
    mean_reduction_pct=100 * (1 - statistics.mean(v3) / statistics.mean(base)),
    paired_reduction_median_pct=statistics.median(p["reduction_pct"] for p in pairs),
    details=pairs,
)
(root / "matched-serial-decode.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k: v for k, v in out.items() if k != "details"}, indent=2))
