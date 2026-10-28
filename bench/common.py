"""Small CPU-only helpers shared by the benchmark scripts."""

import json
import math
from pathlib import Path

import numpy as np


def distribution(values):
    values = np.asarray([x for x in values if x is not None], dtype=np.float64)
    if values.size and not np.all(np.isfinite(values)):
        raise ValueError("non-finite value in distribution")
    result = {"count": int(values.size)}
    for name, percentile in (("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99)):
        result[name] = float(np.percentile(values, percentile)) if values.size else None
    result["mean"] = float(values.mean()) if values.size else None
    result["max"] = float(values.max()) if values.size else None
    return result


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("must be finite and > 0")
    return number


def nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("must be finite and >= 0")
    return number
