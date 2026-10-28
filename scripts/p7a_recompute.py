#!/usr/bin/env python3
"""Read-only retrospective 300s drain projection of the four final runs."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from common import write_json
from metrics import summarize_records

LABELS = ("chunked", "muxwise_r2", "sglang_pd_r2", "muxwise_pp_v3")


def main():
    manifest = {}
    for label in LABELS:
        source = ROOT / "results" / f"{label}_summary.json"
        original = json.loads(source.read_text())
        details = source.parent / Path(original["request_details_file"]).name
        files = [source, details, source.parent / f"{label}.formal-started.json"]
        before = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        records = [json.loads(line) for line in details.read_text().splitlines() if line.strip()]
        cutoff = max(r["sent_at"] for r in records) + 300
        result = dict(original)
        result.update(summarize_records(records, original["planned_requests"], cutoff))
        result.update(schema_version=2, drain_timeout_sec=300,
                      retrospective_cutoff=True, original_summary_file=str(source),
                      request_details_file=str(details), original_file_sha256=before)
        write_json(source.with_name(f"{label}_cut300_summary.json"), result)
        after = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        assert before == after, "Original formal data changed"
        manifest.update(before)
        print(label, json.dumps({k: result[k] for k in ("completed", "unfinished", "failed", "cutoff_time", "duration_sec", "throughput")}))
    write_json(ROOT / "logs/p7a/original-final-sha256.json", manifest)


if __name__ == "__main__":
    main()
