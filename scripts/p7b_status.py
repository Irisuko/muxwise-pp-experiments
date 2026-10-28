#!/usr/bin/env python3
"""Compact, read-only status for the running P7b batch."""
import datetime
import json
from pathlib import Path

root = Path("/root/workspace/muxpp_work")
out = root / "logs/p7b"
current = json.loads((out / "current.json").read_text())
print(datetime.datetime.now(datetime.timezone.utc).isoformat(),
      current.get("label", current.get("status")))
progress = [line for line in (out / "progress.md").read_text().splitlines()
            if line.startswith("| ") and "label" not in line and "---" not in line]
print(f"Finished configurations: {len(progress)}/11")
if "label" in current:
    path = out / current["label"]
    for name in ("formal-client.log", "smoke-client.log"):
        if (path / name).exists():
            lines = (path / name).read_text().splitlines()
            print(name + ": " + (lines[-1] if lines else "(starting)"))
            break
    else:
        p = out / (current["label"] + "-runner.log")
        print(p.read_text().splitlines()[-1:] if p.exists() else "starting")
