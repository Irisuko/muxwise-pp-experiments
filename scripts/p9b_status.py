#!/usr/bin/env python3
"""Small read-only progress snapshot; no service actions."""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

root=Path(__file__).resolve().parents[1]
out=root/'logs/p9b'
current=json.loads((out/'current.json').read_text())
if '--compact' in sys.argv:
    label=current.get('label','finished')
    path=out/label/'formal-client.log'
    if not path.exists(): path=out/f'{label}-runner.log'
    lines=path.read_text(errors='replace').splitlines() if path.exists() else []
    print(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'label':label,
                      'latest':lines[-1] if lines else 'starting',
                      'finished':max(0,len((out/'progress.md').read_text().splitlines())-4)},ensure_ascii=False))
    raise SystemExit
print(datetime.now(timezone.utc).isoformat(), current)
if 'label' in current:
    label=current['label']
    for path in [out/f'{label}-runner.log',out/label/'formal-client.log']:
        if path.exists():
            with path.open('rb') as f:
                f.seek(max(0,path.stat().st_size-3500))
                lines=f.read().decode(errors='replace').splitlines()
            print(path.name+':\n'+'\n'.join(lines[-4:]))
print((out/'progress.md').read_text())
