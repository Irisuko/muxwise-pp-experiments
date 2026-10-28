#!/usr/bin/env python3
"""Wait for a completed development round, then check its frozen source."""
import argparse
import json
import subprocess
import time
from pathlib import Path
import p11a_runner as r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--round', required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--mode', choices=['correctness','verify'], default='verify')
    x = p.parse_args()
    marker = r.OUT / (x.round + '-big_pp4') / 'run.json'
    while True:
        assert not list(r.OUT.glob('STOP-*.json'))
        try:
            meta = json.loads(marker.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            meta = {}
        if 'finished_utc' in meta:
            assert meta['status'] == 'PASS', meta.get('error')
            break
        time.sleep(5)
    time.sleep(15)
    source = Path(meta['source']) if 'source' in meta else r.MAIN.parent / ('sglang-p11a-' + x.round)
    config = json.loads(x.config.read_text())
    for label in r.LABELS[:4]:
        tag = x.round + '-' + x.mode + '-' + label
        args = [r.b.PYTHON, 'scripts/p11a_verify.py', x.mode, label,
                '--tag', tag, '--source', str(source), '--extra', *config['flags'][label]]
        print('START', tag, flush=True)
        with (r.OUT / (tag + '-driver.log')).open('x') as log:
            subprocess.run(args, cwd=r.ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print('END', tag, flush=True)


if __name__ == '__main__':
    main()
