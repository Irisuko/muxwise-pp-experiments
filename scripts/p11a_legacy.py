#!/usr/bin/env python3
"""Run final legacy-path and PP2xDP2 functional regressions, without traces."""
import argparse
import json
from pathlib import Path
import subprocess
import p11a_runner as r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=r.MAIN)
    p.add_argument('--config', type=Path, default=r.ROOT / 'configs/p11a_selected.json')
    p.add_argument('--prefix', default='final-legacy')
    x = p.parse_args()
    flags = json.loads(x.config.read_text())['flags']
    jobs = [(label, 'pdmux') for label in ['llama_tp1', 'llama_tp2', 'big_tp4']]
    jobs += [(label, kind) for kind in ['plain', 'chunked'] for label in r.LABELS[:4]]
    jobs += [('llama_pp2dp2', 'plain')]
    for label, kind in jobs:
        tag = f'{x.prefix}-{kind}-{label}'
        args = [r.b.PYTHON, 'scripts/p11a_verify.py', 'smoke', label,
                '--kind', kind, '--tag', tag, '--source', str(x.source)]
        if label == 'llama_pp2dp2':
            args += ['--extra', *flags[label]]
        print('START', tag, flush=True)
        with (r.OUT / (tag + '-driver.log')).open('x') as log:
            subprocess.run(args, cwd=r.ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print('END', tag, flush=True)


if __name__ == '__main__':
    main()
