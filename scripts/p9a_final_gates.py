#!/usr/bin/env python3
"""Validate selected existing policy on PP4, then correctness/stress/legacy gates."""
import os
from pathlib import Path
import subprocess
import sys

root = Path('/root/workspace/muxpp_work')
env = dict(os.environ, P9A_SOURCE='/root/workspace/sglang-p9a-v4')
for trace in ['toolagent', 'conversation']:
    tag = f'r1v4-pp4-{trace}'
    print('START', tag, flush=True)
    with (root / f'logs/p9a/{tag}-driver.log').open('x') as f:
        subprocess.run([sys.executable, str(root / 'scripts/p9a_runner.py'),
                        '--topology', 'pp4', '--trace', trace, '--tag', tag],
                       env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    subprocess.run([sys.executable, str(root / 'scripts/p9a_analyze.py'),
                    str(root / 'logs/p9a' / tag)], check=True, stdout=subprocess.DEVNULL)
    print('PASS', tag, flush=True)
for script in ['p9a_compare.py', 'p9a_stage_report.py', 'p9a_audit.py']:
    subprocess.run([sys.executable, str(root / 'scripts' / script)], check=True,
                   stdout=subprocess.DEVNULL)
subprocess.run([sys.executable, str(root / 'scripts/p9a_regression_queue.py')], check=True)
with (root / 'logs/p9a/v4-pp2dp2-smoke-driver.log').open('x') as f:
    subprocess.run([sys.executable, str(root / 'scripts/p9a_serve.py'), 'smoke',
                    'pp2dp2', '--tag', 'v4-pp2dp2-smoke'],
                   stdout=f, stderr=subprocess.STDOUT, check=True)
print('ALL FINAL GATES COMPLETE', flush=True)
