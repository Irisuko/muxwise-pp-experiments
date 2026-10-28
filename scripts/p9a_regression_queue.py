#!/usr/bin/env python3
"""Run the selected P9a regression gates serially; no formal trace access."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/root/workspace/muxpp_work')
LOG = ROOT / 'logs/p9a/regressions'
LOG.mkdir(exist_ok=True, parents=True)
selected = json.loads((ROOT / 'configs/p9a_selected.json').read_text())
env = dict(os.environ, P5B_SOURCE='/root/workspace/sglang')
common = ['--dtype', 'bfloat16']
mux = common + [
    '--mem-fraction-static', '0.71', '--pdmux-config-path',
    str(ROOT / 'configs/pdmux_pp_4090_v4.yml'),
    '--cuda-graph-bs', '1', '4', '8', '16', '32',
    '--max-prefill-tokens', '16384', '--pdmux-pp-max-budget', '16384',
    '--pdmux-pp-tpot-coeff', '0.75', '--pdmux-prefill-mlp-chunk-tokens', '4096',
    '--enable-cache-report', '--schedule-policy', 'fcfs',
    '--pdmux-pp-reserve-decode', '--pdmux-pp-balance-merge',
    '--pdmux-pp-rebalance', '--pdmux-pp-alp-admission',
    '--pdmux-pp-rebalance-unit', '8', '--pdmux-pp-summary-interval', '10',
] + selected['flags']
jobs = []
for tp in [2, 1]:
    for cold in [False, True]:
        suffix = 'cold' if cold else 'serial'
        jobs.append((f'tp{tp}-reference-{suffix}', 'normal',
                     ['--suite', 'correctness', '--tp', str(tp), '--pp', '1'] +
                     (['--cold-cache'] if cold else []), common))
for label, tp, pp, cap in [('pp2tp2', 2, 2, 3), ('pp4', 1, 4, 6)]:
    for suffix, suite in [('serial', 'correctness'), ('cold', 'correctness'),
                          ('cancel', 'control'), ('stress', 'stress')]:
        args = ['--suite', suite, '--tp', str(tp), '--pp', str(pp)]
        if suffix == 'cold':
            args += ['--cold-cache']
        if suite == 'stress':
            args += ['--limit', '200', '--time-scale', '0.02',
                     '--force-close', '--long-warmup']
        jobs.append((f'{label}-{suffix}', 'muxpp', args,
                     mux + ['--pdmux-pp-max-inflight-prefill', str(cap)]))
for label, mode, tp, pp in [('pdmux-tp1', 'pdmux', 1, 1),
                            ('pdmux-tp2', 'pdmux', 2, 1),
                            ('plain-pp2', 'pp', 1, 2)]:
    extra = common + (['--mem-fraction-static', '0.76'] if mode == 'pdmux' and tp == 2 else [])
    jobs.append((label, mode, ['--suite', 'smoke', '--tp', str(tp), '--pp', str(pp)], extra))
commands = [[sys.executable, str(ROOT / 'scripts/p9a_verify.py'), mode,
             '--tag', tag] + args + ['--extra'] + extra
            for tag, mode, args, extra in jobs]
(LOG / 'queue-commands.json').write_text(json.dumps(commands, indent=2))
for (tag, _, _, _), cmd in zip(jobs, commands):
    print('START', tag, flush=True)
    with (LOG / f'{tag}-driver.log').open('x') as f:
        subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    print('PASS', tag, flush=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts/p9a_audit.py')], check=True,
                   stdout=subprocess.DEVNULL)
    audit = json.loads((ROOT / 'logs/p9a/audit.json').read_text())
    for name, result in audit['correctness'].items():
        assert result['exact_requests'] == 20 and result['equal_tokens'] == 1280, (name, result)
    result = audit['regressions'][tag]
    assert result['status'] == 'PASS' and not result['errors'], (tag, result)
    if tag.startswith(('pp2tp2', 'pp4')):
        assert result['resource_conservation'] and all(result['tp_control_equal'].values()), (tag, result)
print('ALL REGRESSIONS COMPLETE', flush=True)
