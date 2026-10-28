#!/usr/bin/env python3
"""Check R2's 24B numeric repair before its development performance matrix."""
import argparse
import json
import subprocess
import time
from pathlib import Path
import p11a_runner as r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', default='r2')
    p.add_argument('--after', default='r1-correctness-big_pp4')
    p.add_argument('--source', type=Path, default=Path('/root/workspace/sglang-p11a-r2'))
    p.add_argument('--config', type=Path, default=r.ROOT / 'configs/p11a_r2.json')
    p.add_argument('--labels', nargs='+', choices=r.LABELS[:4], default=['big_pp2tp2', 'big_pp4'])
    p.add_argument('--numeric-gate', type=Path)
    x = p.parse_args()
    marker = r.OUT / x.after / 'run.json'
    while True:
        assert not list(r.OUT.glob('STOP-*.json'))
        try:
            meta = json.loads(marker.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            meta = {}
        if 'finished_utc' in meta:
            assert meta['status'] == 'PASS'
            break
        time.sleep(5)
    time.sleep(15)
    if x.numeric_gate:
        numeric = json.loads(x.numeric_gate.read_text())
        assert all(c['first_token_difference'] is None for c in numeric['cases']), numeric
    source = x.source
    config_path = x.config
    config = json.loads(config_path.read_text())
    gate = {}
    for label in x.labels:
        tag = x.variant + '-correctness-' + label
        args = [r.b.PYTHON, 'scripts/p11a_verify.py', 'correctness', label,
                '--tag', tag, '--source', str(source), '--extra', *config['flags'][label]]
        print('START', tag, flush=True)
        with (r.OUT / (tag + '-driver.log')).open('x') as log:
            subprocess.run(args, cwd=r.ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        checks = {}
        for kind in ['serial20', 'cold20']:
            actual = json.loads((r.OUT / tag / (kind + '.json')).read_text())
            reference = json.loads((r.OUT / ('reference-' + label) / (kind + '.json')).read_text())
            historical = r.ROOT / 'logs/p10b/r4-mux-pp4' / (kind + '.json')
            # The supplied P10b notes document one PP4 cold-cache near-tie.
            # Accept only the exact prior output or this run's plain output;
            # every new discrepancy stops the performance queue for diagnosis.
            previous = json.loads(historical.read_text()) if label == 'big_pp4' else reference
            unexpected = [i for i, (a, b, c) in enumerate(zip(actual, reference, previous))
                          if a['output_ids'] not in [b['output_ids'], c['output_ids']]]
            checks[kind] = dict(unexpected_cases=unexpected,
                                exact_plain=sum(a['output_ids'] == b['output_ids'] for a, b in zip(actual, reference)),
                                exact_previous=sum(a['output_ids'] == b['output_ids'] for a, b in zip(actual, previous)),
                                previous_sha256=r.monitoring.sha(historical) if label == 'big_pp4' else None)
        gate[label] = checks
        r.b.save(r.OUT / (x.variant + '-correctness-gate.json'), gate)
        print('END', tag, checks, flush=True)
    assert not any(c['unexpected_cases'] for kinds in gate.values() for c in kinds.values()), gate
    with (r.OUT / 'r2-performance-driver.log').open('x') as log:
        subprocess.run([r.b.PYTHON, 'scripts/p11a_runner.py', '--round', 'r2',
                        '--source', str(source), '--config', str(config_path)],
                       cwd=r.ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
