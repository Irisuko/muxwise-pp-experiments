#!/usr/bin/env python3
"""Cold, serial logits evidence for correctness discrepancies, not performance."""
import argparse
import fcntl
import json
from pathlib import Path
import signal
import time
import pynvml
import p11a_runner as r
import p11a_verify as v


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('label', choices=r.LABELS[:4])
    p.add_argument('--kind', choices=['plain', 'mux'], required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--cases', type=int, nargs='+', required=True)
    p.add_argument('--outputs', type=int, default=64)
    p.add_argument('--layers', type=int, nargs='+')
    p.add_argument('--source', type=Path, default=r.MAIN)
    p.add_argument('--extra', nargs=argparse.REMAINDER, default=[])
    x = p.parse_args()
    assert all(0 <= i < 20 for i in x.cases)
    r.monitoring.OUT = r.OUT
    pynvml.nvmlInit()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    path = r.OUT / x.tag
    probe = path / 'logits'
    with (r.ROOT / 'logs/p7a/gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not list(r.OUT.glob('STOP-*.json'))
        r.monitoring.idle(x.tag + '-before')
        jobs = v.ordinary(x.label) if x.kind == 'plain' else r.commands(x.label, x.source, path, x.extra)
        for name, cmd, env, _ in jobs:
            if name == 'router':
                continue
            if '--skip-tokenizer-init' not in cmd:
                cmd += ['--skip-tokenizer-init']
            env.update(P11A_LOGITS_PROBE=str(probe),
                       PYTHONPATH=str(r.ROOT / 'scripts/p11a_probe') + ':' + str(x.source / 'python'))
            if x.layers:
                env['P11A_LAYER_PROBE'] = ','.join(map(str, x.layers))
        r.b.commands = lambda _: jobs
        c = r.p7.Cluster(x.label, path)
        c.meta.update(mode='numeric', kind=x.kind, cases=x.cases,
                      commit=r.monitoring.git(x.source, 'rev-parse', 'HEAD'),
                      diagnostic_only=True)
        guard = r.Guard(x.tag)
        guard.start()
        try:
            c.start()
            probe.mkdir()
            cases = v.prompts(x.label)
            (probe / 'case-lengths.json').write_text(json.dumps({f'cold-{i}':len(cases[i]) for i in x.cases}))
            for i in x.cases:
                r.b.flush(c)
                (probe / 'active-case').write_text(f'cold-{i}')
                try:
                    out = r.b.generate(r.b.BASE, cases[i], x.outputs)
                finally:
                    (probe / 'active-case').unlink(missing_ok=True)
                r.b.save(path / f'cold-{i}.json', out)
                assert list(probe.glob(f'cold-{i}-*.pt')), 'Probe not installed in scheduler'
            r.b.flush(c)
            time.sleep(3)
            c.meta['status'] = 'PASS'
        except BaseException as exc:
            c.meta.update(status='FAIL', error=repr(exc))
            raise
        finally:
            try:
                c.close()
            finally:
                guard.close()
        assert not guard.error
        time.sleep(3)
        r.monitoring.idle(x.tag + '-after')


if __name__ == '__main__':
    main()
