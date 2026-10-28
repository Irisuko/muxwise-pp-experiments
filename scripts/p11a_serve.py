#!/usr/bin/env python3
"""Print or launch a selected v5 configuration; contains no trace replay."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import shlex
import signal
import time
import pynvml
import p11a_runner as r


LABELS = r.LABELS[:4] + ['llama_pp2dp2']


def commands(label, config):
    selected = json.loads(config.read_text())
    jobs = r.commands(label, r.MAIN, r.OUT / 'serve', selected['flags'][label])
    for name, cmd, env, _ in jobs:
        if name == 'router':
            continue
        # Detailed per-request developer tracing is kept in benchmark runs.
        index = cmd.index('--pdmux-pp-request-trace')
        del cmd[index:index + 2]
        env.pop('SGLANG_PDMUX_PP_ROUND_TRACE', None)
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['commands', 'launch'])
    p.add_argument('label', choices=LABELS)
    p.add_argument('--config', type=Path, default=r.ROOT / 'configs/p11a_selected.json')
    p.add_argument('--tag')
    x = p.parse_args()
    jobs = commands(x.label, x.config)
    if x.mode == 'commands':
        print('source /root/workspace/muxpp_work/env.sh')
        if len(jobs) > 1:
            print('# Start workers in separate terminals; start the router after both workers are ready.')
        for _, cmd, env, _ in jobs:
            print(shlex.join(['env', *[k + '=' + v for k, v in env.items()], *cmd]))
        return
    tag = x.tag or 'serve-' + x.label + '-' + datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    pynvml.nvmlInit()
    r.monitoring.OUT = r.OUT
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    with (r.ROOT / 'logs/p7a/gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not list(r.OUT.glob('STOP-*.json'))
        r.monitoring.idle(tag + '-before')
        r.b.commands = lambda _: jobs
        c = r.p7.Cluster(x.label, r.OUT / tag)
        c.meta.update(config=str(x.config), commit=r.monitoring.git(r.MAIN, 'rev-parse', 'HEAD'))
        guard = r.Guard(tag)
        guard.start()
        try:
            c.start()
            while True:
                time.sleep(1)
                assert all(p.poll() is None for p in c.procs)
        except KeyboardInterrupt:
            c.meta['status'] = 'STOPPED'
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
        r.monitoring.idle(tag + '-after')


if __name__ == '__main__':
    main()
