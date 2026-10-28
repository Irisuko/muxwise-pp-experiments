#!/usr/bin/env python3
"""Launch, print, or smoke-check the frozen v4 configuration. No trace replay."""
import argparse
import concurrent.futures
import datetime
import json
import shlex
import signal
import subprocess
from pathlib import Path

import p3_runner as base
from p8a_runner import commands as v3_commands

ROOT = base.ROOT
SOURCE = Path('/root/workspace/sglang')


def commands(topology):
    selected = json.loads((ROOT / 'configs/p9a_selected.json').read_text())
    tp, pp, replicas = {'pp2tp2': (2, 2, 1), 'pp4': (1, 4, 1),
                        'pp2dp2': (1, 2, 2)}[topology]
    jobs = []
    for i in range(replicas):
        _, cmd, env, _ = v3_commands()[0]
        port = 30000 if replicas == 1 else 31000 + i
        for key, value in [('--tp', tp), ('--pp-size', pp), ('--port', port),
                           ('--pdmux-pp-max-inflight-prefill', 6 if pp == 4 else 3),
                           ('--pdmux-config-path', ROOT / 'configs/pdmux_pp_4090_v4.yml')]:
            cmd[cmd.index(key) + 1] = str(value)
        if tp == 1:
            cmd.remove('--disable-custom-all-reduce')
        cmd += selected['flags']
        env.update(CUDA_VISIBLE_DEVICES=','.join(str(x) for x in range(i*tp*pp, (i+1)*tp*pp)),
                   PYTHONPATH=str(SOURCE / 'python'))
        jobs.append((f'worker{i}', cmd, env, port))
    if replicas > 1:
        jobs.append(('router', [base.PYTHON, '-m', 'sglang_router.launch_router',
            '--host', '127.0.0.1', '--port', '30000', '--policy', 'cache_aware',
            '--request-timeout-secs', '3600', '--max-concurrent-requests', '4096',
            '--log-level', 'info', '--worker-urls', 'http://127.0.0.1:31000',
            'http://127.0.0.1:31001'], {}, 30000))
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['commands', 'launch', 'smoke'])
    p.add_argument('topology', choices=['pp2tp2', 'pp4', 'pp2dp2'])
    p.add_argument('--tag')
    a = p.parse_args()
    jobs = commands(a.topology)
    if a.mode == 'commands':
        print('source /root/workspace/muxpp_work/env.sh')
        for _, cmd, env, _ in jobs:
            print(shlex.join(['env'] + [k+'='+v for k,v in env.items()] + cmd))
        return
    base.commands = lambda _: jobs
    tag = a.tag or f'{a.mode}-{a.topology}-' + datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    c = base.Cluster(a.topology, ROOT / 'logs/p9a' / tag)
    c.meta['commit'] = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    def stop(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, stop)
    try:
        c.start()
        if a.mode == 'launch':
            signal.pause()
        else:
            # Exercise every PP2 replica directly, then concurrent router traffic.
            outputs = []
            for name, _, _, port in jobs:
                if name != 'router':
                    outputs.append(base.generate(f'http://127.0.0.1:{port}',
                                                 [128000] + [791, 1401, 374, 264, 1296, 13]*100, 32))
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                outputs += list(pool.map(lambda n: base.generate(base.BASE,
                    [128000] + [791, 1401, 374, 264, 1296, 13]*n, 64), [20, 150, 600, 900, 1200, 100]))
            base.save(c.path / 'smoke.json', outputs)
            base.flush(c)
            c.meta['status'] = 'PASS'
    except KeyboardInterrupt:
        c.meta['status'] = 'STOPPED'
    except BaseException as e:
        c.meta.update(status='FAIL', error=repr(e))
        raise
    finally:
        c.close()


if __name__ == '__main__':
    main()
