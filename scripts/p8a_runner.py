#!/usr/bin/env python3
"""PP2 x TP2 lifecycle. Development is the default; formal replay is one-shot."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import signal
import subprocess
import time

import p3_runner as base

ROOT = base.ROOT
SOURCE = Path('/root/workspace/sglang')
LABEL = 'muxpp_pp2tp2'
DEV = ROOT / 'bench/dev_toolagent_1200_240_x2.5.jsonl'
SELECTED = ROOT / 'configs/p8a_selected.json'
CONFIG = ROOT / 'configs/pdmux_pp_4090_v3.yml'


def commands():
    script = (ROOT / 'scripts/p5b_v3_serve.sh').read_text()
    cmd = [base.PYTHON] + shlex.split(script.split('exec python ', 1)[1].replace('\\\n', ' '))
    for key, value in (('--tp', 2), ('--pp-size', 2), ('--pdmux-pp-max-inflight-prefill', 3)):
        cmd[cmd.index(key) + 1] = str(value)
    cmd += ['--disable-custom-all-reduce', '--attention-backend', 'flashinfer']
    return [('pp2tp2', cmd, {'CUDA_VISIBLE_DEVICES': '0,1,2,3',
                           'PYTHONPATH': str(SOURCE / 'python')}, 30000)]


def signature():
    return {'jobs': commands(), 'config_sha256': hashlib.sha256(CONFIG.read_bytes()).hexdigest()}


def assert_unused_formal(root=ROOT):
    for directory in ('results', 'archive/superseded/results'):
        path = root / directory
        assert not (path / f'{LABEL}.formal-started.json').exists(), 'Formal replay already started; no retry'
        assert not list(path.glob(f'{LABEL}_*.json*')), 'Formal results already exist; no retry'


def replay(cluster, kind):
    formal = kind == 'formal'
    out = ROOT / 'results' if formal else cluster.path / kind
    label = LABEL if formal else f'{LABEL}_{kind}'
    cmd = [base.PYTHON, str(ROOT / 'bench/bench_client.py'), '--trace', str(base.TRACE if formal else DEV),
           '--base-url', base.BASE, '--label', label, '--out-dir', str(out),
           '--warmup', '2', '--force-close', '--drain-timeout', '300', '--progress-interval', '15']
    if not formal:
        cmd += ['--limit', '20' if kind == 'smoke' else '300',
                '--time-scale', '0.2' if kind == 'smoke' else '1']
    marker = (ROOT / 'results' / f'{LABEL}.formal-started.json' if formal else
              ROOT / 'logs/p8a' / f'{LABEL}.dev300-started.json' if kind == 'dev300' else None)
    if marker:
        with marker.open('x') as f:
            json.dump({'command': cmd, 'log_dir': str(cluster.path),
                       'started': datetime.datetime.now(datetime.timezone.utc).isoformat()}, f, indent=2)
    base.save(cluster.path / f'{kind}-command.json', cmd)
    with (cluster.path / f'{kind}-client.log').open('x') as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        try:
            rc = proc.wait(timeout=1100)
        except BaseException:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
    summary = json.loads((out / f'{label}_summary.json').read_text())
    cluster.meta[kind] = summary
    base.save(cluster.path / 'run.json', cluster.meta)
    assert rc == 0 and summary['failed'] == 0 and summary['unfinished'] == 0, summary
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['launch', 'preflight', 'sanity', 'experiment', 'commands'])
    parser.add_argument('label', choices=[LABEL], nargs='?', default=LABEL)
    parser.add_argument('--tag')
    args = parser.parse_args()
    if args.mode == 'commands':
        print('\n'.join(shlex.join(['env'] + [k+'='+v for k,v in env.items()] + cmd)
                        for _,cmd,env,_ in commands()))
        return
    if args.mode == 'experiment':
        assert_unused_formal()
    lock = (ROOT / 'logs/p7a/gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    commit = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True)
    if args.mode == 'experiment':
        selected = json.loads(SELECTED.read_text())
        assert not dirty and selected['commit'] == commit, 'Frozen tested source required'
        assert selected['signature'] == json.loads(json.dumps(signature())), 'Launch/config changed since validation'
        gate = json.loads(Path(selected['gate']).read_text())
        assert gate['status'] == 'PASS' and gate['commit'] == commit and not gate['dirty']
        assert gate['dev300']['completed'] == 300, 'Development gate incomplete'
    path = ROOT / 'logs/p8a' / (args.tag or args.mode + '-' + datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
    base.commands = lambda _: commands()
    cluster = base.Cluster(LABEL, path)
    cluster.meta.update(mode=args.mode, commit=commit, dirty=dirty, signature=signature())
    def stop(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, stop)
    try:
        cluster.start()
        if args.mode == 'launch':
            signal.pause()
        else:
            rng = random.Random(730031)
            ids = [128000] + [rng.randrange(128000) for _ in range(31662)]
            base.save(path / 'long-warmup.json', base.generate(base.BASE, ids))
            base.flush(cluster)
            replay(cluster, 'smoke')
            base.flush(cluster)
            base.save(path / 'long-warmup-after-smoke.json', base.generate(base.BASE, ids))
            base.flush(cluster)
            if args.mode in ('sanity', 'experiment'):
                replay(cluster, 'dev300' if args.mode == 'sanity' else 'formal')
            time.sleep(3)
            cluster.meta['status'] = 'PASS'
    except BaseException as exc:
        cluster.meta.update(status='FAIL', error=repr(exc))
        raise
    finally:
        cluster.close()


if __name__ == '__main__':
    main()
