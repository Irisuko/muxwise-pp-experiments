#!/usr/bin/env python3
"""P11a development-only capacity/SLO matrix. No formal replay entrypoint."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time

import pynvml
import p3_runner as b
import p7a_runner as p7
import p9a_serve as llama
import p9b_runner as previous
import p10a_runner as big
import p10b_serve as selected
import p10c_runner as monitoring
from p10c_resume import Guard

ROOT = b.ROOT
OUT = ROOT / 'logs/p11a'
BASELINE = Path('/root/workspace/sglang-p11a-baseline')
MAIN = Path('/root/workspace/sglang')
LABELS = ['llama_pp4', 'llama_pp2tp2', 'big_pp2tp2', 'big_pp4',
          'llama_muxwise_tp2dp2', 'big_muxwise_tp4']
TRACES = {'llama': ROOT/'bench/dev_toolagent_1200_240_x2.5.jsonl',
          'big': ROOT/'bench/dev_big_toolagent_1200_80_x7.5.jsonl'}


def commands(label, source, path, extra):
    if label.startswith('llama_muxwise'):
        jobs = previous.commands('muxwise_tp2dp2', previous.mapping())
    elif label == 'big_muxwise_tp4':
        jobs = selected.commands(label)
    elif label.startswith('llama_'):
        jobs = llama.commands(label.removeprefix('llama_'))
    else:
        jobs = selected.commands(label.replace('big_', 'big_muxpp_'))
    for name, cmd, env, _ in jobs:
        if name == 'router':
            continue
        env['PYTHONPATH'] = str(source/'python')
        if 'muxwise' not in label:
            cmd += ['--pdmux-pp-request-trace', str(path/f'requests-timeline-{name}')]
            env['SGLANG_PDMUX_PP_ROUND_TRACE'] = str(path/f'decode-rounds-{name}.jsonl')
            cmd += extra
    return jobs


def warmup(c, label, rate):
    b.flush(c)
    if label.startswith('big_'):
        big.warmup(c)
    else:
        rng = random.Random(730031)
        ids = [128000] + [rng.randrange(128000) for _ in range(31662)]
        results = []
        for name, _, _, port in b.commands(c.label):
            if name != 'router':
                results.append(b.generate(f'http://127.0.0.1:{port}', ids))
        b.save(c.path/'long-warmup.json', results)
        b.flush(c)
    (c.path/'long-warmup.json').rename(c.path/f'long-warmup-{rate}.json')
    c.reset_router()


def replay(c, label, rate, scale):
    family = label.split('_')[0]
    trace = TRACES[family]
    assert trace.resolve() in [p.resolve() for p in TRACES.values()]
    limit = 150 if family == 'big' else 300
    out = c.path/rate
    cmd = [b.PYTHON, str(ROOT/'bench/bench_client.py'), '--trace', str(trace),
           '--base-url', b.BASE, '--label', rate, '--out-dir', str(out),
           '--limit', str(limit), '--time-scale', str(scale), '--warmup', '2',
           '--force-close', '--drain-timeout', '300', '--progress-interval', '30']
    b.save(c.path/f'{rate}-client-command.json', cmd)
    print('REPLAY', label, rate, flush=True)
    with (c.path/f'{rate}-client.log').open('x') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            rc = proc.wait(timeout=1100)
        except BaseException:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
            raise
    s = json.loads((out/f'{rate}_summary.json').read_text())
    c.meta[rate] = s
    b.save(c.path/'run.json', c.meta)
    print('RESULT', label, rate, json.dumps({k:s[k] for k in
          ('completed', 'failed', 'unfinished', 'duration_sec', 'slo')}), flush=True)
    # Preserve incomplete capacity results, but stop on server/client errors.
    assert s['planned_requests'] == limit
    assert s['failed'] == 0 and all(p.poll() is None for p in c.procs), (rc, s)
    if s['unfinished']:
        raise RuntimeError('Capacity run did not drain; preserve result and stop sequence')
    assert rc == 0
    time.sleep(3)


def experiment(round_name, label, source, extra, rates=None):
    tag = round_name+'-'+label
    monitoring.idle(tag+'-before')
    path = OUT/tag
    jobs = commands(label, source, path, extra)
    b.commands = lambda _: jobs
    c = p7.Cluster(label, path)
    c.meta.update(round=round_name, source=str(source), commit=monitoring.git(source,'rev-parse','HEAD'),
                  gpu_mapping=previous.mapping(),
                  traces={str(p):monitoring.sha(p) for p in TRACES.values()})
    c.meta['runner_sha256']=monitoring.sha(Path(__file__))
    c.meta['pdmux_configs']={str(Path(cmd[i+1])):Path(cmd[i+1]).read_text()
        for _,cmd,_,_ in jobs for i,key in enumerate(cmd) if key=='--pdmux-config-path'}
    guard = Guard(tag)
    guard.start()
    print('START', tag, monitoring.now(), flush=True)
    try:
        c.start()
        (path/'source.patch').write_text(monitoring.git(source,'diff','HEAD','--binary'))
        for rate, scale in [('standard',1), ('double',.5), ('saturated',.05)]:
            if rates and rate not in rates:continue
            warmup(c, label, rate)
            replay(c, label, rate, scale)
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
    monitoring.idle(tag+'-after')
    print('END', tag, monitoring.now(), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--round', required=True)
    p.add_argument('--labels', nargs='+', choices=LABELS)
    p.add_argument('--source', type=Path)
    p.add_argument('--config', type=Path, help='Per-label extra flags for this round')
    p.add_argument('--rates', nargs='+', choices=['standard','double','saturated'])
    p.add_argument('--extra', nargs=argparse.REMAINDER, default=[])
    x = p.parse_args()
    OUT.mkdir(exist_ok=True)
    monitoring.OUT = OUT
    pynvml.nvmlInit()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    source = x.source or (BASELINE if x.round == 'baseline' else MAIN)
    labels = x.labels or (LABELS if x.round == 'baseline' else LABELS[:4])
    with (ROOT/'logs/p7a/gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if list(OUT.glob('STOP-*.json')):
            raise RuntimeError('External occupancy detected; no further GPU runs')
        if x.source is None and x.round != 'baseline':
            assert not monitoring.git(MAIN,'status','--porcelain'), 'Commit source before a performance round'
            source=MAIN.parent/('sglang-p11a-'+x.round)
            assert not source.exists(), source
            subprocess.run(['git','-C',str(MAIN),'worktree','add','--detach',str(source),'HEAD'],check=True)
        if x.round == 'baseline':
            assert monitoring.git(source,'rev-parse','HEAD').startswith('4c10135d2')
            assert not monitoring.git(source,'status','--porcelain')
        for label in labels:
            flags=json.loads(x.config.read_text())['flags'][label] if x.config else []
            experiment(x.round, label, source, flags+x.extra, x.rates)
        monitoring.idle(x.round+'-final')


if __name__ == '__main__':
    main()
