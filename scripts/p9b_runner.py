#!/usr/bin/env python3
"""P9b orchestration only. Frozen inference code, benchmark and selected parameters."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
from types import SimpleNamespace

import p3_runner as base
import p7a_runner as p7
import p9a_serve as v4

ROOT = base.ROOT
OUT = ROOT / 'logs/p9b'
FROZEN = ROOT / 'configs/p9b_frozen.json'
HEAD = 'f6833f24ef0587278ae799696cfe17c3ade99048'
LABELS = ['conv_muxwise_tp2dp2', 'conv_muxwise_tp4', 'conv_chunked_pp2dp2',
          'conv_chunked_tp2dp2', 'conv_muxpp_v4_pp2tp2', 'conv_muxpp_v4_pp4',
          'muxpp_v4_pp2tp2', 'muxpp_v4_pp4', 'muxpp_v4_pp2dp2']


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_gate():
    head = subprocess.check_output(['git', '-C', str(p7.MAIN), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(p7.MAIN), 'status', '--porcelain'], text=True)
    assert head == HEAD and not dirty, (head, dirty)
    assert not [k for k in os.environ if k.startswith(('P3_', 'SGLANG_PDMUX_PP_'))]


def mapping():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,name',
                                   '--format=csv,noheader'], text=True)
    rows = []
    for line in raw.strip().splitlines():
        index, uuid, bus, name = [x.strip() for x in line.split(',')]
        pci = '0000:' + bus.split(':', 1)[1].lower()
        numa = int(Path('/sys/bus/pci/devices', pci, 'numa_node').read_text())
        rows.append(dict(index=int(index), uuid=uuid, pci_bus_id=bus, numa=numa, name=name))
    assert [r['numa'] for r in rows] == [0, 0, 1, 1], rows
    return rows


def commands(label, gpu_map):
    selected = label.removeprefix('conv_')
    if selected.startswith('muxpp_v4_'):
        jobs = v4.commands(selected.removeprefix('muxpp_v4_'))
        # Use the exact v4 commands, with physical UUIDs replacing equivalent indices.
        for name, cmd, env, port in jobs:
            if name != 'router':
                env['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_map[int(i)]['uuid']
                    for i in env['CUDA_VISIBLE_DEVICES'].split(','))
    else:
        options = json.loads((ROOT / 'configs/p7a_selected.json').read_text())[selected]
        assert options['status'] == 'PASS'
        args = SimpleNamespace(label=selected, **{k: options[k] for k in
                               ['mem', 'attention', 'graph_bs', 'no_graph']})
        # p7 groups [1,2] / [0,3] now map to physical 0+1 / 2+3.
        p7.UUIDS = [gpu_map[i]['uuid'] for i in [2, 0, 1, 3]]
        jobs = p7.make_commands(args)
        old = json.loads((Path(options['gate']) / 'commands.json').read_text())
        for job, expected in zip(jobs, old):
            name, cmd, env, port = job
            assert (name, cmd, port) == (expected['name'], expected['argv'], expected['port'])
            assert {k:v for k,v in env.items() if k != 'CUDA_VISIBLE_DEVICES'} == {
                k:v for k,v in expected['env'].items() if k != 'CUDA_VISIBLE_DEVICES'}
    return jobs


def serial(jobs):
    return [dict(name=n, argv=c, env=e, port=p) for n,c,e,p in jobs]


def freeze():
    source_gate()
    OUT.mkdir(exist_ok=True)
    gpu_map = mapping()
    paths = [ROOT / p for p in ['env.sh', 'configs/p7a_selected.json', 'configs/p9a_selected.json',
        'configs/pdmux_4090.yml', 'configs/pdmux_pp_4090_v4.yml', 'scripts/p3_runner.py',
        'scripts/p7a_runner.py', 'scripts/p8a_runner.py', 'scripts/p9a_serve.py',
        'scripts/p9b_runner.py', 'scripts/run_experiment.sh', 'bench/bench_client.py',
        'bench/metrics.py', 'bench/common.py']]
    paths += sorted((ROOT / 'bench').glob('*_600_240_x2.5.jsonl'))
    paths += sorted((ROOT / 'bench').glob('dev_*_1200_240_x2.5.jsonl'))
    frozen = dict(commit=HEAD, created=now(), gpu_mapping=gpu_map, labels=LABELS,
        commands={label:serial(commands(label, gpu_map)) for label in LABELS},
        protected_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        prior_p7_commit=p7.COMMITS[p7.MAIN],
        note='P9b-only freeze update authorized by user; original P7/P8 selection records retained. '
             'Only physical GPU mapping changes; selected argv unchanged. No inference/benchmark edits.')
    with FROZEN.open('x') as f:
        json.dump(frozen, f, indent=2)
    base.save(OUT / 'historical-result-sha256.json', {str(p.relative_to(ROOT)):sha(p)
        for p in (ROOT / 'results').glob('*.json*')})
    (OUT / 'RESULTS-before-P9b.md').write_bytes((ROOT / 'RESULTS.md').read_bytes())
    (OUT / 'topology.txt').write_text(subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True))
    base.save(OUT / 'gpu-mapping.json', gpu_map)
    print('FROZEN', HEAD, flush=True)


def replay(cluster, kind, workload):
    formal = kind == 'formal'
    trace = ROOT / 'bench' / (f'{workload}_600_240_x2.5.jsonl' if formal else
                             f'dev_{workload}_1200_240_x2.5.jsonl')
    out = ROOT / 'results' if formal else cluster.path / 'smoke'
    label = cluster.label if formal else cluster.label + '_smoke'
    cmd = [base.PYTHON, str(ROOT / 'bench/bench_client.py'), '--trace', str(trace),
           '--base-url', base.BASE, '--label', label, '--out-dir', str(out),
           '--warmup', '2', '--force-close', '--drain-timeout', '300', '--progress-interval', '30']
    if not formal:
        cmd += ['--limit', '20', '--time-scale', '0.2', '--timeout', '55']
    else:
        with (out / f'{label}.formal-started.json').open('x') as f:
            json.dump(dict(command=cmd, log_dir=str(cluster.path), started=now(), commit=HEAD), f, indent=2)
    base.save(cluster.path / f'{kind}-command.json', cmd)
    print('RUN ' + shlex.join(cmd), flush=True)
    wall_start = time.monotonic()
    with (cluster.path / f'{kind}-client.log').open('x') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            rc = proc.wait(timeout=1100 if formal else 60)
        except BaseException:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
            raise
    summary = json.loads((out / f'{label}_summary.json').read_text())
    cluster.meta[kind + '_summary'] = summary
    cluster.meta[kind + '_returncode'] = rc
    cluster.meta[kind + '_wall_sec'] = time.monotonic() - wall_start
    base.save(cluster.path / 'run.json', cluster.meta)
    if formal:
        assert summary['planned_requests'] == (735 if workload == 'conversation' else 1551)
    else:
        assert rc == 0 and summary['completed'] == 20 and cluster.meta[kind + '_wall_sec'] <= 60, 'smoke gate failed'
    print(f'{kind}: {summary["completed"]}/{summary["unfinished"]}/{summary["failed"]} '
          f'SLO={summary["slo"]["ttft_le_5s_and_tpot_le_100ms"]["rate"]:.6%}', flush=True)


def experiment(label):
    lock = (ROOT / 'logs/p7a/gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source_gate()
    frozen = json.loads(FROZEN.read_text())
    assert mapping() == frozen['gpu_mapping']
    for path, digest in frozen['protected_sha256'].items():
        assert sha(ROOT / path) == digest, path
    jobs = commands(label, frozen['gpu_mapping'])
    assert serial(jobs) == frozen['commands'][label]
    assert not list((ROOT / 'results').glob(label + '*')), 'Formal data/marker already exists'
    with (OUT / f'{label}.launch-started.json').open('x') as f:
        json.dump(dict(label=label, started=now(), commit=HEAD), f, indent=2)
    base.commands = lambda _: jobs
    # Reuse P7b long-warmup implementation for all model replicas.
    p7.MATRIX[label] = ('muxpp' if 'muxpp_' in label else 'muxwise' if 'muxwise_' in label else 'chunked',
                       0, 0, sum(n != 'router' for n,_,_,_ in jobs))
    c = p7.Cluster(label, OUT / label)
    workload = 'conversation' if label.startswith('conv_') else 'toolagent'
    c.meta.update(mode='experiment', commit=HEAD, source=str(p7.MAIN), workload=workload,
                  frozen_file=str(FROZEN), frozen_sha256=sha(FROZEN), gpu_mapping=frozen['gpu_mapping'],
                  formal_launch_count=1, restart_count=0)
    def stop(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, stop)
    try:
        c.start()
        p7.warmup(c)
        replay(c, 'smoke', workload)
        if 'muxwise_' in label or 'muxpp_' in label:
            base.flush(c)
            base.save(c.path / 'long-warmup-after-smoke-cache-flush.json', {'flushed':True})
            p7.warmup(c, 'long-warmup-after-smoke.json')
        base.flush(c)
        c.reset_router()
        (c.path / 'topology-before-formal.txt').write_text(subprocess.check_output(['nvidia-smi','topo','-m'], text=True))
        (c.path / 'gpu-rank-processes.csv').write_text(subprocess.check_output(['nvidia-smi',
            '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv'], text=True))
        replay(c, 'formal', workload)
        c.meta['status'] = 'PASS' if c.meta['formal_returncode'] == 0 else 'FORMAL_ERRORS_WITH_DATA'
    except BaseException as exc:
        c.meta.update(status='FAIL', error=repr(exc))
        raise
    finally:
        c.close()


def batch():
    with (OUT / 'batch-started.json').open('x') as f:
        json.dump(dict(started=now(), labels=LABELS), f, indent=2)
    with (OUT / 'progress.md').open('x') as f:
        f.write('# P9b 正式进度（UTC）\n\n| label | 开始 | 结束 | 完成/未完/失败 | 主SLO | rc |\n'
                '| --- | --- | --- | --- | --- | --- |\n')
    for label in LABELS:
        base.clean_gpu()
        started = now()
        cmd = ['bash', str(ROOT / 'scripts/run_experiment.sh'), label]
        base.save(OUT / 'current.json', dict(label=label, started=started, command=cmd))
        print('START', label, started, flush=True)
        with (OUT / f'{label}-runner.log').open('x') as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
        summary_path = ROOT / 'results' / f'{label}_summary.json'
        if not summary_path.exists():
            raise RuntimeError(f'{label} no formal data, rc={rc}; manual startup-only diagnosis required')
        s = json.loads(summary_path.read_text())
        counts = '/'.join(str(s[k]) for k in ['completed','unfinished','failed'])
        slo = s['slo']['ttft_le_5s_and_tpot_le_100ms']['rate']
        with (OUT / 'progress.md').open('a') as f:
            f.write(f'| {label} | {started} | {now()} | {counts} | {slo:.6%} | {rc} |\n')
        print('END', label, counts, f'{slo:.6%}', flush=True)
        base.clean_gpu()
    base.save(OUT / 'batch-finished.json', dict(finished=now()))
    base.save(OUT / 'current.json', dict(status='finished', finished=now()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['freeze','batch','experiment'])
    p.add_argument('label', nargs='?', choices=LABELS)
    a = p.parse_args()
    {'freeze':freeze, 'batch':batch, 'experiment':lambda:experiment(a.label)}[a.mode]()
