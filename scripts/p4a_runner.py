#!/usr/bin/env python3
"""P4a PD-only validation and the single authorized sglang_pd_r2 replay."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import signal
import subprocess

import p3_runner as p3

ROOT = p3.ROOT
SOURCE = Path('/root/workspace/sglang-p4a')
PROFILE_PATH = SOURCE/'benchmark/pd_4090/p4a_profiles.json'
PROFILES = json.loads(PROFILE_PATH.read_text())['profiles']


def validate(cluster):
    reference_dir = ROOT/'logs/p4a/normal-text20'
    reference_meta = json.loads((reference_dir/'run.json').read_text())
    assert reference_meta['status'] == 'PASS'
    cases_path = ROOT/'logs/p4a/text20-cases.json'
    cases = json.loads(cases_path.read_text())
    expected = json.loads((reference_dir/'worker0-text20.json').read_text())
    rng = random.Random(730031)
    ids = [128000]+[rng.randrange(128000) for _ in range(31662)]
    outputs = []
    for i in range(2):
        if i:
            ids[1] = (ids[1]+1)%128000
        out = p3.generate(p3.BASE, ids)
        out['url'] = p3.BASE
        outputs.append(out)
        p3.save(cluster.path/'long-warmup.json', outputs)
        print(f'long warmup {i}: {out["elapsed_s"]:.2f}s / {len(ids)} tokens', flush=True)
    p3.flush(cluster)
    results = [p3.generate(p3.BASE, case['input_ids'], case['max_new_tokens']) for case in cases]
    p3.save(cluster.path/'correctness20.json', results)
    matches = [a['response']['text'] == b['response']['text'] and a['response']['output_ids'] == b['response']['output_ids'] for a,b in zip(results,expected)]
    p3.save(cluster.path/'correctness-comparison.json', {
        'matches': matches, 'passed': sum(matches), 'total': 20,
        'cases': str(cases_path), 'cases_sha256': hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        'reference': str(reference_dir/'worker0-text20.json'),
        'comparison': 'exact text and output_ids; fixed before testing PD; all four ordinary TP1 replicas agree',
    })
    assert len(results) == 20 and all(matches), f'PD output equality: {sum(matches)}/20'
    print('PD correctness: 20/20 exact text and output_ids matches', flush=True)
    p3.flush(cluster)


def replay(cluster, formal=False):
    label = 'sglang_pd_r2' if formal else 'precheck300'
    out = ROOT/'results' if formal else cluster.path/'precheck300'
    cmd = [p3.PYTHON, str(ROOT/'bench/bench_client.py'), '--trace', str(p3.TRACE),
           '--base-url', p3.BASE, '--label', label, '--out-dir', str(out),
           '--warmup', '2', '--force-close']
    if not formal:
        cmd += ['--limit', '300', '--time-scale', '1']
    else:
        with (out/f'{label}.formal-started.json').open('x') as f:
            json.dump({'started': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       'command': cmd, 'log_dir': str(cluster.path)}, f, indent=2)
    stage = 'formal' if formal else 'precheck300'
    p3.save(cluster.path/f'{stage}-command.json', cmd)
    print('RUN '+shlex.join(cmd), flush=True)
    with (cluster.path/f'{stage}-client.log').open('x') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            rc = proc.wait(timeout=4300)
        except BaseException:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
    summary = json.loads((out/f'{label}_summary.json').read_text())
    cluster.meta[stage+'_returncode'] = rc
    cluster.meta[stage+'_summary'] = summary
    p3.save(cluster.path/'run.json', cluster.meta)
    print(f'{label}: complete={summary["completed_requests"]}, fail={summary["failed_requests"]}, duration={summary["duration_sec"]:.2f}s', flush=True)
    assert summary['planned_requests'] == (1551 if formal else 300)
    if not formal:
        assert rc == 0 and summary['completed_requests'] == 300 and summary['failed_requests'] == 0 and summary['unsent_requests'] == 0, '300-request gate failed'
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['preflight', 'experiment', 'launch'])
    parser.add_argument('--profile', choices=PROFILES, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--gate', type=Path)
    args = parser.parse_args()
    # Inherited P3 tuning overrides must not silently change the comparison.
    assert not any(k.startswith('P3_') for k in os.environ), 'Unset P3_* overrides'
    assert not subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True).strip(), 'Isolated source tree must be clean'
    source_diff = subprocess.check_output(['git', '-C', str(SOURCE), 'diff', '--name-only', 'bd849d628cd6332defee24945ce71db3fa662f90', 'HEAD'], text=True).splitlines()
    assert source_diff == ['benchmark/pd_4090/p4a_profiles.json'], 'Inference source must match P3b exactly'
    source_commit = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    profile = PROFILES[args.profile]
    os.environ['P3_PD_BACKEND'] = profile['backend']
    original_commands = p3.commands

    def commands(label):
        jobs = original_commands(label)
        for name, cmd, env, port in jobs:
            if name != 'router':
                env.update(profile['env'])
                env['PYTHONPATH'] = str(SOURCE/'python')
            else:
                env['PYTHONPATH'] = str(SOURCE/'sgl-router/py_src')+':'+str(SOURCE/'python')
        return jobs

    p3.commands = commands
    jobs = commands('sglang_pd')
    fingerprint = hashlib.sha256(json.dumps(jobs, sort_keys=True).encode()).hexdigest()
    gate = None
    if args.mode == 'experiment':
        assert not (ROOT/'results/sglang_pd_r2.formal-started.json').exists(), 'Formal rerun already started'
        assert not list((ROOT/'results').glob('sglang_pd_r2_*')), 'Formal result already exists'
        assert args.gate, '--gate is required for experiment'
        gate = json.loads((args.gate/'run.json').read_text())
        assert gate['status'] == 'PASS' and gate['configuration_sha256'] == fingerprint
        assert gate['git_commit'] == source_commit
        assert gate['precheck300_summary']['completed_requests'] == 300
        assert gate['precheck300_summary']['failed_requests'] == 0
        assert json.loads((args.gate/'correctness-comparison.json').read_text())['passed'] == 20
        assert len(json.loads((args.gate/'long-warmup.json').read_text())) == 2
    cluster = p3.Cluster('sglang_pd', ROOT/'logs/p4a'/args.tag)
    cluster.meta.update(label='sglang_pd_r2' if args.mode == 'experiment' else 'sglang_pd_p4a_preflight',
                        profile=args.profile, configuration_sha256=fingerprint, gate=str(args.gate),
                        source_dir=str(SOURCE), profile_file=str(PROFILE_PATH),
                        git_commit=source_commit)

    def term(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, term)
    try:
        cluster.start()
        (cluster.path/'p4a_runner.snapshot.py').write_bytes(Path(__file__).read_bytes())
        validate(cluster)
        if args.mode == 'launch':
            signal.pause()
        elif args.mode == 'preflight':
            p3.flush(cluster)
            cluster.reset_router()
            replay(cluster)
        else:
            p3.bench(cluster, smoke=True)
            p3.flush(cluster)
            cluster.reset_router()
            replay(cluster, formal=True)
        cluster.meta['status'] = 'PASS'
    except BaseException as exc:
        cluster.meta.update(status='FAIL', error=repr(exc))
        raise
    finally:
        cluster.close()


if __name__ == '__main__':
    main()
